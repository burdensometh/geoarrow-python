"""Explicit axis-order validation and repair for GeoArrow geometry.

Some data sources declare a single coordinate reference system (CRS) and
mostly honour it but occasionally write a batch of coordinates with the x
and y ordinates reversed (for example latitude/longitude instead of
longitude/latitude).  Blindly swapping every batch is dangerous because a
batch that is already correct would be corrupted.  The helpers in this
module make the decision explicit by comparing both the original and the
swapped geometry against an expected business extent
``(xmin, ymin, xmax, ymax)``.

The workflow is deliberately split into a validation mode and a correction
mode:

* :func:`check_axis_order` is the validation mode.  It returns one record
  per feature classifying it as ``as_is``, ``swap``, ``ambiguous``,
  ``outside``, or ``empty``.
* :func:`axis_order_evidence` aggregates those records into a batch
  verdict (``as_is``, ``swap``, or ``pending``) with explicit reasons.
* :func:`fix_axis_order` is the correction mode.  It only rewrites
  geometries when the evidence is consistent and unique.  By default it
  works per feature (``as_is`` stays, ``swap`` is swapped) which suits a
  source that reverses a few coordinates within an otherwise-correct
  batch; features for which both orientations fit (``ambiguous``) or
  neither fits (``outside``) are left untouched and reported as pending
  confirmation.  A conservative ``batch`` mode is also available.  When
  given a :class:`pyarrow.Table`, the corrected data is written back to
  the standard geometry column.
* :func:`swap_axes` is the unconditional building block used by the
  correction mode.

GeoArrow arrays always store ordinates in x, y order (like GeoParquet), so
the expected extent is also interpreted as ``(xmin, ymin, xmax, ymax)``
regardless of what the CRS axis order declares.  Use
:func:`geoarrow.types.crs_axis_order` if the declared axis order itself is
needed for diagnostics.
"""

import math
import struct
from collections.abc import Mapping
from typing import NamedTuple, Optional

import pyarrow as pa
import pyarrow.compute as pc

from geoarrow.pyarrow import _compute, _type
from geoarrow.types import crs as _crs
from geoarrow.types.type_pyarrow import BoxType

_EXTENT_NAMES = ("xmin", "ymin", "xmax", "ymax")


class AxisOrderStatus:
    """Constants describing the axis-order status of a single feature.

    Examples
    --------
    >>> from geoarrow.pyarrow import AxisOrderStatus
    >>> AxisOrderStatus.SWAP
    'swap'
    """

    AS_IS = "as_is"
    """The original coordinates fit the expected extent and the swapped
    coordinates do not."""

    SWAP = "swap"
    """The swapped coordinates fit the expected extent and the original
    coordinates do not."""

    AMBIGUOUS = "ambiguous"
    """Both the original and the swapped coordinates fit the expected
    extent (both orientations are possible)."""

    OUTSIDE = "outside"
    """Neither the original nor the swapped coordinates fit the expected
    extent (neither orientation is reasonable)."""

    EMPTY = "empty"
    """The feature has no evaluable coordinates (null, empty, or
    non-finite bounds)."""


class AxisOrderVerdict:
    """Constants for a batch-level axis-order verdict.

    Examples
    --------
    >>> from geoarrow.pyarrow import AxisOrderVerdict
    >>> AxisOrderVerdict.PENDING
    'pending'
    """

    AS_IS = "as_is"
    """Every evaluable feature uniquely fits the original orientation."""

    SWAP = "swap"
    """Every evaluable feature uniquely fits the swapped orientation."""

    PENDING = "pending"
    """The evidence is not consistent and unique; confirmation is needed."""


class AxisOrderEvidence(NamedTuple):
    """Aggregate evidence collected from a batch of geometries.

    Attributes
    ----------
    extent : tuple
        The expected extent as ``(xmin, ymin, xmax, ymax)``.
    verdict : str
        One of :class:`AxisOrderVerdict`.
    n_total : int
        Total number of features (including nulls and empties).
    n_as_is, n_swap, n_ambiguous, n_outside, n_empty : int
        Counts for each :class:`AxisOrderStatus` value.
    pending_reasons : tuple of str
        Empty when the verdict is not ``"pending"``.  Otherwise contains
        one or more of ``"ambiguous"``, ``"outside"``, ``"inconsistent"``,
        or ``"no_evaluable_features"``.
    crs : Crs or None
        The CRS declared by the geometry type, if any.
    crs_axis_order : tuple of str or None
        The axis abbreviations (or directions) declared by the CRS.
    """

    extent: tuple
    verdict: str
    n_total: int
    n_as_is: int
    n_swap: int
    n_ambiguous: int
    n_outside: int
    n_empty: int
    pending_reasons: tuple
    crs: Optional[object]
    crs_axis_order: Optional[tuple]

    @property
    def n_pending(self):
        """Number of features that still need confirmation."""
        return self.n_ambiguous + self.n_outside

    @property
    def is_pending(self):
        """True if the batch verdict requires confirmation (i.e. the batch
        is not uniformly ``as_is`` or ``swap``)."""
        return self.verdict == AxisOrderVerdict.PENDING


class AxisOrderFix(NamedTuple):
    """Result of :func:`fix_axis_order`.

    Attributes
    ----------
    data : pyarrow.Array, pyarrow.ChunkedArray, or pyarrow.Table
        The (possibly) corrected data.
    status : pyarrow.Array or pyarrow.ChunkedArray
        Per-feature :class:`AxisOrderStatus` records for the input data.
    evidence : AxisOrderEvidence
        Batch-level evidence.
    changed : bool
        True if any geometry was rewritten.
    pending : bool
        True if the batch verdict is ``"pending"``.
    """

    data: object
    status: object
    evidence: AxisOrderEvidence
    changed: bool
    pending: bool


_AXIS_ORDER_STATUS_TYPE = pa.struct(
    [
        pa.field("status", pa.string()),
        pa.field("original_in_extent", pa.bool_()),
        pa.field("swapped_in_extent", pa.bool_()),
        pa.field("pending", pa.bool_()),
    ]
)


def axis_order_status_type():
    """The pyarrow struct type returned by :func:`check_axis_order`."""
    return _AXIS_ORDER_STATUS_TYPE


def _normalize_extent(extent):
    if hasattr(extent, "as_py") and not isinstance(extent, (str, bytes)):
        extent = extent.as_py()

    if isinstance(extent, Mapping):
        try:
            values = [extent[name] for name in _EXTENT_NAMES]
        except KeyError as exc:
            raise ValueError(
                f"Expected extent to contain {_EXTENT_NAMES} but it is missing {exc}"
            ) from None
    elif (
        not isinstance(extent, (str, bytes))
        and hasattr(extent, "__len__")
        and hasattr(extent, "__getitem__")
        and len(extent) == 4
    ):
        values = [extent[i] for i in range(4)]
    else:
        raise TypeError(
            "Expected extent to be a (xmin, ymin, xmax, ymax) sequence or a "
            f"mapping with those keys, got {extent!r}"
        )

    try:
        values = [float(value) for value in values]
    except (TypeError, ValueError) as exc:
        raise TypeError(
            "Expected extent bounds to be numeric, got "
            f"{extent!r}"
        ) from exc

    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"Expected extent bounds to be finite, got {extent!r}")

    xmin, ymin, xmax, ymax = values
    if xmin > xmax:
        raise ValueError(
            "Expected extent xmin <= xmax, got "
            f"({xmin}, {ymin}, {xmax}, {ymax})"
        )
    if ymin > ymax:
        raise ValueError(
            "Expected extent ymin <= ymax, got "
            f"({xmin}, {ymin}, {xmax}, {ymax})"
        )

    return (xmin, ymin, xmax, ymax)


def _chunks(obj):
    if isinstance(obj, pa.ChunkedArray):
        return obj.chunks
    return [obj]


def _rebuild_like(obj, chunks):
    if isinstance(obj, pa.ChunkedArray):
        return pa.chunked_array(chunks, type=obj.type)
    return chunks[0]


def _normalize_array(arr):
    """Return a zero-offset array (a cheap copy only when sliced)."""
    if arr.offset != 0:
        return pa.concat_arrays([arr])
    return arr


def _abbreviate(value, limit=60):
    value = str(value)
    if len(value) > limit:
        return value[: limit - 3] + "..."
    return value


def _crs_match(lhs, rhs):
    if lhs is None or rhs is None:
        return lhs is rhs
    if hasattr(lhs, "to_json_dict") and hasattr(rhs, "to_json_dict"):
        return lhs.to_json_dict() == rhs.to_json_dict()
    return lhs == rhs


def _validate_crs(type_, expected_crs, require_crs):
    actual = getattr(type_, "crs", None)
    if require_crs and actual is None:
        raise ValueError(
            "Expected the geometry type to declare a CRS, but it declares none"
        )

    if expected_crs is None:
        return None

    expected = _crs.create(expected_crs)
    if not _crs_match(actual, expected):
        raise ValueError(
            "Expected the geometry type to declare CRS "
            f"{_abbreviate(expected)} but it declares {_abbreviate(actual)}"
        )

    return expected


def _crs_axis_order_or_none(crs):
    if crs is None:
        return None
    try:
        return _crs.crs_axis_order(crs)
    except (ValueError, TypeError):
        return None


def _box_storage(obj):
    """Compute the bounding-box storage for ``obj`` as a struct array."""
    return _compute.ensure_storage(_compute.box(obj))


def _evaluable_mask(storage):
    evaluable = pc.is_valid(storage)
    for name in _EXTENT_NAMES:
        evaluable = pc.and_kleene(
            evaluable, pc.is_finite(pc.struct_field(storage, name))
        )
    return pc.fill_null(evaluable, False)


def _in_extent(storage, extent):
    xmin, ymin, xmax, ymax = extent
    return pc.and_kleene(
        pc.and_kleene(
            pc.greater_equal(pc.struct_field(storage, "xmin"), xmin),
            pc.greater_equal(pc.struct_field(storage, "ymin"), ymin),
        ),
        pc.and_kleene(
            pc.less_equal(pc.struct_field(storage, "xmax"), xmax),
            pc.less_equal(pc.struct_field(storage, "ymax"), ymax),
        ),
    )


def _swapped_in_extent(storage, extent):
    # Swapping x and y turns the box into (ymin, xmin, ymax, xmax).
    xmin, ymin, xmax, ymax = extent
    return pc.and_kleene(
        pc.and_kleene(
            pc.greater_equal(pc.struct_field(storage, "ymin"), xmin),
            pc.greater_equal(pc.struct_field(storage, "xmin"), ymin),
        ),
        pc.and_kleene(
            pc.less_equal(pc.struct_field(storage, "ymax"), xmax),
            pc.less_equal(pc.struct_field(storage, "ymin"), ymax),
        ),
    )


def _status_for_storage(storage, extent):
    evaluable = _evaluable_mask(storage)
    original = pc.fill_null(_in_extent(storage, extent), False)
    swapped = pc.fill_null(_swapped_in_extent(storage, extent), False)

    status = pc.if_else(
        evaluable,
        pc.if_else(
            pc.and_kleene(original, swapped),
            pa.scalar(AxisOrderStatus.AMBIGUOUS),
            pc.if_else(
                original,
                pa.scalar(AxisOrderStatus.AS_IS),
                pc.if_else(
                    swapped,
                    pa.scalar(AxisOrderStatus.SWAP),
                    pa.scalar(AxisOrderStatus.OUTSIDE),
                ),
            ),
        ),
        pa.scalar(AxisOrderStatus.EMPTY),
    )

    pending = pc.or_(
        pc.equal(status, AxisOrderStatus.AMBIGUOUS),
        pc.equal(status, AxisOrderStatus.OUTSIDE),
    )

    return pa.StructArray.from_arrays(
        [status, original, swapped, pending], type=_AXIS_ORDER_STATUS_TYPE
    )


def _status_for_array(obj, extent):
    storage = _box_storage(obj)
    chunks = [
        _status_for_storage(chunk, extent) for chunk in _chunks(storage)
    ]
    if isinstance(storage, pa.ChunkedArray):
        return pa.chunked_array(chunks, type=_AXIS_ORDER_STATUS_TYPE)
    return chunks[0]


def _geometry_column_name(obj, geometry_column):
    if not isinstance(obj, pa.Table):
        raise TypeError("Expected a pyarrow.Table")

    schema = obj.schema
    if geometry_column is not None:
        index = schema.get_field_index(geometry_column)
        if index < 0:
            raise ValueError(
                f"Geometry column {geometry_column!r} not found in table"
            )
        if not isinstance(schema.field(index).type, pa.ExtensionType):
            raise ValueError(
                f"Column {geometry_column!r} is not a GeoArrow extension type"
            )
        return geometry_column

    if "geometry" in schema.names:
        index = schema.get_field_index("geometry")
        if isinstance(schema.field(index).type, pa.ExtensionType):
            return "geometry"

    if "geography" in schema.names:
        index = schema.get_field_index("geography")
        if isinstance(schema.field(index).type, pa.ExtensionType):
            return "geography"

    geometry_columns = [
        name
        for name, type_ in zip(schema.names, schema.types)
        if isinstance(type_, pa.ExtensionType)
    ]
    if len(geometry_columns) == 0:
        raise ValueError(
            "Table has no geometry column; pass `geometry_column` explicitly"
        )
    if len(geometry_columns) > 1:
        joined = ", ".join(repr(name) for name in geometry_columns)
        raise ValueError(
            "Table has more than one geometry column "
            f"({joined}); pass `geometry_column` explicitly"
        )
    return geometry_columns[0]


def _table_geometry(obj, geometry_column):
    name = _geometry_column_name(obj, geometry_column)
    return name, obj.column(name)


def _set_table_column(obj, name, column):
    index = obj.schema.get_field_index(name)
    old_field = obj.schema.field(index)
    new_field = pa.field(
        old_field.name,
        column.type,
        nullable=old_field.nullable,
        metadata=old_field.metadata,
    )
    schema = obj.schema.set(index, new_field)
    arrays = [
        column if i == index else obj.column(i) for i in range(obj.num_columns)
    ]
    return pa.Table.from_arrays(arrays, schema=schema)


def _count_statuses(status):
    counts = {
        AxisOrderStatus.AS_IS: 0,
        AxisOrderStatus.SWAP: 0,
        AxisOrderStatus.AMBIGUOUS: 0,
        AxisOrderStatus.OUTSIDE: 0,
        AxisOrderStatus.EMPTY: 0,
    }
    status_strings = pc.struct_field(status, "status")
    for record in pc.value_counts(status_strings).to_pylist():
        counts[record["values"]] = record["counts"]
    return counts


def _evidence_from_status(status, extent, crs):
    counts = _count_statuses(status)
    n_as_is = counts[AxisOrderStatus.AS_IS]
    n_swap = counts[AxisOrderStatus.SWAP]
    n_ambiguous = counts[AxisOrderStatus.AMBIGUOUS]
    n_outside = counts[AxisOrderStatus.OUTSIDE]
    n_empty = counts[AxisOrderStatus.EMPTY]
    n_evaluable = n_as_is + n_swap + n_ambiguous + n_outside

    reasons = []
    if n_evaluable == 0:
        verdict = AxisOrderVerdict.PENDING
        reasons.append("no_evaluable_features")
    elif n_swap > 0 and n_as_is == 0 and n_ambiguous == 0 and n_outside == 0:
        verdict = AxisOrderVerdict.SWAP
    elif n_as_is > 0 and n_swap == 0 and n_ambiguous == 0 and n_outside == 0:
        verdict = AxisOrderVerdict.AS_IS
    else:
        verdict = AxisOrderVerdict.PENDING
        if n_as_is > 0 and n_swap > 0:
            reasons.append("inconsistent")
        if n_ambiguous > 0:
            reasons.append("ambiguous")
        if n_outside > 0:
            reasons.append("outside")

    return AxisOrderEvidence(
        extent=extent,
        verdict=verdict,
        n_total=len(status),
        n_as_is=n_as_is,
        n_swap=n_swap,
        n_ambiguous=n_ambiguous,
        n_outside=n_outside,
        n_empty=n_empty,
        pending_reasons=tuple(reasons),
        crs=crs,
        crs_axis_order=_crs_axis_order_or_none(crs),
    )


def _resolve_geometry(obj, geometry_column):
    """Return ``(is_table, column_name, geometry_array_or_chunked)``."""
    if isinstance(obj, pa.Table):
        name, column = _table_geometry(obj, geometry_column)
        return True, name, column
    return False, None, _compute.obj_as_array_or_chunked(obj)


def check_axis_order(
    obj,
    extent,
    *,
    geometry_column=None,
    expected_crs=None,
    require_crs=False,
):
    """Validate the axis order of geometries against an expected extent.

    Parameters
    ----------
    obj : geoarrow-like, pyarrow.Table
        An array, chunked array, or table.  For a table the standard
        geometry column is selected automatically (or by
        ``geometry_column``).
    extent : sequence or mapping
        The expected business extent as ``(xmin, ymin, xmax, ymax)`` or a
        mapping with those keys.  GeoArrow ordinates are always x, y, so
        this extent is not reordered for the CRS axis order.
    geometry_column : str, optional
        The geometry column to validate when ``obj`` is a table.  Defaults
        to ``"geometry"``, then ``"geography"``, then the only geometry
        column present.
    expected_crs : crs-like, optional
        If given, raise ``ValueError`` unless the geometry type declares
        this CRS.
    require_crs : bool, optional
        If True, raise ``ValueError`` when the geometry type declares no
        CRS.

    Returns
    -------
    pyarrow.Array or pyarrow.ChunkedArray
        A struct array with fields ``status``, ``original_in_extent``,
        ``swapped_in_extent``, and ``pending``.  Null and empty features
        have status ``"empty"`` and do not count as evidence.

    Examples
    --------
    >>> import geoarrow.pyarrow as ga
    >>> geometry = ga.as_geoarrow(["POINT (-75 40)", "POINT (40 -75)"])
    >>> status = ga.check_axis_order(geometry, (-80, 30, -70, 45))
    >>> status.to_pylist()[0]["status"]
    'as_is'
    >>> status.to_pylist()[1]["status"]
    'swap'
    """
    extent = _normalize_extent(extent)
    _, _, obj = _resolve_geometry(obj, geometry_column)
    _validate_crs(obj.type, expected_crs, require_crs)
    return _status_for_array(obj, extent)


def axis_order_evidence(
    obj,
    extent,
    *,
    geometry_column=None,
    expected_crs=None,
    require_crs=False,
):
    """Aggregate axis-order evidence for a batch of geometries.

    The verdict is ``"swap"`` only when at least one feature uniquely fits
    the swapped orientation and every other evaluable feature also
    uniquely fits the swapped orientation.  It is ``"as_is"`` under the
    symmetric condition for the original orientation.  Everything else is
    ``"pending"``: ambiguous features (both orientations possible),
    outside features (neither orientation reasonable), inconsistent
    evidence, or no evaluable features at all.

    Parameters
    ----------
    obj, extent, geometry_column, expected_crs, require_crs
        See :func:`check_axis_order`.

    Returns
    -------
    AxisOrderEvidence
    """
    extent = _normalize_extent(extent)
    _, _, obj = _resolve_geometry(obj, geometry_column)
    _validate_crs(obj.type, expected_crs, require_crs)
    status = _status_for_array(obj, extent)
    return _evidence_from_status(status, extent, getattr(obj.type, "crs", None))


# ---------------------------------------------------------------------------
# Axis swapping
# ---------------------------------------------------------------------------


def _decode_wkb_type(raw):
    has_z = bool(raw & 0x80000000)
    has_m = bool(raw & 0x40000000)
    has_srid = bool(raw & 0x20000000)
    base = raw & 0x0FFFFFFF

    if base >= 3000:
        base -= 3000
        has_z = True
        has_m = True
    elif base >= 2000:
        base -= 2000
        has_m = True
    elif base >= 1000:
        base -= 1000
        has_z = True

    return base, has_z, has_m, has_srid


def _swap_wkb_coord(mv, pos, n_dims, out):
    n_bytes = 8 * n_dims
    # x and y are the first two doubles; z and m (if any) stay put.
    out += mv[pos + 8 : pos + 16]
    out += mv[pos : pos + 8]
    out += mv[pos + 16 : pos + n_bytes]
    return pos + n_bytes


def _swap_wkb_geometry(mv, pos, out):
    start = pos
    if pos >= len(mv):
        raise ValueError("Expected a byte-order marker but reached end of buffer")

    order = mv[pos]
    if order == 1:
        endian = "<"
    elif order == 0:
        endian = ">"
    else:
        raise ValueError(f"Invalid WKB byte order {order}")

    if pos + 5 > len(mv):
        raise ValueError("Expected a WKB geometry type but reached end of buffer")

    raw = struct.unpack_from(endian + "I", mv, pos + 1)[0]
    geometry_type, has_z, has_m, has_srid = _decode_wkb_type(raw)
    n_dims = 2 + int(has_z) + int(has_m)
    pos += 5

    if has_srid:
        pos += 4

    out += mv[start:pos]

    if geometry_type == 1:
        return _swap_wkb_coord(mv, pos, n_dims, out)

    if geometry_type in (2, 3, 4, 5, 6, 7):
        (count,) = struct.unpack_from(endian + "I", mv, pos)
        out += mv[pos : pos + 4]
        pos += 4

        if geometry_type == 2:
            for _ in range(count):
                pos = _swap_wkb_coord(mv, pos, n_dims, out)
            return pos

        if geometry_type == 3:
            for _ in range(count):
                (ring_count,) = struct.unpack_from(endian + "I", mv, pos)
                out += mv[pos : pos + 4]
                pos += 4
                for _ in range(ring_count):
                    pos = _swap_wkb_coord(mv, pos, n_dims, out)
            return pos

        # MultiPoint, MultiLineString, MultiPolygon, GeometryCollection
        for _ in range(count):
            pos = _swap_wkb_geometry(mv, pos, out)
        return pos

    raise ValueError(f"Unsupported WKB geometry type {geometry_type}")


def _swap_wkb(value):
    mv = memoryview(value)
    out = bytearray()
    pos = _swap_wkb_geometry(mv, 0, out)
    if pos != len(mv):
        raise ValueError(f"Trailing bytes in WKB: {len(mv) - pos}")
    return bytes(out)


def _swap_wkb_array(obj):
    storage = _compute.ensure_storage(obj)
    chunks = []
    for chunk in _chunks(storage):
        values = [
            None if item is None else _swap_wkb(item) for item in chunk.to_pylist()
        ]
        chunks.append(pa.array(values, type=chunk.type))
    return obj.type.wrap_array(_rebuild_like(storage, chunks))


def _swap_wkt_array(obj):
    # Round-trip through WKB so that all geometry types and dimensions are
    # handled by the single, well-tested WKB implementation.
    wkb = _compute.as_wkb(obj)
    swapped = _swap_wkb_array(wkb)
    swapped_wkt = _compute.as_wkt(swapped)
    target_storage = obj.type.storage_type
    storage = _compute.ensure_storage(swapped_wkt)
    if storage.type != target_storage:
        storage = storage.cast(target_storage)
    return obj.type.wrap_array(storage)


def _is_coord_struct(type_):
    if not pa.types.is_struct(type_) or type_.num_fields < 2:
        return False
    names = [type_.field(i).name.lower() for i in range(type_.num_fields)]
    return (
        names[0] == "x"
        and names[1] == "y"
        and pa.types.is_floating(type_.field(0).type)
        and pa.types.is_floating(type_.field(1).type)
    )


def _is_box_struct(type_):
    if not pa.types.is_struct(type_):
        return False
    names = {type_.field(i).name.lower() for i in range(type_.num_fields)}
    return {"xmin", "ymin", "xmax", "ymax"} <= names


def _swap_box_children(type_, children):
    # Box fields are named per dimension (e.g. xmin/ymin/zmin/xmax/ymax/zmax
    # for XYZ) so swap by name rather than by position.
    names = [type_.field(i).name.lower() for i in range(type_.num_fields)]
    index = {name: i for i, name in enumerate(names)}
    swap = {"xmin": "ymin", "ymin": "xmin", "xmax": "ymax", "ymax": "xmax"}
    return [
        children[index[swap[name]]] if name in swap else children[i]
        for i, name in enumerate(names)
    ]


def _null_mask(arr):
    return arr.is_null() if arr.null_count else None


def _swap_storage(arr):
    arr = _normalize_array(arr)
    type_ = arr.type

    if pa.types.is_struct(type_):
        children = [arr.field(i) for i in range(type_.num_fields)]
        if _is_box_struct(type_):
            new_children = _swap_box_children(type_, children)
        elif _is_coord_struct(type_):
            new_children = [children[1], children[0], *children[2:]]
        else:
            raise TypeError(
                "Can't swap axes of struct storage with fields "
                f"{[type_.field(i).name for i in range(type_.num_fields)]}"
            )
        return pa.StructArray.from_arrays(
            new_children, fields=list(type_), mask=_null_mask(arr)
        )

    if pa.types.is_fixed_size_list(type_):
        if not pa.types.is_floating(type_.value_type):
            raise TypeError(
                f"Can't swap axes of fixed-size list storage {type_}"
            )
        size = type_.list_size
        # Only the visible slice of the child array is relevant.
        values_arr = arr.values[: len(arr) * size]
        flat = values_arr.to_numpy(zero_copy_only=False).reshape(-1, size).copy()
        flat[:, [0, 1]] = flat[:, [1, 0]]
        values = pa.array(flat.reshape(-1), type=values_arr.type)
        return pa.FixedSizeListArray.from_arrays(
            values, type=type_, mask=_null_mask(arr)
        )

    if pa.types.is_list(type_) or pa.types.is_large_list(type_):
        values = _swap_storage(arr.values)
        cls = pa.LargeListArray if pa.types.is_large_list(type_) else pa.ListArray
        return cls.from_arrays(
            arr.offsets, values, type=type_, mask=_null_mask(arr)
        )

    if isinstance(type_, pa.DenseUnionType):
        children = [_swap_storage(child) for child in arr.children]
        return pa.UnionArray.from_dense(
            arr.type_codes,
            arr.offsets,
            children,
            field_names=[type_.field(i).name for i in range(type_.num_fields)],
            type_codes=list(type_.type_codes),
        )

    raise TypeError(f"Can't swap axes of storage type {type_}")


def _swap_native_array(obj):
    chunks = [
        obj.type.wrap_array(_swap_storage(chunk.storage))
        for chunk in _chunks(obj)
    ]
    return _rebuild_like(obj, chunks)


def swap_axes(obj):
    """Unconditionally swap the x and y ordinates of every geometry.

    This preserves the encoding (GeoArrow-native, WKB, or WKT), the
    declared CRS, the geometry type, dimensions, coordinate type, and the
    null/empty geometry status.  Use :func:`fix_axis_order` instead when
    the swap must be conditional on evidence.

    Parameters
    ----------
    obj : geoarrow-like
        An array, chunked array, or anything ``geoarrow.pyarrow.array()``
        can consume.

    Returns
    -------
    pyarrow.Array or pyarrow.ChunkedArray
        The axes-swapped geometry with the same type as the input.

    Examples
    --------
    >>> import geoarrow.pyarrow as ga
    >>> ga.format_wkt(ga.swap_axes(["LINESTRING (0 1, 2 3)"]))[0].as_py()
    'LINESTRING (1 0, 3 2)'
    >>> ga.format_wkt(ga.swap_axes(["POINT Z (1 2 3)"]))[0].as_py()
    'POINT Z (2 1 3)'
    """
    obj = _compute.obj_as_array_or_chunked(obj)

    if isinstance(obj.type, _type.WkbType):
        return _swap_wkb_array(obj)
    if isinstance(obj.type, _type.WktType):
        return _swap_wkt_array(obj)
    if isinstance(obj.type, BoxType):
        chunks = [
            obj.type.wrap_array(_swap_storage(chunk.storage))
            for chunk in _chunks(obj)
        ]
        return _rebuild_like(obj, chunks)
    if isinstance(obj.type, _type.GeometryExtensionType):
        return _swap_native_array(obj)

    raise TypeError(f"Can't swap axes of type {obj.type}")


# ---------------------------------------------------------------------------
# Conditional repair
# ---------------------------------------------------------------------------


def _select(mask, if_true, if_false):
    """Select between two arrays, including wide (view) storage."""
    try:
        return pc.if_else(mask, if_true, if_false)
    except pa.ArrowNotImplementedError as exc:
        # pyarrow has no if_else kernel for binary/string views; widen the
        # storages, select, then narrow back to the original type.
        type_ = if_true.type
        if pa.types.is_binary_view(type_):
            wide = pa.large_binary()
        elif pa.types.is_string_view(type_):
            wide = pa.large_utf8()
        else:
            raise exc

    selected = pc.if_else(mask, if_true.cast(wide), if_false.cast(wide))
    return selected.cast(type_)


def _swap_selected(obj, mask):
    swapped = swap_axes(obj)
    swapped_storage = _compute.ensure_storage(swapped)
    original_storage = _compute.ensure_storage(obj)

    if isinstance(mask, pa.ChunkedArray) and not isinstance(
        original_storage, pa.ChunkedArray
    ):
        mask = mask.combine_chunks()

    selected = _select(mask, swapped_storage, original_storage)
    return obj.type.wrap_array(selected)


def _swap_mask(status):
    return pc.fill_null(
        pc.equal(pc.struct_field(status, "status"), AxisOrderStatus.SWAP), False
    )


def fix_axis_order(
    obj,
    extent,
    *,
    mode="per_feature",
    geometry_column=None,
    expected_crs=None,
    require_crs=False,
):
    """Correct axis order only where the evidence is consistent and unique.

    Parameters
    ----------
    obj : geoarrow-like, pyarrow.Table
        An array, chunked array, or table.  For a table the corrected
        geometries are written back to the standard geometry column
        (``"geometry"`` by default), preserving column order, field
        metadata, and schema metadata.  An unchanged table is returned as
        a table (not a bare column).
    extent : sequence or mapping
        The expected business extent as ``(xmin, ymin, xmax, ymax)``.
    mode : {"per_feature", "batch"}, optional
        ``"per_feature"`` (the default) repairs each feature whose
        evidence is unique: ``as_is`` features stay, ``swap`` features are
        swapped, and ``ambiguous``/``outside`` features are left
        untouched.  This matches a source that writes most coordinates
        correctly but reverses a few within a batch.  ``"batch"`` swaps
        the whole batch only when every evaluable feature uniquely fits
        the swapped orientation (``verdict == "swap"``); otherwise
        nothing is changed.  In both modes ``empty`` features are left
        untouched.
    geometry_column, expected_crs, require_crs
        See :func:`check_axis_order`.

    Returns
    -------
    AxisOrderFix
        ``data`` holds the corrected array/chunked array/table, ``status``
        the per-feature validation records, ``evidence`` the batch
        evidence, and ``changed`` / ``pending`` the outcome flags.
        ``pending`` is True when some outcome still needs confirmation:
        in ``"per_feature"`` mode any ``ambiguous`` (both directions
        possible) or ``outside`` (neither direction reasonable) feature,
        and in ``"batch"`` mode any verdict other than a uniform
        ``as_is``/``swap``.

    Examples
    --------
    >>> import geoarrow.pyarrow as ga
    >>> geometry = ga.as_geoarrow(["POINT (-75 40)", "POINT (40 -75)"])
    >>> result = ga.fix_axis_order(geometry, (-80, 30, -70, 45))
    >>> result.changed
    True
    >>> ga.format_wkt(result.data).to_pylist()
    ['POINT (-75 40)', 'POINT (-75 40)']

    Ambiguous features (both orientations fit) are left untouched and
    reported as pending:

    >>> geometry = ga.as_geoarrow(["POINT (40 40)"])
    >>> result = ga.fix_axis_order(geometry, (-50, -50, 50, 50))
    >>> result.changed, result.pending
    (False, True)

    In ``"batch"`` mode a mixed batch is not touched at all:

    >>> geometry = ga.as_geoarrow(["POINT (-75 40)", "POINT (40 -75)"])
    >>> result = ga.fix_axis_order(geometry, (-80, 30, -70, 45), mode="batch")
    >>> result.changed, result.evidence.verdict
    (False, 'pending')
    """
    if mode not in ("per_feature", "batch"):
        raise ValueError(
            f"Expected mode to be 'per_feature' or 'batch', got {mode!r}"
        )

    extent = _normalize_extent(extent)
    source_table = obj if isinstance(obj, pa.Table) else None
    is_table, column_name, obj = _resolve_geometry(obj, geometry_column)

    _validate_crs(obj.type, expected_crs, require_crs)
    status = _status_for_array(obj, extent)
    evidence = _evidence_from_status(status, extent, getattr(obj.type, "crs", None))

    if mode == "batch":
        should_swap = evidence.verdict == AxisOrderVerdict.SWAP
        mask = None
        # Batch mode changes nothing unless the whole batch resolves
        # uniformly, so any pending verdict needs confirmation.
        pending = evidence.verdict == AxisOrderVerdict.PENDING
    else:
        mask = _swap_mask(status)
        should_swap = pc.any(mask).as_py() is True
        # Each feature with unique evidence is repaired individually; only
        # features for which both directions fit (ambiguous) or neither fits
        # (outside) still need confirmation.
        pending = evidence.n_ambiguous + evidence.n_outside > 0

    if not should_swap:
        fixed = obj
        changed = False
    elif mode == "batch":
        fixed = swap_axes(obj)
        changed = True
    else:
        fixed = _swap_selected(obj, mask)
        changed = True

    if is_table:
        if changed:
            fixed = _set_table_column(source_table, column_name, fixed)
        else:
            fixed = source_table

    return AxisOrderFix(
        data=fixed,
        status=status,
        evidence=evidence,
        changed=changed,
        pending=pending,
    )
