"""Canonical geometry encodings and stable signatures

Snapshot reconciliation compares two versions of a dataset and decides which
features changed. A naive comparison of stored geometry bytes reports false
updates whenever a producer rewrites a geometry without changing its meaning
(e.g., starting a polygon ring at a different vertex, reversing ring winding,
reordering holes, or writing big-endian instead of little-endian WKB).

This module adds an explicit canonicalization mode to the data flow. A
canonical form is computed for each geometry and hashed into a fixed-size
signature that can be compared cheaply across snapshots. The original geometry
is never modified: callers keep it alongside the signature so that every change
remains traceable back to the inputs.

Only *simple polygonal* geometries (``POLYGON`` and ``MULTIPOLYGON``) are
normalized semantically. Under :attr:`CanonicalMode.POLYGON`, two geometries
are equivalent when they describe the same area using exactly the same
coordinates (no tolerance is applied) in the same coordinate reference system.
Ring start points, ring winding, hole order, polygon order, and the
``POLYGON``/single-part ``MULTIPOLYGON`` distinction are ignored. Null and
empty geometries are kept distinct: a null geometry has a null signature,
whereas an empty geometry has a signature of its own.

>>> import geoarrow.pyarrow as ga
>>> a = "POLYGON ((0 0, 4 0, 4 4, 0 4, 0 0))"
>>> b = "POLYGON ((4 0, 4 4, 0 4, 0 0, 4 0))"
>>> sig_a = ga.geometry_signature([a])
>>> sig_b = ga.geometry_signature([b])
>>> sig_a.type
FixedSizeBinaryType(fixed_size_binary[32])
>>> sig_a[0].as_py() == sig_b[0].as_py()
True

The original geometry remains available and is not rewritten in place:

>>> ga.format_wkt(ga.array([b]))[0].as_py()
'POLYGON ((4 0, 4 4, 0 4, 0 0, 4 0))'
>>> ga.format_wkt(ga.canonicalize_geometry([b]))[0].as_py()
'POLYGON ((0 0, 0 4, 4 4, 4 0, 0 0))'
"""

import enum
import hashlib
import json
import struct

from geoarrow.pyarrow._compute import as_wkb, obj_as_array_or_chunked
from geoarrow.pyarrow._type import wkb
from geoarrow.types import EdgeType

import pyarrow as pa
import pyarrow.compute as pc

SIGNATURE_SIZE_BYTES = 32
"""Size of the fixed-size binary signatures produced by this module."""

_CANONICAL_VERSION = 1
_SIGNATURE_DOMAIN = b"geoarrow.pyarrow:geometry-signature:v1\x00"
_NAN_BYTES = b"\x00\x00\x00\x00\x00\x00\xf8\x7f"
_SIGNATURE_TYPE = pa.binary(SIGNATURE_SIZE_BYTES)


class CanonicalMode(enum.Enum):
    """Canonicalization mode used to build geometry signatures

    ``EXACT`` only normalizes the encoding (little-endian, ISO geometry type
    codes) and preserves coordinate order. ``POLYGON`` additionally treats
    semantically equivalent simple polygons as equal: ring start point, ring
    winding, hole order, polygon order, and the single-part-multipolygon
    distinction are ignored.

    The mode is part of the signature domain, so signatures produced with
    different modes never compare equal.
    """

    EXACT = "exact"
    """Normalize the WKB encoding only; preserve coordinate order"""

    POLYGON = "polygon"
    """Semantic canonicalization for POLYGON and MULTIPOLYGON geometries"""


def _coerce_mode(mode):
    if isinstance(mode, CanonicalMode):
        return mode
    if isinstance(mode, str):
        try:
            return CanonicalMode(mode.lower())
        except ValueError:
            pass
    allowed = ", ".join(repr(m.value) for m in CanonicalMode)
    raise ValueError(f"Expected mode to be one of {allowed} but got {mode!r}")


class _WkbError(ValueError):
    pass


def _read_u32(buf, pos, fmt):
    if pos + 4 > len(buf):
        raise _WkbError("truncated WKB geometry")
    return pos + 4, struct.unpack_from(fmt + "I", buf, pos)[0]


_GEOMETRY_TYPE_NAMES = {
    0: "GEOMETRY",
    1: "POINT",
    2: "LINESTRING",
    3: "POLYGON",
    4: "MULTIPOINT",
    5: "MULTILINESTRING",
    6: "MULTIPOLYGON",
    7: "GEOMETRYCOLLECTION",
}


def _read_header(buf, pos, inherit_z=False, inherit_m=False):
    """Read a WKB header.

    Returns ``(pos, fmt, base_code, has_z, has_m)``. ISO type codes carry
    dimensions by adding 1000/2000/3000; EWKB uses high bits and may also carry
    an SRID. Members of a collection sometimes omit dimensional flags that are
    present on the parent, in which case ``inherit_z``/``inherit_m`` are used.
    """
    if pos >= len(buf):
        raise _WkbError("truncated WKB geometry")

    byte_order = buf[pos]
    pos += 1
    if byte_order == 1:
        fmt = "<"
    elif byte_order == 0:
        fmt = ">"
    else:
        raise _WkbError(f"invalid WKB byte order marker: {byte_order}")

    pos, raw = _read_u32(buf, pos, fmt)

    has_z = bool(raw & 0x80000000)
    has_m = bool(raw & 0x40000000)
    has_srid = bool(raw & 0x20000000)
    code = raw & 0x1FFFFFFF

    if code >= 3000:
        code -= 3000
        has_z = True
        has_m = True
    elif code >= 2000:
        code -= 2000
        has_m = True
    elif code >= 1000:
        code -= 1000
        has_z = True

    if has_srid:
        if pos + 4 > len(buf):
            raise _WkbError("truncated WKB SRID")
        pos += 4

    if not has_z and inherit_z:
        has_z = True
    if not has_m and inherit_m:
        has_m = True

    return pos, fmt, code, has_z, has_m


def _read_coords(buf, pos, fmt, ncoords):
    nbytes = 8 * ncoords
    if pos + nbytes > len(buf):
        raise _WkbError("truncated WKB coordinate")
    values = struct.unpack_from(fmt + "d" * ncoords, buf, pos)
    return pos + nbytes, values


def _emit_coords(buf, pos, out, fmt, ncoords):
    """Copy ``ncoords`` doubles from ``buf`` to ``out`` as little-endian."""
    nbytes = 8 * ncoords
    if pos + nbytes > len(buf):
        raise _WkbError("truncated WKB coordinate")
    chunk = buf[pos : pos + nbytes]
    if fmt == "<":
        out += chunk
    else:
        for i in range(ncoords):
            out += bytes(chunk[8 * i : 8 * i + 8])[::-1]
    return pos + nbytes


def _emit_geometry_header(out, code, has_z, has_m):
    type_code = code + 1000 * int(has_z) + 2000 * int(has_m)
    out += b"\x01" + struct.pack("<I", type_code)


def _normalize_wkb(raw):
    """Return little-endian ISO WKB with coordinates unchanged."""
    buf = memoryview(raw)
    out = bytearray()
    pos = _emit_geometry(buf, 0, out, False, False)
    if pos != len(buf):
        raise _WkbError("trailing bytes after WKB geometry")
    return bytes(out)


def _emit_geometry(buf, pos, out, inherit_z, inherit_m, expected=None):
    pos, fmt, code, has_z, has_m = _read_header(buf, pos, inherit_z, inherit_m)
    if expected is not None and code != expected:
        expected_name = _GEOMETRY_TYPE_NAMES[expected]
        got_name = _GEOMETRY_TYPE_NAMES.get(code, f"type code {code}")
        raise _WkbError(f"expected a {expected_name} member but got {got_name}")

    ncoords = 2 + int(has_z) + int(has_m)
    _emit_geometry_header(out, code, has_z, has_m)

    if code == 0:  # GEOMETRY EMPTY has no body
        return pos
    elif code == 1:  # POINT
        return _emit_coords(buf, pos, out, fmt, ncoords)
    elif code == 2:  # LINESTRING
        pos, n = _read_u32(buf, pos, fmt)
        out += struct.pack("<I", n)
        return _emit_coords(buf, pos, out, fmt, n * ncoords)
    elif code == 3:  # POLYGON
        pos, nrings = _read_u32(buf, pos, fmt)
        out += struct.pack("<I", nrings)
        for _ in range(nrings):
            pos, npoints = _read_u32(buf, pos, fmt)
            out += struct.pack("<I", npoints)
            pos = _emit_coords(buf, pos, out, fmt, npoints * ncoords)
        return pos
    elif code in (4, 5, 6, 7):  # MULTI* / GEOMETRYCOLLECTION
        member_type = {4: 1, 5: 2, 6: 3}.get(code)
        pos, n = _read_u32(buf, pos, fmt)
        out += struct.pack("<I", n)
        for _ in range(n):
            pos = _emit_geometry(buf, pos, out, has_z, has_m, expected=member_type)
        return pos
    else:
        name = _GEOMETRY_TYPE_NAMES.get(code, f"type code {code}")
        raise _WkbError(f"unsupported WKB geometry type: {name}")


def _read_polygon_body(buf, pos, fmt, ncoords):
    pos, nrings = _read_u32(buf, pos, fmt)
    rings = []
    for _ in range(nrings):
        pos, npoints = _read_u32(buf, pos, fmt)
        ring = []
        for _ in range(npoints):
            pos, values = _read_coords(buf, pos, fmt, ncoords)
            ring.append(values)
        rings.append(ring)
    return pos, rings


def _parse_polygonal_wkb(raw):
    """Parse WKB into ``(polygons, has_z, has_m)``.

    Each polygon is a list of rings and each ring is a list of coordinate
    tuples.
    """
    buf = memoryview(raw)
    pos, fmt, code, has_z, has_m = _read_header(buf, 0)
    ncoords = 2 + int(has_z) + int(has_m)

    if code == 3:
        pos, rings = _read_polygon_body(buf, pos, fmt, ncoords)
        polys = [rings]
    elif code == 6:
        pos, nparts = _read_u32(buf, pos, fmt)
        polys = []
        for _ in range(nparts):
            pos, fmt, code, sub_z, sub_m = _read_header(
                buf, pos, inherit_z=has_z, inherit_m=has_m
            )
            if code != 3:
                raise _WkbError(
                    "MULTIPOLYGON contains a non-POLYGON member "
                    f"(geometry type code {code})"
                )
            if (sub_z, sub_m) != (has_z, has_m):
                raise _WkbError(
                    "MULTIPOLYGON members have inconsistent dimensional flags"
                )
            pos, rings = _read_polygon_body(buf, pos, fmt, ncoords)
            polys.append(rings)
    else:
        name = _GEOMETRY_TYPE_NAMES.get(code, f"type code {code}")
        raise _WkbError(
            "CanonicalMode.POLYGON supports only POLYGON and MULTIPOLYGON "
            f"geometries but got {name}"
        )

    if pos != len(buf):
        raise _WkbError("trailing bytes after WKB geometry")

    return polys, has_z, has_m


def _coordinate_bytes(value):
    if value != value:  # NaN: use a stable payload
        return _NAN_BYTES
    if value == 0.0:  # normalize -0.0
        value = 0.0
    return struct.pack("<d", value)


def _point_key(values):
    return b"".join(_coordinate_bytes(value) for value in values)


def _clean_ring(ring):
    """Drop consecutive duplicates and the repeated closing point."""
    cleaned = []
    for values in ring:
        key = _point_key(values)
        if cleaned and cleaned[-1][0] == key:
            continue
        cleaned.append((key, values))

    while len(cleaned) > 1 and cleaned[-1][0] == cleaned[0][0]:
        cleaned.pop()

    return cleaned


def _minimal_rotation(seq):
    """Return ``(rotated_seq, start)`` for the lexicographically least rotation."""
    n = len(seq)
    if n == 0:
        return [], 0

    doubled = seq + seq
    i, j, k = 0, 1, 0
    while i < n and j < n and k < n:
        a = doubled[i + k]
        b = doubled[j + k]
        if a == b:
            k += 1
        elif a > b:
            i = i + k + 1
            if i <= j:
                i = j + 1
            k = 0
        else:
            j = j + k + 1
            if j <= i:
                j = i + 1
            k = 0

    start = min(i, j)
    return doubled[start : start + n], start


def _rotate(seq, start):
    return seq[start:] + seq[:start]


def _canonical_ring(ring):
    """Return a ring with a stable start point and direction, or ``None``.

    The direction is chosen by comparing the canonicalized byte sequences, not
    by a floating-point area computation. Computing an area would be unstable
    for nearly degenerate rings: the same ring with a different start vertex
    can round to a different sign, producing two signatures for identical
    coordinates.
    """
    cleaned = _clean_ring(ring)
    if not cleaned:
        return None

    keys = [key for key, _ in cleaned]
    values = [value for _, value in cleaned]

    forward, forward_start = _minimal_rotation(keys)
    reverse, reverse_start = _minimal_rotation(keys[::-1])
    if forward <= reverse:
        return _rotate(values, forward_start)
    else:
        return _rotate(values[::-1], reverse_start)


def _ring_bytes(ring):
    closed = list(ring)
    closed.append(closed[0])
    parts = [struct.pack("<I", len(closed))]
    parts.extend(_point_key(values) for values in closed)
    return b"".join(parts)


def _canonical_polygon_body(rings):
    """Return the canonical polygon body (ring count + rings) or ``None``."""
    if not rings:
        return None

    shell = _canonical_ring(rings[0])
    if shell is None:
        return None

    holes = []
    for ring in rings[1:]:
        canonical = _canonical_ring(ring)
        if canonical is not None:
            holes.append(_ring_bytes(canonical))
    holes.sort()

    parts = [struct.pack("<I", 1 + len(holes)), _ring_bytes(shell)]
    parts.extend(holes)
    return b"".join(parts)


def _canonical_polygon_wkb(raw):
    """Return canonical little-endian ISO WKB for a polygonal geometry."""
    polys, has_z, has_m = _parse_polygonal_wkb(raw)

    bodies = []
    for rings in polys:
        body = _canonical_polygon_body(rings)
        if body is not None:
            bodies.append(body)

    if not bodies:
        return _polygon_header(3, has_z, has_m) + struct.pack("<I", 0)

    bodies.sort()
    if len(bodies) == 1:
        return _polygon_header(3, has_z, has_m) + bodies[0]

    parts = [_polygon_header(6, has_z, has_m), struct.pack("<I", len(bodies))]
    for body in bodies:
        parts.append(_polygon_header(3, has_z, has_m))
        parts.append(body)
    return b"".join(parts)


def _polygon_header(code, has_z, has_m):
    header = bytearray()
    _emit_geometry_header(header, code, has_z, has_m)
    return bytes(header)


def _canonical_bytes(raw, mode):
    if mode is CanonicalMode.EXACT:
        return _normalize_wkb(raw)
    else:
        return _canonical_polygon_wkb(raw)


def _coerce_input(obj):
    # pyarrow cannot wrap a ChunkedArray with zero chunks in an extension type;
    # collapsing it first yields an equivalent empty Array.
    if isinstance(obj, pa.ChunkedArray) and obj.num_chunks == 0:
        obj = obj.combine_chunks()

    try:
        return obj_as_array_or_chunked(obj)
    except TypeError:
        # ``pa.array([None, None])`` and ``pa.array([])`` have a null type,
        # which geoarrow.array() rejects. Snapshots may legitimately contain an
        # all-null geometry column, so preserve it instead of failing.
        inferred = pa.array(obj)
        if pa.types.is_null(inferred.type):
            return inferred
        raise


def _geometry_metadata(type_):
    crs = getattr(type_, "crs", None)
    edge_type = getattr(type_, "edge_type", None)
    return crs, edge_type


def _output_type(crs):
    if crs is None:
        return wkb()
    else:
        return wkb().with_crs(crs)


def _edge_key(type_):
    """Return a comparable key for a type's edge interpretation.

    Plain (non-extension) storage arrays have no edge metadata, which GeoArrow
    treats as the planar default, so ``None`` and ``PLANAR`` are equivalent.
    """
    _, edge_type = _geometry_metadata(type_)
    if edge_type is None or edge_type == EdgeType.PLANAR:
        return "planar"
    return edge_type.name.lower()


def _check_edges(type_, mode):
    if mode is not CanonicalMode.POLYGON:
        return
    if _edge_key(type_) != "planar":
        raise ValueError(
            f"CanonicalMode.{mode.name} requires planar edges but the input has "
            f"{_edge_key(type_)} edges"
        )


def _check_compatible(before_type, after_type, check_crs):
    if check_crs:
        before_crs, _ = _geometry_metadata(before_type)
        after_crs, _ = _geometry_metadata(after_type)
        if _crs_key(before_crs) != _crs_key(after_crs):
            raise ValueError(
                "Can't compare snapshots with different coordinate reference "
                f"systems ({before_crs!r} vs {after_crs!r}). Reproject one side "
                "or pass check_crs=False if the difference is intentional."
            )

    if _edge_key(before_type) != _edge_key(after_type):
        raise ValueError(
            "Can't compare snapshots with different edge interpretations "
            f"({_edge_key(before_type)} vs {_edge_key(after_type)}); "
            "planar and spherical edges describe different areas"
        )


def _map_values(pylist, func, offset):
    values = []
    for i, value in enumerate(pylist):
        if value is None:
            values.append(None)
            continue
        try:
            values.append(func(value))
        except _WkbError as exc:
            raise _WkbError(
                f"Failed to canonicalize geometry at index {offset + i}: {exc}"
            ) from exc
    return values


def _canonicalize_chunked(chunked, mode):
    _check_edges(chunked.type, mode)
    crs, _ = _geometry_metadata(chunked.type)
    out_type = _output_type(crs)

    if chunked.num_chunks == 0:
        return pa.chunked_array([], type=out_type)
    if chunked.type == pa.null():
        nulls = pa.nulls(len(chunked), type=out_type)
        return pa.chunked_array([nulls], type=out_type)

    wkb_chunked = as_wkb(chunked)

    chunks = []
    offset = 0
    for chunk in wkb_chunked.chunks:
        values = _map_values(
            chunk.to_pylist(), lambda raw: _canonical_bytes(raw, mode), offset
        )
        offset += len(chunk)
        chunks.append(out_type.wrap_array(pa.array(values, type=pa.binary())))
    return pa.chunked_array(chunks, type=out_type)


def _canonicalize_array(array, mode):
    _check_edges(array.type, mode)
    crs, _ = _geometry_metadata(array.type)
    out_type = _output_type(crs)

    if array.type == pa.null():
        return pa.nulls(len(array), type=out_type)

    wkb_array = as_wkb(array)
    values = _map_values(
        wkb_array.to_pylist(), lambda raw: _canonical_bytes(raw, mode), 0
    )
    return out_type.wrap_array(pa.array(values, type=pa.binary()))


def canonicalize_geometry(obj, mode=CanonicalMode.POLYGON):
    """Return a canonical GeoArrow WKB array for each geometry in ``obj``

    The original geometries are not modified. This function returns a new
    ``geoarrow.wkb`` array (or chunked array) and is intended to be stored
    alongside the original geometry when signatures are used to compare
    snapshots.

    In :attr:`CanonicalMode.POLYGON` mode, each ring is given a stable starting
    vertex and a stable direction (chosen by comparing coordinates
    lexicographically, so it does not depend on floating-point area), holes are
    sorted, and multipolygon members are sorted. A single polygon and a
    one-member multipolygon produce the same canonical value; empty polygons
    and empty multipolygons also produce the same canonical value. ``NULL``
    values remain ``NULL``.

    In :attr:`CanonicalMode.EXACT` mode, only the WKB encoding is normalized
    (little-endian, ISO geometry type codes). Ring order and direction are
    preserved. This mode works for every geometry type.

    Parameters
    ----------
    obj : array-like
        Any object accepted by :func:`geoarrow.pyarrow.array`, including
        WKT/WKB storage arrays, GeoArrow arrays, and GeoPandas objects.
    mode : CanonicalMode or str, optional
        ``"polygon"`` (the default) or ``"exact"``.

    Examples
    --------
    >>> import geoarrow.pyarrow as ga
    >>> canonical = ga.canonicalize_geometry([
    ...     "POLYGON ((0 0, 1 0, 1 1, 0 0))",
    ...     "POLYGON ((1 0, 1 1, 0 0, 1 0))",
    ... ])
    >>> ga.format_wkt(canonical).to_pylist()
    ['POLYGON ((0 0, 1 0, 1 1, 0 0))', 'POLYGON ((0 0, 1 0, 1 1, 0 0))']
    """
    mode = _coerce_mode(mode)
    obj = _coerce_input(obj)

    if isinstance(obj, pa.ChunkedArray):
        return _canonicalize_chunked(obj, mode)
    else:
        return _canonicalize_array(obj, mode)


def _signature_for(raw, mode):
    h = hashlib.sha256()
    h.update(_SIGNATURE_DOMAIN)
    h.update(mode.value.encode("UTF-8"))
    h.update(b"\x00")
    h.update(raw)
    return h.digest()


def _signature_chunk(chunk, mode):
    values = [
        None if value is None else _signature_for(value, mode)
        for value in chunk.to_pylist()
    ]
    return pa.array(values, type=_SIGNATURE_TYPE)


def geometry_signature(obj, mode=CanonicalMode.POLYGON):
    """Compute a stable 32-byte signature for each geometry in ``obj``

    The signature is the SHA-256 hash of the canonical geometry bytes produced
    by :func:`canonicalize_geometry` (with a mode-specific domain prefix).
    Comparing signatures is therefore equivalent to comparing canonical
    geometries, but uses a fixed-size value that is cheap to shuffle, sort, and
    join.

    ``NULL`` geometries produce ``NULL`` signatures. Empty geometries produce a
    non-null signature that differs from every non-empty geometry and from
    ``NULL``. Signatures only describe the geometry: they are computed under a
    consistent coordinate reference system, and :func:`compare_snapshots`
    checks that precondition explicitly.

    Parameters
    ----------
    obj : array-like
        Any object accepted by :func:`canonicalize_geometry`.
    mode : CanonicalMode or str, optional
        ``"polygon"`` (the default) or ``"exact"``.

    Examples
    --------
    >>> import geoarrow.pyarrow as ga
    >>> sig = ga.geometry_signature(["POLYGON ((0 0, 1 0, 1 1, 0 0))", None])
    >>> sig.type
    FixedSizeBinaryType(fixed_size_binary[32])
    >>> len(sig[0].as_py())
    32
    >>> sig[1].as_py() is None
    True
    """
    mode = _coerce_mode(mode)
    canonical = canonicalize_geometry(obj, mode=mode)

    if isinstance(canonical, pa.ChunkedArray):
        chunks = [_signature_chunk(chunk, mode) for chunk in canonical.chunks]
        return pa.chunked_array(chunks, type=_SIGNATURE_TYPE)
    else:
        return _signature_chunk(canonical, mode)


def _freeze(value):
    if isinstance(value, dict):
        return tuple(sorted((key, _freeze(item)) for key, item in value.items()))
    elif isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    else:
        return value


def _crs_key(crs):
    if crs is None:
        return ("none",)
    if hasattr(crs, "to_json_dict"):
        try:
            return ("json", _freeze(crs.to_json_dict()))
        except Exception:  # noqa: BLE001
            pass
    if hasattr(crs, "__geoarrow_crs_json_values__"):
        try:
            return ("values", _freeze(crs.__geoarrow_crs_json_values__()))
        except Exception:  # noqa: BLE001
            pass
    return ("text", str(crs))


def _check_key_columns(table, keys, label):
    schema = table.schema
    for key in keys:
        if key not in schema.names:
            raise ValueError(f"Key column {key!r} is missing from the {label} table")

    key_table = table.select(keys)
    for key in keys:
        if key_table.column(key).null_count > 0:
            raise ValueError(
                f"Key column {key!r} contains nulls in the {label} table; "
                "feature identifiers must not be null"
            )

    grouped = key_table.group_by(keys).aggregate([(keys[0], "count")])
    count_column = grouped.column(f"{keys[0]}_count")
    if len(count_column) > 0 and pc.max(count_column).as_py() > 1:
        raise ValueError(
            f"Key columns {keys!r} are not unique in the {label} table; "
            "snapshots must identify each feature at most once"
        )


def _status_column(sig_before, sig_after, present_before, present_after):
    present_before = pc.fill_null(present_before, False)
    present_after = pc.fill_null(present_after, False)

    both_null = pc.and_(pc.is_null(sig_before), pc.is_null(sig_after))
    either_null = pc.or_(pc.is_null(sig_before), pc.is_null(sig_after))
    equal = pc.if_else(
        either_null, both_null, pc.fill_null(pc.equal(sig_before, sig_after), False)
    )

    text = pa.string()
    return pc.if_else(
        pc.invert(present_before),
        pa.scalar("added", type=text),
        pc.if_else(
            pc.invert(present_after),
            pa.scalar("removed", type=text),
            pc.if_else(
                equal,
                pa.scalar("unchanged", type=text),
                pa.scalar("updated", type=text),
            ),
        ),
    )


def _gather_geometry(column, row_indices):
    """Gather ``column`` at ``row_indices``, preserving its type and nulls.

    ``row_indices`` comes from an outer join and contains nulls for rows that
    are absent from this side. ``pyarrow.compute.take()`` maps null indices to
    null values, which is exactly the ``NULL``/absent distinction we want.
    """
    if isinstance(column, pa.ChunkedArray) and column.num_chunks == 0:
        column = column.combine_chunks()

    taken = pc.take(column, row_indices)
    if isinstance(column, pa.Array):
        return taken

    # Keep the input's chunking semantics predictable: one output chunk.
    if isinstance(taken, pa.ChunkedArray):
        return pa.chunked_array([taken.combine_chunks()], type=taken.type)
    return pa.chunked_array([taken], type=taken.type)


def compare_snapshots(
    before,
    after,
    key,
    geometry_column="geometry",
    geometry_column_after=None,
    mode=CanonicalMode.POLYGON,
    check_crs=True,
):
    """Compare two snapshots using canonical geometry signatures

    ``before`` and ``after`` are :class:`pyarrow.Table` objects representing
    two real snapshots of the same feature collection. Features are matched on
    ``key`` and classified as ``"added"``, ``"removed"``, ``"updated"``, or
    ``"unchanged"``. Two features are unchanged when their canonical
    signatures match; this ignores polygon ring start points, ring winding,
    hole order, polygon order, and big-endian vs little-endian WKB.

    The returned table keeps the original geometry from both snapshots so that
    every reported change can be traced back to the exact input values. Its
    columns are the key column(s), ``status``, ``geometry_before``,
    ``geometry_after``, ``signature_before``, and ``signature_after``. Rows are
    sorted by key. A ``NULL`` geometry produces a ``NULL`` signature, whereas
    an empty geometry produces a non-null signature (so a feature changing from
    ``NULL`` to empty is reported as ``"updated"``).

    Parameters
    ----------
    before : pyarrow.Table
        The earlier snapshot.
    after : pyarrow.Table
        The later snapshot.
    key : str or list of str
        Column name(s) that uniquely identify a feature in both snapshots.
    geometry_column : str, optional
        Geometry column name in ``before`` (default ``"geometry"``).
    geometry_column_after : str, optional
        Geometry column name in ``after``. Defaults to ``geometry_column``.
    mode : CanonicalMode or str, optional
        Canonicalization mode (default ``"polygon"``).
    check_crs : bool, optional
        If ``True`` (the default), raise if the two geometry columns have
        different coordinate reference systems. Signatures are only comparable
        under a consistent CRS.

    Examples
    --------
    >>> import geoarrow.pyarrow as ga
    >>> import pyarrow as pa
    >>> before = pa.table({
    ...     "id": [1, 2],
    ...     "geometry": ga.array([
    ...         "POLYGON ((0 0, 1 0, 1 1, 0 0))",
    ...         "POLYGON ((5 5, 6 5, 6 6, 5 5))",
    ...     ]),
    ... })
    >>> after = pa.table({
    ...     "id": [2, 3],
    ...     "geometry": ga.array([
    ...         "POLYGON ((6 5, 6 6, 5 5, 6 5))",
    ...         "POLYGON ((9 9, 10 9, 10 10, 9 9))",
    ...     ]),
    ... })
    >>> ga.compare_snapshots(before, after, key="id").column("status").to_pylist()
    ['removed', 'unchanged', 'added']
    """
    mode = _coerce_mode(mode)

    if not isinstance(before, pa.Table) or not isinstance(after, pa.Table):
        raise TypeError("before and after must be pyarrow.Table instances")

    if isinstance(key, str):
        keys = [key]
    else:
        keys = list(key)
    if not keys:
        raise ValueError("key must name at least one column")
    if len(set(keys)) != len(keys):
        raise ValueError(f"key contains duplicate column names: {keys}")

    if geometry_column_after is None:
        geometry_column_after = geometry_column

    reserved = {
        "status",
        "geometry_before",
        "geometry_after",
        "signature_before",
        "signature_after",
    }
    conflicting = reserved.intersection(keys)
    if conflicting:
        raise ValueError(
            f"Key columns may not use reserved names: {sorted(conflicting)}"
        )

    _check_key_columns(before, keys, "before")
    _check_key_columns(after, keys, "after")

    if geometry_column not in before.schema.names:
        raise ValueError(
            f"Geometry column {geometry_column!r} is missing from the before table"
        )
    if geometry_column_after not in after.schema.names:
        raise ValueError(
            f"Geometry column {geometry_column_after!r} is missing from the after table"
        )

    for key_name in keys:
        before_type = before.schema.field(key_name).type
        after_type = after.schema.field(key_name).type
        if before_type != after_type:
            raise ValueError(
                f"Key column {key_name!r} has type {before_type} in the before "
                f"table but {after_type} in the after table"
            )

    _check_compatible(
        before.schema.field(geometry_column).type,
        after.schema.field(geometry_column_after).type,
        check_crs,
    )

    sig_before = geometry_signature(before.column(geometry_column), mode=mode)
    sig_after = geometry_signature(after.column(geometry_column_after), mode=mode)

    # Join on key columns plus row positions rather than geometry columns:
    # pyarrow's join rejects null-typed non-key fields, and geometry columns may
    # legitimately be all-null. The original geometry is gathered afterwards
    # with take(), which preserves both the geometry type and its nulls.
    before_side = pa.table(
        {
            **{name: before.column(name) for name in keys},
            "__row_before": pa.array(range(before.num_rows), type=pa.int64()),
            "__signature_before": sig_before,
            "__present_before": pa.array([True] * before.num_rows, type=pa.bool_()),
        }
    )
    after_side = pa.table(
        {
            **{name: after.column(name) for name in keys},
            "__row_after": pa.array(range(after.num_rows), type=pa.int64()),
            "__signature_after": sig_after,
            "__present_after": pa.array([True] * after.num_rows, type=pa.bool_()),
        }
    )

    joined = before_side.join(after_side, keys=keys, join_type="full outer")
    joined = joined.sort_by([(name, "ascending") for name in keys])

    status = _status_column(
        joined["__signature_before"],
        joined["__signature_after"],
        joined["__present_before"],
        joined["__present_after"],
    )

    geometry_before = _gather_geometry(
        before.column(geometry_column), joined["__row_before"]
    )
    geometry_after = _gather_geometry(
        after.column(geometry_column_after), joined["__row_after"]
    )

    out = pa.table(
        {
            **{name: joined.column(name) for name in keys},
            "status": status,
            "geometry_before": geometry_before,
            "geometry_after": geometry_after,
            "signature_before": joined.column("__signature_before"),
            "signature_after": joined.column("__signature_after"),
        }
    )

    metadata = {
        b"geoarrow.pyarrow.canonical": json.dumps(
            {
                "version": _CANONICAL_VERSION,
                "mode": mode.value,
                "hash": "sha256",
                "signature_size": SIGNATURE_SIZE_BYTES,
            }
        ).encode("UTF-8")
    }
    return out.replace_schema_metadata(metadata)
