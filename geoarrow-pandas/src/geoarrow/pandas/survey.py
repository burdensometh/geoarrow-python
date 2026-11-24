"""
Survey point consolidation with explicit dimension-promotion rules.

This module combines survey-point batches whose geometries may be 2D (``XY``),
3D (``XYZ``) or entirely missing (``null``) into a single geometry column that
is persisted once as GeoParquet.

Dimension-promotion rules (applied per geometry, never per column coercion):

1. A geometry carrying a real Z keeps that Z unchanged.
2. A 2D geometry is *promoted* to 3D (``XY`` -> ``XYZ``) and its missing Z is
   represented by ``NaN`` -- **never ``0``**. A missing height is unknown, not
   a measured height of zero, and treating it as zero would corrupt vertical
   statistics.
3. A ``null`` geometry stays ``null`` (the whole feature is missing); it does
   not become a point and it does not receive any coordinate.
4. If at least one source geometry has a Z, the output column is uniformly
   ``XYZ`` so the table has one declared dimension. The names of the source
   batches that contributed no Z at all are recorded in the provenance
   metadata (``survey.z_missing_batches``). A batch that mixes XY and XYZ rows
   is reported under ``survey.z_partial_batches``: its 2D rows likewise get
   ``NaN`` Z.

Both the write and read paths keep the geometry on a geoarrow extension type,
so the column round-trips as an Arrow extension array and remains directly
usable for spatial analysis (``Series.geoarrow`` / GeoPandas).
"""

import json as _json
import re as _re

import numpy as _np
import pyarrow as _pa

import geoarrow.pyarrow as _ga
import geoarrow.pandas as _gapd
from geoarrow.pyarrow import io as _ga_io
from geoarrow.types import Dimensions as _Dimensions

from pandas import Series as _Series

# Schema metadata keys are byte->byte (Parquet/Arrow convention). The JSON
# payload under each key uses the same names without the prefix.
META_Z_MISSING_BATCHES = b"survey.z_missing_batches"
META_Z_PARTIAL_BATCHES = b"survey.z_partial_batches"
META_Z_RULE = b"survey.z_missing_rule"

# Human-readable rule persisted alongside the data so a downstream reader does
# not have to guess what a NaN Z means.
Z_MISSING_RULE = (
    "Missing Z values are represented by NaN (never 0); "
    "null geometries indicate entirely missing features."
)

_DIM_XY = _Dimensions.XY.value
_DIM_XYZ = _Dimensions.XYZ.value

__all__ = [
    "consolidate_survey_batches",
    "write_survey_geoparquet",
    "read_survey_geoparquet",
    "z_is_missing",
    "Z_MISSING_RULE",
]


# The geoarrow-c WKT parser follows ISO WKT: a 3D/4D point must carry the
# explicit Z/ZM marker (e.g. "POINT Z (1 2 3)"). This tolerates the common
# non-ISO shorthand "POINT (1 2 3)" by inserting the marker from the coordinate
# count. Geometry keywords other than POINT are left untouched.
_WKT_BARE_POINT = _re.compile(
    r"^(\s*)POINT(\s+)(?!Z\b|M\b|ZM\b)(\(\s*[^)]*\))(\s*)$", _re.IGNORECASE
)


def _normalize_wkt(value):
    match = _WKT_BARE_POINT.match(value)
    if not match:
        return value
    prefix, _gap, coords, suffix = match.groups()
    numbers = coords.strip("()").split()
    if len(numbers) == 3:
        marker = "Z"
    elif len(numbers) == 4:
        marker = "ZM"
    else:
        return value
    return f"{prefix}POINT {marker} {coords}{suffix}"


def _as_geoarrow_array(geometries):
    """Normalize shapely geometries / WKB / WKT / geoarrow input to a
    geoarrow extension array. ``None`` entries become null array elements."""
    if isinstance(geometries, _gapd.GeoArrowExtensionArray):
        return geometries._data
    if isinstance(geometries, _Series) and isinstance(
        geometries.dtype, _gapd.GeoArrowExtensionDtype
    ):
        return geometries.array._data
    if isinstance(geometries, _Series) and _is_arrow_geometry_dtype(
        geometries.dtype
    ):
        # Series backed by a pyarrow ArrowDtype wrapping a geoarrow type
        # (e.g. after a plain pandas/pyarrow parquet round trip).
        return _ga.array(_pa.array(geometries))
    if isinstance(geometries, (_pa.Array, _pa.ChunkedArray)):
        return _ga.array(geometries)

    # Shapely 2 objects are not understood by pyarrow's sequence inference;
    # serialize to WKB first. Plain bytes (WKB) pass through; strings are
    # treated as WKT and normalized to the ISO dialect.
    wkb_like = []
    for g in geometries:
        if g is None:
            wkb_like.append(None)
        elif hasattr(g, "wkb"):
            wkb_like.append(g.wkb)
        elif isinstance(g, str):
            wkb_like.append(_normalize_wkt(g))
        else:
            wkb_like.append(g)

    if all(g is None for g in wkb_like):
        # pyarrow cannot infer a type from an all-null sequence: declare WKB
        # explicitly so the batch keeps a proper geometry extension type.
        return _ga.array(_pa.array(wkb_like, type=_pa.binary()))
    return _ga.array(wkb_like)


def _batch_z_status(array):
    """Classify a batch by the dimensions actually present in its data.

    Returns a tuple ``(has_xy, has_xyz)`` based on the geometry types observed
    (null-only geometry types are ignored). A WKB array may carry several
    geometry types/dimensions at once.
    """
    unique = _ga.unique_geometry_types(array).to_pylist()
    dims = {item["dimensions"] for item in unique}
    return (_DIM_XY in dims, _DIM_XYZ in dims)


def consolidate_survey_batches(batches, geometry_column="geometry"):
    """Merge named survey-point batches into one XYZ geometry column.

    Parameters
    ----------
    batches:
        Mapping of ``batch_name -> iterable of geometries``. Each geometry may
        be a shapely geometry (``Point`` with or without Z), WKB bytes, WKT
        text, or ``None`` for a missing feature.
    geometry_column:
        Name to give the output geometry column.

    Returns
    -------
    (pandas.Series, dict)
        A geometry series backed by :class:`GeoArrowExtensionArray` whose
        declared dimensions are uniformly ``XYZ`` (or left untouched when no
        input has a Z), plus a provenance dict recording which batches had no
        Z data and which only partially had Z data.

    Examples
    --------
    >>> from shapely import Point
    >>> series, provenance = consolidate_survey_batches({
    ...     "old_survey": [Point(0, 1), None],
    ...     "new_survey": [Point(2, 3, 4)],
    ... })
    >>> provenance["z_missing_batches"]
    ['old_survey']
    """
    promoted_chunks = []
    z_missing_batches = []
    z_partial_batches = []
    any_xyz = False
    batch_order = []

    normalized = []
    for batch_name, geometries in batches.items():
        array = _as_geoarrow_array(geometries)
        has_xy, has_xyz = _batch_z_status(array)
        normalized.append((batch_name, array, has_xy, has_xyz))
        any_xyz = any_xyz or has_xyz

    # First unify every (non-null-only) chunk on one extension type. The
    # geoarrow kernel cannot infer a target type from a null-only array, so
    # those are converted explicitly once the common type is known.
    typed_chunks = [
        item[1] for item in normalized if item[1].null_count != len(item[1])
    ]
    if typed_chunks:
        common_type = _ga.geometry_type_common([c.type for c in typed_chunks])
    else:
        common_type = _ga.wkb()

    if any_xyz:
        common_type = common_type.with_dimensions(_Dimensions.XYZ)

    for batch_name, array, has_xy, has_xyz in normalized:
        batch_order.append(batch_name)
        if has_xyz and has_xy:
            z_partial_batches.append(batch_name)
        if not has_xyz:
            z_missing_batches.append(batch_name)

        if array.null_count == len(array):
            # Null-only batch: rewrap its nulls on the target extension type
            # directly; it has no coordinates to promote or fill.
            array = common_type.wrap_array(array.storage)
        else:
            # Cast first, then promote dimensions. geoarrow fills any added Z
            # coordinate with NaN; it never inserts zero.
            array = _ga.as_geoarrow(array, type=common_type)
            if any_xyz and array.type.dimensions != _Dimensions.XYZ:
                promoted = _ga.with_dimensions(array, _Dimensions.XYZ)
                # with_dimensions may return a different geoarrow encoding
                # (e.g. point instead of wkb); cast back to the common type so
                # every chunk shares one exact type for the chunked array.
                array = _ga.as_geoarrow(promoted, type=common_type)
        promoted_chunks.append(array)

    if len(promoted_chunks) == 1:
        combined = promoted_chunks[0]
    else:
        combined = _ga.as_geoarrow(_pa.chunked_array(promoted_chunks))

    series = _Series(
        _gapd.GeoArrowExtensionArray(combined), name=geometry_column
    )

    provenance = {
        "z_missing_batches": z_missing_batches,
        "z_partial_batches": z_partial_batches,
        "z_missing_rule": Z_MISSING_RULE,
        "output_dimensions": "XYZ" if any_xyz else "XY",
        "batch_order": batch_order,
    }
    return series, provenance


def _provenance_to_schema_metadata(provenance):
    return {
        META_Z_MISSING_BATCHES: _json.dumps(
            provenance["z_missing_batches"], ensure_ascii=False
        ).encode("utf-8"),
        META_Z_PARTIAL_BATCHES: _json.dumps(
            provenance["z_partial_batches"], ensure_ascii=False
        ).encode("utf-8"),
        META_Z_RULE: provenance["z_missing_rule"].encode("utf-8"),
    }


def write_survey_geoparquet(df, path, provenance, geometry_column="geometry"):
    """Write a survey table to standard GeoParquet with provenance metadata.

    The geometry column is written through the geoarrow GeoParquet writer,
    which emits the ``geo`` schema metadata and WKB geometry encoding readable
    by GDAL, DuckDB, GeoPandas and Arrow. The provenance keys (which batches
    lacked Z, and the NaN-not-zero rule) are added as additional schema
    metadata and survive the round trip.

    Parameters
    ----------
    df:
        DataFrame containing ``geometry_column`` as a geoarrow extension
        dtype Series (as produced by :func:`consolidate_survey_batches`).
    path:
        Output Parquet path or file-like object.
    provenance:
        Provenance dict returned by :func:`consolidate_survey_batches`.
    geometry_column:
        Name of the geometry column.
    """
    if not isinstance(
        df[geometry_column].dtype,
        (_gapd.GeoArrowExtensionDtype,),
    ) and not _is_arrow_geometry_dtype(df[geometry_column].dtype):
        raise TypeError(
            f"Column '{geometry_column}' must use a geoarrow extension dtype; "
            f"got {df[geometry_column].dtype!r}"
        )

    table = _pa.Table.from_pandas(df, preserve_index=False)
    metadata = dict(table.schema.metadata or {})
    metadata.update(_provenance_to_schema_metadata(provenance))
    table = table.replace_schema_metadata(metadata)

    _ga_io.write_geoparquet_table(
        table, path, primary_geometry_column=geometry_column
    )
    return path


def _is_arrow_geometry_dtype(dtype):
    import pandas as _pd

    return (
        isinstance(dtype, _pd.ArrowDtype)
        and isinstance(dtype.pyarrow_dtype, _ga.GeometryExtensionType)
    )


def read_survey_geoparquet(path, geometry_column="geometry"):
    """Read a survey GeoParquet back as a DataFrame with an extension dtype.

    Returns
    -------
    (pandas.DataFrame, dict)
        The geometry column is restored on a geoarrow extension dtype (via
        :class:`GeoArrowExtensionArray`) so it is directly usable with the
        ``.geoarrow`` accessor and GeoPandas; the provenance dict is rebuilt
        from the schema metadata.
    """
    table = _ga_io.read_geoparquet_table(path)
    raw_metadata = _pa.parquet.read_schema(path).metadata or {}

    # Prefer the declared geometry types in the GeoParquet "geo" metadata:
    # a WKB column itself reports UNKNOWN dimensions after deserialization.
    output_dimensions = _dimensions_from_geo_metadata(raw_metadata, geometry_column)

    provenance = {
        "z_missing_batches": _decode_list(
            raw_metadata.get(META_Z_MISSING_BATCHES), default=[]
        ),
        "z_partial_batches": _decode_list(
            raw_metadata.get(META_Z_PARTIAL_BATCHES), default=[]
        ),
        "z_missing_rule": raw_metadata.get(META_Z_RULE, b"").decode("utf-8"),
        "output_dimensions": output_dimensions,
    }

    geometry_series = _Series(
        _gapd.GeoArrowExtensionArray(table[geometry_column]),
        name=geometry_column,
    )

    df = table.drop([geometry_column]).to_pandas()
    df[geometry_column] = geometry_series
    return df, provenance


def _decode_list(raw, default):
    if raw is None:
        return default
    return _json.loads(raw.decode("utf-8"))


def _dimensions_from_geo_metadata(raw_metadata, geometry_column):
    """Resolve output dimensions from GeoParquet ``geo`` metadata.

    A WKB column reports ``UNKNOWN`` dimensions once deserialized, so the
    declared ``geometry_types`` labels (e.g. ``"Point Z"``) are authoritative.
    """
    geo_raw = raw_metadata.get(b"geo")
    if geo_raw is None:
        return "UNKNOWN"
    geo_meta = _json.loads(geo_raw.decode("utf-8"))
    type_labels = geo_meta.get("columns", {}).get(geometry_column, {}).get(
        "geometry_types", []
    )
    if any(label.endswith(" ZM") for label in type_labels):
        return "XYZM"
    if any(label.endswith(" Z") for label in type_labels):
        return "XYZ"
    if any(label.endswith(" M") for label in type_labels):
        return "XYM"
    if type_labels:
        return "XY"
    return "UNKNOWN"


def z_is_missing(series):
    """Boolean mask marking rows whose Z is missing (NaN) versus measured.

    Null geometries are excluded (they are not "missing Z", they are missing
    features) and show up as ``False``; use ``series.isna()`` to find them.
    """
    array = _as_geoarrow_array(series)
    # WKB/WKT-encoded columns report UNKNOWN dimensions; resolve the actual
    # coordinate structure through the geoarrow kernel before extracting Z.
    geoarrow_array = _ga.as_geoarrow(array)
    if geoarrow_array.type.dimensions not in (_Dimensions.XYZ, _Dimensions.XYZM):
        raise ValueError(
            "Series has no Z dimension "
            f"(dimensions={geoarrow_array.type.dimensions})"
        )

    # point_coords() works uniformly for Array/ChunkedArray and any geometry
    # type; pass the dimensions explicitly because it defaults to XY and would
    # otherwise drop Z. The third component is Z (X, Y, Z[, M]).
    coords = list(
        _ga.point_coords(
            geoarrow_array, dimensions=geoarrow_array.type.dimensions
        )
    )
    if len(coords) < 3:  # pragma: no cover - defensive
        raise ValueError("Could not extract Z coordinates from series")
    z_values = coords[2]
    if hasattr(z_values, "to_numpy"):
        z_numpy = z_values.to_numpy(zero_copy_only=False)
    else:  # pragma: no cover - defensive across pyarrow versions
        z_numpy = _np.asarray(z_values.to_pylist(), dtype=float)

    valid = array.is_valid()
    if hasattr(valid, "to_numpy"):
        valid = valid.to_numpy(zero_copy_only=False)
    return _np.isnan(z_numpy) & valid
