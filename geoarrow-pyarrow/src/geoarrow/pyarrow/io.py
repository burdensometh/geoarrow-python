"""GeoArrow IO helpers

A module wrapping IO functionality from external libraries to simplify
testing and documenting the GeoArrow format and encodings.

>>> from geoarrow.pyarrow import io
"""

import json
import math
from collections.abc import Mapping

import geoarrow.pyarrow as _ga
from geoarrow.pyarrow import _compute, _type

import pyarrow as _pa
import pyarrow.compute as _pc
import pyarrow.parquet as _pq
import pyarrow.types as _types

# Target number of rows per row group when writing a spatially bucketed
# Parquet file. Smaller row groups make row-group pruning more effective.
DEFAULT_ROW_GROUP_SIZE = 65536

# Name of the column used to record the original row number of each feature
# when rows are spatially reordered. The original order can be recovered with
# (e.g.) ``table.take(pc.sort_indices(table["_original_row_number"]))``.
DEFAULT_ORIGINAL_ROW_COLUMN = "_original_row_number"


def read_pyogrio_table(*args, **kwargs):
    """Read a file using GDAL/OGR

    Reads a file as a ``pyarrow.Table`` using ``pyogrio.raw.read_arrow()``.
    This does not parse the input, which OGR returns as
    :func:`geoarrow.pyarrow.wkb`.

    >>> from geoarrow.pyarrow import io
    >>> import tempfile
    >>> import geopandas
    >>> import os
    >>> with tempfile.TemporaryDirectory() as tmpdir:
    ...     temp_gpkg = os.path.join(tmpdir, "test.gpkg")
    ...     geopandas.GeoDataFrame(
    ...         geometry=geopandas.GeoSeries.from_wkt(["POINT (0 1)"],
    ...         crs="OGC:CRS84")
    ...     ).to_file(temp_gpkg)
    ...     table = io.read_pyogrio_table(temp_gpkg)
    ...     table.column("geom").chunk(0)
    GeometryExtensionArray:WkbType(geoarrow.wkb <...>)[1]
    <POINT (0 1)>
    """
    import pyproj
    from pyogrio.raw import read_arrow

    meta, table = read_arrow(*args, **kwargs)

    # When meta["geometry_name"] is `""`, the geometry column name is wkb_geometry
    # in GDAL's Arrow output. This occurs for sources like shapefile whose geometry
    # has no source-provided name.
    geometry_name = meta["geometry_name"] if meta["geometry_name"] else "wkb_geometry"
    geometry_index = table.schema.get_field_index(geometry_name)

    # Extract geometry information
    geometry_field = table.schema.field(geometry_index)
    geometry = table.column(geometry_index)

    # Always output the geometry column as "geometry"
    if geometry_name == "wkb_geometry":
        geometry_name_out = "geometry"
    else:
        geometry_name_out = geometry_name

    # If extension types are registered and the version of pyogrio + GDAL is
    # sufficient, the geometry field may already be a GeoArrow type. If that
    # is the case, we are done.
    if isinstance(geometry_field.type, _pa.ExtensionType):
        return table.set_column(geometry_index, geometry_name_out, geometry)

    geometry_metadata = geometry_field.metadata
    field_is_geometry = (
        geometry_metadata
        and b"ARROW:extension:name" in geometry_metadata
        and geometry_metadata[b"ARROW:extension:name"] == b"ogc.wkb"
    )

    if not field_is_geometry:
        raise ValueError(
            f"Expected field {geometry_index} ({geometry_name})' to have extension "
            f"name 'ogc.wkb' but got {geometry_field}"
        )

    # Apply geoarrow type to geometry column. This doesn't scale to multiple geometry
    # columns; however, this case is handled by newer GDAL and pyogrio.
    prj_as_json = pyproj.CRS(meta["crs"]).to_json()
    geometry = _ga.wkb().wrap_array(geometry)
    geometry = _ga.with_crs(geometry, prj_as_json)

    return table.set_column(geometry_index, geometry_name_out, geometry)


def read_geoparquet_table(*args, **kwargs):
    """Read GeoParquet using PyArrow

    A thin wrapper around ``pyarrow.parquet.read_parquet()`` that ensures any columns
    marked as geometry are encoded as extension arrays. This will read Parquet files
    with and without the ``geo`` metadata key as described in the GeoParquet format
    and guidance on compatible Parquet. Briefly, this means that any valid GeoParquet
    file (i.e., written by GDAL or :func:`write_geoparquet_table()`) or regular Parquet
    file with a column named ``geometry`` can be read by ``read_geoparquet_table()``.
    For regular Parquet files, a ``geometry`` column encoded as binary is assumed to
    contain well-known binary and a ``geometry`` column encoded as text is assumed to
    contain well-known text.

    Because a ``pyarrow.Table`` has no concept of "primary geometry column", that
    information is currently lost on read. To minimize the chances of misinterpreting
    the primary geometry column, use ``geometry`` as the name of the column or
    refer to columns explicitly.

    >>> import geoarrow.pyarrow as ga
    >>> from geoarrow.pyarrow import io
    >>> import tempfile
    >>> import os
    >>> import pyarrow as pa
    >>> with tempfile.TemporaryDirectory() as tmpdir:
    ...     temp_pq = os.path.join(tmpdir, "test.parquet")
    ...     tab = pa.table([ga.array(["POINT (0 1)"])], names=["geometry"])
    ...     io.write_geoparquet_table(tab, temp_pq)
    ...     tab2 = io.read_geoparquet_table(temp_pq)
    ...     tab2["geometry"].chunk(0)
    GeometryExtensionArray:WkbType(geoarrow.wkb)[1]
    <POINT (0 1)>
    """

    tab = _pq.read_table(*args, **kwargs)
    tab_metadata = tab.schema.metadata if tab.schema.metadata else {}
    if b"geo" in tab_metadata:
        geo_meta = json.loads(tab_metadata[b"geo"])
    else:
        geo_meta = {}

    # Remove "geo" schema metadata key since few transformations following
    # the read operation check that schema metadata is valid (e.g., column
    # subset or rename)
    non_geo_meta = {k: v for k, v in tab_metadata.items() if k != b"geo"}
    tab = tab.replace_schema_metadata(non_geo_meta)

    # Assign extension types to columns
    if "columns" in geo_meta:
        columns = geo_meta["columns"]
    else:
        columns = _geoparquet_guess_geometry_columns(tab.schema)

    return _geoparquet_table_to_geoarrow(tab, columns)


def write_geoparquet_table(
    table,
    *args,
    primary_geometry_column=None,
    geometry_columns=None,
    write_bbox=False,
    write_geometry_types=None,
    check_wkb=True,
    geometry_encoding="WKB",
    spatial_bucketing=None,
    row_group_size=None,
    original_row_column=DEFAULT_ORIGINAL_ROW_COLUMN,
    **kwargs,
):
    """Write GeoParquet using PyArrow

    Writes a Parquet file with the ``geo`` metadata key used by GeoParquet
    readers to recreate geometry types. Note that if you are writing Parquet
    that will be read by an Arrow-based Parquet reader and a GeoArrow
    implementation, using ``pyarrow.parquet.write_parquet()`` will preserve
    geometry types/metadata and is usually faster.

    Note that passing ``write_bbox=True`` and/or ``write_geometry_types=True``
    may be computationally expensive for large input. Use
    `write_geometry_types=False`` to force omitting geometry types even when
    this value is type-constant.

    Pass ``spatial_bucketing`` to write rows in spatial bucket order such that
    each row group contains features from a single grid cell (or from the
    no-coordinate bucket). This makes row group envelopes small, which lets
    :class:`geoarrow.pyarrow.dataset.ParquetRowGroupGeoDataset` skip most row
    groups for a local query. The value can be:

    - ``True``: choose an automatic grid based on ``row_group_size``.
    - A number: the width and height of a grid cell (with the grid origin at
      ``(0, 0)``).
    - A ``(cell_width, cell_height)`` tuple.
    - A ``(origin_x, origin_y, cell_width, cell_height)`` tuple.
    - A mapping with any of ``cell_size``, ``cell_width``/``dx``,
      ``cell_height``/``dy``, ``origin_x``/``xmin``/``x0``,
      ``origin_y``/``ymin``/``y0``, or ``num_cells``/``num_cells_x``/``nx``
      and ``num_cells_y``/``ny`` (the number of cells across the extent of
      the input).

    Records whose geometry is null, empty, or has non-finite coordinates are
    placed at the end of the file in dedicated (null geometry) row groups so
    they do not widen the envelope of any spatial bucket. When
    ``spatial_bucketing`` is enabled, ``original_row_column`` is added to the
    output (unless it is ``None``) and contains the zero-based position of each
    row in ``table`` so the original row order can be recovered with (e.g.)
    ``table.sort_by(original_row_column)``. If a column with that name already
    exists (e.g., when re-bucketing a file that was previously written with
    this option), it is preserved so the original order is not lost.

    See :func:`read_geoparquet_table()` for examples.

    >>> import geoarrow.pyarrow as ga
    >>> from geoarrow.pyarrow import io
    >>> import tempfile
    >>> import os
    >>> import pyarrow as pa
    >>> import pyarrow.parquet as pq
    >>> with tempfile.TemporaryDirectory() as tmpdir:
    ...     temp_pq = os.path.join(tmpdir, "test.parquet")
    ...     tab = pa.table(
    ...         [ga.array(["POINT (9 9)", "POINT (0 0)"])],
    ...         names=["geometry"],
    ...     )
    ...     io.write_geoparquet_table(
    ...         tab,
    ...         temp_pq,
    ...         spatial_bucketing=5,
    ...         row_group_size=1,
    ...     )
    ...     written = pq.read_table(temp_pq)
    ...     row_groups = pq.ParquetFile(temp_pq).metadata.num_row_groups
    ...     print(row_groups)
    ...     print(written["_original_row_number"].to_pylist())
    2
    [1, 0]
    """
    geo_meta = _geoparquet_metadata_from_schema(
        table.schema,
        primary_geometry_column=primary_geometry_column,
        geometry_columns=geometry_columns,
        add_geometry_types=write_geometry_types,
        encoding=geometry_encoding,
    )

    grid = _normalize_spatial_bucketing(
        spatial_bucketing, row_group_size=row_group_size
    )

    # Compute the spatially bucketed row order from the geometry column as
    # provided by the caller (before the encoding step below changes its
    # storage type). This is a no-op when spatial bucketing is not requested.
    order = group_lengths = None
    add_original_row_column = False
    if grid is not None:
        if geo_meta["primary_column"] not in geo_meta["columns"]:
            raise ValueError(
                "Can't use spatial_bucketing without a geometry column: pass "
                "geometry_columns or use a GeoArrow extension type column"
            )

        original_row_column_present = (
            original_row_column is not None
            and original_row_column in table.schema.names
        )
        if original_row_column_present and not _is_integer_type(
            table.schema.field(original_row_column).type
        ):
            raise ValueError(
                "Can't preserve original row number column "
                f"'{original_row_column}': column is not integer-typed"
            )

        # If the original row number column is already present (e.g., when
        # re-bucketing a file that was previously written with bucketing),
        # preserve it as-is so the order relative to the true input is kept.
        add_original_row_column = (
            original_row_column is not None and not original_row_column_present
        )

        order, group_lengths = _spatial_buckets(
            table[geo_meta["primary_column"]],
            grid,
            row_group_size=row_group_size,
        )

    # Note: this will also update geo_meta with geometry_types and bbox if requested
    for i, name in enumerate(table.schema.names):
        if name in geo_meta["columns"]:
            table = table.set_column(
                i,
                name,
                _geoparquet_encode_chunked_array(
                    table[i],
                    geo_meta["columns"][name],
                    add_geometry_types=write_geometry_types,
                    add_bbox=write_bbox,
                    check_wkb=check_wkb,
                ),
            )

    metadata = table.schema.metadata if table.schema.metadata else {}
    metadata["geo"] = json.dumps(geo_meta)

    if order is None:
        table = table.replace_schema_metadata(metadata)
        if row_group_size is not None:
            kwargs["row_group_size"] = row_group_size
        return _pq.write_table(table, *args, **kwargs)

    if add_original_row_column:
        table = table.append_column(
            original_row_column,
            _pa.array(range(table.num_rows), type=_pa.int64()),
        )

    table = table.replace_schema_metadata(metadata).take(order)
    return _write_geoparquet_buckets(
        table,
        group_lengths,
        args,
        **kwargs,
    )


def _normalize_spatial_bucketing(spec, row_group_size=None):
    """Normalize the ``spatial_bucketing`` argument to a grid specification

    Returns ``None`` when spatial bucketing is not requested. Otherwise
    returns a dict with ``origin_x``, ``origin_y``, ``cell_width``,
    ``cell_height``, and ``row_group_size``. The special value ``True``
    selects an automatic, square grid based on ``row_group_size``.
    """
    if spec is None or spec is False:
        return None

    if row_group_size is None:
        row_group_size = DEFAULT_ROW_GROUP_SIZE
    elif isinstance(row_group_size, bool) or not isinstance(row_group_size, int):
        raise TypeError("row_group_size must be an integer")
    elif row_group_size <= 0:
        raise ValueError("row_group_size must be greater than zero")

    if spec is True:
        return {
            "auto": True,
            "origin_x": 0.0,
            "origin_y": 0.0,
            "cell_width": None,
            "cell_height": None,
            "num_cells_x": None,
            "num_cells_y": None,
            "row_group_size": row_group_size,
        }

    origin_x = 0.0
    origin_y = 0.0
    cell_width = cell_height = None
    num_cells_x = num_cells_y = None

    if isinstance(spec, Mapping):
        if "cell_size" in spec:
            cell_width = cell_height = spec["cell_size"]
        cell_width = spec.get("cell_width", spec.get("dx", cell_width))
        cell_height = spec.get("cell_height", spec.get("dy", cell_height))
        origin_x = spec.get("origin_x", spec.get("xmin", spec.get("x0", origin_x)))
        origin_y = spec.get("origin_y", spec.get("ymin", spec.get("y0", origin_y)))

        if "num_cells" in spec:
            num_cells_x = num_cells_y = spec["num_cells"]
        num_cells_x = spec.get("num_cells_x", spec.get("nx", num_cells_x))
        num_cells_y = spec.get("num_cells_y", spec.get("ny", num_cells_y))
    elif isinstance(spec, (tuple, list)):
        if len(spec) == 2:
            cell_width, cell_height = spec
        elif len(spec) == 4:
            origin_x, origin_y, cell_width, cell_height = spec
        else:
            raise ValueError(
                "spatial_bucketing tuple must have 2 or 4 elements, got "
                f"{len(spec)}"
            )
    else:
        cell_width = cell_height = spec

    if cell_width is not None:
        cell_width = _validate_cell_size("cell_width", cell_width)
    if cell_height is not None:
        cell_height = _validate_cell_size("cell_height", cell_height)
    if num_cells_x is not None:
        num_cells_x = _validate_num_cells("num_cells_x", num_cells_x)
    if num_cells_y is not None:
        num_cells_y = _validate_num_cells("num_cells_y", num_cells_y)

    if (
        cell_width is None
        and cell_height is None
        and num_cells_x is None
        and num_cells_y is None
    ):
        raise ValueError(
            "spatial_bucketing must specify a cell size or a number of cells"
        )

    # A single cell size (or number of cells) applies to both axes.
    if num_cells_x is None and num_cells_y is None:
        if cell_width is None:
            cell_width = cell_height
        if cell_height is None:
            cell_height = cell_width

    origin_x = _validate_origin("origin_x", origin_x)
    origin_y = _validate_origin("origin_y", origin_y)

    return {
        "auto": False,
        "origin_x": origin_x,
        "origin_y": origin_y,
        "cell_width": cell_width,
        "cell_height": cell_height,
        "num_cells_x": num_cells_x,
        "num_cells_y": num_cells_y,
        "row_group_size": row_group_size,
    }


def _validate_cell_size(name, value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        raise TypeError(f"spatial_bucketing {name} must be a number") from None

    if not math.isfinite(value) or value <= 0:
        raise ValueError(
            f"spatial_bucketing {name} must be a finite positive number"
        )

    return value


def _validate_num_cells(name, value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"spatial_bucketing {name} must be a number")

    if not math.isfinite(value) or value < 1:
        raise ValueError(f"spatial_bucketing {name} must be at least 1")

    return int(value)


def _validate_origin(name, value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        raise TypeError(f"spatial_bucketing {name} must be a number") from None

    if not math.isfinite(value):
        raise ValueError(f"spatial_bucketing {name} must be finite")

    return value


def _is_integer_type(type):
    return _pa.types.is_integer(type)


def _box_components(geometry):
    """Compute finite box components and a validity mask for ``geometry``"""
    if _pa.types.is_null(geometry.type):
        # A null-typed column has no coordinates; treat every row as a
        # no-coordinate record.
        null = _pa.chunked_array(
            [_pa.array([None] * len(geometry), type=_pa.float64())]
        )
        return null, null, null, null, _pa.chunked_array(
            [_pa.array([False] * len(geometry), type=_pa.bool_())]
        )

    storage = _compute.ensure_storage(_ga.box(geometry))

    xmin = _pc.struct_field(storage, "xmin")
    ymin = _pc.struct_field(storage, "ymin")
    xmax = _pc.struct_field(storage, "xmax")
    ymax = _pc.struct_field(storage, "ymax")

    # Null geometries have a null box and empty geometries have an infinite
    # box; both are treated as "no coordinates" and bucketed separately.
    has_coords = _pc.and_(
        _pc.and_(_pc.is_finite(xmin), _pc.is_finite(ymin)),
        _pc.and_(_pc.is_finite(xmax), _pc.is_finite(ymax)),
    )
    has_coords = _pc.fill_null(has_coords, False)

    return xmin, ymin, xmax, ymax, has_coords


def _finite_extent(lo, hi, has_coords):
    """Compute the extent of ``hi - lo`` over rows that have coordinates"""
    null = _pa.scalar(None, type=_pa.float64())
    lo = _pc.if_else(has_coords, lo, null)
    hi = _pc.if_else(has_coords, hi, null)
    lo = _pc.min(lo).as_py()
    hi = _pc.max(hi).as_py()
    if lo is None or hi is None or not math.isfinite(hi - lo):
        return 0.0
    return hi - lo


def _resolve_auto_cell_size(xmin, ymin, xmax, ymax, has_coords, num_coords, target):
    """Choose a square cell size targeting ``target`` rows per cell"""
    if num_coords == 0:
        return 1.0

    # Aim for roughly `target` rows per cell on each axis (STR-like).
    num_axis = max(1, round(math.sqrt(num_coords / target)))

    cell_width = _finite_extent(xmin, xmax, has_coords) / num_axis
    cell_height = _finite_extent(ymin, ymax, has_coords) / num_axis
    cell_size = max(cell_width, cell_height)

    return cell_size if cell_size > 0 else 1.0


def _spatial_buckets(geometry, grid, row_group_size=None):
    """Compute a bucketed row order and per-row-group lengths

    Returns a tuple ``(order, group_lengths)`` where ``order`` is an integer
    array of original row numbers in spatially bucketed order and
    ``group_lengths`` is a list of the number of rows in each written row
    group. Features without coordinates are placed at the end, one bucket
    containing only null/empty geometries, so they cannot widen the envelope
    of any spatial bucket.
    """
    if row_group_size is None:
        row_group_size = grid.get("row_group_size", DEFAULT_ROW_GROUP_SIZE)

    xmin, ymin, xmax, ymax, has_coords = _box_components(geometry)
    num_rows = len(geometry)

    cell_width = grid["cell_width"]
    cell_height = grid["cell_height"]

    if grid.get("auto"):
        num_coords = _pc.sum(_pc.cast(has_coords, _pa.int64())).as_py() or 0
        cell_width = cell_height = _resolve_auto_cell_size(
            xmin, ymin, xmax, ymax, has_coords, num_coords, row_group_size
        )
    elif grid["num_cells_x"] is not None or grid["num_cells_y"] is not None:
        # Convert a requested number of cells into a cell size using the
        # extent of the features that have coordinates.
        num_cells_x = grid["num_cells_x"]
        num_cells_y = grid["num_cells_y"]
        if cell_width is None:
            if num_cells_x is None:
                num_cells_x = num_cells_y
            cell_width = (
                _finite_extent(xmin, xmax, has_coords) / num_cells_x
                if num_cells_x
                else 0.0
            )
        if cell_height is None:
            if num_cells_y is None:
                num_cells_y = num_cells_x
            cell_height = (
                _finite_extent(ymin, ymax, has_coords) / num_cells_y
                if num_cells_y
                else 0.0
            )
        cell_width = cell_width if cell_width > 0 else 1.0
        cell_height = cell_height if cell_height > 0 else 1.0

    origin_x = grid["origin_x"]
    origin_y = grid["origin_y"]

    # Use the box midpoint so features larger than a cell stay in one bucket
    # and are keyed by the cell containing their center.
    null = _pa.scalar(None, type=_pa.float64())
    center_x = _pc.if_else(
        has_coords, _pc.divide(_pc.add(xmin, xmax), _pa.scalar(2.0)), null
    )
    center_y = _pc.if_else(
        has_coords, _pc.divide(_pc.add(ymin, ymax), _pa.scalar(2.0)), null
    )

    ix = _pc.floor(_pc.divide(_pc.subtract(center_x, origin_x), cell_width))
    iy = _pc.floor(_pc.divide(_pc.subtract(center_y, origin_y), cell_height))

    # Clamp to the range of integers that a double can represent exactly so
    # that very large (but finite) coordinates cannot overflow the cast to
    # int64. Features in the same clamped cell still share a row group.
    limit = float(2**52)
    ix = _pc.max_element_wise(_pc.min_element_wise(ix, limit), -limit)
    iy = _pc.max_element_wise(_pc.min_element_wise(iy, limit), -limit)
    ix = _pc.cast(ix, _pa.int64())
    iy = _pc.cast(iy, _pa.int64())

    keys = _pa.table(
        {
            "_iy": iy,
            "_ix": ix,
            "_order": _pa.array(range(num_rows), type=_pa.int64()),
        }
    )

    # Nulls (no-coordinate rows) sort last, forming a single trailing bucket.
    order = _pc.sort_indices(
        keys,
        sort_keys=[
            ("_iy", "ascending"),
            ("_ix", "ascending"),
            ("_order", "ascending"),
        ],
    )

    if num_rows == 0:
        return order, []

    sorted_iy = keys["_iy"].take(order)
    sorted_ix = keys["_ix"].take(order)

    def differs(values):
        left = values.slice(1)
        right = values.slice(0, len(values) - 1)
        not_equal = _pc.fill_null(_pc.not_equal(left, right), False)
        null_transition = _pc.not_equal(_pc.is_null(left), _pc.is_null(right))
        return _pc.or_(not_equal, null_transition)

    changed = _pc.or_(differs(sorted_iy), differs(sorted_ix))
    boundaries = [0] + [i + 1 for i in _pc.indices_nonzero(changed).to_pylist()]
    boundaries.append(num_rows)

    group_lengths = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        length = end - start
        while length > 0:
            group_lengths.append(min(length, row_group_size))
            length -= row_group_size

    return order, group_lengths


def _write_geoparquet_buckets(table, group_lengths, args, **kwargs):
    """Write ``table`` with one row group per spatial bucket (or sub-bucket)"""
    if args and "where" in kwargs:
        raise TypeError(
            "write_geoparquet_table() got multiple values for argument 'where'"
        )

    if args:
        if len(args) > 1:
            raise TypeError(
                "write_geoparquet_table() accepts a single 'where' argument"
            )
        where = args[0]
    elif "where" in kwargs:
        where = kwargs.pop("where")
    else:
        raise TypeError("write_geoparquet_table() missing required 'where'")

    if table.num_rows == 0:
        return _pq.write_table(table, where, **kwargs)

    writer = _pq.ParquetWriter(where, table.schema, **kwargs)
    try:
        offset = 0
        for length in group_lengths:
            piece = table.slice(offset, length).combine_chunks()
            writer.write_table(piece, row_group_size=length)
            offset += length
    finally:
        writer.close()

    return None


def _geoparquet_guess_geometry_columns(schema):
    # Only attempt guessing the "geometry" or "geography" column
    columns = {}

    for name in ("geometry", "geography"):
        if name not in schema.names:
            continue

        spec = {}
        type = schema.field(name).type

        if _types.is_binary(type) or _types.is_large_binary(type):
            spec["encoding"] = "WKB"
        elif _types.is_string(type) or _types.is_large_string(type):
            # WKT is not actually a geoparquet encoding but the guidance on
            # putting geospatial things in parquet without metadata says you
            # can do it and this is the internal sentinel for that case.
            spec["encoding"] = "WKT"

        # A column named "geography" has spherical edges according to the
        # compatible Parquet guidance.
        if name == "geography":
            spec["edges"] = "spherical"

        columns[name] = spec

    return columns


def _geoparquet_chunked_array_to_geoarrow(item, spec):
    # If item was written as a GeoArrow extension type to the Parquet file,
    # ignore any information in the column spec
    if isinstance(item.type, _ga.GeometryExtensionType):
        return item

    if "encoding" not in spec:
        raise ValueError("Invalid GeoParquet column specification: missing 'encoding'")

    encoding = spec["encoding"]
    if encoding in ("WKB", "WKT"):
        item = _ga.array(item)
    elif encoding in _GEOARROW_ENCODINGS:
        type = _GEOARROW_ENCODINGS[encoding].__arrow_ext_deserialize__(item.type, b"")
        item = type.wrap_array(item)
    else:
        raise ValueError(f"Invalid GeoParquet encoding value: '{encoding}'")

    if "crs" not in spec:
        crs = _CRS_LONLAT
    else:
        crs = spec["crs"]

    if "edges" not in spec or spec["edges"] == "planar":
        edge_type = None
    elif spec["edges"] == "spherical":
        edge_type = _ga.EdgeType.SPHERICAL
    else:
        raise ValueError("Invalid GeoParquet column edges value")

    if crs is not None:
        item = _ga.with_crs(item, crs)

    if edge_type is not None:
        item = _ga.with_edge_type(item, edge_type)

    return item


def _geoparquet_table_to_geoarrow(tab, columns):
    tab_names = set(tab.schema.names)
    for col_name, spec in columns.items():
        # col_name might not exist if only a subset of columns were read from file
        if col_name not in tab_names:
            continue

        col_i = tab.schema.get_field_index(col_name)
        new_geometry = _geoparquet_chunked_array_to_geoarrow(tab[col_i], spec)
        tab = tab.set_column(col_i, col_name, new_geometry)

    return tab


def _geoparquet_guess_primary_geometry_column(schema, primary_geometry_column=None):
    if primary_geometry_column is not None:
        return primary_geometry_column

    # If there's a "geometry" or "geography" column, pick that one
    if "geometry" in schema.names:
        return "geometry"
    elif "geography" in schema.names:
        return "geography"

    # Otherwise, pick the first thing we know is actually geometry
    for name, type in zip(schema.names, schema.types):
        if isinstance(type, _ga.GeometryExtensionType):
            return name

    raise ValueError(
        "write_geoparquet_table() requires source with at least one geometry column"
    )


def _geoparquet_column_spec_from_type(type, add_geometry_types=None, encoding=None):
    spec = {"encoding": encoding, "geometry_types": []}

    # Pass along extra information from GeoArrow extension type metadata
    if isinstance(type, _ga.GeometryExtensionType):
        # If encoding is unspecified and data is already geoarrow, don't serialize to WKB
        if encoding is None and type.coord_type != _ga.CoordType.UNSPECIFIED:
            spec["encoding"] = geoparquet_encoding_geoarrow()

        crs = type.crs
        if crs is None:
            spec["crs"] = None
        else:
            spec["crs"] = type.crs.to_json_dict()

        if type.edge_type != _ga.EdgeType.PLANAR:
            spec["edges"] = type.edge_type.name.lower()

        # GeoArrow-encoded types can confidently declare a single geometry type
        maybe_known_geometry_type = type.geometry_type
        maybe_known_dimensions = type.dimensions
        if (
            add_geometry_types is not False
            and maybe_known_geometry_type != _ga.GeometryType.GEOMETRY
            and maybe_known_dimensions != _ga.Dimensions.UNKNOWN
        ):
            geometry_type = _GEOPARQUET_GEOMETRY_TYPE_LABELS[
                maybe_known_geometry_type.value
            ]
            dimensions = _GEOPARQUET_DIMENSION_LABELS[maybe_known_dimensions.value]
            spec["geometry_types"] = [f"{geometry_type}{dimensions}"]

    if spec["encoding"] is None:
        spec["encoding"] = "WKB"

    return spec


def _geoparquet_columns_from_schema(
    schema,
    geometry_columns=None,
    primary_geometry_column=None,
    add_geometry_types=None,
    encoding=None,
):
    schema_names = schema.names
    schema_types = schema.types

    if geometry_columns is None:
        geometry_columns = set()
        if primary_geometry_column is not None:
            geometry_columns.add(primary_geometry_column)

        for name, type in zip(schema_names, schema_types):
            if isinstance(type, _ga.GeometryExtensionType):
                geometry_columns.add(name)
    else:
        geometry_columns = set(geometry_columns)

    specs = {}
    for name, type in zip(schema_names, schema_types):
        if name in geometry_columns:
            specs[name] = _geoparquet_column_spec_from_type(
                type, add_geometry_types=add_geometry_types, encoding=encoding
            )

    return specs


def _geoparquet_metadata_from_schema(
    schema,
    geometry_columns=None,
    primary_geometry_column=None,
    add_geometry_types=None,
    encoding=None,
):
    primary_geometry_column = _geoparquet_guess_primary_geometry_column(
        schema, primary_geometry_column
    )
    columns = _geoparquet_columns_from_schema(
        schema,
        geometry_columns,
        add_geometry_types=add_geometry_types,
        encoding=encoding,
    )

    return {
        "version": _geoparquet_get_version_from_columns(columns),
        "primary_column": primary_geometry_column,
        "columns": columns,
    }


def _geoparquet_get_version_from_columns(columns):
    for column in columns.values():
        if column["encoding"] != "WKB":
            return "1.1.0"

    return "1.0.0"


def _geoparquet_update_spec_geometry_types(item, spec, unique_geometry_types=None):
    # Reuse a previous computation if possible
    if unique_geometry_types is None:
        unique_geometry_types = _ga.unique_geometry_types(item)

    geometry_type_labels = []
    for element in unique_geometry_types.to_pylist():
        geometry_type = _GEOPARQUET_GEOMETRY_TYPE_LABELS[element["geometry_type"]]
        dimensions = _GEOPARQUET_DIMENSION_LABELS[element["dimensions"]]
        geometry_type_labels.append(f"{geometry_type}{dimensions}")

    spec["geometry_types"] = geometry_type_labels


def _geoparquet_update_spec_bbox(item, spec):
    box = _ga.box_agg(item).as_py()
    spec["bbox"] = [box["xmin"], box["ymin"], box["xmax"], box["ymax"]]


def _geoparquet_encode_chunked_array(
    item, spec, add_geometry_types=None, add_bbox=False, check_wkb=True
):
    # For geoarrow encodings we still need to collect the geometry types
    # and infer what we think the encoding might be to ensure it is appropriate
    unique_geometry_types = None
    geoarrow_type = None
    inferred_geoarrow_encoding = None
    if spec["encoding"] in (
        tuple(_GEOARROW_ENCODINGS.keys()) + (geoparquet_encoding_geoarrow(),)
    ):
        unique_geometry_types = _ga.unique_geometry_types(item)
        geoarrow_type = _ga.infer_type_common(
            item,
            coord_type=_ga.CoordType.SEPARATED,
            _geometry_types=unique_geometry_types,
        )

        if geoarrow_type.coord_type == _ga.CoordType.UNSPECIFIED:
            raise ValueError(
                "Can't encode column with incompatable geometry types as geoarrow"
            )

        inferred_geoarrow_encoding = geoarrow_type._extension_name[len("geoarrow.") :]

    # If the encoding was the "give me any geoarrow" sentinel, we need to
    # update it to be an actual valid encoding value
    if spec["encoding"] == geoparquet_encoding_geoarrow():
        spec["encoding"] = inferred_geoarrow_encoding

    if spec["encoding"] == "WKB":
        item_out = _ga.as_wkb(item, strict_iso_wkb=check_wkb)

        # If input was WKT, use the WKB-encoded value to perform calculations
        if (
            _pa.types.is_string(item.type)
            or _pa.types.is_large_string(item.type)
            or isinstance(item.type, _ga.WktType)
        ):
            item_calc = item_out
        else:
            item_calc = item

    elif spec["encoding"] in _GEOARROW_ENCODINGS:
        # Check that this encoding is appropriate for item
        if inferred_geoarrow_encoding != spec["encoding"]:
            raise ValueError(
                f"Can't encode column with encoding='{spec['encoding']}'. "
                f"Inferred encoding: '{inferred_geoarrow_encoding}'"
            )

        item_out = _ga.as_geoarrow(item, geoarrow_type)
        # Use geoarrow-encoded value to perform further calculations
        item_calc = item_out

    else:
        encoding = spec["encoding"]
        geoarrow_list = ", ".join("'" + item + "'" for item in _GEOARROW_ENCODINGS)
        raise ValueError(
            f"Expected column encoding to be one of 'WKB', {geoarrow_list} but got '{encoding}'"
        )

    # geometry_types that are fixed at the data type level have already been
    # added to the spec in an earlier step. The unique_geometry_types()
    # function is sufficiently optimized such that this potential
    # re-computation is not expensive. We also might have calculated them
    # already to infer the encoding value.
    if add_geometry_types is True:
        _geoparquet_update_spec_geometry_types(item_calc, spec, unique_geometry_types)

    if add_bbox:
        _geoparquet_update_spec_bbox(item_calc, spec)

    return _compute.ensure_storage(item_out)


def geoparquet_encoding_geoarrow():
    return "GEOARROW_ENCODING_SENTINEL"


_GEOPARQUET_GEOMETRY_TYPE_LABELS = [
    "Geometry",
    "Point",
    "LineString",
    "Polygon",
    "MultiPoint",
    "MultiLineString",
    "MultiPolygon",
    "GeometryCollection",
]

_GEOPARQUET_DIMENSION_LABELS = [None, "", " Z", " M", " ZM"]

_GEOARROW_ENCODINGS = {
    "point": _type.point(),
    "linestring": _type.linestring(),
    "polygon": _type.polygon(),
    "multipoint": _type.multipoint(),
    "multilinestring": _type.multilinestring(),
    "multipolygon": _type.multipolygon(),
}

_CRS_LONLAT = {
    "$schema": "https://proj.org/schemas/v0.7/projjson.schema.json",
    "type": "GeographicCRS",
    "name": "WGS 84 (CRS84)",
    "datum_ensemble": {
        "name": "World Geodetic System 1984 ensemble",
        "members": [
            {
                "name": "World Geodetic System 1984 (Transit)",
                "id": {"authority": "EPSG", "code": 1166},
            },
            {
                "name": "World Geodetic System 1984 (G730)",
                "id": {"authority": "EPSG", "code": 1152},
            },
            {
                "name": "World Geodetic System 1984 (G873)",
                "id": {"authority": "EPSG", "code": 1153},
            },
            {
                "name": "World Geodetic System 1984 (G1150)",
                "id": {"authority": "EPSG", "code": 1154},
            },
            {
                "name": "World Geodetic System 1984 (G1674)",
                "id": {"authority": "EPSG", "code": 1155},
            },
            {
                "name": "World Geodetic System 1984 (G1762)",
                "id": {"authority": "EPSG", "code": 1156},
            },
            {
                "name": "World Geodetic System 1984 (G2139)",
                "id": {"authority": "EPSG", "code": 1309},
            },
        ],
        "ellipsoid": {
            "name": "WGS 84",
            "semi_major_axis": 6378137,
            "inverse_flattening": 298.257223563,
        },
        "accuracy": "2.0",
        "id": {"authority": "EPSG", "code": 6326},
    },
    "coordinate_system": {
        "subtype": "ellipsoidal",
        "axis": [
            {
                "name": "Geodetic longitude",
                "abbreviation": "Lon",
                "direction": "east",
                "unit": "degree",
            },
            {
                "name": "Geodetic latitude",
                "abbreviation": "Lat",
                "direction": "north",
                "unit": "degree",
            },
        ],
    },
    "scope": "Not known.",
    "area": "World.",
    "bbox": {
        "south_latitude": -90,
        "west_longitude": -180,
        "north_latitude": 90,
        "east_longitude": 180,
    },
    "id": {"authority": "OGC", "code": "CRS84"},
}
