"""GeoArrow IO helpers

A module wrapping IO functionality from external libraries to simplify
testing and documenting the GeoArrow format and encodings.

>>> from geoarrow.pyarrow import io
"""

import json

import geoarrow.pyarrow as _ga
from geoarrow.pyarrow import _compute, _type

import pyarrow as _pa
import pyarrow.parquet as _pq
import pyarrow.types as _types


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


def read_geoparquet_table(*args, primary_geometry_column=None, **kwargs):
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

    A ``pyarrow.Table`` has no built-in concept of a "primary geometry column";
    however, this function records the role that each geometry column had when it
    was written. The primary geometry column is marked using the field metadata key
    ``b"geoarrow:geometry_role"`` (``b"primary"`` or ``b"secondary"``) and the name
    of the recorded primary column is stored in the schema metadata key
    ``b"geoarrow:primary_geometry_column"``. This means that
    :func:`write_geoparquet_table()` can restore the same primary geometry column
    (and will refuse to guess a new one) even if the table is projected or
    otherwise transformed in between.

    If a projection excludes the primary geometry column (e.g.,
    ``columns=["coverage"]``), the presence of the primary is recorded as absent.
    In this case the returned table has no primary role. Pass
    ``primary_geometry_column`` to explicitly designate one of the projected
    geometry columns as the new primary. Otherwise, a subsequent call to
    :func:`write_geoparquet_table()` will require an explicit
    ``primary_geometry_column`` rather than silently promoting a different column.

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

    tab = _geoparquet_table_to_geoarrow(tab, columns)

    # Record the primary/secondary role that each geometry column had when it was
    # written. A projection that removed the primary column still records that
    # geometry roles exist, but with no primary, so that a later write is forced
    # to be explicit instead of silently promoting a different column.
    geometry_columns = [name for name in tab.schema.names if name in columns]
    if primary_geometry_column is not None and (
        primary_geometry_column not in geometry_columns
    ):
        raise ValueError(
            f"Requested primary geometry column '{primary_geometry_column}' is not "
            "a geometry column present in the read result"
        )

    roles_are_recorded = "columns" in geo_meta
    if geometry_columns and (roles_are_recorded or primary_geometry_column is not None):
        recorded_primary = geo_meta.get("primary_column")
        if primary_geometry_column is None and recorded_primary in geometry_columns:
            primary_out = recorded_primary
        else:
            primary_out = primary_geometry_column

        tab = _geoparquet_with_geometry_roles(tab, geometry_columns, primary_out)
    elif not geometry_columns and _PRIMARY_GEOMETRY_COLUMN_KEY in non_geo_meta:
        # A projection that removed every geometry column must not leave a stale
        # primary pointing at a column that is no longer there.
        non_geo_meta = dict(non_geo_meta)
        del non_geo_meta[_PRIMARY_GEOMETRY_COLUMN_KEY]
        tab = tab.replace_schema_metadata(non_geo_meta)

    return tab


def write_geoparquet_table(
    table,
    *args,
    primary_geometry_column=None,
    geometry_columns=None,
    write_bbox=False,
    write_geometry_types=None,
    check_wkb=True,
    geometry_encoding="WKB",
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

    If the input table was produced by :func:`read_geoparquet_table()` (or
    otherwise carries geometry roles recorded by it), the primary geometry
    column is restored from that recorded role. In that case a missing primary
    geometry column is an error: if a projection removed the originally primary
    column, pass ``primary_geometry_column`` explicitly to designate a new one.
    This prevents silently promoting an unrelated coordinate column (e.g., a
    coverage column) to the primary role.

    See :func:`read_geoparquet_table()` for examples.
    """
    primary_geometry_column, primary_is_designated = (
        _geoparquet_resolve_primary_geometry_column(
            table.schema, primary_geometry_column
        )
    )

    geo_meta = _geoparquet_metadata_from_schema(
        table.schema,
        primary_geometry_column=primary_geometry_column,
        geometry_columns=geometry_columns,
        add_geometry_types=write_geometry_types,
        encoding=geometry_encoding,
    )

    # The resolved primary must actually be written as a geometry column. If it
    # was designated (explicitly or recorded by read_geoparquet_table()), doing
    # anything else would silently substitute a different column as the center.
    if primary_geometry_column not in geo_meta["columns"]:
        if primary_is_designated:
            raise ValueError(
                f"Primary geometry column '{primary_geometry_column}' is not among "
                "the geometry columns being written"
            )

        if not geo_meta["columns"]:
            raise ValueError(
                "write_geoparquet_table() requires source with at least one "
                "geometry column"
            )

        primary_geometry_column = sorted(geo_meta["columns"])[0]
        geo_meta["primary_column"] = primary_geometry_column

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

    # Keep the geometry roles in sync with the primary column actually written
    table = _geoparquet_with_geometry_roles(
        table,
        list(geo_meta["columns"].keys()),
        geo_meta["primary_column"],
    )

    metadata = table.schema.metadata if table.schema.metadata else {}
    metadata["geo"] = json.dumps(geo_meta)
    table = table.replace_schema_metadata(metadata)
    return _pq.write_table(table, *args, **kwargs)


def geoparquet_primary_geometry_column(table):
    """Get the recorded primary geometry column of a table

    Returns the name of the primary geometry column recorded by
    :func:`read_geoparquet_table()` or ``None`` if no primary role is recorded
    (including when the primary column was removed by a projection). This is the
    role that :func:`write_geoparquet_table()` will restore when writing.

    Parameters
    ----------
    table : pyarrow.Table or pyarrow.Schema
        The table (or schema) to query.

    >>> import geoarrow.pyarrow as ga
    >>> from geoarrow.pyarrow import io
    >>> import pyarrow as pa
    >>> tab = pa.table([ga.array(["POINT (0 1)"]), ga.array(["POINT (1 2)"])],
    ...                names=["center", "coverage"])
    >>> io.geoparquet_primary_geometry_column(tab) is None
    True
    """
    schema = table if isinstance(table, _pa.Schema) else table.schema
    _, primary = _geoparquet_get_recorded_primary(schema)
    return primary


def _geoparquet_resolve_primary_geometry_column(schema, primary_geometry_column=None):
    """Resolve the primary geometry column, also reporting whether it was designated

    Returns a tuple ``(name, is_designated)``. ``is_designated`` is ``True`` if the
    name came from the caller or from roles recorded by
    :func:`read_geoparquet_table()` (in which case it must not be replaced by a
    guess) and ``False`` if it was guessed from the schema.
    """
    if primary_geometry_column is not None:
        if primary_geometry_column not in schema.names:
            raise ValueError(
                f"Requested primary geometry column '{primary_geometry_column}' is "
                "not present in the table"
            )

        return primary_geometry_column, True

    # Roles recorded by read_geoparquet_table() take precedence over guessing.
    roles_are_recorded, recorded_primary = _geoparquet_get_recorded_primary(schema)
    if roles_are_recorded:
        if recorded_primary is None:
            raise ValueError(
                "Can't determine the primary geometry column: the table has "
                "geometry roles recorded but no primary geometry column (for "
                "example, because the primary column was removed by a projection). "
                "Pass primary_geometry_column= to designate one explicitly"
            )

        if recorded_primary not in schema.names:
            raise ValueError(
                f"Recorded primary geometry column '{recorded_primary}' is not "
                "present in the table. Pass primary_geometry_column= to designate "
                "one explicitly"
            )

        return recorded_primary, True

    return _geoparquet_guess_primary_geometry_column(schema), False


def _geoparquet_geometry_role(field):
    if field.metadata is None:
        return None

    role = field.metadata.get(_GEOMETRY_ROLE_KEY)
    if role is None:
        return None

    return role.decode()


def _geoparquet_get_recorded_primary(schema):
    # Roles are recorded in two places so that the information survives common
    # table transformations: per-field metadata (travels with the column through
    # select/drop/filter/rename) and schema metadata (also survives
    # select/drop/filter but is dropped by rename_columns). Field roles are
    # authoritative because they follow renames; schema metadata is a fallback
    # for tables whose column-level metadata was stripped.
    primary_roles = []
    secondary_roles = []
    for field in schema:
        role = _geoparquet_geometry_role(field)
        if role == _GEOMETRY_ROLE_PRIMARY:
            primary_roles.append(field.name)
        elif role == _GEOMETRY_ROLE_SECONDARY:
            secondary_roles.append(field.name)

    if len(primary_roles) > 1:
        raise ValueError(
            "Table has more than one geometry column marked as primary: "
            + ", ".join(primary_roles)
        )

    if primary_roles:
        return True, primary_roles[0]

    if secondary_roles:
        # Geometry roles were recorded but no column carries the primary role:
        # the primary column was removed by a projection.
        return True, None

    schema_metadata = schema.metadata if schema.metadata else {}
    if _PRIMARY_GEOMETRY_COLUMN_KEY in schema_metadata:
        primary = schema_metadata[_PRIMARY_GEOMETRY_COLUMN_KEY].decode()
        if primary and primary in schema.names:
            return True, primary

        # The recorded primary column is no longer present.
        return True, None

    return False, None


def _geoparquet_with_geometry_roles(tab, geometry_columns, primary_geometry_column):
    geometry_columns = set(geometry_columns)
    fields = []
    for field in tab.schema:
        if field.name in geometry_columns:
            metadata = dict(field.metadata) if field.metadata else {}
            if field.name == primary_geometry_column:
                metadata[_GEOMETRY_ROLE_KEY] = _GEOMETRY_ROLE_PRIMARY.encode()
            else:
                metadata[_GEOMETRY_ROLE_KEY] = _GEOMETRY_ROLE_SECONDARY.encode()
            field = field.with_metadata(metadata)
        fields.append(field)

    metadata = dict(tab.schema.metadata) if tab.schema.metadata else {}
    if primary_geometry_column is None:
        metadata[_PRIMARY_GEOMETRY_COLUMN_KEY] = b""
    else:
        metadata[_PRIMARY_GEOMETRY_COLUMN_KEY] = primary_geometry_column.encode()

    schema = _pa.schema(fields, metadata=metadata)
    return _pa.Table.from_arrays(tab.columns, schema=schema)


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

        # Preserve any field-level metadata (e.g., a recorded geometry role)
        field = tab.schema.field(col_i)
        new_field = _pa.field(
            field.name,
            new_geometry.type,
            nullable=field.nullable,
            metadata=field.metadata,
        )
        tab = tab.set_column(col_i, new_field, new_geometry)

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


# Schema metadata key recording the name of the primary geometry column. An empty
# value means that the source had geometry roles but no primary (e.g., the primary
# column was projected away).
_PRIMARY_GEOMETRY_COLUMN_KEY = b"geoarrow:primary_geometry_column"

# Field metadata key recording the role of a geometry column ("primary" or
# "secondary").
_GEOMETRY_ROLE_KEY = b"geoarrow:geometry_role"
_GEOMETRY_ROLE_PRIMARY = "primary"
_GEOMETRY_ROLE_SECONDARY = "secondary"


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
