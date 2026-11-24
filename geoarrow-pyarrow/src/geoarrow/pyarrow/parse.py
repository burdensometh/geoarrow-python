"""Row-wise parsing of geometry batches

This module provides a "batch parse" entry point that keeps a per-row status
alongside the parsed geometry. It is intended for data-quality workflows where
a batch may contain a mix of parseable geometry, legally empty geometry, missing
values, and values that cannot be parsed at all (for example, invalid well-known
text).

The parsed geometry column is still a regular GeoArrow extension column, so rows
whose status is ``"valid"`` or ``"empty"`` can be written directly using
:func:`geoarrow.pyarrow.io.write_geoparquet_table` or any other Arrow-aware
writer.

>>> import geoarrow.pyarrow as ga
>>> from geoarrow.pyarrow import parse
>>> parsed = parse.parse_batch(
...     ["POINT (0 1)", "POINT EMPTY", None, "not valid wkt"],
...     source_key=["north", "north", "south", "south"],
... )
>>> parsed.to_pylist()  # doctest: +SKIP
[...]
>>> parsed.column("parse_status").to_pylist()
['valid', 'empty', 'missing', 'unparseable']
"""

import geoarrow.pyarrow as _ga
from geoarrow.pyarrow import _compute

import pyarrow as _pa
import pyarrow.compute as _pc
import pyarrow.dataset as _ds
import pyarrow.types as _types


#: The values used by the ``parse_status`` column.
PARSE_STATUS_VALUES = ("valid", "empty", "missing", "unparseable")

#: ValueError-compatible aliases for the ``on_error`` argument.
_ON_ERROR_ALIASES = {
    None: "classify",
    "classify": "classify",
    "ignore": "classify",
    "tolerant": "classify",
    "raise": "raise",
    "error": "raise",
    "strict": "raise",
    "skip": "skip",
    "drop": "skip",
}


def _on_error_mode(on_error, errors=None):
    if errors is not None:
        if on_error not in (None, "classify"):
            raise ValueError("Pass either on_error= or errors=, not both")
        on_error = errors

    try:
        return _ON_ERROR_ALIASES[on_error]
    except KeyError:
        raise ValueError(
            f"Unsupported on_error value {on_error!r}. Expected one of "
            "'classify', 'raise', or 'skip'"
        ) from None


def _parse_errors():
    # Only geoarrow-c errors indicate an unparseable feature. Other exceptions
    # (for example a TypeError for a non-planar edge type) are programming errors
    # and should not be swallowed by the tolerant path.
    errors = []
    try:
        from geoarrow.c import _lib

        errors.append(_lib.GeoArrowCException)
    except ImportError:  # pragma: no cover - geoarrow-c is a hard dependency
        pass

    if not errors:  # pragma: no cover
        errors.append(RuntimeError)

    return tuple(errors)


_PARSE_ERRORS = _parse_errors()


def _as_geometry(obj):
    """Normalize ``obj`` to a pyarrow Array/ChunkedArray usable by kernels."""
    if isinstance(obj, (_pa.Array, _pa.ChunkedArray)):
        # pyarrow's *view* types are not yet understood by geoarrow.array();
        # cast to their canonical counterpart first (a zero-copy operation).
        if _types.is_string_view(obj.type):
            obj = _pc.cast(obj, _pa.utf8())
        elif _types.is_binary_view(obj.type):
            obj = _pc.cast(obj, _pa.binary())
    else:
        # An empty or all-null sequence has no inferable type; treat it as
        # (empty) WKT so that the result still carries a usable GeoArrow type.
        obj = _pa.array(obj)

    if _pa.types.is_null(obj.type):
        obj = _pc.cast(obj, _pa.string())

    obj = _compute.obj_as_array_or_chunked(obj)
    return obj


def _as_planar(obj):
    # ``box()`` is only defined for planar edges. Classification is about
    # whether a feature can be parsed, not about its edge interpretation, so
    # evaluate empties as planar regardless of the input's declared edges.
    if isinstance(obj.type, _ga.GeometryExtensionType):
        if obj.type.edge_type != _ga.EdgeType.PLANAR:
            return _ga.with_edge_type(obj, _ga.EdgeType.PLANAR)
    return obj


def _split_box(obj, indices, successes, mode):
    """Collect ``(indices, box)`` pairs, isolating unparseable features.

    ``box()`` parses every feature in the input. If a batch contains an
    unparseable feature it raises, so split the batch in half and try again.
    This keeps the common (all-parseable) case to a single pass.
    """
    if len(indices) == 0:
        return

    subset = obj.take(indices)
    try:
        box = _compute.box(subset)
    except _PARSE_ERRORS:
        if mode == "raise":
            raise
        if len(indices) == 1:
            return
        mid = len(indices) // 2
        _split_box(obj, indices[:mid], successes, mode)
        _split_box(obj, indices[mid:], successes, mode)
        return

    successes.append((indices, box))


def classify_geometry(geometry, *, on_error="classify", errors=None):
    """Parse ``geometry`` and classify every row.

    Returns a ``(geometry, status)`` tuple where ``geometry`` is a
    :func:`geoarrow.pyarrow.wkb`-encoded extension array and ``status`` is a
    string array with one of ``"valid"``, ``"empty"``, ``"missing"``, or
    ``"unparseable"`` for each input row.

    Parseable rows (``"valid"`` and ``"empty"``) have a non-null geometry.
    Missing and unparseable rows have a null geometry. This means the result
    can be written directly as a standard spatial column.

    ``on_error`` (alias: ``errors``) controls what happens for rows that cannot
    be parsed:

    * ``"classify"`` (default): mark them as ``"unparseable"``.
    * ``"raise"``: raise the underlying parsing error (the previous, strict
      behaviour of :func:`geoarrow.pyarrow.parse_all`).
    * ``"skip"``: handled by :func:`parse_batch`, which drops those rows.
    """
    mode = _on_error_mode(on_error, errors)
    obj = _as_geometry(geometry)

    if isinstance(obj, _pa.ChunkedArray):
        obj = obj.combine_chunks()

    n = len(obj)
    if n == 0:
        return _compute.as_wkb(obj), _pa.array([], type=_pa.string())

    planarity_obj = _as_planar(obj)

    if mode == "raise":
        # Validate everything up front so that the error is the same as
        # parse_all() and points at the offending batch.
        _compute.parse_all(planarity_obj)

    null_mask = _pc.is_null(obj)
    non_null = _pc.cast(_pc.indices_nonzero(_pc.invert(null_mask)), _pa.int64())

    successes = []
    _split_box(planarity_obj, non_null, successes, mode)

    if successes:
        success_indices = _pa.concat_arrays([indices for indices, _ in successes])
        xmins = [
            _pc.struct_field(_compute.ensure_storage(box), "xmin")
            for _, box in successes
        ]
        xmaxs = [
            _pc.struct_field(_compute.ensure_storage(box), "xmax")
            for _, box in successes
        ]
        xmin = _pa.concat_arrays(xmins)
        xmax = _pa.concat_arrays(xmaxs)
    else:
        success_indices = _pa.array([], type=_pa.int64())
        xmin = _pa.array([], type=_pa.float64())
        xmax = _pa.array([], type=_pa.float64())

    # A box with xmin > xmax (or NaN coordinates) is the canonical
    # representation of a legally empty feature in geoarrow-c.
    empty_ok = _pc.or_(
        _pc.greater(xmin, xmax),
        _pc.is_nan(xmin),
    )

    parsed = _compute.as_wkb(obj.take(success_indices))

    # Scatter the values parsed from the successful subset back onto the full
    # input length; positions that were not successful become null.
    ok_mask = _pc.is_in(
        _pa.array(range(n), type=_pa.int64()), value_set=success_indices
    )
    rank = _pc.subtract(
        _pc.cumulative_sum(_pc.cast(ok_mask, _pa.int64())), 1
    )
    take_positions = _pc.if_else(ok_mask, rank, _pa.scalar(None, _pa.int64()))
    parsed = parsed.take(take_positions)
    empty_full = _pc.fill_null(empty_ok.take(take_positions), False)

    # PARSE_STATUS_VALUES order: 0 = valid, 1 = empty, 2 = missing,
    # 3 = unparseable.
    parsed_code = _pc.if_else(
        empty_full, _pa.scalar(1, _pa.int8()), _pa.scalar(0, _pa.int8())
    )
    codes = _pc.if_else(
        null_mask,
        _pa.scalar(2, _pa.int8()),
        _pc.if_else(ok_mask, parsed_code, _pa.scalar(3, _pa.int8())),
    )
    status = _pc.take(
        _pa.array(list(PARSE_STATUS_VALUES), type=_pa.string()), codes
    )

    return parsed, status


def _broadcast_source_key(source_key, num_rows):
    if isinstance(source_key, _pa.ChunkedArray):
        if len(source_key) != num_rows:
            raise ValueError("source_key must have the same length as geometry")
        return source_key

    if isinstance(source_key, _pa.Array):
        if len(source_key) != num_rows:
            raise ValueError("source_key must have the same length as geometry")
        return source_key

    if isinstance(source_key, (_pa.Scalar, str, bytes, int, float, bool)):
        return _pa.repeat(_pa.scalar(source_key), num_rows)

    array = _pa.array(source_key)
    if len(array) != num_rows:
        raise ValueError("source_key must have the same length as geometry")

    return array


def _set_or_append(table, name, column):
    if name in table.column_names:
        index = table.schema.get_field_index(name)
        return table.set_column(index, name, column)
    else:
        return table.append_column(name, column)


def _resolve_geometry_column(schema, geometry_column, column_names):
    if geometry_column is not None:
        if geometry_column not in column_names:
            raise KeyError(f"Column {geometry_column!r} not found")
        return geometry_column

    if "geometry" in column_names:
        return "geometry"

    candidates = [
        name
        for name, type in zip(schema.names, schema.types)
        if isinstance(type, _ga.GeometryExtensionType)
    ]
    if len(candidates) == 1:
        return candidates[0]

    # GeoParquet written by ``write_geoparquet_table()`` stores geometry as
    # plain binary plus a ``geo`` metadata key, so pyarrow does not surface an
    # extension type. Use the declared primary column in that case.
    metadata = schema.metadata or {}
    if b"geo" in metadata:
        import json

        geo_meta = json.loads(metadata[b"geo"])
        primary = geo_meta.get("primary_column")
        if primary and primary in column_names:
            return primary

        columns = list(geo_meta.get("columns", {}))
        if len(columns) == 1 and columns[0] in column_names:
            return columns[0]

    raise ValueError(
        "Could not determine geometry column; pass geometry_column="
    )


def _parse_table(
    table,
    *,
    geometry_column,
    source_key,
    source_key_column,
    on_error,
    crs,
):
    geometry_column = _resolve_geometry_column(
        table.schema, geometry_column, table.column_names
    )

    geometry = table.column(geometry_column)
    parsed, status = classify_geometry(geometry, on_error=on_error)

    if crs is not None:
        parsed = _ga.with_crs(parsed, crs)

    geometry_index = table.schema.get_field_index(geometry_column)
    out = table.set_column(geometry_index, geometry_column, parsed)
    out = _set_or_append(out, "parse_status", status)

    if source_key is not None:
        if isinstance(source_key, str) and source_key in table.column_names:
            key_values = table.column(source_key)
            key_name = source_key
        else:
            key_values = _broadcast_source_key(source_key, table.num_rows)
            key_name = source_key_column

        if key_name != geometry_column:
            out = _set_or_append(out, key_name, key_values)

    return out


def parse_batch(
    data,
    *,
    geometry_column=None,
    source_key=None,
    source_key_column="source_key",
    on_error="classify",
    errors=None,
    crs=None,
):
    """Parse a batch of geometries while retaining a per-row status.

    Parameters
    ----------
    data
        A ``pyarrow.Table``, ``pyarrow.Array``, ``pyarrow.ChunkedArray``, or
        any sequence that can be converted to one. Text values are interpreted
        as well-known text, binary values as well-known binary, and GeoArrow
        extension arrays as their existing encoding.
    geometry_column
        The geometry column to parse when ``data`` is a table. If ``None``,
        ``"geometry"`` is used when present; otherwise the single GeoArrow
        extension column is used.
    source_key
        A value to retain for every row. When ``data`` is a table, this may be
        the name of an existing column; otherwise it is broadcast as a scalar.
        An array-like value must have the same length as ``data``.
    source_key_column
        The name used for a broadcast source key. Defaults to ``"source_key"``.
    on_error
        Alias ``errors``. Values are ``"classify"`` (default), which
        records ``"unparseable"`` rows instead of raising; ``"raise"``, which
        preserves the strict behaviour; and ``"skip"``, which drops unparseable
        rows from the result.
    crs
        An optional CRS to attach to the parsed geometry column.

    Returns
    -------
    pyarrow.Table
        With the parsed geometry in ``geometry_column`` (a WKB-encoded
        GeoArrow extension column), a ``parse_status`` string column, and the
        retained source key column when one was provided. Other input columns
        are preserved, so the table can be written as GeoParquet.
    """
    mode = _on_error_mode(on_error, errors)

    if isinstance(data, _pa.Table):
        out = _parse_table(
            data,
            geometry_column=geometry_column,
            source_key=source_key,
            source_key_column=source_key_column,
            on_error=mode,
            crs=crs,
        )
    else:
        parsed, status = classify_geometry(data, on_error=mode)
        if crs is not None:
            parsed = _ga.with_crs(parsed, crs)

        names = [geometry_column or "geometry", "parse_status"]
        arrays = [parsed, status]

        if source_key is not None:
            key_values = _broadcast_source_key(source_key, len(parsed))
            names.append(source_key_column)
            arrays.append(key_values)

        out = _pa.table(arrays, names=names)

    if mode == "skip":
        keep = _pc.not_equal(out.column("parse_status"), "unparseable")
        out = out.filter(_pc.fill_null(keep, True))

    return out


def _partition_key_names(dataset):
    names = []
    for fragment in dataset.get_fragments():
        for name in _ds.get_partition_keys(fragment.partition_expression):
            if name not in names:
                names.append(name)

    # Preserve the schema's column order where possible.
    return [name for name in dataset.schema.names if name in names]


def _resolve_source_key(table, source_key, partition_keys, source_key_column):
    """Return ``(column_name, values, should_add)`` for a batch of a dataset."""
    if source_key is None:
        if not partition_keys:
            return None, None, False
        source_key = partition_keys

    if isinstance(source_key, str):
        # A name that is not a column is treated as a constant source key.
        if source_key not in table.column_names:
            values = _broadcast_source_key(source_key, table.num_rows)
            return source_key_column, values, True
        source_key = [source_key]

    if not isinstance(source_key, (list, tuple)):
        values = _broadcast_source_key(source_key, table.num_rows)
        return source_key_column, values, True

    if len(source_key) == 0:
        return None, None, False

    for name in source_key:
        if name not in table.column_names:
            raise KeyError(f"Source key column {name!r} not found")

    if len(source_key) == 1:
        name = source_key[0]
        if name == source_key_column:
            return name, table.column(name), False
        return source_key_column, table.column(name), True

    # Composite key: join the string representation of each key with "/".
    parts = [_pc.cast(table.column(name), _pa.string()) for name in source_key]
    values = _pc.binary_join_element_wise(*parts, "/")
    return source_key_column, values, True


def _parse_dataset_table(
    table,
    *,
    geometry_column,
    source_key,
    partition_keys,
    source_key_column,
    mode,
    crs,
):
    key_name, key_values, add_key = _resolve_source_key(
        table, source_key, partition_keys, source_key_column
    )

    parsed, classified = classify_geometry(
        table.column(geometry_column), on_error=mode
    )
    if crs is not None:
        parsed = _ga.with_crs(parsed, crs)

    geometry_index = table.schema.get_field_index(geometry_column)
    out = table.set_column(geometry_index, geometry_column, parsed)

    # Preserve a status that was already stored (for example, by a previous
    # parse) so that re-reading does not downgrade "unparseable" rows to
    # "missing" (both of which have null geometry).
    if "parse_status" in table.column_names:
        status = table.column("parse_status")
    else:
        status = classified
    out = _set_or_append(out, "parse_status", status)

    if add_key and key_name != geometry_column:
        out = _set_or_append(out, key_name, key_values)

    return out


def parse_dataset(
    source,
    *,
    geometry_column=None,
    partition_keys=None,
    source_key=None,
    source_key_column="source_key",
    on_error="classify",
    errors=None,
    crs=None,
    **dataset_kwargs,
):
    """Parse every batch of a dataset, retaining partition information.

    ``source`` may be a ``pyarrow.dataset.Dataset`` or anything accepted by
    :func:`pyarrow.dataset.dataset` (for example a directory containing
    Hive-partitioned Parquet files). Partition columns are read as ordinary
    columns and are retained in the output, so they can be grouped on directly
    or used as the ``source_key``.

    ``source_key`` defaults to the dataset's partition columns. With a single
    partition column, its values are copied to ``source_key_column``; with
    multiple partition columns, their string values are joined with ``"/"``.
    Pass a column name or list of column names to choose explicitly.

    ``on_error`` accepts the same values as :func:`parse_batch`; its alias
    ``errors`` is also accepted.
    """
    if isinstance(source, _pa.Table):
        return parse_batch(
            source,
            geometry_column=geometry_column,
            source_key=source_key,
            source_key_column=source_key_column,
            on_error=on_error,
            errors=errors,
            crs=crs,
        )

    if isinstance(source, _ds.Dataset):
        dataset = source
    else:
        # Hive-style directory names (``region=north``) are the common way to
        # lay out regional batches; pyarrow does not infer this by default.
        dataset_kwargs.setdefault("partitioning", "hive")
        dataset = _ds.dataset(source, **dataset_kwargs)

    if partition_keys is None:
        partition_keys = _partition_key_names(dataset)
    elif isinstance(partition_keys, str):
        partition_keys = [partition_keys]

    geometry_column_use = _resolve_geometry_column(
        dataset.schema, geometry_column, dataset.schema.names
    )
    mode = _on_error_mode(on_error, errors)

    tables = []
    for batch in dataset.to_batches():
        table = _pa.Table.from_batches([batch])
        tables.append(
            _parse_dataset_table(
                table,
                geometry_column=geometry_column_use,
                source_key=source_key,
                partition_keys=partition_keys,
                source_key_column=source_key_column,
                mode=mode,
                crs=crs,
            )
        )

    if not tables:
        # Preserve the schema (including the source key column) for an empty
        # dataset rather than failing.
        tables.append(
            _parse_dataset_table(
                dataset.to_table(),
                geometry_column=geometry_column_use,
                source_key=source_key,
                partition_keys=partition_keys,
                source_key_column=source_key_column,
                mode=mode,
                crs=crs,
            )
        )

    out = _pa.concat_tables(tables) if len(tables) > 1 else tables[0]
    if mode == "skip":
        keep = _pc.not_equal(out.column("parse_status"), "unparseable")
        out = out.filter(_pc.fill_null(keep, True))

    return out


def _ensure_parse_status(
    data, *, geometry_column=None, on_error="classify", errors=None
):
    if isinstance(data, _pa.Table) and "parse_status" in data.column_names:
        return data

    if isinstance(data, (_pa.Table, _pa.Array, _pa.ChunkedArray, list, tuple)):
        return parse_batch(
            data, geometry_column=geometry_column, on_error=on_error, errors=errors
        )

    if isinstance(data, _ds.Dataset):
        dataset = data
    else:
        dataset = _ds.dataset(data, partitioning="hive")

    # A previously parsed dataset keeps its per-row status (including
    # "unparseable" rows, whose geometry is null). Re-parsing would downgrade
    # those rows to "missing", so read the stored status instead.
    if "parse_status" in dataset.schema.names:
        return dataset.to_table()

    return parse_dataset(
        dataset, geometry_column=geometry_column, on_error=on_error, errors=errors
    )


def _status_indicator(status, value):
    indicator = _pc.cast(_pc.equal(status, value), _pa.int64())
    return _pc.fill_null(indicator, 0)


def summarize_parse_status(
    data,
    *,
    partition_key="source_key",
    status_column="parse_status",
    geometry_column=None,
    on_error="classify",
    errors=None,
):
    """Summarize valid/empty/missing/unparseable counts by partition.

    ``data`` may be a table returned by :func:`parse_batch`, any table with a
    ``parse_status`` column, or a dataset/path accepted by
    :func:`parse_dataset`. If no status column is present, the geometry column
    is parsed first.

    ``partition_key`` is a column name or list of column names to group by.
    Pass ``None`` to produce a single overall row. The result contains the
    partition columns, ``n_valid``, ``n_empty``, ``n_missing``,
    ``n_unparseable``, ``n_total``, and ``valid_rate`` (the effective drawing
    rate: valid rows divided by all rows in the partition).
    """
    table = _ensure_parse_status(
        data,
        geometry_column=geometry_column,
        on_error=on_error,
        errors=errors,
    )

    if status_column not in table.column_names:
        raise KeyError(f"Column {status_column!r} not found")

    status = table.column(status_column)
    indicator_names = []
    for value in PARSE_STATUS_VALUES:
        name = f"_{value}"
        table = table.append_column(name, _status_indicator(status, value))
        indicator_names.append(name)

    if partition_key is None:
        keys = []
    elif isinstance(partition_key, str):
        keys = [partition_key]
    else:
        keys = list(partition_key)

    # If the requested partition key is absent (for example, a batch parsed
    # without a source key), fall back to an overall summary rather than failing.
    keys = [key for key in keys if key in table.column_names]

    aggregations = [(name, "sum") for name in indicator_names]

    if table.num_rows == 0:
        # group_by() on an empty table emits a spurious null-key row; build the
        # empty result explicitly instead.
        arrays = [table.column(key) for key in keys]
        names = list(keys)
        for value in PARSE_STATUS_VALUES:
            names.append(f"n_{value}")
            arrays.append(_pa.array([], type=_pa.int64()))
        names.extend(["n_total", "valid_rate"])
        arrays.extend(
            [
                _pa.array([], type=_pa.int64()),
                _pa.array([], type=_pa.float64()),
            ]
        )
        return _pa.table(arrays, names=names)

    grouped = table.group_by(keys, use_threads=False).aggregate(aggregations)

    rename = {}
    for value, name in zip(PARSE_STATUS_VALUES, indicator_names):
        rename[f"{name}_sum"] = f"n_{value}"
    grouped = grouped.rename_columns(
        keys + [rename[name] for name in grouped.column_names if name in rename]
    )

    count_names = [f"n_{value}" for value in PARSE_STATUS_VALUES]
    for name in count_names:
        grouped = grouped.set_column(
            grouped.schema.get_field_index(name),
            name,
            _pc.fill_null(grouped.column(name), 0),
        )

    n_valid = grouped.column("n_valid")
    n_total = None
    for name in count_names:
        column = grouped.column(name)
        n_total = column if n_total is None else _pc.add(n_total, column)

    n_total = _pc.fill_null(n_total, 0)
    grouped = grouped.append_column("n_total", n_total)

    valid_rate = _pc.if_else(
        _pc.equal(n_total, 0),
        _pa.scalar(0.0, type=_pa.float64()),
        _pc.divide(
            _pc.cast(n_valid, _pa.float64()), _pc.cast(n_total, _pa.float64())
        ),
    )
    grouped = grouped.append_column("valid_rate", _pc.fill_null(valid_rate, 0.0))

    if keys:
        grouped = grouped.sort_by([(key, "ascending") for key in keys])

    return grouped
