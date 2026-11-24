"""Merge spatial shards that were declared with different CRSs into one table.

Each shard (a Parquet file or an in-memory :class:`pyarrow.Table`) carries one
or more geometry columns whose source CRS is declared on the GeoArrow extension
type (or, for plain WKB/WKT columns, resolved from the GeoParquet ``geo``
metadata). This module reprojects every shard from its *declared* source CRS to
a single target CRS before concatenating, so the merged lake table contains
coordinates that were actually converted rather than merely relabelled.

Design properties:

- Coordinates are transformed with ``pyproj`` (via ``shapely``); the target CRS
  is attached to every output geometry column and recorded in the output table
  metadata.
- Row groups map 1:1: every input row group becomes exactly one output row
  group, in shard order. Null geometry values stay at their original row
  positions (validity bitmaps are carried through, never rebuilt).
- A shard (or a geometry column) that does not declare a CRS is never silently
  tagged with the target CRS. By default a :class:`MissingCrsError` is raised
  carrying locatable records; with ``on_missing_crs="quarantine"`` the affected
  rows are excluded from the merged table and delivered in a separate table.

>>> from geoarrow.pyarrow import merge
"""

import json
from dataclasses import dataclass, field
from typing import Optional

import geoarrow.pyarrow as _ga
from geoarrow.pyarrow._type import GeometryExtensionType
from geoarrow.types import crs as _crs_mod

import pyarrow as _pa
import pyarrow.parquet as _pq

# Metadata keys written on the merged table
METADATA_TARGET_CRS = b"geoarrow.merge.target_crs"
METADATA_MANIFEST = b"geoarrow.merge.manifest"
METADATA_QUARANTINE_REASONS = b"geoarrow.merge.quarantine_reasons"


class MissingCrsError(ValueError):
    """Raised when one or more shards cannot be transformed because their
    geometry columns do not declare a source CRS.

    The offending rows were *not* transformed and were *not* given the target
    CRS label. :attr:`records` locates every affected row.
    """

    def __init__(self, records):
        self.records = list(records)
        n_shards = len({r.shard_index for r in self.records})
        n_rows = sum(r.row_count for r in self.records)
        if n_rows:
            message = (
                f"{n_rows} row(s) across {n_shards} shard(s) have geometry columns "
                "without a declared source CRS; refusing to assume the target CRS. "
                "Fix the source data or use on_missing_crs='quarantine'. "
                f"First record: {self.records[0]}"
            )
        else:
            message = (
                f"{n_shards} shard(s) have geometry columns without a declared "
                "source CRS (the columns contain only nulls); refusing to assume "
                "the target CRS. Declare a source CRS or use "
                "on_missing_crs='quarantine'. "
                f"First record: {self.records[0]}"
            )
        super().__init__(message)


@dataclass(frozen=True)
class MissingCrsRecord:
    """A locatable reference to rows that could not be reprojected.

    Attributes
    ----------
    shard_index:
        Position of the shard in the input order.
    shard_path:
        File path of the shard, or ``None`` for an in-memory table.
    geometry_column:
        Name of the geometry column missing a CRS.
    row_group_index:
        Row group within the shard (``None`` for in-memory tables without row
        groups).
    row_indices:
        Row indices within the row group/shard that need a CRS fix.
    row_count:
        Number of affected rows (``len(row_indices)``).
    """

    shard_index: int
    shard_path: Optional[str]
    geometry_column: str
    row_group_index: Optional[int]
    row_indices: tuple
    row_count: int

    def to_dict(self):
        return {
            "shard_index": self.shard_index,
            "shard_path": self.shard_path,
            "geometry_column": self.geometry_column,
            "row_group_index": self.row_group_index,
            "row_indices": list(self.row_indices),
            "row_count": self.row_count,
        }


@dataclass
class ShardReprojection:
    """How one shard was handled in the merge."""

    shard_index: int
    shard_path: Optional[str]
    source_crs: Optional[object]
    target_crs: object
    transformed: bool
    row_groups: int
    rows: int

    def to_dict(self):
        return {
            "shard_index": self.shard_index,
            "shard_path": self.shard_path,
            "source_crs": _crs_to_json_dict(self.source_crs),
            "target_crs": _crs_to_json_dict(self.target_crs),
            "transformed": self.transformed,
            "row_groups": self.row_groups,
            "rows": self.rows,
        }


@dataclass
class MergeResult:
    """Output of :func:`merge_shards`.

    Attributes
    ----------
    table:
        The merged table. Every geometry column carries the target CRS on its
        extension type; the schema metadata additionally records the target
        CRS and a per-shard manifest.
    target_crs:
        The resolved target CRS (a :class:`geoarrow.types.crs.ProjJsonCrs`).
    manifest:
        One :class:`ShardReprojection` per input shard, in input order.
    missing_crs_records:
        Locatable records for rows excluded under
        ``on_missing_crs="quarantine"`` (empty otherwise).
    quarantined:
        Rows excluded from ``table`` because of a missing CRS, or ``None``.
    """

    table: _pa.Table
    target_crs: object
    manifest: list = field(default_factory=list)
    missing_crs_records: list = field(default_factory=list)
    quarantined: Optional[_pa.Table] = None


def merge_shards(
    shards,
    target_crs=_crs_mod.OGC_CRS84,
    *,
    geometry_columns=None,
    crs_overrides=None,
    on_missing_crs="error",
    output_encoding="auto",
):
    """Reproject each shard from its declared CRS and merge into one table.

    Parameters
    ----------
    shards:
        Sequence of shard inputs. Each item is either a path to a Parquet file
        or a :class:`pyarrow.Table`. Tables are handled as a single row group.
    target_crs:
        CRS to convert every shard into. Accepts anything
        :func:`geoarrow.types.crs.create` accepts (an EPSG string, PROJJSON,
        a ``pyproj.CRS``, or a geoarrow Crs). Defaults to OGC CRS84
        (longitude/latitude WGS 84).
    geometry_columns:
        Explicit names of geometry columns to reproject. By default every
        GeoArrow extension-type column is used; plain WKB/WKT columns are also
        detected when the shard carries GeoParquet ``geo`` metadata.
    crs_overrides:
        Optional mapping ``{shard_index: crs}`` (or ``{shard_path: crs}``)
        supplying the source CRS for a shard that does not declare one. An
        override is applied to every geometry column of that shard and is
        recorded in the manifest as the declared source CRS.
    on_missing_crs:
        What to do when a geometry column has no declared source CRS:

        - ``"error"`` (default): raise :class:`MissingCrsError` with locatable
          records. Nothing is written under an assumed CRS.
        - ``"quarantine"``: exclude the affected rows from the merged table and
          return them in :attr:`MergeResult.quarantined` with records.
    output_encoding:
        Encoding of the output geometry columns:

        - ``"auto"`` (default): keep each column's input encoding (WKT stays
          WKT, WKB stays WKB, native GeoArrow stays native).
        - ``"wkb"``: unify every geometry column as WKB.

    Returns
    -------
    MergeResult

    Examples
    --------
    >>> import geoarrow.pyarrow as ga
    >>> from geoarrow.pyarrow.merge import merge_shards
    >>> from geoarrow.types import crs
    >>> utm = ga.with_crs(ga.array(["POINT (500000 4649776)"]), "EPSG:32620")
    >>> lonlat = ga.with_crs(ga.array(["POINT (-63 42)"]), crs.OGC_CRS84)
    >>> import pyarrow as pa
    >>> result = merge_shards(
    ...     [pa.table({"geometry": utm}), pa.table({"geometry": lonlat})]
    ... )
    >>> result.table.schema.metadata[b"geoarrow.merge.target_crs"] is not None
    True
    """
    if on_missing_crs not in ("error", "quarantine"):
        raise ValueError(
            "on_missing_crs must be 'error' or 'quarantine', "
            f"got {on_missing_crs!r}"
        )
    if output_encoding not in ("auto", "wkb"):
        raise ValueError(
            f"output_encoding must be 'auto' or 'wkb', got {output_encoding!r}"
        )

    target_pyproj, target_ga_crs = _resolve_crs(target_crs)
    overrides = _normalize_overrides(crs_overrides)

    merged_batches = []
    manifest = []
    missing_records = []
    quarantine_tables = []
    output_schema = None

    for shard_index, shard in enumerate(shards):
        shard_path = shard if isinstance(shard, str) else None
        override = _lookup_override(overrides, shard_index, shard_path)
        row_group_tables = _iter_row_group_tables(shard)

        shard_rows = 0
        shard_row_groups = 0
        shard_source_crs = None
        shard_transformed = False
        on_disk_crss = None

        for row_group_index, rg_table in enumerate(row_group_tables):
            geo_cols = _resolve_geometry_columns(rg_table, geometry_columns)
            if not geo_cols:
                raise ValueError(
                    f"Shard {shard_index} ({shard_path or '<in-memory>'}) has no "
                    "geometry columns"
                )

            # For Parquet shards, resolve the declared CRS from the embedded
            # Arrow schema rather than the read-back extension type, whose
            # missing CRS may have been silently replaced by a reader default.
            if shard_path is not None and on_disk_crss is None:
                on_disk_crss = _parquet_declared_crsses(shard_path, geo_cols)

            processed = rg_table
            keep_mask = None

            for col_name in geo_cols:
                col = processed.column(col_name)
                if shard_path is not None:
                    declared = on_disk_crss.get(col_name)
                else:
                    declared = _declared_crs(col, rg_table, col_name)
                if declared is None and override is not None:
                    declared = override
                if declared is None:
                    # Locate every row carrying a non-null geometry with no CRS.
                    bad_rows = tuple(
                        i for i, is_null in enumerate(col.is_null()) if not is_null
                    )
                    record = MissingCrsRecord(
                        shard_index=shard_index,
                        shard_path=shard_path,
                        geometry_column=col_name,
                        row_group_index=(
                            row_group_index if shard_path is not None else None
                        ),
                        row_indices=bad_rows,
                        row_count=len(bad_rows),
                    )
                    missing_records.append(record)
                    if on_missing_crs == "error":
                        raise MissingCrsError(missing_records)
                    if bad_rows:
                        # Only non-null rows need to be removed; an all-null
                        # geometry column can safely stay (nothing to transform
                        # and no coordinates to mislabel).
                        mask = _pa.array(
                            [idx not in bad_rows for idx in range(len(col))],
                            type=_pa.bool_(),
                        )
                        keep_mask = (
                            mask
                            if keep_mask is None
                            else _pa.compute.and_(keep_mask, mask)
                        )
                    continue

                source_pyproj, _ = _resolve_crs(declared)
                if shard_source_crs is None:
                    shard_source_crs = source_pyproj
                elif source_pyproj != shard_source_crs:
                    raise ValueError(
                        f"Shard {shard_index} ({shard_path or '<in-memory>'}) "
                        f"declares inconsistent source CRSs across geometry "
                        f"columns: {shard_source_crs} vs {source_pyproj}"
                    )

                do_transform = source_pyproj != target_pyproj
                new_col = _reproject_column(
                    col,
                    source_pyproj=source_pyproj,
                    target_pyproj=target_pyproj,
                    target_ga_crs=target_ga_crs,
                    transform=do_transform,
                    output_encoding=output_encoding,
                )
                processed = processed.set_column(
                    processed.schema.get_field_index(col_name), col_name, new_col
                )
                shard_transformed = shard_transformed or do_transform

            if keep_mask is not None:
                quarantine_tables.append(rg_table.filter(_pa.compute.invert(keep_mask)))
                processed = processed.filter(keep_mask)

            # Normalize the schema toward a common output schema so that
            # concatenation is type-safe even if shards used different encodings.
            processed, output_schema = _align_schema(
                processed, output_schema, target_ga_crs, output_encoding
            )

            if processed.num_rows > 0 or not merged_batches:
                merged_batches.append(processed)
            shard_rows += rg_table.num_rows
            shard_row_groups += 1

        manifest.append(
            ShardReprojection(
                shard_index=shard_index,
                shard_path=shard_path,
                source_crs=shard_source_crs,
                target_crs=target_pyproj,
                transformed=shard_transformed,
                row_groups=shard_row_groups,
                rows=shard_rows,
            )
        )

    if not merged_batches:
        raise ValueError("merge_shards() requires at least one shard")

    merged = _pa.concat_tables(merged_batches, promote_options="default")
    merged = _attach_metadata(merged, target_ga_crs, manifest, missing_records)

    quarantined = None
    if quarantine_tables:
        quarantined = _pa.concat_tables(
            quarantine_tables, promote_options="default"
        ).replace_schema_metadata(
            {
                METADATA_QUARANTINE_REASONS: json.dumps(
                    [r.to_dict() for r in missing_records]
                ).encode("utf-8")
            }
        )

    return MergeResult(
        table=merged,
        target_crs=target_ga_crs,
        manifest=manifest,
        missing_crs_records=missing_records,
        quarantined=quarantined,
    )


def write_merged_parquet(result, where):
    """Write a :class:`MergeResult` to Parquet preserving row groups 1:1.

    Each table fragment produced from one input row group is written as its own
    Parquet row group, so output row groups correspond exactly to input row
    groups. Target CRS and manifest metadata travel in the schema.
    """
    if not isinstance(result, MergeResult):
        raise TypeError("write_merged_parquet() expects a MergeResult")
    table = result.table
    with _pq.ParquetWriter(where, table.schema) as writer:
        # Each fragment appended during the merge came from one input row
        # group; writing the table batch-by-batch keeps that mapping exact in
        # the output (one Parquet row group per input row group).
        for batch in table.to_batches():
            writer.write_table(_pa.Table.from_batches([batch], schema=table.schema))


# --------------------------------------------------------------------------- #
# Internal helpers
# --------------------------------------------------------------------------- #


def _resolve_crs(obj):
    """Return ``(pyproj.CRS, geoarrow ProjJsonCrs)`` for a CRS-like object."""
    import pyproj

    ga_crs = _crs_mod.create(obj) if not hasattr(obj, "to_json_dict") else obj
    try:
        pyproj_crs = pyproj.CRS.from_user_input(ga_crs.to_json_dict())
    except Exception as exc:  # pragma: no cover - defensive
        raise ValueError(f"Could not resolve CRS from {obj!r}: {exc}") from exc
    canonical = _crs_mod.ProjJsonCrs(pyproj_crs.to_json_dict())
    return pyproj_crs, canonical


def _crs_to_json_dict(crs_obj):
    if crs_obj is None:
        return None
    if hasattr(crs_obj, "to_json_dict"):
        try:
            return crs_obj.to_json_dict()
        except Exception:  # pragma: no cover - defensive
            return str(crs_obj)
    return str(crs_obj)


def _normalize_overrides(crs_overrides):
    if crs_overrides is None:
        return {}
    if not isinstance(crs_overrides, dict):
        raise TypeError("crs_overrides must be a dict of {shard_index/path: crs}")
    return dict(crs_overrides)


def _lookup_override(overrides, shard_index, shard_path):
    if shard_index in overrides:
        return overrides[shard_index]
    if shard_path is not None and shard_path in overrides:
        return overrides[shard_path]
    return None


def _iter_row_group_tables(shard):
    """Yield one table per row group (files) or one table (in-memory)."""
    if isinstance(shard, _pa.Table):
        yield shard
        return
    if isinstance(shard, str):
        parquet_file = _pq.ParquetFile(shard)
        for row_group_index in range(parquet_file.num_row_groups):
            yield parquet_file.read_row_group(row_group_index)
        return
    raise TypeError(
        "Each shard must be a Parquet path (str) or pyarrow.Table, "
        f"got {type(shard)!r}"
    )


def _parquet_declared_crsses(shard_path, geometry_columns):
    """Read declared CRSs directly from a Parquet file's embedded Arrow schema.

    PyArrow's built-in GeoArrow support substitutes a default OGC:CRS84 when a
    geometry column was written without a CRS, which would make an actually
    missing CRS look declared after a normal read. The embedded ``ARROW:schema``
    key-value preserves the literal extension metadata (``{}`` with no ``crs``
    key for a missing CRS), so we inspect it before any default is applied.

    Returns a mapping ``{column_name: Crs or None}`` for the geometry columns,
    where ``None`` means "no CRS was declared on disk".
    """
    import base64

    parquet_file = _pq.ParquetFile(shard_path)
    kv_metadata = parquet_file.metadata.metadata or {}
    if b"ARROW:schema" not in kv_metadata:
        return {name: None for name in geometry_columns}

    embedded = _pa.ipc.read_schema(
        _pa.BufferReader(base64.b64decode(kv_metadata[b"ARROW:schema"]))
    )

    # GeoParquet files carry authoritative per-column CRSs under the "geo"
    # key-value (a JSON null explicitly means no CRS).
    geo_meta = None
    if b"geo" in kv_metadata:
        try:
            geo_meta = json.loads(kv_metadata[b"geo"])
        except (ValueError, TypeError):
            geo_meta = None

    out = {}
    for name in geometry_columns:
        if name not in list(embedded.names):
            out[name] = None
            continue

        field = embedded.field(name)
        declared = None

        # When the geoarrow extension types are registered (the normal state
        # once this package was imported), read_schema() resolves the field to
        # the extension type and carries the CRS on it, leaving the field
        # metadata empty. Without registration, the literal extension metadata
        # stays on the field. Handle both representations.
        if isinstance(field.type, GeometryExtensionType):
            declared = field.type.crs
        else:
            field_meta = field.metadata or {}
            ext_meta = field_meta.get(b"ARROW:extension:metadata")
            if ext_meta:
                try:
                    parsed = json.loads(ext_meta.decode("utf-8"))
                except (ValueError, UnicodeDecodeError):
                    parsed = {}
                if parsed.get("crs") is not None:
                    declared = _crs_mod.create(parsed["crs"])

        if declared is None and geo_meta and name in geo_meta.get("columns", {}):
            spec = geo_meta["columns"][name]
            if spec.get("crs") is not None:
                declared = _crs_mod.ProjJsonCrs(spec["crs"])

        out[name] = declared
    return out


def _resolve_geometry_columns(table, explicit):
    if explicit is not None:
        missing = [name for name in explicit if name not in table.schema.names]
        if missing:
            raise KeyError(f"Geometry column(s) not found in shard: {missing}")
        return tuple(explicit)

    columns = []
    geo_meta = _read_geo_metadata(table)
    for name, pa_type in zip(table.schema.names, table.schema.types):
        if isinstance(pa_type, GeometryExtensionType):
            columns.append(name)
        elif geo_meta and name in geo_meta.get("columns", {}):
            # GeoParquet-declared WKB/WKT column without an extension type
            columns.append(name)
    return tuple(columns)


def _read_geo_metadata(table):
    metadata = table.schema.metadata
    if not metadata or b"geo" not in metadata:
        return None
    try:
        return json.loads(metadata[b"geo"])
    except (ValueError, TypeError):
        return None


def _declared_crs(column, table, col_name):
    """Extract the declared source CRS, or None if absent."""
    pa_type = column.type
    if isinstance(pa_type, GeometryExtensionType):
        crs_obj = pa_type.crs
        # None means explicitly unset; treat as missing rather than fabricating
        return crs_obj

    geo_meta = _read_geo_metadata(table)
    if geo_meta and col_name in geo_meta.get("columns", {}):
        spec = geo_meta["columns"][col_name]
        if spec.get("crs") is not None:
            return _crs_mod.ProjJsonCrs(spec["crs"])
    return None


def _reproject_column(
    column,
    *,
    source_pyproj,
    target_pyproj,
    target_ga_crs,
    transform,
    output_encoding,
):
    """Reproject one (possibly chunked) geometry column.

    Validity is preserved exactly: null rows never reach the transformer and
    the output array is rebuilt with the input null bitmap.
    """
    import numpy as np
    import pyproj
    from shapely import from_wkb, to_wkb
    from shapely import transform as shapely_transform

    # Normalize to an extension array first so plain WKB/WKT storage columns
    # get the same treatment.
    if not isinstance(column.type, GeometryExtensionType):
        column = _ga.array(column)

    is_chunked = isinstance(column, _pa.ChunkedArray)
    chunks = column.chunks if is_chunked else [column]
    out_chunks = []

    transformer = None
    if transform:
        transformer = pyproj.Transformer.from_crs(
            source_pyproj, target_pyproj, always_xy=True
        )

    for chunk in chunks:
        original_type = chunk.type
        wkb_chunk = _ga.as_wkb(chunk)
        null_mask = wkb_chunk.is_null()
        geometries = from_wkb(np.asarray(wkb_chunk.storage))

        if transformer is not None:
            include_z = _chunk_has_z(wkb_chunk)
            geometries = shapely_transform(
                geometries,
                transformer.transform,
                include_z=include_z,
                interleaved=False,
            )

        out_storage = _pa.array(to_wkb(geometries), type=_pa.binary())
        # Re-apply the exact input validity bitmap. to_wkb(None) already yields
        # None, but this makes the null correspondence explicit and robust to
        # geometries that transform to an empty/null representation.
        out_storage = _pa.compute.if_else(
            null_mask, _pa.scalar(None, type=_pa.binary()), out_storage
        )

        wkb_type = _ga.wkb().with_crs(target_ga_crs)
        wkb_out = wkb_type.wrap_array(out_storage)
        out_chunks.append(
            _encode_output(wkb_out, original_type, target_ga_crs, output_encoding)
        )

    if is_chunked:
        out_type = out_chunks[0].type
        return _pa.chunked_array(out_chunks, type=out_type)
    return out_chunks[0]


def _encode_output(wkb_array, original_type, target_ga_crs, output_encoding):
    """Match the requested output encoding.

    ``"wkb"`` always returns WKB. ``"auto"`` keeps the source encoding:
    WKT stays WKT and a native GeoArrow encoding is re-derived from the
    transformed WKB with the target CRS attached.
    """
    if output_encoding == "wkb":
        return wkb_array

    if isinstance(original_type, _ga.WktType):
        return _ga.as_wkt(wkb_array)

    if not isinstance(original_type, (_ga.WkbType, _ga.WktType)):
        native_type = original_type.with_crs(target_ga_crs)
        try:
            return _ga.as_geoarrow(wkb_array, native_type)
        except Exception:
            # Heterogeneous geometry (e.g. mixed POINT/LINESTRING) cannot fit
            # the original single-type native encoding; fall back to WKB rather
            # than fail the merge.
            return wkb_array

    return wkb_array


def _chunk_has_z(wkb_chunk):
    """Return True if any non-null geometry in the chunk carries a Z ordinate."""
    import numpy as np
    from shapely import from_wkb

    geometries = from_wkb(np.asarray(wkb_chunk.storage))
    return any(g is not None and not g.is_empty and g.has_z for g in geometries)


def _align_schema(table, previous_schema, target_ga_crs, output_encoding):
    """Cast a processed fragment toward a common output schema.

    Geometry columns are unified through a geometry-aware conversion (WKB as
    the common carrier); ordinary columns use a regular Arrow cast.
    """
    if previous_schema is None:
        return table, table.schema

    if table.schema.equals(previous_schema, check_metadata=False):
        return table, previous_schema

    casted = table
    for name in previous_schema.names:
        if name not in casted.schema.names:
            continue
        wanted = previous_schema.field(name).type
        current = casted.schema.field(name).type
        if current == wanted:
            continue

        idx = casted.schema.get_field_index(name)
        column = casted.column(name)
        if isinstance(wanted, GeometryExtensionType):
            # Geometry-to-geometry: route through WKB, which every geoarrow
            # geometry type can be cast both to and from.
            column = _ga.as_geoarrow(_ga.as_wkb(column), wanted)
        else:
            column = column.cast(wanted)
        casted = casted.set_column(idx, name, column)
    return casted, previous_schema


def _attach_metadata(table, target_ga_crs, manifest, missing_records):
    metadata = dict(table.schema.metadata or {})
    metadata[METADATA_TARGET_CRS] = json.dumps(target_ga_crs.to_json_dict()).encode(
        "utf-8"
    )
    metadata[METADATA_MANIFEST] = json.dumps(
        [entry.to_dict() for entry in manifest]
    ).encode("utf-8")
    if missing_records:
        metadata[METADATA_QUARANTINE_REASONS] = json.dumps(
            [record.to_dict() for record in missing_records]
        ).encode("utf-8")
    return table.replace_schema_metadata(metadata)
