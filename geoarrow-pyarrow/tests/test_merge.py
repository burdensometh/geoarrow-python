"""Tests for CRS-aware shard merging (geoarrow.pyarrow.merge)."""

import json
from tempfile import TemporaryDirectory

import geoarrow.pyarrow as ga
import pyarrow as pa
import pyproj
import pyarrow.parquet as pq
import pytest
from geoarrow.pyarrow import io, merge
from geoarrow.pyarrow.merge import (
    METADATA_MANIFEST,
    METADATA_TARGET_CRS,
    MergeResult,
    MissingCrsError,
    MissingCrsRecord,
)
from geoarrow.types import crs as ga_crs

UTM = "EPSG:32620"
# A point whose UTM coordinates reproject to a known lon/lat
UTM_POINT = (500000.0, 4649776.0)


def _wkt(values, source_crs):
    return ga.with_crs(
        ga.wkt().wrap_array(pa.array(values, pa.utf8())), source_crs
    )


def _wkb(values, source_crs):
    return ga.with_crs(ga.as_wkb(ga.array(values)), source_crs)


def _native_points(xs, ys, source_crs):
    return ga.with_crs(ga.make_point(xs, ys), source_crs)


def _expected_lonlat(x, y, source_crs=UTM, target_crs="OGC:CRS84"):
    transformer = pyproj.Transformer.from_crs(
        pyproj.CRS(source_crs), pyproj.CRS(target_crs), always_xy=True
    )
    return transformer.transform(x, y)


def _first_point_xy(wkt_value):
    """Extract the first (x, y) coordinate pair from a WKT string."""
    import re

    match = re.search(r"(-?\d+(?:\.\d+)?(?:e[+-]?\d+)?)\s+(-?\d+(?:\.\d+)?)", wkt_value)
    return float(match.group(1)), float(match.group(2))


def test_merge_reprojects_from_declared_source_crs():
    utm_table = pa.table(
        {"id": [1], "geometry": _wkt(["POINT (500000 4649776)"], UTM)}
    )
    result = merge.merge_shards([utm_table])

    assert isinstance(result, MergeResult)
    # Coordinates were actually converted (not merely relabelled)
    out = ga.format_wkt(result.table.column("geometry")).to_pylist()[0]
    lon, lat = _expected_lonlat(*UTM_POINT)
    got_x, got_y = _first_point_xy(out)
    assert got_x == pytest.approx(lon, abs=1e-9)
    assert got_y == pytest.approx(lat, abs=1e-9)
    # Target CRS is attached to the geometry type
    assert (
        result.table.column("geometry").type.crs.to_json_dict()
        == ga_crs.OGC_CRS84.to_json_dict()
    )


def test_merge_combines_shards_with_different_crs_into_target():
    shard_utm = pa.table(
        {"id": [1, 2], "geometry": _wkt(
            ["POINT (500000 4649776)", "POINT (500100 4649876)"], UTM)}
    )
    shard_lonlat = pa.table(
        {"id": [3], "geometry": _wkt(["POINT (-63 42)"], ga_crs.OGC_CRS84)}
    )
    result = merge.merge_shards([shard_utm, shard_lonlat])
    table = result.table

    assert table.num_rows == 3
    # Every output geometry carries the same target CRS
    output_crs = table.column("geometry").type.crs.to_json_dict()
    assert output_crs == ga_crs.OGC_CRS84.to_json_dict()

    # The already-target-CRS shard was not transformed; the UTM one was
    manifest = result.manifest
    assert manifest[0].transformed is True
    assert manifest[1].transformed is False

    # Target CRS is recorded in the table metadata
    meta_crs = json.loads(table.schema.metadata[METADATA_TARGET_CRS])
    assert meta_crs["id"] == {"authority": "OGC", "code": "CRS84"}
    manifest_meta = json.loads(table.schema.metadata[METADATA_MANIFEST])
    assert len(manifest_meta) == 2
    assert manifest_meta[0]["source_crs"]["id"]["code"] == 32620


def test_merge_explicit_target_crs():
    shard = pa.table(
        {"geometry": _wkt(["POINT (-63 42)"], ga_crs.OGC_CRS84)}
    )
    result = merge.merge_shards([shard], target_crs="EPSG:32620")
    out_type = result.table.column("geometry").type
    assert out_type.crs.to_json_dict()["id"]["code"] == 32620
    # lon/lat -> metres gives large projected coordinates
    out = ga.format_wkt(result.table.column("geometry")).to_pylist()[0]
    assert "POINT (5" in out


def test_null_correspondence_is_preserved():
    values = ["POINT (500000 4649776)", None, "POINT (500100 4649876)"]
    shard = pa.table({"id": [1, 2, 3], "geometry": _wkt(values, UTM)})
    result = merge.merge_shards([shard])

    geometry = result.table.column("geometry")
    assert geometry.is_null().to_pylist() == [False, True, False]
    # Attribute rows line up with geometry rows
    assert result.table.column("id").to_pylist() == [1, 2, 3]
    # Non-null values were still transformed
    assert geometry.null_count == 1


def test_mixed_geometry_types_and_wkb_input():
    values = [
        "POINT (500000 4649776)",
        "LINESTRING (500000 4649776, 500100 4649876)",
        None,
    ]
    shard = pa.table({"id": [1, 2, 3], "geometry": _wkb(values, UTM)})
    result = merge.merge_shards([shard])
    geometry = result.table.column("geometry")
    assert isinstance(geometry.type, ga.WkbType)
    assert geometry.is_null().to_pylist() == [False, False, True]
    rendered = ga.format_wkt(geometry).to_pylist()
    assert rendered[0].startswith("POINT")
    assert rendered[1].startswith("LINESTRING")


def test_native_geoarrow_encoding_is_preserved_in_auto_mode():
    point = _native_points([500000.0], [4649776.0], UTM)
    shard = pa.table({"id": [1], "geometry": point})
    result = merge.merge_shards([shard])
    out_type = result.table.column("geometry").type
    assert isinstance(out_type, ga.PointType)
    assert out_type.crs.to_json_dict() == ga_crs.OGC_CRS84.to_json_dict()

    # Forcing WKB unifies the encoding
    forced = merge.merge_shards([shard], output_encoding="wkb")
    assert isinstance(forced.table.column("geometry").type, ga.WkbType)


def test_z_ordinate_is_preserved():
    shard = pa.table(
        {
            "geometry": _wkb(
                ["POINT Z (500000 4649776 123.5)"], UTM
            )
        }
    )
    result = merge.merge_shards([shard])
    rendered = ga.format_wkt(result.table.column("geometry")).to_pylist()[0]
    assert "POINT Z" in rendered
    assert "123.5" in rendered


def test_missing_crs_raises_locatable_error():
    # No CRS declared on the extension type
    no_crs = ga.wkt().wrap_array(
        pa.array(["POINT (1 2)", None, "POINT (3 4)"], pa.utf8())
    )
    shard = pa.table({"id": [1, 2, 3], "geometry": no_crs})

    with pytest.raises(MissingCrsError) as exc_info:
        merge.merge_shards([shard])

    records = exc_info.value.records
    assert len(records) == 1
    record = records[0]
    assert isinstance(record, MissingCrsRecord)
    assert record.shard_index == 0
    assert record.shard_path is None
    assert record.geometry_column == "geometry"
    # Only the non-null rows need a CRS fix; the null row is not flagged
    assert record.row_indices == (0, 2)
    assert record.row_count == 2


def test_all_null_geometry_without_crs_is_still_flagged():
    all_null = ga.wkt().wrap_array(pa.array([None, None], pa.utf8()))
    shard = pa.table({"id": [1, 2], "geometry": all_null})

    # Error mode reports the missing declaration even though no coordinates exist
    with pytest.raises(MissingCrsError) as exc_info:
        merge.merge_shards([shard])
    record = exc_info.value.records[0]
    assert record.row_count == 0
    assert record.row_indices == ()

    # Quarantine mode keeps the null rows (there is nothing to transform) but
    # still delivers the locatable record and never fabricates a source CRS
    result = merge.merge_shards([shard], on_missing_crs="quarantine")
    assert result.table.num_rows == 2
    assert result.quarantined is None
    assert result.missing_crs_records[0].row_count == 0


def test_missing_crs_quarantine_delivers_rows():
    good = pa.table({"id": [1], "geometry": _wkt(["POINT (500000 4649776)"], UTM)})
    bad = pa.table(
        {
            "id": [2, 3],
            "geometry": ga.wkt().wrap_array(
                pa.array(["POINT (1 2)", "POINT (3 4)"], pa.utf8())
            ),
        }
    )
    result = merge.merge_shards([good, bad], on_missing_crs="quarantine")

    # Merged table only contains the good shard's rows
    assert result.table.num_rows == 1
    assert result.table.column("id").to_pylist() == [1]
    # Quarantined rows are delivered with their original data intact
    assert result.quarantined is not None
    assert result.quarantined.num_rows == 2
    assert result.quarantined.column("id").to_pylist() == [2, 3]
    record = result.missing_crs_records[0]
    assert record.shard_index == 1
    assert record.row_indices == (0, 1)


def test_quarantined_rows_do_not_get_target_crs_label():
    bad = pa.table(
        {
            "id": [1],
            "geometry": ga.wkt().wrap_array(pa.array(["POINT (1 2)"], pa.utf8())),
        }
    )
    result = merge.merge_shards([bad], on_missing_crs="quarantine")
    # The bad row is excluded from the (empty) merged table...
    assert result.table.num_rows == 0
    # ...and its original, CRS-less geometry is preserved untouched in quarantine
    quarantined_type = result.quarantined.column("geometry").type
    assert quarantined_type.crs is None
    assert result.quarantined.column("geometry").to_pylist() == ["POINT (1 2)"]


def test_crs_override_rescues_missing_crs():
    no_crs = ga.wkt().wrap_array(pa.array(["POINT (500000 4649776)"], pa.utf8()))
    shard = pa.table({"id": [1], "geometry": no_crs})

    result = merge.merge_shards([shard], crs_overrides={0: UTM})
    assert result.table.num_rows == 1
    assert result.manifest[0].source_crs.to_json_dict()["id"]["code"] == 32620
    # Coordinates were transformed using the override
    rendered = ga.format_wkt(result.table.column("geometry")).to_pylist()[0]
    lon, lat = _expected_lonlat(*UTM_POINT)
    got_x, got_y = _first_point_xy(rendered)
    assert got_x == pytest.approx(lon, abs=1e-9)
    assert got_y == pytest.approx(lat, abs=1e-9)


def test_inconsistent_crs_across_geometry_columns_rejected():
    table = pa.table(
        {
            "g1": _wkt(["POINT (500000 4649776)"], UTM),
            "g2": _wkt(["POINT (-63 42)"], ga_crs.OGC_CRS84),
        }
    )
    with pytest.raises(ValueError, match="inconsistent source CRS"):
        merge.merge_shards([table])


def test_multiple_geometry_columns_are_both_transformed():
    table = pa.table(
        {
            "g1": _wkt(["POINT (500000 4649776)"], UTM),
            "g2": _wkt(["POINT (600000 4700000)"], UTM),
        }
    )
    result = merge.merge_shards([table])
    for name in ("g1", "g2"):
        column = result.table.column(name)
        assert column.type.crs.to_json_dict() == ga_crs.OGC_CRS84.to_json_dict()
        assert ga.format_wkt(column).to_pylist()[0].startswith("POINT (-")


def test_invalid_arguments_rejected():
    shard = pa.table({"geometry": _wkt(["POINT (1 2)"], UTM)})
    with pytest.raises(ValueError, match="on_missing_crs"):
        merge.merge_shards([shard], on_missing_crs="bogus")
    with pytest.raises(ValueError, match="output_encoding"):
        merge.merge_shards([shard], output_encoding="bogus")
    with pytest.raises(KeyError):
        merge.merge_shards([shard], geometry_columns=["missing"])
    with pytest.raises(TypeError):
        merge.merge_shards([123])


def test_parquet_row_groups_map_one_to_one():
    with TemporaryDirectory() as tmp:
        path_a = f"{tmp}/shard_a.parquet"
        path_b = f"{tmp}/shard_b.parquet"
        merged_path = f"{tmp}/merged.parquet"

        # Shard A: two row groups (2 rows + 1 all-null row)
        rg1 = pa.table(
            {
                "id": [1, 2],
                "geometry": _wkt(
                    ["POINT (500000 4649776)", "POINT (500100 4649876)"], UTM
                ),
            }
        )
        rg2 = pa.table({"id": [3], "geometry": _wkt([None], UTM)})
        with pq.ParquetWriter(path_a, rg1.schema) as writer:
            writer.write_table(rg1)
            writer.write_table(rg2)

        # Shard B: one row group, already in target CRS
        rg3 = pa.table(
            {"id": [4], "geometry": _wkt(["POINT (-63 42)"], ga_crs.OGC_CRS84)}
        )
        pq.write_table(rg3, path_b)

        result = merge.merge_shards([path_a, path_b])
        assert result.table.num_rows == 4
        assert [m.row_groups for m in result.manifest] == [2, 1]

        merge.write_merged_parquet(result, merged_path)
        parquet_file = pq.ParquetFile(merged_path)
        # 2 input row groups + 1 input row group = 3 output row groups
        assert parquet_file.num_row_groups == 3
        assert [
            parquet_file.metadata.row_group(i).num_rows for i in range(3)
        ] == [2, 1, 1]

        # Row order and null correspondence survived the file round trip
        merged = pq.read_table(merged_path)
        assert merged.column("id").to_pylist() == [1, 2, 3, 4]
        assert merged.column("geometry").is_null().to_pylist() == [
            False,
            False,
            True,
            False,
        ]
        # The in-memory table keeps one fragment per input row group, so the
        # 1:1 mapping does not depend only on the writer
        assert len(result.table.to_batches()) == 3
        # Target CRS metadata survived the file round trip
        meta = json.loads(merged.schema.metadata[METADATA_TARGET_CRS])
        assert meta["id"] == {"authority": "OGC", "code": "CRS84"}


def test_parquet_missing_crs_detected_despite_reader_default():
    # A WKB column written with no CRS. Some Parquet readers inject a default
    # CRS84 on read; the merge entry point must still treat it as missing.
    with TemporaryDirectory() as tmp:
        path = f"{tmp}/no_crs.parquet"
        plain = ga.wkb().wrap_array(
            ga.as_wkb(ga.array(["POINT (1 2)"])).storage
        )
        pq.write_table(pa.table({"id": [1], "geometry": plain}), path)

        with pytest.raises(MissingCrsError) as exc_info:
            merge.merge_shards([path])
        record = exc_info.value.records[0]
        assert record.shard_path == path
        assert record.row_group_index == 0
        assert record.row_indices == (0,)

        # Quarantine mode also works for files
        result = merge.merge_shards([path], on_missing_crs="quarantine")
        assert result.table.num_rows == 0
        assert result.quarantined.num_rows == 1


def test_geoparquet_geo_metadata_supplies_crs():
    with TemporaryDirectory() as tmp:
        path = f"{tmp}/geo.parquet"
        table = pa.table(
            {"id": [1], "geometry": _wkb(["POINT (500000 4649776)"], UTM)}
        )
        io.write_geoparquet_table(table, path)

        result = merge.merge_shards([path])
        rendered = ga.format_wkt(result.table.column("geometry")).to_pylist()[0]
        lon, lat = _expected_lonlat(*UTM_POINT)
        got_x, got_y = _first_point_xy(rendered)
        assert got_x == pytest.approx(lon, abs=1e-9)
        assert got_y == pytest.approx(lat, abs=1e-9)


def test_manifest_records_per_shard_provenance():
    s1 = pa.table({"geometry": _wkt(["POINT (500000 4649776)"], UTM)})
    s2 = pa.table({"geometry": _wkt(["POINT (-63 42)"], ga_crs.OGC_CRS84)})
    result = merge.merge_shards([s1, s2])

    assert len(result.manifest) == 2
    first = result.manifest[0]
    assert first.shard_index == 0
    assert first.shard_path is None
    assert first.rows == 1
    assert first.row_groups == 1
    assert first.source_crs.to_json_dict()["id"]["code"] == 32620
    assert first.target_crs == pyproj.CRS(ga_crs.OGC_CRS84.to_json_dict())
