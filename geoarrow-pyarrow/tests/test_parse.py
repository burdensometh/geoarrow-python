import os
import tempfile

import geoarrow.pyarrow as ga
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import pytest
from geoarrow.pyarrow import io, parse


# A batch that mixes valid geometry, a legally empty geometry, a missing value,
# and text that cannot be parsed.
MIXED = [
    "POINT (0 1)",
    "POINT EMPTY",
    None,
    "not valid wkt",
    "LINESTRING (0 0, 1 1)",
    "GEOMETRYCOLLECTION EMPTY",
]

MIXED_STATUS = ["valid", "empty", "missing", "unparseable", "valid", "empty"]


def test_classify_geometry_status():
    geometry, status = parse.classify_geometry(MIXED)

    assert status.to_pylist() == MIXED_STATUS
    assert isinstance(geometry.type, ga.GeometryExtensionType)
    assert geometry.type == ga.wkb()

    # Parseable rows have geometry; missing and unparseable rows do not.
    nulls = geometry.is_null().to_pylist()
    assert nulls == [False, False, True, True, False, False]

    # Empty rows parse to legally empty WKB, not to null.
    assert ga.format_wkt(geometry).to_pylist()[:2] == ["POINT (0 1)", "POINT EMPTY"]


def test_classify_geometry_raises_or_skips():
    with pytest.raises(Exception):
        parse.classify_geometry(MIXED, on_error="raise")

    with pytest.raises(ValueError):
        parse.classify_geometry(MIXED, on_error="not-a-mode")


def test_classify_geometry_empty_and_all_invalid():
    empty_geometry, empty_status = parse.classify_geometry([])
    assert len(empty_geometry) == 0
    assert empty_status.to_pylist() == []
    assert isinstance(empty_geometry.type, ga.GeometryExtensionType)

    _, all_missing = parse.classify_geometry([None, None])
    assert all_missing.to_pylist() == ["missing", "missing"]

    _, all_invalid = parse.classify_geometry(["nope", "also nope"])
    assert all_invalid.to_pylist() == ["unparseable", "unparseable"]


def test_classify_geometry_accepts_wkb_and_chunked():
    wkb = ga.as_wkb(["POINT (0 1)", "POINT EMPTY"])
    geometry, status = parse.classify_geometry(wkb)
    assert status.to_pylist() == ["valid", "empty"]

    chunked = pa.chunked_array(
        [pa.array(["POINT (0 1)", "bad"]), pa.array(["POINT EMPTY", None])]
    )
    _, chunked_status = parse.classify_geometry(chunked)
    assert chunked_status.to_pylist() == ["valid", "unparseable", "empty", "missing"]


def test_parse_batch_sequence_with_source_key():
    table = parse.parse_batch(MIXED, source_key=["north"] * 3 + ["south"] * 3)

    assert table.column_names == ["geometry", "parse_status", "source_key"]
    assert table.column("parse_status").to_pylist() == MIXED_STATUS
    assert table.column("source_key").to_pylist() == ["north"] * 3 + ["south"] * 3

    # The parsed geometry is a standard spatial column.
    assert isinstance(table.column("geometry").type, ga.GeometryExtensionType)


def test_parse_batch_table_preserves_other_columns():
    table = pa.table(
        {"region": ["a", "a", "b", "b"], "wkt": MIXED[:4], "value": [1, 2, 3, 4]}
    )

    parsed = parse.parse_batch(table, geometry_column="wkt", source_key="region")
    assert parsed.column_names == ["region", "wkt", "value", "parse_status"]
    assert parsed.column("value").to_pylist() == [1, 2, 3, 4]
    assert parsed.column("parse_status").to_pylist() == MIXED_STATUS[:4]

    with pytest.raises(KeyError):
        parse.parse_batch(table, geometry_column="missing")


def test_parse_batch_skip_drops_unparseable_rows():
    skipped = parse.parse_batch(MIXED, on_error="skip")
    assert skipped.num_rows == len(MIXED) - 1
    assert "unparseable" not in skipped.column("parse_status").to_pylist()


def test_parse_batch_on_error_raise_is_opt_in():
    with pytest.raises(Exception):
        parse.parse_batch(MIXED, on_error="raise")

    # The default keeps the previous permissive behaviour.
    assert parse.parse_batch(MIXED).num_rows == len(MIXED)


def test_parse_batch_output_writes_geoparquet():
    table = parse.parse_batch(MIXED, source_key=["north"] * 3 + ["south"] * 3)

    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "parsed.parquet")
        io.write_geoparquet_table(table, path)
        back = io.read_geoparquet_table(path)

        # A standard Parquet reader sees WKB binary plus GeoParquet metadata.
        plain = pq.read_table(path)
        assert plain.schema.field("geometry").type == pa.binary()
        assert b"geo" in plain.schema.metadata

    assert back.column_names == ["geometry", "parse_status", "source_key"]
    assert back.column("parse_status").to_pylist() == MIXED_STATUS
    assert ga.format_wkt(back.column("geometry")).to_pylist()[:2] == [
        "POINT (0 1)",
        "POINT EMPTY",
    ]
    # Valid/empty rows keep geometry; missing/unparseable rows are null.
    assert back.column("geometry").is_null().to_pylist() == [
        False,
        False,
        True,
        True,
        False,
        False,
    ]


def test_parse_batch_crs():
    table = parse.parse_batch(["POINT (0 1)"], crs=ga.OGC_CRS84)
    assert table.column("geometry").type.crs is not None


def test_summarize_parse_status_by_source_key():
    table = parse.parse_batch(MIXED, source_key=["north"] * 3 + ["south"] * 3)
    summary = parse.summarize_parse_status(table)

    assert summary.column_names == [
        "source_key",
        "n_valid",
        "n_empty",
        "n_missing",
        "n_unparseable",
        "n_total",
        "valid_rate",
    ]
    assert summary.to_pylist() == [
        {
            "source_key": "north",
            "n_valid": 1,
            "n_empty": 1,
            "n_missing": 1,
            "n_unparseable": 0,
            "n_total": 3,
            "valid_rate": pytest.approx(1 / 3),
        },
        {
            "source_key": "south",
            "n_valid": 1,
            "n_empty": 1,
            "n_missing": 0,
            "n_unparseable": 1,
            "n_total": 3,
            "valid_rate": pytest.approx(1 / 3),
        },
    ]


def test_summarize_parse_status_overall_and_empty():
    table = parse.parse_batch(MIXED)
    overall = parse.summarize_parse_status(table, partition_key=None)
    assert overall.to_pylist() == [
        {
            "n_valid": 2,
            "n_empty": 2,
            "n_missing": 1,
            "n_unparseable": 1,
            "n_total": 6,
            "valid_rate": pytest.approx(2 / 6),
        }
    ]

    empty_summary = parse.summarize_parse_status(
        parse.parse_batch([]), partition_key=None
    )
    assert empty_summary.num_rows == 0
    assert empty_summary.column_names == [
        "n_valid",
        "n_empty",
        "n_missing",
        "n_unparseable",
        "n_total",
        "valid_rate",
    ]


def _write_hive_dataset(root):
    batches = {
        "north": ["POINT (0 1)", "POINT EMPTY", None, "bad text"],
        "south": ["LINESTRING (0 0, 1 1)", None, "POINT (2 3)", None],
    }
    for region, values in batches.items():
        path = os.path.join(root, f"region={region}")
        os.makedirs(path)
        table = pa.table({"wkt": pa.array(values, type=pa.string())})
        pq.write_table(table, os.path.join(path, "part.parquet"))


def test_parse_dataset_hive_partitions():
    with tempfile.TemporaryDirectory() as td:
        _write_hive_dataset(td)
        table = parse.parse_dataset(td, geometry_column="wkt")

    assert "region" in table.column_names
    assert table.column("source_key").to_pylist() == (
        ["north"] * 4 + ["south"] * 4
    )
    assert table.column("parse_status").to_pylist() == (
        ["valid", "empty", "missing", "unparseable"]
        + ["valid", "missing", "valid", "missing"]
    )


def test_parse_dataset_summary_and_roundtrip():
    with tempfile.TemporaryDirectory() as td, tempfile.TemporaryDirectory() as out_td:
        _write_hive_dataset(td)
        table = parse.parse_dataset(td, geometry_column="wkt")

        out_path = os.path.join(out_td, "summary.parquet")
        io.write_geoparquet_table(table, out_path)

        from_file = parse.summarize_parse_status(out_path)
        direct = parse.summarize_parse_status(td, geometry_column="wkt")

    assert from_file.to_pylist() == direct.to_pylist()
    by_region = {row["source_key"]: row for row in from_file.to_pylist()}
    assert by_region["north"]["n_valid"] == 1
    assert by_region["north"]["n_empty"] == 1
    assert by_region["north"]["n_missing"] == 1
    assert by_region["north"]["n_unparseable"] == 1
    assert by_region["south"]["n_valid"] == 2
    assert by_region["south"]["n_missing"] == 2
    assert by_region["north"]["valid_rate"] == pytest.approx(0.25)


def test_parse_dataset_composite_partition_key():
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "region=north", "year=2024")
        os.makedirs(path)
        pq.write_table(
            pa.table({"geometry": ["POINT (0 1)"]}),
            os.path.join(path, "part.parquet"),
        )

        table = parse.parse_dataset(td)
        assert table.column("source_key").to_pylist() == ["north/2024"]

        summary = parse.summarize_parse_status(table)
        assert summary.column("source_key").to_pylist() == ["north/2024"]


def test_classify_geometry_accepts_view_types():
    wkt_view = pa.array(
        ["POINT (0 1)", "POINT EMPTY", None], type=pa.string_view()
    )
    _, wkt_status = parse.classify_geometry(wkt_view)
    assert wkt_status.to_pylist() == ["valid", "empty", "missing"]

    wkb_value = ga.as_wkb(["POINT (0 1)"]).to_pylist()[0]
    wkb_view = pa.array([wkb_value, None], type=pa.binary_view())
    _, wkb_status = parse.classify_geometry(wkb_view)
    assert wkb_status.to_pylist() == ["valid", "missing"]


def test_classify_geometry_ignores_spherical_edges():
    spherical = ga.with_edge_type(
        ga.as_wkb(["LINESTRING (0 0, 1 1)", "POINT EMPTY"]),
        ga.EdgeType.SPHERICAL,
    )
    geometry, status = parse.classify_geometry(spherical)
    assert status.to_pylist() == ["valid", "empty"]
    # The original edge interpretation is retained on the output.
    assert geometry.type.edge_type == ga.EdgeType.SPHERICAL


def test_summarize_parse_status_multiple_partition_keys():
    table = pa.table(
        {
            "region": ["a", "a", "b", "b"],
            "year": [2024, 2024, 2023, 2023],
            "parse_status": ["valid", "empty", "valid", "unparseable"],
        }
    )
    summary = parse.summarize_parse_status(table, partition_key=["region", "year"])
    assert summary.column_names[:2] == ["region", "year"]
    assert summary.to_pylist() == [
        {
            "region": "a",
            "year": 2024,
            "n_valid": 1,
            "n_empty": 1,
            "n_missing": 0,
            "n_unparseable": 0,
            "n_total": 2,
            "valid_rate": 0.5,
        },
        {
            "region": "b",
            "year": 2023,
            "n_valid": 1,
            "n_empty": 0,
            "n_missing": 0,
            "n_unparseable": 1,
            "n_total": 2,
            "valid_rate": 0.5,
        },
    ]


def test_summarize_parse_status_missing_partition_key_falls_back():
    table = parse.parse_batch(["POINT (0 1)", "not valid wkt"])
    summary = parse.summarize_parse_status(table)
    assert summary.to_pylist() == [
        {
            "n_valid": 1,
            "n_empty": 0,
            "n_missing": 0,
            "n_unparseable": 1,
            "n_total": 2,
            "valid_rate": 0.5,
        }
    ]


def test_parse_batch_errors_alias():
    assert parse.parse_batch(["POINT (0 1)", "bad"], errors="skip").num_rows == 1

    with pytest.raises(Exception):
        parse.parse_batch(["POINT (0 1)", "bad"], errors="strict")

    with pytest.raises(ValueError):
        parse.parse_batch(["POINT (0 1)"], on_error="raise", errors="skip")


def test_parse_dataset_is_idempotent_for_written_output():
    with tempfile.TemporaryDirectory() as src, tempfile.TemporaryDirectory() as dst:
        path = os.path.join(src, "region=north")
        os.makedirs(path)
        pq.write_table(
            pa.table({"wkt": pa.array(["POINT (0 1)", "bad", None])}),
            os.path.join(path, "part.parquet"),
        )

        first = parse.parse_dataset(src, geometry_column="wkt")
        out_path = os.path.join(dst, "parsed.parquet")
        io.write_geoparquet_table(first, out_path)

        # Re-reading the written output must not downgrade "unparseable" rows
        # to "missing" (both have null geometry).
        second = parse.parse_dataset(dst)
        assert second.column("parse_status").to_pylist() == [
            "valid",
            "unparseable",
            "missing",
        ]


def test_parse_dataset_empty():
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "region=north")
        os.makedirs(path)
        pq.write_table(
            pa.table({"wkt": pa.array([], type=pa.string())}),
            os.path.join(path, "part.parquet"),
        )

        table = parse.parse_dataset(td, geometry_column="wkt")
        assert table.num_rows == 0
        assert {"wkt", "region", "parse_status", "source_key"} <= set(
            table.column_names
        )
        assert parse.summarize_parse_status(table).num_rows == 0


def test_parse_dataset_accepts_dataset_object():
    table = pa.table({"geometry": ["POINT (0 1)", "POINT EMPTY"]})
    dataset = ds.dataset(table)
    parsed = parse.parse_dataset(dataset, source_key="region_value")
    assert parsed.column("source_key").to_pylist() == ["region_value", "region_value"]

    # A column that exists is used by name rather than treated as a constant.
    table = pa.table(
        {"geometry": ["POINT (0 1)", "POINT EMPTY"], "region": ["n", "s"]}
    )
    parsed = parse.parse_dataset(ds.dataset(table), source_key="region")
    assert parsed.column("source_key").to_pylist() == ["n", "s"]
