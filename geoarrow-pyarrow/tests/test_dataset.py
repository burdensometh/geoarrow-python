from tempfile import TemporaryDirectory
from types import SimpleNamespace

import geoarrow.pyarrow as ga
import geoarrow.pyarrow.dataset as gads
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import pytest


def test_geodataset_column_name_guessing():
    table = pa.table([ga.array(["POINT (0.5 1.5)"])], ["geometry"])
    geods = gads.dataset(table)
    assert geods.geometry_columns == ("geometry",)


def test_geodataset_column_type_guessing():
    # Already a geoarrow type
    table = pa.table([ga.array(["POINT (0.5 1.5)"])], ["geometry"])
    geods = gads.dataset(table, geometry_columns=["geometry"])
    assert geods.geometry_types == (ga.wkt(),)

    # utf8 maps to wkt
    table = pa.table([ga.array(["POINT (0.5 1.5)"]).storage], ["geometry"])
    geods = gads.dataset(table, geometry_columns=["geometry"])
    assert geods.geometry_types == (ga.wkt(),)

    # binary maps to wkb
    table = pa.table([ga.as_wkb(["POINT (0.5 1.5)"]).storage], ["geometry"])
    geods = gads.dataset(table, geometry_columns=["geometry"])
    assert geods.geometry_types == (ga.wkb(),)

    # Error for other types
    with pytest.raises(TypeError):
        table = pa.table([[123]], ["geometry"])
        geods = gads.dataset(table, geometry_columns=["geometry"])
        _ = geods.geometry_types


def test_geodataset_in_memory():
    table1 = pa.table([ga.array(["POINT (0.5 1.5)"])], ["geometry"])
    table2 = pa.table([ga.array(["POINT (2.5 3.5)"])], ["geometry"])

    geods = gads.dataset([table1, table2])
    assert isinstance(geods._parent, ds.InMemoryDataset)
    assert len(list(geods._parent.get_fragments())) == 2

    filtered1 = geods.filter_fragments("POLYGON ((2 3, 3 3, 3 4, 2 4, 2 3))")
    assert isinstance(filtered1, gads.GeoDataset)
    assert filtered1.to_table().num_rows == 1
    assert filtered1._index.column("_fragment_index") == pa.chunked_array([[0]])
    assert filtered1._index.column("geometry") == geods._index.column("geometry").take(
        [1]
    )

    # Make sure we can filter to empty
    filtered0 = geods.filter_fragments("POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0))")
    assert filtered0.to_table().num_rows == 0

    with pytest.raises(TypeError):
        gads.dataset([table1], use_row_groups=True)


def test_geodataset_in_memory_guessed_type():
    table1 = pa.table([ga.array(["POINT (0.5 1.5)"]).storage], ["geometry"])
    table2 = pa.table([ga.array(["POINT (2.5 3.5)"]).storage], ["geometry"])
    geods = gads.dataset([table1, table2], geometry_columns=["geometry"])

    filtered1 = geods.filter_fragments("POLYGON ((2 3, 3 3, 3 4, 2 4, 2 3))")
    assert filtered1.to_table().num_rows == 1


def test_geodataset_multiple_geometry_columns():
    table1 = pa.table(
        [ga.array(["POINT (0.5 1.5)"]), ga.array(["POINT (2.5 3.5)"])],
        ["geometry1", "geometry2"],
    )
    table2 = pa.table(
        [ga.array(["POINT (4.5 5.5)"]), ga.array(["POINT (6.5 7.5)"])],
        ["geometry1", "geometry2"],
    )

    geods = gads.dataset([table1, table2])
    assert isinstance(geods._parent, ds.InMemoryDataset)
    assert len(list(geods._parent.get_fragments())) == 2

    filtered1 = geods.filter_fragments("POLYGON ((0 1, 1 1, 1 2, 0 2, 0 1))").to_table()
    assert filtered1.num_rows == 1

    filtered2 = geods.filter_fragments("POLYGON ((2 3, 3 3, 3 4, 2 4, 2 3))").to_table()
    assert filtered2.num_rows == 1


def test_geodataset_parquet():
    table1 = pa.table([ga.array(["POINT (0.5 1.5)"])], ["geometry"])
    table2 = pa.table([ga.array(["POINT (2.5 3.5)"])], ["geometry"])
    with TemporaryDirectory() as td:
        pq.write_table(table1, f"{td}/table1.parquet")
        pq.write_table(table2, f"{td}/table2.parquet")
        geods = gads.dataset(
            [f"{td}/table1.parquet", f"{td}/table2.parquet"], use_row_groups=False
        )

        filtered1 = geods.filter_fragments(
            "POLYGON ((0 1, 1 1, 1 2, 0 2, 0 1))"
        ).to_table()
        assert filtered1.num_rows == 1


def test_geodataset_parquet_rowgroups():
    table = pa.table([ga.array(["POINT (0.5 1.5)", "POINT (2.5 3.5)"])], ["geometry"])
    with TemporaryDirectory() as td:
        pq.write_table(table, f"{td}/table.parquet", row_group_size=1)

        geods = gads.dataset(f"{td}/table.parquet")
        assert isinstance(geods, gads.ParquetRowGroupGeoDataset)
        assert len(geods.get_fragments()) == 2

        filtered1 = geods.filter_fragments("POLYGON ((2 3, 3 3, 3 4, 2 4, 2 3))")
        assert isinstance(filtered1, gads.ParquetRowGroupGeoDataset)
        assert filtered1.to_table().num_rows == 1
        assert filtered1._index.column("_fragment_index") == pa.chunked_array([[0]])
        assert filtered1._index.column("geometry") == geods._index.column(
            "geometry"
        ).take([1])

        assert filtered1._row_group_ids == [1]


def test_geodataset_parquet_index_rowgroups():
    array_wkt = ga.array(
        ["LINESTRING (0.5 1.5, 2.5 3.5)", "LINESTRING (4.5 5.5, 6.5 7.5)"]
    )
    array_geoarrow = ga.as_geoarrow(
        ["LINESTRING (8.5 9.5, 10.5 11.5)", "LINESTRING (12.5 13.5, 14.5 15.5)"]
    )

    table_wkt = pa.table([array_wkt], ["geometry"])
    table_geoarrow = pa.table([array_geoarrow], ["geometry"])
    table_both = pa.table(
        [array_wkt, array_geoarrow], ["geometry_wkt", "geometry_geoarrow"]
    )

    with TemporaryDirectory() as td:
        pq.write_table(table_wkt, f"{td}/table_wkt.parquet", row_group_size=1)
        pq.write_table(table_geoarrow, f"{td}/table_geoarrow.parquet", row_group_size=1)
        pq.write_table(
            table_geoarrow,
            f"{td}/table_geoarrow_nostats.parquet",
            row_group_size=1,
            write_statistics=False,
        )
        pq.write_table(table_both, f"{td}/table_both.parquet", row_group_size=1)

        ds_wkt = gads.dataset(f"{td}/table_wkt.parquet")
        ds_geoarrow = gads.dataset(f"{td}/table_geoarrow.parquet")
        ds_geoarrow_nostats = gads.dataset(f"{td}/table_geoarrow_nostats.parquet")
        ds_both = gads.dataset(f"{td}/table_both.parquet")

        index_wkt = ds_wkt.index_fragments()
        index_geoarrow = ds_geoarrow.index_fragments()
        index_geoarrow_nostats = ds_geoarrow_nostats.index_fragments()
        index_both = ds_both.index_fragments()

        # All the fragment indices should be the same
        assert index_geoarrow.column(0) == index_wkt.column(0)
        assert index_geoarrow_nostats.column(0) == index_wkt.column(0)
        assert index_both.column(0) == index_wkt.column(0)

        # The wkt index should be the same in index_both and index_wkt
        assert index_both.column("geometry_wkt") == index_wkt.column("geometry")

        # The geoarrow index should be the same everywhere
        assert index_geoarrow_nostats.column("geometry") == index_geoarrow.column(
            "geometry"
        )
        assert index_both.column("geometry_geoarrow") == index_geoarrow.column(
            "geometry"
        )


def test_geodataset_parquet_filter_rowgroups_with_stats():
    arr = ga.as_geoarrow(["POINT (0.5 1.5)", "POINT (2.5 3.5)"])
    table = pa.table([arr], ["geometry"])
    with TemporaryDirectory() as td:
        pq.write_table(table, f"{td}/table.parquet", row_group_size=1)

        geods = gads.dataset(f"{td}/table.parquet")
        assert len(geods.get_fragments()) == 2

        geods._build_index_using_stats(["geometry"])

        filtered1 = geods.filter_fragments(
            "POLYGON ((0 1, 1 1, 1 2, 0 2, 0 1))"
        ).to_table()
        assert filtered1.num_rows == 1


def test_geodataset_parquet_rowgroups_multiple_files():
    # Row group ids restart at 0 for each file: make sure statistics are
    # attributed to the file the row group actually came from.
    table1 = pa.table([ga.as_geoarrow(["POINT (1 1)", "POINT (2 2)"])], ["geometry"])
    table2 = pa.table(
        [ga.as_geoarrow(["POINT (100 100)", "POINT (200 200)"])], ["geometry"]
    )
    with TemporaryDirectory() as td:
        pq.write_table(table1, f"{td}/table1.parquet")
        pq.write_table(table2, f"{td}/table2.parquet")

        geods = gads.dataset([f"{td}/table1.parquet", f"{td}/table2.parquet"])
        assert geods._row_group_ids == [0, 0]

        index = geods.index_fragments()
        boxes = index.column("geometry").to_pylist()
        assert boxes[0] == {"xmin": 1.0, "ymin": 1.0, "xmax": 2.0, "ymax": 2.0}
        assert boxes[1] == {"xmin": 100.0, "ymin": 100.0, "xmax": 200.0, "ymax": 200.0}

        filtered = geods.filter_fragments(
            "POLYGON ((90 90, 90 210, 210 210, 210 90, 90 90))"
        )
        assert filtered.to_table().num_rows == 2


def test_geodataset_parquet_rowgroups_missing_statistics():
    arr1 = ga.as_geoarrow(["POINT (0.5 1.5)", "POINT (2.5 3.5)"])
    arr2 = ga.as_geoarrow(["POINT (4.5 5.5)", "POINT (6.5 7.5)"])
    with TemporaryDirectory() as td:
        pq.write_table(
            pa.table([arr1], ["geometry"]), f"{td}/stats.parquet", row_group_size=1
        )
        pq.write_table(
            pa.table([arr2], ["geometry"]),
            f"{td}/nostats.parquet",
            row_group_size=1,
            write_statistics=False,
        )

        geods = gads.dataset([f"{td}/stats.parquet", f"{td}/nostats.parquet"])
        index = geods.index_fragments()

        # Row groups without statistics are read to compute their envelope
        assert index.column("_geometry_status").to_pylist() == [
            "statistics",
            "statistics",
            "computed",
            "computed",
        ]
        boxes = index.column("geometry").to_pylist()
        assert boxes[2] == {"xmin": 4.5, "ymin": 5.5, "xmax": 4.5, "ymax": 5.5}
        assert boxes[3] == {"xmin": 6.5, "ymin": 7.5, "xmax": 6.5, "ymax": 7.5}

        # Filtering uses all row groups, regardless of statistics
        filtered = geods.filter_fragments("POLYGON ((4 5, 4 6, 5 6, 5 5, 4 5))")
        assert filtered.to_table().num_rows == 1
        filtered = geods.filter_fragments("POLYGON ((0 1, 0 2, 1 2, 1 1, 0 1))")
        assert filtered.to_table().num_rows == 1


def test_geodataset_parquet_rowgroups_empty_geometries():
    empty_points = ga.as_geoarrow(["POINT (0 0)", "POINT EMPTY", "POINT EMPTY"])[1:]
    real_points = ga.as_geoarrow(["POINT (5.5 6.5)", "POINT (7.5 8.5)"])
    table = pa.table([pa.chunked_array([empty_points, real_points])], ["geometry"])
    with TemporaryDirectory() as td:
        pq.write_table(table, f"{td}/table.parquet", row_group_size=2)

        geods = gads.dataset(f"{td}/table.parquet")
        assert len(geods.get_fragments()) == 2

        index = geods.index_fragments()
        boxes = index.column("geometry").to_pylist()
        statuses = index.column("_geometry_status").to_pylist()

        # The all-empty row group is marked as empty and its envelope is
        # null: an empty geometry has no coordinates and must not be
        # treated as a point at the origin.
        assert statuses == ["empty", "statistics"]
        assert boxes[0] is None
        assert boxes[1] == {"xmin": 5.5, "ymin": 6.5, "xmax": 7.5, "ymax": 8.5}

        # A query containing the origin must not match the empty row group
        filtered = geods.filter_fragments("POLYGON ((-1 -1, -1 1, 1 1, 1 -1, -1 -1))")
        assert filtered.to_table().num_rows == 0

        # A query around the real points keeps only the second row group
        filtered = geods.filter_fragments("POLYGON ((5 6, 5 7, 6 7, 6 6, 5 6))")
        assert filtered._row_group_ids == [1]
        assert filtered.to_table().num_rows == 2


def test_geodataset_parquet_rowgroups_all_null():
    point_type = ga.as_geoarrow(["POINT (0 0)"]).type
    null_storage = pa.array([None, None], type=point_type.storage_type)
    real_points = ga.as_geoarrow(["POINT (5.5 6.5)", "POINT (7.5 8.5)"])
    storage = pa.concat_arrays([null_storage, real_points.storage])
    table = pa.table([pa.chunked_array([storage]).cast(point_type)], ["geometry"])
    with TemporaryDirectory() as td:
        pq.write_table(table, f"{td}/table.parquet", row_group_size=2)

        geods = gads.dataset(f"{td}/table.parquet")
        index = geods.index_fragments()

        # All-null row groups are detected as empty directly from statistics
        assert index.column("_geometry_status").to_pylist() == ["empty", "statistics"]
        assert index.column("geometry").to_pylist()[0] is None

        filtered = geods.filter_fragments("POLYGON ((-1 -1, -1 1, 1 1, 1 -1, -1 -1))")
        assert filtered.to_table().num_rows == 0


def test_geodataset_explain_filter():
    table1 = pa.table([ga.as_geoarrow(["POINT (0.5 1.5)"])], ["geometry"])
    table2 = pa.table([ga.as_geoarrow(["POINT (5.5 6.5)"])], ["geometry"])
    with TemporaryDirectory() as td:
        pq.write_table(table1, f"{td}/table1.parquet")
        pq.write_table(table2, f"{td}/table2.parquet")
        geods = gads.dataset([f"{td}/table1.parquet", f"{td}/table2.parquet"])

        explained = geods.explain_filter("POLYGON ((0 1, 0 2, 1 2, 1 1, 0 1))")
        assert explained.column("_fragment_index").to_pylist() == [0, 1]
        assert explained.column("decision").to_pylist() == ["keep", "prune"]
        assert explained.column("_row_group_id").to_pylist() == [0, 0]
        reasons = explained.column("reason").to_pylist()
        assert "intersects target" in reasons[0] and "statistics" in reasons[0]
        assert "does not intersect target" in reasons[1]

        # Decisions match filter_fragments()
        filtered = geods.filter_fragments("POLYGON ((0 1, 0 2, 1 2, 1 1, 0 1))")
        assert filtered.to_table().num_rows == 1


def test_geodataset_explain_filter_in_memory():
    table1 = pa.table([ga.array(["POINT (0.5 1.5)"])], ["geometry"])
    table2 = pa.table([ga.array(["POINT (2.5 3.5)"])], ["geometry"])
    geods = gads.dataset([table1, table2])

    explained = geods.explain_filter("POLYGON ((0 1, 0 2, 1 2, 1 1, 0 1))")
    assert explained.column("decision").to_pylist() == ["keep", "prune"]
    assert "computed" in explained.column("reason").to_pylist()[0]
    assert "_row_group_id" not in explained.column_names


def test_classify_box():
    # Empty geometries have no coordinates: their aggregate envelope is
    # null, NaN, or inverted, and must never be treated as the origin.
    assert gads.GeoDataset._classify_box(None) == (True, None)
    assert gads.GeoDataset._classify_box(
        {
            "xmin": float("inf"),
            "ymin": float("inf"),
            "xmax": float("-inf"),
            "ymax": float("-inf"),
        }
    ) == (True, None)
    nan = float("nan")
    assert gads.GeoDataset._classify_box(
        {"xmin": nan, "ymin": nan, "xmax": nan, "ymax": nan}
    ) == (True, None)

    # A genuine point at the origin is a valid envelope
    empty, box = gads.GeoDataset._classify_box(
        {"xmin": 0.0, "ymin": 0.0, "xmax": 0.0, "ymax": 0.0}
    )
    assert not empty
    assert box == {"xmin": 0.0, "ymin": 0.0, "xmax": 0.0, "ymax": 0.0}


def test_classify_statistics():
    def fake_stats(
        has_min_max=False,
        min=None,
        max=None,
        has_null_count=True,
        null_count=0,
        num_values=10,
    ):
        return SimpleNamespace(
            has_min_max=has_min_max,
            min=min,
            max=max,
            has_null_count=has_null_count,
            null_count=null_count,
            num_values=num_values,
        )

    # Missing statistics -> unknown (the row group must be read)
    assert gads.GeoDataset._classify_statistics(None, None) == (
        gads._ENVELOPE_UNKNOWN,
        None,
    )

    # Valid min/max -> statistics
    status, box = gads.GeoDataset._classify_statistics(
        fake_stats(has_min_max=True, min=1.0, max=2.0),
        fake_stats(has_min_max=True, min=3.0, max=4.0),
    )
    assert status == gads.ENVELOPE_STATISTICS
    assert box == {"xmin": 1.0, "ymin": 3.0, "xmax": 2.0, "ymax": 4.0}

    # No min/max but all values null -> empty
    status, box = gads.GeoDataset._classify_statistics(
        fake_stats(null_count=10, num_values=10),
        fake_stats(null_count=10, num_values=10),
    )
    assert status == gads.ENVELOPE_EMPTY
    assert box is None

    # No min/max with data present (e.g., old writer) -> unknown, not empty
    status, box = gads.GeoDataset._classify_statistics(
        fake_stats(null_count=0, num_values=10),
        fake_stats(null_count=0, num_values=10),
    )
    assert status == gads._ENVELOPE_UNKNOWN
    assert box is None

    # Null count unavailable -> unknown
    status, box = gads.GeoDataset._classify_statistics(
        fake_stats(has_null_count=False), fake_stats(has_null_count=False)
    )
    assert status == gads._ENVELOPE_UNKNOWN
    assert box is None

    # Unusable (NaN) min/max -> unknown
    status, box = gads.GeoDataset._classify_statistics(
        fake_stats(has_min_max=True, min=float("nan"), max=float("nan")),
        fake_stats(has_min_max=True, min=float("nan"), max=float("nan")),
    )
    assert status == gads._ENVELOPE_UNKNOWN
    assert box is None


def test_parquet_fields_before():
    schema = pa.schema([pa.field("col1", pa.int32()), pa.field("col2", pa.int32())])
    fields_before = gads.ParquetRowGroupGeoDataset._count_fields_before(schema)
    assert fields_before == [(("col1",), 0), (("col2",), 1)]

    schema = pa.schema(
        [pa.field("col1", pa.list_(pa.int32())), pa.field("col2", pa.int32())]
    )
    fields_before = gads.ParquetRowGroupGeoDataset._count_fields_before(schema)
    assert fields_before == [(("col1",), 0), (("col1", "item"), 0), (("col2",), 1)]

    schema = pa.schema(
        [pa.field("col1", ga.linestring()), pa.field("col2", pa.int32())]
    )
    fields_before = gads.ParquetRowGroupGeoDataset._count_fields_before(schema)
    assert fields_before == [
        (("col1",), 0),
        (("col1", "vertices"), 0),
        (("col1", "vertices", "x"), 0),
        (("col1", "vertices", "y"), 1),
        (("col2",), 2),
    ]
