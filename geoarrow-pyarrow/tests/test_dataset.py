from tempfile import TemporaryDirectory

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


def make_roads_dataset(**kwargs):
    # A table with start point, end point, and corridor columns
    table1 = pa.table(
        [
            ga.array(["POINT (0.5 1.5)"]),
            ga.array(["POINT (10.5 11.5)"]),
            ga.array(["LINESTRING (0.5 1.5, 10.5 11.5)"]),
        ],
        ["start", "end", "corridor"],
    )
    table2 = pa.table(
        [
            ga.array(["POINT (2.5 3.5)"]),
            ga.array(["POINT (12.5 13.5)"]),
            ga.array(["LINESTRING (2.5 3.5, 12.5 13.5)"]),
        ],
        ["start", "end", "corridor"],
    )
    return gads.dataset([table1, table2], **kwargs)


def test_geodataset_index_fragments_columns():
    geods = make_roads_dataset()

    # Index a single column: only that column is read and returned
    index_end = geods.index_fragments(columns=["end"])
    assert index_end.column_names == ["_fragment_index", "end"]
    assert index_end.column("end").to_pylist() == [
        {"xmin": 10.5, "ymin": 11.5, "xmax": 10.5, "ymax": 11.5},
        {"xmin": 12.5, "ymin": 13.5, "xmax": 12.5, "ymax": 13.5},
    ]

    # A later request for more columns reuses the cached column and
    # returns columns in the requested order
    index_all = geods.index_fragments()
    assert index_all.column_names == ["_fragment_index", "start", "end", "corridor"]
    assert index_all.column("end") == index_end.column("end")

    # A single column name can be passed as a string
    index_start = geods.index_fragments(columns="start")
    assert index_start.column_names == ["_fragment_index", "start"]


def test_geodataset_index_fragments_columns_errors():
    geods = make_roads_dataset()

    with pytest.raises(ValueError, match="does not exist in the dataset schema"):
        geods.index_fragments(columns=["not_a_column"])

    with pytest.raises(ValueError, match="is not a geometry column"):
        table = pa.table(
            [ga.array(["POINT (0.5 1.5)"]), pa.array([1])], ["geometry", "not_geo"]
        )
        geods = gads.dataset(table)
        geods.index_fragments(columns=["not_geo"])


def test_geodataset_filter_fragments_columns():
    geods = make_roads_dataset()

    # Window intersects the "end" of fragment 2 only, but intersects
    # the "corridor" of both fragments
    window = "POLYGON ((12 13, 13 13, 13 14, 12 14, 12 13))"
    filtered = geods.filter_fragments(window, columns=["end"])
    assert filtered.to_table().num_rows == 1
    assert filtered.to_table().column("end").to_pylist() == ["POINT (12.5 13.5)"]

    # Only the requested column should have been indexed (i.e., read)
    assert geods._index.column_names == ["_fragment_index", "end"]
    assert filtered._index.column_names == ["_fragment_index", "end"]

    # Filtering the wrapped dataset on another column only reads that column
    filtered2 = filtered.filter_fragments(
        "POLYGON ((0 1, 1 1, 1 2, 0 2, 0 1))", columns=["start"]
    )
    assert filtered2.to_table().num_rows == 0
    assert filtered._index.column_names == ["_fragment_index", "end", "start"]


def test_geodataset_filter_fragments_multi_column_conditions():
    geods = make_roads_dataset()
    window_end = "POLYGON ((12 13, 13 13, 13 14, 12 14, 12 13))"
    window_corridor = "POLYGON ((0 1, 1 1, 1 2, 0 2, 0 1))"

    # "or": candidates matching any condition are merged
    filtered_or = geods.filter_fragments(
        {"end": window_end, "corridor": window_corridor}, mode="or"
    )
    assert filtered_or.to_table().num_rows == 2

    # "and": only candidates matching every condition are kept
    filtered_and = geods.filter_fragments(
        {"end": window_end, "corridor": window_corridor}, mode="and"
    )
    assert filtered_and.to_table().num_rows == 0

    # The same window applied to multiple columns with mode="and"
    filtered_and2 = geods.filter_fragments(
        "POLYGON ((2 3, 13 3, 13 14, 2 14, 2 3))",
        columns=["start", "end"],
        mode="and",
    )
    assert filtered_and2.to_table().num_rows == 1


def test_geodataset_filter_fragments_errors():
    geods = make_roads_dataset()
    window = "POLYGON ((12 13, 13 13, 13 14, 12 14, 12 13))"

    with pytest.raises(ValueError, match="does not exist in the dataset schema"):
        geods.filter_fragments(window, columns=["not_a_column"])

    with pytest.raises(ValueError, match="Invalid mode"):
        geods.filter_fragments(window, columns=["end"], mode="invalid")

    with pytest.raises(ValueError, match="columns cannot be specified"):
        geods.filter_fragments({"end": window}, columns=["end"])

    with pytest.raises(ValueError, match="no geometry columns"):
        geods.filter_fragments(window, columns=[])

    with pytest.raises(ValueError, match="no geometry columns"):
        geods.filter_fragments({})


def test_geodataset_filter_fragments_columns_avoids_other_columns():
    # The corridor column contains invalid WKT: filtering on "end" must
    # not read it, but filtering on "corridor" must fail
    table = pa.table(
        [ga.array(["POINT (10.5 11.5)"]), pa.array(["NOT VALID WKT"])],
        ["end", "corridor"],
    )
    geods = gads.dataset(table, geometry_columns=["end", "corridor"])

    filtered = geods.filter_fragments(
        "POLYGON ((10 11, 11 11, 11 12, 10 12, 10 11))", columns=["end"]
    )
    assert filtered.to_table().num_rows == 1

    with pytest.raises(Exception):
        geods.filter_fragments(
            "POLYGON ((10 11, 11 11, 11 12, 10 12, 10 11))", columns=["corridor"]
        )


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


def make_roads_table():
    return pa.table(
        [
            ga.as_geoarrow(["POINT (0.5 1.5)", "POINT (2.5 3.5)"]),
            ga.as_geoarrow(["POINT (10.5 11.5)", "POINT (12.5 13.5)"]),
            ga.as_geoarrow(
                [
                    "LINESTRING (0.5 1.5, 10.5 11.5)",
                    "LINESTRING (2.5 3.5, 12.5 13.5)",
                ]
            ),
        ],
        ["start", "end", "corridor"],
    )


def test_geodataset_parquet_rowgroups_multiple_stats_columns():
    # Multiple columns whose boxes can be built from column statistics
    table = make_roads_table()
    with TemporaryDirectory() as td:
        pq.write_table(table, f"{td}/table.parquet", row_group_size=1)

        geods = gads.dataset(f"{td}/table.parquet")
        index = geods.index_fragments()
        assert index.column_names == ["_fragment_index", "start", "end", "corridor"]
        assert index.column("end").to_pylist() == [
            {"xmin": 10.5, "ymin": 11.5, "xmax": 10.5, "ymax": 11.5},
            {"xmin": 12.5, "ymin": 13.5, "xmax": 12.5, "ymax": 13.5},
        ]
        assert index.column("corridor").to_pylist() == [
            {"xmin": 0.5, "ymin": 1.5, "xmax": 10.5, "ymax": 11.5},
            {"xmin": 2.5, "ymin": 3.5, "xmax": 12.5, "ymax": 13.5},
        ]

        # Filtering on one column only builds the index for that column
        geods = gads.dataset(f"{td}/table.parquet")
        filtered = geods.filter_fragments(
            "POLYGON ((12 13, 13 13, 13 14, 12 14, 12 13))", columns=["end"]
        )
        assert filtered.to_table().num_rows == 1
        assert geods._index.column_names == ["_fragment_index", "end"]

        # Multi-column conditions with mode="and"
        geods = gads.dataset(f"{td}/table.parquet")
        filtered_and = geods.filter_fragments(
            {
                "end": "POLYGON ((12 13, 13 13, 13 14, 12 14, 12 13))",
                "corridor": "POLYGON ((0 1, 1 1, 1 2, 0 2, 0 1))",
            },
            mode="and",
        )
        assert filtered_and.to_table().num_rows == 0


def test_geodataset_parquet_rowgroups_missing_stats():
    # When statistics are missing for the requested column, the index
    # falls back to computing exact boxes from the data (and only reads
    # that column)
    table = make_roads_table()
    with TemporaryDirectory() as td:
        pq.write_table(
            table,
            f"{td}/table.parquet",
            row_group_size=1,
            write_statistics=["start", "corridor"],
        )

        geods = gads.dataset(f"{td}/table.parquet")
        filtered = geods.filter_fragments(
            "POLYGON ((12 13, 13 13, 13 14, 12 14, 12 13))", columns=["end"]
        )
        assert filtered.to_table().num_rows == 1
        assert filtered.to_table().column("end").to_pylist() == [{"x": 12.5, "y": 13.5}]


def test_geodataset_parquet_rowgroups_partial_stats_coverage():
    # Statistics cover some but not all files: the result must still be
    # exact (uncertain coverage falls back to reading the data)
    table1 = pa.table(
        [ga.as_geoarrow(["POINT (10.5 11.5)"])], ["end"]
    )
    table2 = pa.table(
        [ga.as_geoarrow(["POINT (12.5 13.5)"])], ["end"]
    )
    with TemporaryDirectory() as td:
        pq.write_table(table1, f"{td}/table1.parquet", write_statistics=True)
        pq.write_table(table2, f"{td}/table2.parquet", write_statistics=False)

        geods = gads.dataset([f"{td}/table1.parquet", f"{td}/table2.parquet"])
        assert len(geods.get_fragments()) == 2

        index = geods.index_fragments(columns=["end"])
        assert index.column("end").to_pylist() == [
            {"xmin": 10.5, "ymin": 11.5, "xmax": 10.5, "ymax": 11.5},
            {"xmin": 12.5, "ymin": 13.5, "xmax": 12.5, "ymax": 13.5},
        ]

        filtered = geods.filter_fragments(
            "POLYGON ((12 13, 13 13, 13 14, 12 14, 12 13))", columns=["end"]
        )
        assert filtered.to_table().num_rows == 1


def test_geodataset_parquet_rowgroups_stats_multiple_files():
    # Each file's statistics must be read from its own metadata
    table1 = pa.table([ga.as_geoarrow(["POINT (10.5 11.5)"])], ["end"])
    table2 = pa.table([ga.as_geoarrow(["POINT (12.5 13.5)"])], ["end"])
    with TemporaryDirectory() as td:
        pq.write_table(table1, f"{td}/table1.parquet")
        pq.write_table(table2, f"{td}/table2.parquet")

        geods = gads.dataset([f"{td}/table1.parquet", f"{td}/table2.parquet"])
        index = geods.index_fragments(columns=["end"])
        assert index.column("end").to_pylist() == [
            {"xmin": 10.5, "ymin": 11.5, "xmax": 10.5, "ymax": 11.5},
            {"xmin": 12.5, "ymin": 13.5, "xmax": 12.5, "ymax": 13.5},
        ]


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
