import json
import os
import time
from tempfile import TemporaryDirectory

import geoarrow.pyarrow as ga
import geoarrow.pyarrow.dataset as gads
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.fs as fs
import pyarrow.ipc as ipc
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


def test_geodataset_multiple_geometry_columns_with_stats():
    table = pa.table(
        [
            ga.as_geoarrow(["POINT (0 1)", "POINT (2 3)"]),
            ga.as_geoarrow(["POINT (4 5)", "POINT (6 7)"]),
        ],
        ["geometry1", "geometry2"],
    )
    with TemporaryDirectory() as td:
        pq.write_table(table, f"{td}/table.parquet", row_group_size=1)

        with_stats = gads.dataset(f"{td}/table.parquet")
        without_stats = gads.ParquetRowGroupGeoDataset.create(
            pa.dataset.dataset(f"{td}/table.parquet"), use_column_statistics=False
        )
        assert with_stats.index_fragments().to_pylist() == (
            without_stats.index_fragments().to_pylist()
        )


def test_geodataset_mixed_statistics_columns():
    # A geoarrow column (usable statistics) alongside a WKT column
    # (must be scanned) previously raised because the fallback scan reused
    # the wrong types.
    table = pa.table(
        [
            ga.as_geoarrow(["POINT (0 1)", "POINT (2 3)"]),
            ga.array(["POINT (4 5)", "POINT (6 7)"]),
        ],
        ["geometry_geo", "geometry_wkt"],
    )
    with TemporaryDirectory() as td:
        pq.write_table(table, f"{td}/table.parquet", row_group_size=1)

        with_stats = gads.dataset(f"{td}/table.parquet")
        without_stats = gads.ParquetRowGroupGeoDataset.create(
            pa.dataset.dataset(f"{td}/table.parquet"), use_column_statistics=False
        )
        assert with_stats.index_fragments().to_pylist() == (
            without_stats.index_fragments().to_pylist()
        )


def test_geodataset_multiple_files_use_each_files_statistics():
    with TemporaryDirectory() as td:
        pq.write_table(
            pa.table([ga.as_geoarrow(["POINT (0 1)"])], ["geometry"]),
            f"{td}/a.parquet",
            row_group_size=1,
        )
        pq.write_table(
            pa.table([ga.as_geoarrow(["POINT (100 200)"])], ["geometry"]),
            f"{td}/b.parquet",
            row_group_size=1,
        )

        geods = gads.dataset(td)
        assert len(geods.get_fragments()) == 2
        assert geods.index_fragments().to_pylist() == [
            {
                "_fragment_index": 0,
                "geometry": {"xmin": -0.0, "ymin": 1.0, "xmax": 0.0, "ymax": 1.0},
            },
            {
                "_fragment_index": 1,
                "geometry": {
                    "xmin": 100.0,
                    "ymin": 200.0,
                    "xmax": 100.0,
                    "ymax": 200.0,
                },
            },
        ]


def test_geodataset_index_path_none_by_default(tmp_path):
    pq.write_table(
        pa.table([ga.as_geoarrow(["POINT (0 1)"])], ["geometry"]),
        str(tmp_path / "table.parquet"),
    )
    geods = gads.dataset(str(tmp_path))
    assert geods.index_path == str(tmp_path / gads._DEFAULT_INDEX_FILENAME)

    # Opening a dataset must not create the sidecar
    geods.index_fragments()
    assert not os.path.exists(geods.index_path)


def test_geodataset_save_and_reuse_index(tmp_path):
    pq.write_table(
        pa.table([ga.as_geoarrow(["POINT (0 1)", "POINT (2 3)"])], ["geometry"]),
        str(tmp_path / "table.parquet"),
        row_group_size=1,
    )

    geods = gads.dataset(str(tmp_path))
    original = geods.index_fragments().to_pylist()
    saved_path = geods.save_index()
    assert os.path.exists(saved_path)

    reused = gads.dataset(str(tmp_path))
    assert reused.index_fragments().to_pylist() == original
    assert all(row["reused"] for row in reused._index_rows)


def test_geodataset_index_file_is_not_a_fragment(tmp_path):
    pq.write_table(
        pa.table([ga.as_geoarrow(["POINT (0 1)"])], ["geometry"]),
        str(tmp_path / "table.parquet"),
    )

    before = len(gads.dataset(str(tmp_path)).get_fragments())
    gads.dataset(str(tmp_path)).save_index()

    # The hidden default sidecar lives in the data directory but must never be
    # picked up as a data fragment.
    reopened = gads.dataset(str(tmp_path), index_path=False)
    assert len(reopened.get_fragments()) == before == 1
    assert reopened.index_fragments().num_rows == 1


def test_geodataset_recomputes_only_changed_file(tmp_path):
    def write(path, coordinates):
        pq.write_table(
            pa.table([ga.as_geoarrow(coordinates)], ["geometry"]),
            str(path),
            row_group_size=1,
        )

    write(tmp_path / "a.parquet", ["POINT (0 1)", "POINT (2 3)"])
    write(tmp_path / "b.parquet", ["POINT (10 20)"])
    gads.dataset(str(tmp_path)).save_index()

    # Rewriting one file must invalidate only that file's entries
    time.sleep(0.01)
    write(tmp_path / "b.parquet", ["POINT (30 40)"])

    geods = gads.dataset(str(tmp_path))
    index = geods.index_fragments()
    assert [row["reused"] for row in geods._index_rows] == [True, True, False]
    assert index.column("geometry").to_pylist()[2] == {
        "xmin": 30.0,
        "ymin": 40.0,
        "xmax": 30.0,
        "ymax": 40.0,
    }

    # The refresh is persisted so the next startup can reuse everything
    again = gads.dataset(str(tmp_path))
    again.index_fragments()
    assert all(row["reused"] for row in again._index_rows)


def test_geodataset_corrupt_index_does_not_hide_data(tmp_path):
    pq.write_table(
        pa.table(
            [ga.as_geoarrow(["POINT (0 1)", "POINT (50 60)"])], ["geometry"]
        ),
        str(tmp_path / "table.parquet"),
        row_group_size=1,
    )
    geods = gads.dataset(str(tmp_path))
    geods.save_index()

    with open(geods.index_path, "wb") as f:
        f.write(b"not an arrow file")

    reopened = gads.dataset(str(tmp_path))
    assert len(reopened.index_fragments()) == 2
    filtered = reopened.filter_fragments(
        "POLYGON ((-1 -1, 1 -1, 1 2, -1 2, -1 -1))"
    )
    assert filtered.to_table().num_rows == 1


def test_geodataset_mismatched_index_does_not_hide_data(tmp_path):
    pq.write_table(
        pa.table([ga.as_geoarrow(["POINT (0 1)", "POINT (2 3)"])], ["geometry"]),
        str(tmp_path / "table.parquet"),
        row_group_size=1,
    )
    gads.dataset(str(tmp_path)).save_index()

    # Build an index for unrelated data and put it where the dataset expects it
    with TemporaryDirectory() as other:
        os.makedirs(f"{other}/data")
        pq.write_table(
            pa.table([ga.as_geoarrow(["POINT (1000 1000)"])], ["geometry"]),
            f"{other}/data/table.parquet",
            row_group_size=1,
        )
        gads.dataset(f"{other}/data").save_index()
        with open(f"{other}/data/{gads._DEFAULT_INDEX_FILENAME}", "rb") as f:
            mismatched_bytes = f.read()

    with open(os.path.join(tmp_path, gads._DEFAULT_INDEX_FILENAME), "wb") as f:
        f.write(mismatched_bytes)

    reopened = gads.dataset(str(tmp_path))
    filtered = reopened.filter_fragments(
        "POLYGON ((-1 -1, 1 -1, 1 2, -1 2, -1 -1))"
    )
    assert filtered.to_table().num_rows == 1

    # A mismatched index must not be silently replaced on open
    with open(os.path.join(tmp_path, gads._DEFAULT_INDEX_FILENAME), "rb") as f:
        assert f.read() == mismatched_bytes


def test_geodataset_tampered_box_is_recomputed(tmp_path):
    pq.write_table(
        pa.table(
            [ga.as_geoarrow(["POINT (0 1)", "POINT (2 3)", "POINT (100 100)"])],
            ["geometry"],
        ),
        str(tmp_path / "table.parquet"),
        row_group_size=2,
    )
    geods = gads.dataset(str(tmp_path))
    geods.save_index()

    index_path = geods.index_path
    with fs.LocalFileSystem().open_input_stream(index_path) as src:
        table = ipc.open_stream(src).read_all()

    boxes = table.column("_box.geometry").to_pylist()
    boxes[0] = {"xmin": 1000.0, "ymin": 1000.0, "xmax": 1000.0, "ymax": 1000.0}
    table = table.set_column(
        table.schema.get_field_index("_box.geometry"),
        "_box.geometry",
        pa.array(boxes, type=table.column("_box.geometry").type),
    )
    with fs.LocalFileSystem().open_output_stream(index_path) as sink:
        with ipc.new_stream(sink, table.schema) as writer:
            writer.write_table(table)

    reopened = gads.dataset(str(tmp_path))
    index = reopened.index_fragments()
    assert index.column("geometry").to_pylist()[0] == {
        "xmin": -0.0,
        "ymin": 1.0,
        "xmax": 2.0,
        "ymax": 3.0,
    }
    assert [row["reused"] for row in reopened._index_rows] == [False, True]


def test_geodataset_save_index_from_filtered_dataset(tmp_path):
    pq.write_table(
        pa.table([ga.as_geoarrow(["POINT (0 1)", "POINT (2 3)"])], ["geometry"]),
        str(tmp_path / "a.parquet"),
        row_group_size=1,
    )
    pq.write_table(
        pa.table([ga.as_geoarrow(["POINT (10 20)"])], ["geometry"]),
        str(tmp_path / "b.parquet"),
        row_group_size=1,
    )

    geods = gads.dataset(str(tmp_path))
    geods.save_index()

    filtered = geods.filter_fragments("POLYGON ((1 2, 3 2, 3 4, 1 4, 1 2))")
    assert len(filtered.get_fragments()) == 1
    filtered.save_index()

    # Saving from a filtered dataset must preserve the other fragments
    reopened = gads.dataset(str(tmp_path))
    assert len(reopened.get_fragments()) == 3
    assert reopened.filter_fragments(
        "POLYGON ((9 19, 11 19, 11 21, 9 21, 9 19))"
    ).to_table().num_rows == 1


def test_geodataset_saved_index_metadata(tmp_path):
    pq.write_table(
        pa.table([ga.as_geoarrow(["POINT (0 1)"])], ["geometry"]),
        str(tmp_path / "table.parquet"),
    )
    geods = gads.dataset(str(tmp_path))
    index_path = geods.save_index()

    with fs.LocalFileSystem().open_input_stream(index_path) as src:
        table = ipc.open_stream(src).read_all()

    metadata = json.loads(table.schema.metadata[gads._INDEX_METADATA_KEY])
    assert metadata["format"] == gads._INDEX_FORMAT
    assert metadata["version"] == gads._INDEX_VERSION
    assert metadata["geometry_columns"] == ["geometry"]

    # Identity columns are present and the public index shape is unchanged
    assert gads._FILE_PATH_COLUMN in table.column_names
    assert gads._ROW_GROUP_INDEX_COLUMN in table.column_names
    assert gads._ROW_GROUP_DIGEST_COLUMN in table.column_names
    assert geods.index_fragments().column_names == ["_fragment_index", "geometry"]


def test_geodataset_in_memory_index_not_supported():
    table = pa.table([ga.array(["POINT (0.5 1.5)"])], ["geometry"])
    geods = gads.dataset(table)
    with pytest.raises(TypeError):
        geods.save_index()
    with pytest.raises(TypeError):
        geods.load_index()
