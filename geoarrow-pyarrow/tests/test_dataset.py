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


def test_envelope_id_batch_post_init():
    table = pa.table(
        {"envelope": ga.box(["POINT (0 1)"]), "id": pa.array([0], pa.uint64())}
    )
    batch = gads.EnvelopeIdBatch(0, 0, 1, table)
    assert batch.num_rows == 1
    assert not batch.oversized

    with pytest.raises(ValueError, match="row range"):
        gads.EnvelopeIdBatch(0, 0, 2, table)


def test_geodataset_envelope_id_batches_basic():
    table1 = pa.table([ga.as_wkb(["POINT (0.5 1.5)", "POINT (2.5 3.5)"])], ["geometry"])
    table2 = pa.table([ga.as_wkb(["POINT (4.5 5.5)"])], ["geometry"])
    geods = gads.dataset([table1, table2])

    batches = list(geods.iter_envelope_id_batches())
    assert [(b.fragment_index, b.row_start, b.row_end) for b in batches] == [
        (0, 0, 2),
        (1, 0, 1),
    ]
    assert all(not b.oversized for b in batches)
    assert all(b.table.schema.names == ["envelope", "id"] for b in batches)

    assert batches[0].table.column("envelope").to_pylist() == [
        {"xmin": 0.5, "ymin": 1.5, "xmax": 0.5, "ymax": 1.5},
        {"xmin": 2.5, "ymin": 3.5, "xmax": 2.5, "ymax": 3.5},
    ]

    # Generated business IDs follow dataset row order across fragments
    assert batches[0].table.column("id").to_pylist() == [0, 1]
    assert batches[1].table.column("id").to_pylist() == [2]


def test_geodataset_envelope_id_batches_lazy():
    table1 = pa.table([ga.as_wkb(["POINT (0.5 1.5)"])], ["geometry"])
    table2 = pa.table([ga.as_wkb(["POINT (2.5 3.5)"])], ["geometry"])
    geods = gads.dataset([table1, table2])

    batches = geods.iter_envelope_id_batches()
    assert iter(batches) is batches

    # The caller confirms the first batch before the next one is produced
    first = next(batches)
    assert (first.fragment_index, first.row_start, first.row_end) == (0, 0, 1)
    rest = list(batches)
    assert [(b.fragment_index, b.row_start, b.row_end) for b in rest] == [(1, 0, 1)]


def test_geodataset_envelope_id_batches_budget():
    geometry = ga.as_wkb([f"POINT ({i} {i})" for i in range(8)])
    geods = gads.dataset([pa.table([geometry], ["geometry"])])

    # A budget of exactly two rows splits the fragment into two-row batches
    max_bytes = geometry.slice(0, 2).nbytes
    batches = list(geods.iter_envelope_id_batches(max_bytes=max_bytes))
    assert [(b.fragment_index, b.row_start, b.row_end) for b in batches] == [
        (0, 0, 2),
        (0, 2, 4),
        (0, 4, 6),
        (0, 6, 8),
    ]
    assert all(not b.oversized for b in batches)
    expected_ids = [[0, 1], [2, 3], [4, 5], [6, 7]]
    assert [b.table.column("id").to_pylist() for b in batches] == expected_ids


def test_geodataset_envelope_id_batches_pack_record_batches():
    # With a scanner batch size smaller than the budget, one output batch
    # packs pieces from several record batches (and stays contiguous)
    geometry = ga.as_wkb([f"POINT ({i} {i})" for i in range(6)])
    geods = gads.dataset([pa.table([geometry], ["geometry"])])

    batches = list(geods.iter_envelope_id_batches(batch_size=2))
    assert len(batches) == 1
    assert (batches[0].fragment_index, batches[0].row_start) == (0, 0)
    assert batches[0].row_end == 6
    assert batches[0].table.column("id").to_pylist() == list(range(6))
    assert (
        batches[0].table.column("envelope").to_pylist()
        == ga.box(geometry).to_pylist()
    )


def test_geodataset_envelope_id_batches_oversized():
    big = "LINESTRING (" + ", ".join(f"{i} {i}" for i in range(1000)) + ")"
    geometry = ga.as_wkb(["POINT (0 1)", big, "POINT (2 3)"])
    geods = gads.dataset([pa.table([geometry], ["geometry"])])

    # Budget fits several points but not the one big linestring
    max_bytes = geometry.slice(0, 1).nbytes * 4
    batches = list(geods.iter_envelope_id_batches(max_bytes=max_bytes))
    assert [(b.row_start, b.row_end, b.oversized) for b in batches] == [
        (0, 1, False),
        (1, 2, True),
        (2, 3, False),
    ]

    # The oversized geometry is isolated but its envelope is still computed
    assert batches[1].table.column("envelope").to_pylist() == [
        {"xmin": 0.0, "ymin": 0.0, "xmax": 999.0, "ymax": 999.0}
    ]
    # Business IDs still align with source rows
    assert [b.table.column("id").to_pylist() for b in batches] == [[0], [1], [2]]


def test_geodataset_envelope_id_batches_id_column():
    table = pa.table(
        [ga.as_wkb(["POINT (0 1)", "POINT (2 3)"]), pa.array(["a", "b"])],
        ["geometry", "facility_id"],
    )
    geods = gads.dataset(table)

    batches = list(geods.iter_envelope_id_batches(id_column="facility_id"))
    assert len(batches) == 1
    assert batches[0].table.schema.names == ["envelope", "id"]
    assert batches[0].table.column("id").to_pylist() == ["a", "b"]

    # Custom output column names
    batches = list(
        geods.iter_envelope_id_batches(
            id_column="facility_id", envelope_name="bbox", id_name="facility_id"
        )
    )
    assert batches[0].table.schema.names == ["bbox", "facility_id"]


def test_geodataset_envelope_id_batches_errors():
    table = pa.table(
        [ga.as_wkb(["POINT (0 1)"]), pa.array([1])], ["geometry", "not_geometry"]
    )
    geods = gads.dataset(table)

    with pytest.raises(ValueError, match="id_column"):
        list(geods.iter_envelope_id_batches(id_column="does_not_exist"))
    with pytest.raises(ValueError, match="id_column"):
        list(geods.iter_envelope_id_batches(id_column="geometry"))
    with pytest.raises(ValueError, match="geometry_column"):
        list(geods.iter_envelope_id_batches(geometry_column="not_geometry"))
    with pytest.raises(ValueError, match="max_bytes"):
        list(geods.iter_envelope_id_batches(max_bytes=0))
    with pytest.raises(ValueError, match="batch_size"):
        list(geods.iter_envelope_id_batches(batch_size=0))

    # More than one geometry column requires an explicit geometry_column
    table2 = pa.table(
        [ga.as_wkb(["POINT (0 1)"]), ga.as_wkb(["POINT (2 3)"])], ["g1", "g2"]
    )
    geods2 = gads.dataset(table2)
    with pytest.raises(ValueError, match="geometry_column"):
        list(geods2.iter_envelope_id_batches())
    batches = list(geods2.iter_envelope_id_batches(geometry_column="g2"))
    assert batches[0].table.column("envelope").to_pylist() == [
        {"xmin": 2.0, "ymin": 3.0, "xmax": 2.0, "ymax": 3.0}
    ]


def test_geodataset_envelope_id_batches_empty():
    table = pa.table([ga.wkb().wrap_array(pa.array([], pa.binary()))], ["geometry"])
    geods = gads.dataset([table])

    assert list(geods.iter_envelope_id_batches()) == []

    collected = geods.collect_envelope_id_table()
    assert collected.num_rows == 0
    assert collected.schema.names == ["envelope", "id"]
    assert collected.column("envelope").type == ga.box(["POINT (0 1)"]).type
    assert collected.column("id").type == pa.uint64()


def test_geodataset_envelope_id_collect_restores_table():
    tables = [
        pa.table([ga.as_wkb([f"POINT ({i} {i})" for i in range(3)])], ["geometry"]),
        pa.table([ga.as_wkb([f"POINT ({i} {i})" for i in range(3, 5)])], ["geometry"]),
        pa.table([ga.as_wkb([f"POINT ({i} {i})" for i in range(5, 9)])], ["geometry"]),
    ]
    geods = gads.dataset(tables)

    # Concatenating budgeted batches restores the full-table correspondence
    collected = geods.collect_envelope_id_table(max_bytes=1)
    source = geods.to_table()
    assert collected.num_rows == source.num_rows
    assert collected.column("id").to_pylist() == list(range(source.num_rows))
    assert (
        collected.column("envelope").to_pylist()
        == ga.box(source.column("geometry")).to_pylist()
    )


def test_geodataset_envelope_id_coverage_validation():
    table = pa.table([ga.as_wkb(["POINT (0 1)", "POINT (2 3)"])], ["geometry"])
    geods = gads.dataset(table)

    def make_batch(row_start, row_end):
        n = row_end - row_start
        batch_table = pa.table(
            {
                "envelope": ga.box(["POINT (0 1)"] * n),
                "id": pa.array(range(n), pa.uint64()),
            }
        )
        return gads.EnvelopeIdBatch(0, row_start, row_end, batch_table)

    geods._validate_envelope_id_coverage([make_batch(0, 1), make_batch(1, 2)])

    # A gap between batches is rejected
    with pytest.raises(ValueError, match="gapless"):
        geods._validate_envelope_id_coverage([make_batch(1, 2)])

    # Incomplete coverage is rejected
    with pytest.raises(ValueError, match="entire dataset"):
        geods._validate_envelope_id_coverage([make_batch(0, 1), make_batch(1, 2)][:1])


def test_geodataset_envelope_id_batches_parquet_rowgroups():
    table = pa.table(
        [ga.as_wkb([f"POINT ({i} {i})" for i in range(4)])], ["geometry"]
    )
    with TemporaryDirectory() as td:
        pq.write_table(table, f"{td}/table.parquet", row_group_size=2)

        geods = gads.dataset(f"{td}/table.parquet")
        assert isinstance(geods, gads.ParquetRowGroupGeoDataset)
        assert geods.row_group_ids == (0, 1)

        # Each batch carries its source row group and row range within it
        batches = list(geods.iter_envelope_id_batches())
        assert [(b.fragment_index, b.row_start, b.row_end) for b in batches] == [
            (0, 0, 2),
            (1, 0, 2),
        ]

        collected = geods.collect_envelope_id_table()
        assert collected.column("id").to_pylist() == [0, 1, 2, 3]
        assert collected.column("envelope").to_pylist() == ga.box(
            table.column("geometry")
        ).to_pylist()
