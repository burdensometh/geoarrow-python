import json
import os
import random
import tempfile

import geoarrow.pyarrow as ga
import geoarrow.pyarrow.dataset as gads
import pyarrow as pa
import pyarrow.compute as pc
import pytest
from geoarrow import types
from geoarrow.pyarrow import io
from pyarrow import parquet


def _station_table(num_side=20, cell_size=1.0, no_coord=4, seed=42):
    """A table of stations whose rows are in a pseudo-random order"""
    rows = []
    for iy in range(num_side):
        for ix in range(num_side):
            rows.append(
                {
                    "id": iy * num_side + ix,
                    "x": (ix + 0.5) * cell_size,
                    "y": (iy + 0.5) * cell_size,
                    "geometry": f"POINT ({(ix + 0.5) * cell_size} "
                    f"{(iy + 0.5) * cell_size})",
                }
            )

    random.Random(seed).shuffle(rows)
    for i in range(no_coord):
        rows.append(
            {
                "id": 1000 + i,
                "x": None,
                "y": None,
                "geometry": None,
            }
        )

    return pa.table(
        {
            "geometry": ga.as_geoarrow(pa.array([r["geometry"] for r in rows])),
            "id": pa.array([r["id"] for r in rows], pa.int64()),
            "x": pa.array([r["x"] for r in rows], pa.float64()),
            "y": pa.array([r["y"] for r in rows], pa.float64()),
        }
    )


def _recovered(table):
    return table.take(pc.sort_indices(table[io.DEFAULT_ORIGINAL_ROW_COLUMN]))


def _read_geo(path):
    # WKB-encoded GeoParquet reads back as a plain binary column with
    # ``parquet.read_table()``; use the GeoParquet reader to decode it.
    return io.read_geoparquet_table(path)


def test_readpyogrio_table_gpkg():
    pyogrio = pytest.importorskip("pyogrio")
    geopandas = pytest.importorskip("geopandas")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Check gpkg (which has internal geometry column name)
        temp_gpkg = os.path.join(tmpdir, "test.gpkg")
        df = geopandas.GeoDataFrame(
            geometry=geopandas.GeoSeries.from_wkt(["POINT (0 1)"], crs="OGC:CRS84")
        )
        crs_json = df.geometry.crs.to_json()
        pyogrio.write_dataframe(df, temp_gpkg)

        table = io.read_pyogrio_table(temp_gpkg)
        assert table.column("geom").type == ga.wkb().with_crs(crs_json)
        assert ga.format_wkt(table.column("geom")).to_pylist() == ["POINT (0 1)"]

        # Check fgb (which does not have an internal geometry column name)
        temp_fgb = os.path.join(tmpdir, "test.fgb")
        pyogrio.write_dataframe(df, temp_fgb)

        table = io.read_pyogrio_table(temp_fgb)
        assert table.column("geometry").type == ga.wkb().with_crs(crs_json)
        assert ga.format_wkt(table.column("geometry")).to_pylist() == ["POINT (0 1)"]


def test_write_geoparquet_table_default():
    with tempfile.TemporaryDirectory() as tmpdir:
        temp_pq = os.path.join(tmpdir, "test.parquet")
        tab = pa.table([ga.as_geoarrow(["POINT (0 1)"])], names=["geometry"])

        # When geometry_encoding=None, geoarrow types stay geoarrow types
        # (probably need to workshop this based on geoparquet_version or something)
        io.write_geoparquet_table(tab, temp_pq, geometry_encoding=None)
        tab2 = parquet.read_table(temp_pq)
        assert b"geo" in tab2.schema.metadata
        assert ga.as_wkt(ga.point().wrap_array(tab2["geometry"])).to_pylist() == [
            "POINT (0 1)"
        ]


def test_write_geoparquet_table_wkb():
    with tempfile.TemporaryDirectory() as tmpdir:
        temp_pq = os.path.join(tmpdir, "test.parquet")
        tab = pa.table([ga.array(["POINT (0 1)"])], names=["geometry"])
        io.write_geoparquet_table(tab, temp_pq, geometry_encoding="WKB")
        tab2 = parquet.read_table(temp_pq)
        assert b"geo" in tab2.schema.metadata
        meta = json.loads(tab2.schema.metadata[b"geo"])
        assert meta["version"] == "1.0.0"
        assert tab2.schema.types[0] == pa.binary()


def test_write_geoparquet_table_geoarrow():
    with tempfile.TemporaryDirectory() as tmpdir:
        temp_pq = os.path.join(tmpdir, "test.parquet")
        tab = pa.table([ga.array(["POINT (0 1)"])], names=["geometry"])
        io.write_geoparquet_table(
            tab, temp_pq, geometry_encoding=io.geoparquet_encoding_geoarrow()
        )
        tab2 = parquet.read_table(temp_pq)
        assert b"geo" in tab2.schema.metadata
        meta = json.loads(tab2.schema.metadata[b"geo"])
        assert meta["version"] == "1.1.0"
        assert meta["columns"]["geometry"]["encoding"] == "point"
        assert ga.as_wkt(ga.point().wrap_array(tab2["geometry"])).to_pylist() == [
            "POINT (0 1)"
        ]


def test_read_geoparquet_table_wkb():
    with tempfile.TemporaryDirectory() as tmpdir:
        temp_pq = os.path.join(tmpdir, "test.parquet")

        # With "geo" metadata key
        tab = pa.table([ga.array(["POINT (0 1)"])], names=["geometry"])
        io.write_geoparquet_table(tab, temp_pq, geometry_encoding="WKB")
        tab2 = io.read_geoparquet_table(temp_pq)
        assert isinstance(tab2["geometry"].type, ga.GeometryExtensionType)
        assert b"geo" not in tab2.schema.metadata

        # Without "geo" metadata key
        tab = pa.table([pa.array(["POINT (0 1)"])], names=["geometry"])
        parquet.write_table(tab, temp_pq)
        tab2 = io.read_geoparquet_table(temp_pq)
        assert isinstance(tab2["geometry"].type, ga.GeometryExtensionType)


def test_read_geoparquet_table_geoarrow():
    with tempfile.TemporaryDirectory() as tmpdir:
        temp_pq = os.path.join(tmpdir, "test.parquet")

        tab = pa.table([ga.array(["POINT (0 1)"])], names=["geometry"])
        io.write_geoparquet_table(
            tab, temp_pq, geometry_encoding=io.geoparquet_encoding_geoarrow()
        )
        tab2 = io.read_geoparquet_table(temp_pq)
        assert tab2["geometry"].type == ga.point()


def test_geoparquet_column_spec_from_type_geom_type():
    spec_wkb = io._geoparquet_column_spec_from_type(ga.wkb())
    assert spec_wkb["geometry_types"] == []

    spec_point = io._geoparquet_column_spec_from_type(ga.point())
    assert spec_point["geometry_types"] == ["Point"]

    spec_linestring = io._geoparquet_column_spec_from_type(ga.linestring())
    assert spec_linestring["geometry_types"] == ["LineString"]

    spec_polygon = io._geoparquet_column_spec_from_type(ga.polygon())
    assert spec_polygon["geometry_types"] == ["Polygon"]

    spec_multipoint = io._geoparquet_column_spec_from_type(ga.multipoint())
    assert spec_multipoint["geometry_types"] == ["MultiPoint"]

    spec_multilinestring = io._geoparquet_column_spec_from_type(ga.multilinestring())
    assert spec_multilinestring["geometry_types"] == ["MultiLineString"]

    spec_multipolygon = io._geoparquet_column_spec_from_type(ga.multipolygon())
    assert spec_multipolygon["geometry_types"] == ["MultiPolygon"]


def test_geoparquet_column_spec_from_type_crs():
    spec_storage = io._geoparquet_column_spec_from_type(pa.binary())
    assert "crs" not in spec_storage

    spec_none = io._geoparquet_column_spec_from_type(ga.wkb())
    assert spec_none["crs"] is None

    spec_projjson = io._geoparquet_column_spec_from_type(
        ga.wkb().with_crs(types.OGC_CRS84)
    )
    assert spec_projjson["crs"]["id"]["code"] == "CRS84"


def test_geoparquet_column_spec_from_type_edges():
    spec_planar = io._geoparquet_column_spec_from_type(ga.wkb())
    assert "edges" not in spec_planar

    spec_spherical = io._geoparquet_column_spec_from_type(
        ga.wkb().with_edge_type(ga.EdgeType.SPHERICAL)
    )
    assert spec_spherical["edges"] == "spherical"


def test_geoparquet_guess_primary_geometry_column():
    assert (
        io._geoparquet_guess_primary_geometry_column(pa.schema([]), "explicit_name")
        == "explicit_name"
    )

    assert (
        io._geoparquet_guess_primary_geometry_column(
            pa.schema([pa.field("geometry", pa.binary())])
        )
        == "geometry"
    )

    assert (
        io._geoparquet_guess_primary_geometry_column(
            pa.schema([pa.field("geography", pa.binary())])
        )
        == "geography"
    )

    with pytest.raises(ValueError, match="at least one geometry column"):
        io._geoparquet_guess_primary_geometry_column(
            pa.schema([pa.field("not_geom", pa.binary())])
        )

    assert (
        io._geoparquet_guess_primary_geometry_column(
            pa.schema([pa.field("first_def_geom", ga.wkb())])
        )
        == "first_def_geom"
    )


def test_geoparquet_columns_from_schema():
    schema = pa.schema([pa.field("col_a", ga.wkb()), pa.field("col_b", pa.binary())])

    # Guessing should just return GeoArrow columns
    cols = io._geoparquet_columns_from_schema(schema)
    assert list(cols.keys()) == ["col_a"]
    assert cols["col_a"] == {"encoding": "WKB", "geometry_types": [], "crs": None}

    # Explicit should just return specified columns
    cols_explicit = io._geoparquet_columns_from_schema(schema, ["col_b"])
    assert list(cols_explicit.keys()) == ["col_b"]
    assert cols_explicit["col_b"] == {"encoding": "WKB", "geometry_types": []}

    # Guessing should always include primary geometry column
    cols_primary = io._geoparquet_columns_from_schema(
        schema, primary_geometry_column="col_b"
    )
    assert list(cols_primary.keys()) == ["col_a", "col_b"]


def test_geoparquet_metadata_from_schema():
    schema = pa.schema([pa.field("col_a", ga.wkb()), pa.field("col_b", pa.binary())])
    metadata = io._geoparquet_metadata_from_schema(schema)
    assert list(metadata.keys()) == ["version", "primary_column", "columns"]
    assert metadata["version"] == "1.0.0"
    assert metadata["primary_column"] == "col_a"
    assert list(metadata["columns"].keys()) == ["col_a"]


def test_geoparquet_metadata_from_schema_geometry_types():
    # GeoArrow encoding with add_geometry_types=False should not add geometry types
    schema = pa.schema([pa.field("col_a", ga.point())])
    metadata = io._geoparquet_metadata_from_schema(schema, add_geometry_types=False)
    assert metadata["columns"]["col_a"]["geometry_types"] == []

    # ...with None or True, it should be added
    metadata = io._geoparquet_metadata_from_schema(schema, add_geometry_types=None)
    assert metadata["columns"]["col_a"]["geometry_types"] == ["Point"]

    metadata = io._geoparquet_metadata_from_schema(schema, add_geometry_types=True)
    assert metadata["columns"]["col_a"]["geometry_types"] == ["Point"]

    # For WKB type, all values of add_geometry_types should not add geometry types
    schema = pa.schema([pa.field("col_a", ga.wkb())])
    metadata = io._geoparquet_metadata_from_schema(schema, add_geometry_types=False)
    assert metadata["columns"]["col_a"]["geometry_types"] == []

    metadata = io._geoparquet_metadata_from_schema(schema, add_geometry_types=None)
    assert metadata["columns"]["col_a"]["geometry_types"] == []

    metadata = io._geoparquet_metadata_from_schema(schema, add_geometry_types=True)
    assert metadata["columns"]["col_a"]["geometry_types"] == []


def test_guess_geometry_columns():
    assert io._geoparquet_guess_geometry_columns(pa.schema([])) == {}

    guessed_wkb = io._geoparquet_guess_geometry_columns(
        pa.schema([pa.field("geometry", pa.binary())])
    )
    assert list(guessed_wkb.keys()) == ["geometry"]
    assert guessed_wkb["geometry"] == {"encoding": "WKB"}

    guessed_wkt = io._geoparquet_guess_geometry_columns(
        pa.schema([pa.field("geometry", pa.utf8())])
    )
    assert list(guessed_wkt.keys()) == ["geometry"]
    assert guessed_wkt["geometry"] == {"encoding": "WKT"}


def test_guess_geography_columns():
    assert io._geoparquet_guess_geometry_columns(pa.schema([])) == {}

    guessed_wkb = io._geoparquet_guess_geometry_columns(
        pa.schema([pa.field("geography", pa.binary())])
    )
    assert list(guessed_wkb.keys()) == ["geography"]
    assert guessed_wkb["geography"] == {"encoding": "WKB", "edges": "spherical"}


def test_encode_chunked_array():
    with pytest.raises(ValueError, match="Expected column encoding to be one of"):
        io._geoparquet_encode_chunked_array(
            ga.array(["POINT (0 1)"]), {"encoding": "NotAnEncoding"}
        )

    with pytest.raises(ValueError, match="Can't encode column with"):
        io._geoparquet_encode_chunked_array(
            ga.array(["POINT (0 1)", "LINESTRING (0 0, 1 1)"]),
            {"encoding": io.geoparquet_encoding_geoarrow()},
        )

    with pytest.raises(ValueError, match="Can't encode column with encoding"):
        io._geoparquet_encode_chunked_array(
            ga.as_geoarrow(["POINT (0 1)"]),
            {"encoding": "linestring"},
        )

    # Check geoarrow encoding when nothing is to be done
    already_point = ga.as_geoarrow(["POINT (0 1)"])
    encoded = io._geoparquet_encode_chunked_array(
        already_point, spec={"encoding": "point"}
    )
    assert encoded == already_point.storage

    # Check geoarrow encoding when some inference and encoding has to happen
    spec = {"encoding": io.geoparquet_encoding_geoarrow()}
    encoded = io._geoparquet_encode_chunked_array(ga.as_wkb(["POINT (0 1)"]), spec=spec)
    assert encoded == already_point.storage
    assert spec["encoding"] == "point"

    spec = {"encoding": "WKB"}
    encoded = io._geoparquet_encode_chunked_array(ga.as_wkb(["POINT (0 1)"]), spec)
    assert encoded.type == pa.binary()
    assert spec == {"encoding": "WKB"}

    spec = {"encoding": "WKB"}
    encoded = io._geoparquet_encode_chunked_array(
        ga.array(["POINT (0 -1)", "POINT Z (1 2 3)"]),
        spec,
        add_geometry_types=True,
        add_bbox=True,
    )
    assert encoded.type == pa.binary()
    assert spec["bbox"] == [0, -1, 1, 2]
    assert spec["geometry_types"] == ["Point", "Point Z"]


def test_chunked_array_to_geoarrow_encodings():
    item_already_geoarrow = pa.chunked_array([ga.array(["POINT (0 1)"])])
    assert (
        io._geoparquet_chunked_array_to_geoarrow(item_already_geoarrow, {})
        is item_already_geoarrow
    )

    with pytest.raises(ValueError, match="missing 'encoding'"):
        io._geoparquet_chunked_array_to_geoarrow(pa.array([]), {})

    with pytest.raises(ValueError, match="Invalid GeoParquet encoding"):
        io._geoparquet_chunked_array_to_geoarrow(
            pa.array([]), {"encoding": "NotAnEncoding"}
        )

    item_binary = pa.chunked_array([ga.as_wkb(["POINT (0 1)"]).storage])
    item_geoarrow = io._geoparquet_chunked_array_to_geoarrow(
        item_binary, {"encoding": "WKB", "crs": None}
    )
    assert item_geoarrow.type == ga.wkb()

    item_wkt = pa.chunked_array([ga.as_wkt(["POINT (0 1)"]).storage])
    item_geoarrow = io._geoparquet_chunked_array_to_geoarrow(
        item_wkt, {"encoding": "WKT", "crs": None}
    )
    assert item_geoarrow.type == ga.wkt()


def test_chunked_array_to_geoarrow_crs():
    item_binary = pa.chunked_array([ga.as_wkb(["POINT (0 1)"]).storage])

    item_missing_crs = io._geoparquet_chunked_array_to_geoarrow(
        item_binary, {"encoding": "WKB"}
    )
    assert item_missing_crs.type.crs.to_json_dict() == types.OGC_CRS84.to_json_dict()

    item_explicit_crs = io._geoparquet_chunked_array_to_geoarrow(
        item_binary, {"encoding": "WKB", "crs": {}}
    )
    assert item_explicit_crs.type.crs.to_json_dict() == {}


def test_chunked_array_to_geoarrow_edges():
    item_binary = pa.chunked_array([ga.as_wkb(["POINT (0 1)"]).storage])

    item_planar_default = io._geoparquet_chunked_array_to_geoarrow(
        item_binary, {"encoding": "WKB"}
    )
    assert item_planar_default.type.edge_type == ga.EdgeType.PLANAR

    item_planar_explicit = io._geoparquet_chunked_array_to_geoarrow(
        item_binary, {"encoding": "WKB", "edges": "planar"}
    )
    assert item_planar_explicit.type.edge_type == ga.EdgeType.PLANAR

    item_spherical = io._geoparquet_chunked_array_to_geoarrow(
        item_binary, {"encoding": "WKB", "edges": "spherical"}
    )
    assert item_spherical.type.edge_type == ga.EdgeType.SPHERICAL

    with pytest.raises(ValueError, match="Invalid GeoParquet column edges value"):
        io._geoparquet_chunked_array_to_geoarrow(
            item_binary, {"encoding": "WKB", "edges": "invalid_edges_value"}
        )


def test_table_to_geoarrow():
    tab = pa.table([pa.array([], pa.binary())], names=["col_name"])
    tab_geo = io._geoparquet_table_to_geoarrow(tab, {"col_name": {"encoding": "WKB"}})
    assert "col_name" in tab_geo.schema.names
    assert isinstance(tab_geo["col_name"].type, ga.GeometryExtensionType)
    assert tab_geo["col_name"].type.crs.to_json_dict() == types.OGC_CRS84.to_json_dict()

    # Check with no columns selected
    tab_no_cols = tab.drop_columns(["col_name"])
    tab_no_cols_geo = io._geoparquet_table_to_geoarrow(
        tab_no_cols, {"col_name": {"encoding": "WKB"}}
    )
    assert tab_no_cols_geo == tab_no_cols


def test_write_geoparquet_spatial_bucketing_default_off():
    with tempfile.TemporaryDirectory() as tmpdir:
        temp_pq = os.path.join(tmpdir, "test.parquet")
        tab = _station_table(num_side=4)
        io.write_geoparquet_table(tab, temp_pq)

        written = _read_geo(temp_pq)
        assert io.DEFAULT_ORIGINAL_ROW_COLUMN not in written.column_names
        assert written["id"].to_pylist() == tab["id"].to_pylist()
        assert ga.format_wkt(written["geometry"]).to_pylist() == ga.format_wkt(
            tab["geometry"]
        ).to_pylist()


def test_write_geoparquet_spatial_bucketing_recovers_original_order():
    with tempfile.TemporaryDirectory() as tmpdir:
        temp_pq = os.path.join(tmpdir, "test.parquet")
        tab = _station_table(num_side=10)
        io.write_geoparquet_table(tab, temp_pq, spatial_bucketing=2.0, row_group_size=8)

        written = _read_geo(temp_pq)

        # Every row is present exactly once and the original row number
        # column is a permutation of 0..n-1.
        original = written[io.DEFAULT_ORIGINAL_ROW_COLUMN]
        assert sorted(original.to_pylist()) == list(range(tab.num_rows))

        # Reordering by the original row number recovers the input order.
        recovered = _recovered(written)
        assert recovered["id"].to_pylist() == tab["id"].to_pylist()
        assert ga.format_wkt(recovered["geometry"]).to_pylist() == ga.format_wkt(
            tab["geometry"]
        ).to_pylist()


def test_write_geoparquet_spatial_bucketing_no_coordinate_rows_last():
    with tempfile.TemporaryDirectory() as tmpdir:
        temp_pq = os.path.join(tmpdir, "test.parquet")
        tab = _station_table(num_side=6, no_coord=5)
        io.write_geoparquet_table(
            tab,
            temp_pq,
            geometry_encoding="point",
            spatial_bucketing=3.0,
            row_group_size=4,
        )

        written = _read_geo(temp_pq)
        # The no-coordinate records are written last, in their own row groups.
        assert written["id"].to_pylist()[-5:] == list(range(1000, 1005))

        pf = parquet.ParquetFile(temp_pq)
        last_row_group = pf.metadata.row_group(pf.metadata.num_row_groups - 1)
        for i in range(last_row_group.num_columns):
            column = last_row_group.column(i)
            if column.path_in_schema.startswith("geometry"):
                assert column.statistics.has_min_max is False

        # And the spatial buckets do not include the no-coordinate records.
        dataset = gads.dataset(temp_pq)
        filtered = dataset.filter_fragments(
            "POLYGON ((-1 -1, -1 100, 100 100, 100 -1, -1 -1))"
        )
        assert 1000 not in filtered.to_table()["id"].to_pylist()


@pytest.mark.parametrize(
    "spec",
    [
        2.5,
        (2.5, 2.5),
        (0.0, 0.0, 2.5, 2.5),
        {"cell_size": 2.5},
        {"num_cells": 4},
        {"num_cells_x": 4, "num_cells_y": 2},
        {"dx": 2.5, "dy": 2.5, "origin_x": 0.0, "origin_y": 0.0},
        True,
    ],
)
def test_write_geoparquet_spatial_bucketing_spec_forms(spec):
    with tempfile.TemporaryDirectory() as tmpdir:
        temp_pq = os.path.join(tmpdir, "test.parquet")
        tab = _station_table(num_side=8, no_coord=3)
        io.write_geoparquet_table(
            tab, temp_pq, spatial_bucketing=spec, row_group_size=7
        )

        written = _read_geo(temp_pq)
        assert sorted(written[io.DEFAULT_ORIGINAL_ROW_COLUMN].to_pylist()) == list(
            range(tab.num_rows)
        )
        assert _recovered(written)["id"].to_pylist() == tab["id"].to_pylist()


def test_write_geoparquet_spatial_bucketing_row_group_size():
    with tempfile.TemporaryDirectory() as tmpdir:
        temp_pq = os.path.join(tmpdir, "test.parquet")
        tab = _station_table(num_side=10, no_coord=0)
        io.write_geoparquet_table(
            tab, temp_pq, spatial_bucketing={"cell_size": 1.0}, row_group_size=3
        )

        pf = parquet.ParquetFile(temp_pq)
        sizes = [
            pf.metadata.row_group(i).num_rows
            for i in range(pf.metadata.num_row_groups)
        ]
        assert sizes
        assert max(sizes) <= 3

        # Every row is still present exactly once and in the same order after
        # recovery.
        recovered = _recovered(_read_geo(temp_pq))
        assert recovered["id"].to_pylist() == tab["id"].to_pylist()


def test_write_geoparquet_spatial_bucketing_metadata_and_roundtrip():
    with tempfile.TemporaryDirectory() as tmpdir:
        temp_pq = os.path.join(tmpdir, "test.parquet")
        tab = _station_table(num_side=6, no_coord=2)
        io.write_geoparquet_table(tab, temp_pq, spatial_bucketing=2.0, row_group_size=5)

        raw = parquet.read_table(temp_pq)
        assert b"geo" in raw.schema.metadata
        meta = json.loads(raw.schema.metadata[b"geo"])
        assert meta["primary_column"] == "geometry"

        # The geo metadata is not applied to the original row number column.
        assert io.DEFAULT_ORIGINAL_ROW_COLUMN not in meta["columns"]

        geo = io.read_geoparquet_table(temp_pq)
        assert isinstance(geo["geometry"].type, ga.GeometryExtensionType)
        assert geo["geometry"].type == ga.wkb()
        assert geo.num_rows == tab.num_rows


def test_write_geoparquet_spatial_bucketing_filters_row_groups():
    with tempfile.TemporaryDirectory() as tmpdir:
        random_pq = os.path.join(tmpdir, "random.parquet")
        bucketed_pq = os.path.join(tmpdir, "bucketed.parquet")

        tab = _station_table(num_side=40, cell_size=10.0, no_coord=0)
        io.write_geoparquet_table(tab, random_pq, row_group_size=256)
        io.write_geoparquet_table(
            tab, bucketed_pq, spatial_bucketing=20.0, row_group_size=256
        )

        window = "POLYGON ((0 0, 0 20, 20 20, 20 0, 0 0))"
        random_dataset = gads.dataset(random_pq)
        bucketed_dataset = gads.dataset(bucketed_pq)

        random_fragments = random_dataset.filter_fragments(window).get_fragments()
        bucketed_fragments = bucketed_dataset.filter_fragments(window).get_fragments()

        # The bucketed file has more row groups, but a local window only
        # needs a small fraction of them.
        assert len(bucketed_fragments) < len(random_fragments)
        assert len(bucketed_fragments) <= 2

        # The candidate fragments still contain all of the features that
        # intersect the window.
        expected_ids = [
            row["id"]
            for row in tab.to_pylist()
            if 0 < row["x"] < 20 and 0 < row["y"] < 20
        ]
        actual_ids = bucketed_dataset.filter_fragments(window).to_table()["id"]
        assert sorted(actual_ids.to_pylist()) == sorted(expected_ids)


def test_write_geoparquet_spatial_bucketing_original_row_column_options():
    with tempfile.TemporaryDirectory() as tmpdir:
        temp_pq = os.path.join(tmpdir, "test.parquet")
        tab = _station_table(num_side=4)

        # The name of the original row column can be chosen.
        io.write_geoparquet_table(
            tab, temp_pq, spatial_bucketing=1.0, original_row_column="row_id"
        )
        written = parquet.read_table(temp_pq)
        assert "row_id" in written.column_names
        assert sorted(written["row_id"].to_pylist()) == list(range(tab.num_rows))

        # ...or omitted.
        io.write_geoparquet_table(
            tab, temp_pq, spatial_bucketing=1.0, original_row_column=None
        )
        written = parquet.read_table(temp_pq)
        assert written.column_names == tab.column_names

        # An existing non-integer column cannot be repurposed.
        with pytest.raises(ValueError, match="not integer-typed"):
            io.write_geoparquet_table(
                tab, temp_pq, spatial_bucketing=1.0, original_row_column="geometry"
            )


def test_write_geoparquet_spatial_bucketing_preserves_existing_original_rows():
    with tempfile.TemporaryDirectory() as tmpdir:
        first_pq = os.path.join(tmpdir, "first.parquet")
        second_pq = os.path.join(tmpdir, "second.parquet")

        tab = _station_table(num_side=6)
        io.write_geoparquet_table(
            tab, first_pq, spatial_bucketing=2.0, row_group_size=5
        )

        # Re-bucket an already bucketed file: the original row numbers refer
        # to the first write and must survive.
        first = _read_geo(first_pq)
        io.write_geoparquet_table(
            first, second_pq, spatial_bucketing=1.0, row_group_size=3
        )

        second = _read_geo(second_pq)
        assert sorted(second[io.DEFAULT_ORIGINAL_ROW_COLUMN].to_pylist()) == list(
            range(tab.num_rows)
        )
        recovered = _recovered(second)
        assert recovered["id"].to_pylist() == tab["id"].to_pylist()


def test_write_geoparquet_spatial_bucketing_validation():
    tab = _station_table(num_side=4)

    with pytest.raises(ValueError, match="must have 2 or 4 elements"):
        io.write_geoparquet_table(
            tab, "/tmp/discard.parquet", spatial_bucketing=(1, 2, 3)
        )

    with pytest.raises(ValueError, match="cell_width"):
        io.write_geoparquet_table(tab, "/tmp/discard.parquet", spatial_bucketing=0)

    with pytest.raises(ValueError, match="must specify a cell size"):
        io.write_geoparquet_table(
            tab, "/tmp/discard.parquet", spatial_bucketing={"origin_x": 0}
        )

    with pytest.raises(ValueError, match="without a geometry column"):
        io.write_geoparquet_table(
            pa.table(
                {
                    "geometry": pa.chunked_array(
                        [pa.array([{"x": 1.0, "y": 2.0}])]
                    ),
                    "id": pa.array([1]),
                }
            ),
            "/tmp/discard.parquet",
            spatial_bucketing=1.0,
        )


def test_write_geoparquet_spatial_bucketing_edge_cases():
    with tempfile.TemporaryDirectory() as tmpdir:
        temp_pq = os.path.join(tmpdir, "test.parquet")

        # All rows lack coordinates.
        tab = pa.table(
            {
                "geometry": ga.as_wkb(pa.array([None, None, None], pa.string())),
                "id": pa.array([0, 1, 2], pa.int64()),
            }
        )
        io.write_geoparquet_table(tab, temp_pq, spatial_bucketing=1.0)
        written = parquet.read_table(temp_pq)
        assert written.num_rows == 3
        assert written["id"].to_pylist() == [0, 1, 2]

        # Empty input.
        empty = tab.slice(0, 0)
        io.write_geoparquet_table(empty, temp_pq, spatial_bucketing=1.0)
        written = parquet.read_table(temp_pq)
        assert written.num_rows == 0

        # Non-finite (very large) coordinates do not overflow bucketing.
        tab = pa.table(
            {
                "geometry": ga.array(["POINT (1e300 -1e300)", "POINT (1 2)"]),
                "id": pa.array([0, 1], pa.int64()),
            }
        )
        io.write_geoparquet_table(tab, temp_pq, spatial_bucketing=1.0)
        written = _read_geo(temp_pq)
        assert sorted(
            written[io.DEFAULT_ORIGINAL_ROW_COLUMN].to_pylist()
        ) == [0, 1]


def test_write_geoparquet_spatial_bucketing_file_like_and_where():
    import io as stdlib_io

    tab = _station_table(num_side=4)

    with tempfile.TemporaryDirectory() as tmpdir:
        temp_pq = os.path.join(tmpdir, "test.parquet")
        # ``where`` can be passed by keyword.
        io.write_geoparquet_table(tab, where=temp_pq, spatial_bucketing=1.0)
        assert parquet.read_table(temp_pq).num_rows == tab.num_rows

    # ...or as a file-like object.
    buffer = stdlib_io.BytesIO()
    io.write_geoparquet_table(tab, buffer, spatial_bucketing=1.0)
    buffer.seek(0)
    assert parquet.read_table(buffer).num_rows == tab.num_rows


def test_geodataset_detects_geoparquet_geometry_columns():
    with tempfile.TemporaryDirectory() as tmpdir:
        temp_pq = os.path.join(tmpdir, "test.parquet")

        expected_types = {"WKB": ga.wkb(), "point": ga.point(), None: ga.point()}
        for encoding, expected_type in expected_types.items():
            tab = pa.table(
                {
                    "geometry": ga.as_geoarrow(["POINT (0 0)", "POINT (1 1)"]),
                    "id": pa.array([0, 1], pa.int64()),
                }
            )
            io.write_geoparquet_table(
                tab,
                temp_pq,
                row_group_size=1,
                geometry_encoding=encoding,
                spatial_bucketing=1.0,
            )

            dataset = gads.dataset(temp_pq)
            assert dataset.geometry_columns == ("geometry",)
            assert dataset.geometry_types == (expected_type,)
            assert len(dataset.get_fragments()) == 2

            filtered = dataset.filter_fragments(
                "POLYGON ((0.5 0.5, 0.5 2, 2 2, 2 0.5, 0.5 0.5))"
            )
            assert filtered.to_table()["id"].to_pylist() == [1]


def test_geodataset_without_geometry_columns_keeps_everything():
    table = pa.table({"id": pa.array([1, 2, 3])})
    dataset = gads.dataset(table)
    assert dataset.geometry_columns == ()
    filtered = dataset.filter_fragments("POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0))")
    assert filtered.to_table().num_rows == 3
