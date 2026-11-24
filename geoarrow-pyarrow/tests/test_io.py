import json
import os
import tempfile

import geoarrow.pyarrow as ga
import pyarrow as pa
import pytest
from geoarrow import types
from geoarrow.pyarrow import io
from pyarrow import parquet


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


def _facility_table():
    return pa.table(
        [
            ga.array(["POINT (0 1)"]),
            ga.array(["POLYGON ((0 0, 0 2, 2 2, 2 0, 0 0))"]),
            pa.array([1], pa.int64()),
        ],
        names=["center", "coverage", "n"],
    )


def _write_and_get_primary(table, tmpdir, name="test.parquet", **kwargs):
    path = os.path.join(tmpdir, name)
    io.write_geoparquet_table(table, path, **kwargs)
    meta = json.loads(parquet.read_schema(path).metadata[b"geo"])
    return meta


def test_read_geoparquet_table_records_geometry_roles():
    with tempfile.TemporaryDirectory() as tmpdir:
        temp_pq = os.path.join(tmpdir, "test.parquet")
        io.write_geoparquet_table(_facility_table(), temp_pq)

        tab = io.read_geoparquet_table(temp_pq)
        assert io.geoparquet_primary_geometry_column(tab) == "center"
        assert tab.schema.field("center").metadata[io._GEOMETRY_ROLE_KEY] == b"primary"
        assert (
            tab.schema.field("coverage").metadata[io._GEOMETRY_ROLE_KEY] == b"secondary"
        )
        assert tab.schema.field("n").metadata is None


def test_write_geoparquet_table_restores_recorded_primary():
    with tempfile.TemporaryDirectory() as tmpdir:
        temp_pq = os.path.join(tmpdir, "test.parquet")
        io.write_geoparquet_table(_facility_table(), temp_pq)

        # Rename so that name-based guessing would pick the coverage column as the
        # default: only the recorded role keeps "shape" as the primary.
        tab = io.read_geoparquet_table(temp_pq)
        tab = tab.rename_columns(["shape", "geometry", "n"])
        meta = _write_and_get_primary(tab, tmpdir, "again.parquet")
        assert meta["primary_column"] == "shape"


def test_write_geoparquet_table_projected_primary_requires_explicit():
    with tempfile.TemporaryDirectory() as tmpdir:
        temp_pq = os.path.join(tmpdir, "test.parquet")
        io.write_geoparquet_table(_facility_table(), temp_pq)

        # A projection that reads only the coverage column has no primary role
        tab = io.read_geoparquet_table(temp_pq, columns=["coverage"])
        assert io.geoparquet_primary_geometry_column(tab) is None
        role = tab.schema.field("coverage").metadata[io._GEOMETRY_ROLE_KEY]
        assert role == b"secondary"

        # Writing it again must not silently promote coverage to the primary
        with pytest.raises(ValueError, match="primary geometry column"):
            io.write_geoparquet_table(tab, os.path.join(tmpdir, "bad.parquet"))

        # ...and designating the new primary explicitly works
        meta = _write_and_get_primary(
            tab,
            tmpdir,
            "explicit.parquet",
            primary_geometry_column="coverage",
        )
        assert meta["primary_column"] == "coverage"


def test_read_geoparquet_table_projection_explicit_primary():
    with tempfile.TemporaryDirectory() as tmpdir:
        temp_pq = os.path.join(tmpdir, "test.parquet")
        io.write_geoparquet_table(_facility_table(), temp_pq)

        tab = io.read_geoparquet_table(
            temp_pq, columns=["coverage"], primary_geometry_column="coverage"
        )
        assert io.geoparquet_primary_geometry_column(tab) == "coverage"
        role = tab.schema.field("coverage").metadata[io._GEOMETRY_ROLE_KEY]
        assert role == b"primary"

        with pytest.raises(ValueError, match="not a geometry column present"):
            io.read_geoparquet_table(
                temp_pq, columns=["coverage"], primary_geometry_column="center"
            )


def test_write_geoparquet_table_designated_primary_must_be_geometry():
    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.raises(ValueError, match="not present in the table"):
            io.write_geoparquet_table(
                _facility_table(),
                os.path.join(tmpdir, "bad.parquet"),
                primary_geometry_column="not_a_column",
            )

        # An explicit primary that exists but is not a geometry column is refused
        with pytest.raises(ValueError, match="not among the geometry columns"):
            io.write_geoparquet_table(
                _facility_table(),
                os.path.join(tmpdir, "bad1.parquet"),
                primary_geometry_column="n",
            )

        # A designated primary that exists but is excluded by geometry_columns is
        # also an error (rather than silently writing a different primary).
        io.write_geoparquet_table(_facility_table(), os.path.join(tmpdir, "ok.parquet"))
        tab_roled = io.read_geoparquet_table(os.path.join(tmpdir, "ok.parquet"))
        with pytest.raises(ValueError, match="not among the geometry columns"):
            io.write_geoparquet_table(
                tab_roled,
                os.path.join(tmpdir, "bad2.parquet"),
                geometry_columns=["coverage"],
            )


def test_geoparquet_primary_geometry_column_survives_transformations():
    with tempfile.TemporaryDirectory() as tmpdir:
        temp_pq = os.path.join(tmpdir, "test.parquet")
        io.write_geoparquet_table(_facility_table(), temp_pq)
        tab = io.read_geoparquet_table(temp_pq)

        # Column-level roles survive the transformations that keep columns
        filtered = tab.filter(pa.array([True]))
        assert io.geoparquet_primary_geometry_column(filtered) == "center"
        renamed = tab.rename_columns(["a", "b", "c"])
        assert io.geoparquet_primary_geometry_column(renamed) == "a"

        # Removing the primary column leaves no primary (rather than promoting one)
        dropped = tab.drop_columns(["center"])
        assert io.geoparquet_primary_geometry_column(dropped) is None
        assert io.geoparquet_primary_geometry_column(tab.select(["n"])) is None


def test_geoparquet_get_recorded_primary_conflicting_roles():
    schema = pa.schema(
        [
            pa.field("a", ga.wkb(), metadata={io._GEOMETRY_ROLE_KEY: b"primary"}),
            pa.field("b", ga.wkb(), metadata={io._GEOMETRY_ROLE_KEY: b"primary"}),
        ]
    )
    with pytest.raises(ValueError, match="more than one geometry column"):
        io._geoparquet_get_recorded_primary(schema)


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
