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

    # Explicitly declared geometry columns are the fallback primary column
    assert (
        io._geoparquet_guess_primary_geometry_column(
            pa.schema([pa.field("plain_binary", pa.binary())]),
            geometry_columns=["plain_binary"],
        )
        == "plain_binary"
    )

    # ...unless they don't exist in the schema
    with pytest.raises(ValueError, match="at least one geometry column"):
        io._geoparquet_guess_primary_geometry_column(
            pa.schema([pa.field("not_geom", pa.binary())]),
            geometry_columns=["also_not_geom"],
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


def test_plan_geometry_encoding_native_point():
    # WKT-encoded points with uniform XY dimensions
    tab = pa.table({"geometry": ga.array(["POINT (0 1)", None, "POINT (2 3)"])})
    plan = io.plan_geometry_encoding(tab)
    assert plan.encodings == {"geometry": "point"}
    assert plan["geometry"].reason == "all features are Point"
    assert plan["geometry"].geometry_types == ("Point",)
    assert plan["geometry"].expected_type == ga.point()

    # WKB-encoded points with Z dimensions
    tab = pa.table({"geometry": ga.as_wkb(["POINT Z (0 1 2)", None])})
    plan = io.plan_geometry_encoding(tab)
    assert plan.encodings == {"geometry": "point"}
    assert plan["geometry"].reason == "all features are Point Z"
    assert plan["geometry"].expected_type == ga.point().with_dimensions(
        ga.Dimensions.XYZ
    )

    # Already geoarrow-encoded points (XYM)
    tab = pa.table({"geometry": ga.as_geoarrow(ga.array(["POINT M (0 1 2)"]))})
    plan = io.plan_geometry_encoding(tab)
    assert plan.encodings == {"geometry": "point"}
    assert plan["geometry"].expected_type == ga.point().with_dimensions(
        ga.Dimensions.XYM
    )


def test_plan_geometry_encoding_wkb_fallback():
    # Non-point geometry type
    tab = pa.table({"geometry": ga.array(["LINESTRING (0 0, 1 1)", None])})
    plan = io.plan_geometry_encoding(tab)
    assert plan.encodings == {"geometry": "WKB"}
    assert "cannot be represented with native point encoding" in plan["geometry"].reason
    assert plan["geometry"].geometry_types == ("LineString",)
    assert plan["geometry"].expected_type == ga.wkb()

    # Mixed geometry types
    tab = pa.table({"geometry": ga.array(["POINT (0 1)", "LINESTRING (0 0, 1 1)"])})
    plan = io.plan_geometry_encoding(tab)
    assert plan.encodings == {"geometry": "WKB"}
    assert plan["geometry"].geometry_types == ("LineString", "Point")
    assert "'LineString', 'Point'" in plan["geometry"].reason

    # Points with mixed dimensions cannot share one native point type
    tab = pa.table({"geometry": ga.array(["POINT (0 1)", "POINT Z (0 1 2)"])})
    plan = io.plan_geometry_encoding(tab)
    assert plan.encodings == {"geometry": "WKB"}
    assert "mixed dimensions" in plan["geometry"].reason
    assert plan["geometry"].geometry_types == ("Point", "Point Z")

    # Multi geometries are not the native point representation
    tab = pa.table({"geometry": ga.array(["MULTIPOINT (0 1, 2 3)"])})
    plan = io.plan_geometry_encoding(tab)
    assert plan.encodings == {"geometry": "WKB"}
    assert plan["geometry"].geometry_types == ("MultiPoint",)

    # No non-null features: native point eligibility cannot be verified
    tab = pa.table(
        {"geometry": ga.wkb().wrap_array(pa.array([None, None], pa.binary()))}
    )
    plan = io.plan_geometry_encoding(tab)
    assert plan.encodings == {"geometry": "WKB"}
    assert "no non-null features" in plan["geometry"].reason
    assert plan["geometry"].geometry_types == ()


def test_plan_geometry_encoding_explicit_columns():
    wkb_point_0_1 = bytes.fromhex("01010000000000000000000000000000000000F03F")
    tab = pa.table(
        {"wkb_col": pa.array([wkb_point_0_1, None]), "value": pa.array([1, 2])}
    )

    # No extension-typed columns: nothing is planned by default
    plan = io.plan_geometry_encoding(tab)
    assert plan.column_names == ()

    # Plain binary columns can be planned explicitly
    plan = io.plan_geometry_encoding(tab, geometry_columns=["wkb_col"])
    assert plan.encodings == {"wkb_col": "point"}
    # A column without GeoArrow metadata gets the GeoParquet default CRS on read
    assert plan["wkb_col"].expected_type.crs == types.OGC_CRS84

    with pytest.raises(ValueError, match="not present in the table"):
        io.plan_geometry_encoding(tab, geometry_columns=["this_column_does_not_exist"])


def test_write_geoparquet_table_with_plan_plain_columns():
    wkb_point_0_1 = bytes.fromhex("01010000000000000000000000000000000000F03F")
    tab = pa.table(
        {"wkb_col": pa.array([wkb_point_0_1, None]), "value": pa.array([1, 2])}
    )
    plan = io.plan_geometry_encoding(tab, geometry_columns=["wkb_col"])

    with tempfile.TemporaryDirectory() as tmpdir:
        temp_pq = os.path.join(tmpdir, "test.parquet")

        # The plan alone declares the geometry columns of the export
        io.write_geoparquet_table(tab, temp_pq, geometry_encoding=plan)
        geo_meta = json.loads(parquet.read_table(temp_pq).schema.metadata[b"geo"])
        assert geo_meta["primary_column"] == "wkb_col"
        assert geo_meta["columns"]["wkb_col"]["encoding"] == "point"

        # ...and explicit geometry_columns that match the plan are also fine
        temp_pq2 = os.path.join(tmpdir, "test2.parquet")
        io.write_geoparquet_table(
            tab, temp_pq2, geometry_columns=["wkb_col"], geometry_encoding=plan
        )

        for path in (temp_pq, temp_pq2):
            tab2 = io.read_geoparquet_table(path)
            assert tab2["wkb_col"].type == plan["wkb_col"].expected_type
            assert tab2["wkb_col"].type.crs == types.OGC_CRS84
            assert tab2["wkb_col"].is_null().to_pylist() == [False, True]
            assert ga.format_wkt(tab2["wkb_col"]).to_pylist() == ["POINT (0 1)", None]
            assert tab2["value"].to_pylist() == [1, 2]


def test_plan_geometry_encoding_crs_and_edges():
    tab = pa.table(
        {
            "with_crs": ga.with_crs(ga.array(["POINT (0 1)"]), types.OGC_CRS84),
            "spherical": ga.with_edge_type(
                ga.array(["LINESTRING (0 0, 1 1)"]), ga.EdgeType.SPHERICAL
            ),
        }
    )
    plan = io.plan_geometry_encoding(tab)
    assert plan["with_crs"].expected_type.crs == types.OGC_CRS84
    assert plan["spherical"].expected_type.edge_type == ga.EdgeType.SPHERICAL


def test_geometry_encoding_plan_object():
    plan = io.GeometryEncodingPlan(
        {
            "a": io.ColumnEncodingPlan("point", "all features are Point"),
            "b": io.ColumnEncodingPlan("WKB", "fallback"),
        }
    )

    assert plan.column_names == ("a", "b")
    assert plan.encodings == {"a": "point", "b": "WKB"}
    assert plan["a"].encoding == "point"
    assert "a" in plan and "c" not in plan
    assert len(plan) == 2
    assert plan.encoding_for("b") == "WKB"

    with pytest.raises(ValueError, match="not covered by this encoding plan"):
        plan.encoding_for("c")

    assert plan == io.GeometryEncodingPlan(
        {
            "a": io.ColumnEncodingPlan("point", "all features are Point"),
            "b": io.ColumnEncodingPlan("WKB", "fallback"),
        }
    )
    assert plan != io.GeometryEncodingPlan(
        {"a": io.ColumnEncodingPlan("point", "all features are Point")}
    )

    plan_repr = repr(plan)
    assert "a: point (all features are Point)" in plan_repr
    assert "b: WKB (fallback)" in plan_repr
    assert repr(io.GeometryEncodingPlan({})) == "GeometryEncodingPlan()"

    # Column plans carry defaults so that hand-built plans are cheap to write
    assert io.ColumnEncodingPlan("WKB") == ("WKB", "", (), None)

    with pytest.raises(ValueError, match="Invalid encoding"):
        io.GeometryEncodingPlan({"geometry": io.ColumnEncodingPlan("NotAnEncoding")})

    with pytest.raises(TypeError, match="not a ColumnEncodingPlan"):
        io.GeometryEncodingPlan({"geometry": "point"})


def test_write_geoparquet_table_with_plan():
    with tempfile.TemporaryDirectory() as tmpdir:
        temp_pq = os.path.join(tmpdir, "test.parquet")
        tab = pa.table(
            {
                "id": pa.array([1, 2, 3]),
                "pt": ga.with_crs(
                    ga.array(["POINT (0 1)", None, "POINT (2 3)"]), types.OGC_CRS84
                ),
                "pt_z": ga.as_wkb(["POINT Z (0 1 2)", "POINT Z (3 4 5)", None]),
                "lines": ga.array(
                    [
                        "LINESTRING (0 0, 1 1)",
                        "LINESTRING (2 2, 3 3)",
                        "LINESTRING (4 4, 5 5)",
                    ]
                ),
                "mixed": ga.array(["POINT (0 1)", "LINESTRING (0 0, 1 1)", None]),
            }
        )

        plan = io.plan_geometry_encoding(tab)
        assert plan.encodings == {
            "pt": "point",
            "pt_z": "point",
            "lines": "WKB",
            "mixed": "WKB",
        }

        io.write_geoparquet_table(tab, temp_pq, geometry_encoding=plan)

        # The file metadata declares exactly the planned encodings
        tab_meta = parquet.read_table(temp_pq)
        geo_meta = json.loads(tab_meta.schema.metadata[b"geo"])
        assert geo_meta["version"] == "1.1.0"
        file_encodings = {
            name: col["encoding"] for name, col in geo_meta["columns"].items()
        }
        assert file_encodings == plan.encodings

        # Read back and verify type identity, CRS, nulls, and values per column
        tab2 = io.read_geoparquet_table(temp_pq)
        for name in plan:
            expected_type = plan[name].expected_type
            assert tab2[name].type == expected_type
            assert tab2[name].type.crs == expected_type.crs
            assert tab2[name].is_null().to_pylist() == tab[name].is_null().to_pylist()
            assert (
                ga.format_wkt(tab2[name]).to_pylist()
                == ga.format_wkt(tab[name]).to_pylist()
            )

        # CRS values survive the round trip per column
        assert tab2["pt"].type.crs == types.OGC_CRS84
        assert tab2["pt_z"].type.crs is None

        # Native-encoded columns read back as native point; fallback as WKB
        assert isinstance(tab2["pt"].type, ga.PointType)
        assert tab2["pt_z"].type == ga.point().with_dimensions(ga.Dimensions.XYZ)
        assert isinstance(tab2["lines"].type, ga.WkbType)
        assert isinstance(tab2["mixed"].type, ga.WkbType)

        # Non-geometry columns are untouched
        assert tab2["id"].to_pylist() == [1, 2, 3]


def test_write_geoparquet_table_plan_errors():
    with tempfile.TemporaryDirectory() as tmpdir:
        temp_pq = os.path.join(tmpdir, "test.parquet")
        tab = pa.table(
            {
                "pt": ga.array(["POINT (0 1)"]),
                "lines": ga.array(["LINESTRING (0 0, 1 1)"]),
            }
        )

        # The plan may not contain columns that are not in the table
        plan = io.GeometryEncodingPlan(
            {
                "pt": io.ColumnEncodingPlan("point"),
                "not_a_column": io.ColumnEncodingPlan("WKB"),
            }
        )
        with pytest.raises(ValueError, match="not present in the table"):
            io.write_geoparquet_table(tab, temp_pq, geometry_encoding=plan)

        # The plan must cover every detected geometry column
        plan = io.GeometryEncodingPlan({"pt": io.ColumnEncodingPlan("point")})
        with pytest.raises(ValueError, match="not covered by this encoding plan"):
            io.write_geoparquet_table(tab, temp_pq, geometry_encoding=plan)

        # ...including columns passed explicitly via geometry_columns
        with pytest.raises(ValueError, match="not covered by this encoding plan"):
            io.write_geoparquet_table(
                tab,
                temp_pq,
                geometry_columns=["pt", "lines"],
                geometry_encoding=plan,
            )


def test_write_geoparquet_table_hand_built_plan():
    with tempfile.TemporaryDirectory() as tmpdir:
        temp_pq = os.path.join(tmpdir, "test.parquet")
        tab = pa.table({"geometry": ga.array(["POINT (0 1)", "POINT (2 3)"])})

        # Force WKB even though the column is eligible for native point
        plan = io.GeometryEncodingPlan(
            {"geometry": io.ColumnEncodingPlan("WKB", reason="forced")}
        )
        io.write_geoparquet_table(tab, temp_pq, geometry_encoding=plan)
        geo_meta = json.loads(parquet.read_table(temp_pq).schema.metadata[b"geo"])
        assert geo_meta["version"] == "1.0.0"
        assert geo_meta["columns"]["geometry"]["encoding"] == "WKB"

        tab2 = io.read_geoparquet_table(temp_pq)
        assert isinstance(tab2["geometry"].type, ga.WkbType)
        assert ga.format_wkt(tab2["geometry"]).to_pylist() == [
            "POINT (0 1)",
            "POINT (2 3)",
        ]
