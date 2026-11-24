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


def _write_historical_geoparquet(
    tmp_path, item, spec, col_name="geometry", version="1.1.0"
):
    # Write a "historical" GeoParquet file that carries both the Arrow
    # extension type (field-level metadata) and the "geo" schema metadata key.
    # pyarrow preserves the extension type on write/read.
    tab = pa.table([item], names=[col_name])
    geo_meta = {
        "version": version,
        "primary_column": col_name,
        "columns": {col_name: spec},
    }
    metadata = dict(tab.schema.metadata or {})
    metadata["geo"] = json.dumps(geo_meta)
    path = os.path.join(tmp_path, "historical.parquet")
    parquet.write_table(tab.replace_schema_metadata(metadata), path)
    return path


def test_read_geoparquet_strict_consistent_geoarrow(tmp_path):
    # A correctly written file with an extension type + matching "geo" spec
    # reads back into exactly the same geometry type under strict mode.
    item = ga.as_geoarrow(["POINT (0 1)"])
    spec = {"encoding": "point", "geometry_types": ["Point"], "crs": None}
    path = _write_historical_geoparquet(tmp_path, item, spec)

    tab = io.read_geoparquet_table(path, strict=True)
    assert tab["geometry"].type == ga.point()

    # Default (non-strict) behaviour is unchanged
    assert io.read_geoparquet_table(path)["geometry"].type == ga.point()


def test_read_geoparquet_strict_consistent_spherical(tmp_path):
    item = ga.with_edge_type(
        ga.as_geoarrow(["POINT (0 1)"]), ga.EdgeType.SPHERICAL
    )
    spec = {
        "encoding": "point",
        "geometry_types": ["Point"],
        "crs": None,
        "edges": "spherical",
    }
    path = _write_historical_geoparquet(tmp_path, item, spec)

    tab = io.read_geoparquet_table(path, strict=True)
    assert tab["geometry"].type.edge_type == ga.EdgeType.SPHERICAL


def test_read_geoparquet_strict_consistent_wkb_equivalent_crs(tmp_path):
    # A serialized (WKB) extension type restored by the Arrow reader carries a
    # CRS; the same CRS expressed via an equivalent PROJJSON expression in the
    # column spec must not be reported as a conflict.
    restored_crs = types.crs.create("OGC:CRS84").to_json_dict()
    item = ga.as_wkb(["POINT (0 1)"])
    spec = {"encoding": "WKB", "geometry_types": [], "crs": restored_crs}
    path = _write_historical_geoparquet(tmp_path, item, spec, version="1.0.0")

    tab = io.read_geoparquet_table(path, strict=True)
    assert isinstance(tab["geometry"].type, ga.WkbType)

    # The same CRS expressed as a PROJJSON string is also an equivalent
    # expression and must likewise be accepted.
    spec["crs"] = json.dumps(restored_crs)
    path2 = _write_historical_geoparquet(tmp_path, item, spec, version="1.0.0")
    tab2 = io.read_geoparquet_table(path2, strict=True)
    assert isinstance(tab2["geometry"].type, ga.WkbType)


def test_read_geoparquet_strict_crs_conflict_explicit_unknown(tmp_path):
    # The extension type carries a CRS but the spec explicitly says the CRS is
    # unknown ("crs": null): reading it as-is would assign spatial meaning that
    # the file disclaims, so strict mode must fail.
    item = ga.as_wkb(["POINT (0 1)"])
    spec = {"encoding": "WKB", "geometry_types": [], "crs": None}
    path = _write_historical_geoparquet(tmp_path, item, spec, version="1.0.0")

    with pytest.raises(ValueError, match="crs mismatch"):
        io.read_geoparquet_table(path, strict=True)

    # Non-strict reading preserves the extension type (previous behaviour)
    tab = io.read_geoparquet_table(path)
    assert isinstance(tab["geometry"].type, ga.WkbType)


def test_read_geoparquet_strict_crs_conflict_extension_unset(tmp_path):
    # Extension type with no CRS, but the spec omits "crs" which per the
    # GeoParquet specification implicitly means OGC:CRS84.
    item = ga.as_geoarrow(["POINT (0 1)"])
    spec = {"encoding": "point", "geometry_types": ["Point"]}
    path = _write_historical_geoparquet(tmp_path, item, spec)

    with pytest.raises(ValueError, match="crs mismatch"):
        io.read_geoparquet_table(path, strict=True)


def test_read_geoparquet_strict_crs_conflict_different_crs(tmp_path):
    item = ga.as_geoarrow(["POINT (0 1)"])
    spec = {
        "encoding": "point",
        "geometry_types": ["Point"],
        "crs": types.OGC_CRS84.to_json_dict(),
    }
    path = _write_historical_geoparquet(tmp_path, item, spec)

    with pytest.raises(ValueError, match="crs mismatch"):
        io.read_geoparquet_table(path, strict=True)


def test_read_geoparquet_strict_edges_conflict(tmp_path):
    item = ga.as_geoarrow(["POINT (0 1)"])
    spec = {
        "encoding": "point",
        "geometry_types": ["Point"],
        "crs": None,
        "edges": "spherical",
    }
    path = _write_historical_geoparquet(tmp_path, item, spec)

    with pytest.raises(ValueError, match="edges mismatch"):
        io.read_geoparquet_table(path, strict=True)

    # Non-strict reading keeps the (planar) extension type
    tab = io.read_geoparquet_table(path)
    assert tab["geometry"].type.edge_type == ga.EdgeType.PLANAR


def test_read_geoparquet_strict_encoding_conflict(tmp_path):
    item = ga.as_geoarrow(["POINT (0 1)"])
    spec = {"encoding": "WKB", "crs": None}
    path = _write_historical_geoparquet(tmp_path, item, spec, version="1.0.0")

    with pytest.raises(ValueError, match="encoding mismatch"):
        io.read_geoparquet_table(path, strict=True)


def test_read_geoparquet_strict_reports_all_conflicts(tmp_path):
    # Multiple distinct conflicts in the same column are reported together in
    # a single error.
    item = ga.as_geoarrow(["POINT (0 1)"])
    spec = {
        "encoding": "WKB",
        "crs": types.OGC_CRS84.to_json_dict(),
        "edges": "spherical",
    }
    path = _write_historical_geoparquet(tmp_path, item, spec, version="1.0.0")

    with pytest.raises(ValueError) as excinfo:
        io.read_geoparquet_table(path, strict=True)
    message = str(excinfo.value)
    assert "encoding mismatch" in message
    assert "crs mismatch" in message
    assert "edges mismatch" in message


def test_read_geoparquet_strict_aggregates_columns(tmp_path):
    # Conflicts in several columns are aggregated into one error that names
    # every affected column.
    tab = pa.table(
        {
            "g1": ga.as_geoarrow(["POINT (0 1)"]),
            "g2": ga.as_geoarrow(["POINT (0 1)"]),
        }
    )
    geo_meta = {
        "version": "1.1.0",
        "primary_column": "g1",
        "columns": {
            "g1": {"encoding": "WKB", "crs": None},
            "g2": {"encoding": "linestring", "crs": None},
        },
    }
    metadata = dict(tab.schema.metadata or {})
    metadata["geo"] = json.dumps(geo_meta)
    path = os.path.join(tmp_path, "two_columns.parquet")
    parquet.write_table(tab.replace_schema_metadata(metadata), path)

    with pytest.raises(ValueError) as excinfo:
        io.read_geoparquet_table(path, strict=True)
    message = str(excinfo.value)
    assert "column 'g1'" in message
    assert "column 'g2'" in message


def test_read_geoparquet_strict_invalid_edges_value(tmp_path):
    item = ga.as_geoarrow(["POINT (0 1)"])
    spec = {
        "encoding": "point",
        "geometry_types": ["Point"],
        "crs": None,
        "edges": "not-a-real-edge-type",
    }
    path = _write_historical_geoparquet(tmp_path, item, spec)

    with pytest.raises(ValueError, match="invalid GeoParquet column edges value"):
        io.read_geoparquet_table(path, strict=True)


def test_read_geoparquet_strict_ignored_without_geo_metadata(tmp_path):
    # A regular Parquet file has no authoritative column specification, so the
    # guessed geometry column is not strict-checked.
    path = os.path.join(tmp_path, "plain.parquet")
    parquet.write_table(
        pa.table([pa.array(["POINT (0 1)"])], names=["geometry"]), path
    )
    tab = io.read_geoparquet_table(path, strict=True)
    assert isinstance(tab["geometry"].type, ga.GeometryExtensionType)


def test_read_geoparquet_strict_skips_unread_columns(tmp_path):
    item = ga.as_geoarrow(["POINT (0 1)"])
    spec = {"encoding": "WKB", "crs": None}
    path = _write_historical_geoparquet(tmp_path, item, spec, version="1.0.0")

    # The conflicting column is not read; its specification must be ignored.
    tab = io.read_geoparquet_table(path, strict=True, columns=[])
    assert tab.schema.names == []


def test_read_geoparquet_strict_non_extension_file(tmp_path):
    # A normal library-written WKB file (storage type, no extension type)
    # passes strict mode and reads as before.
    path = os.path.join(tmp_path, "normal.parquet")
    io.write_geoparquet_table(
        pa.table([ga.array(["POINT (0 1)"])], names=["geometry"]),
        path,
        geometry_encoding="WKB",
    )
    tab = io.read_geoparquet_table(path, strict=True)
    assert isinstance(tab["geometry"].type, ga.WkbType)


def test_geoparquet_extension_type_conflicts():
    # Unit-level coverage of the per-column conflict reporter, including the
    # encoding labels for serialized types.
    wkb_spec = {"encoding": "WKB", "crs": None}
    assert io._geoparquet_extension_type_conflicts(ga.wkb(), wkb_spec) == []

    wkt_spec = {"encoding": "WKT", "crs": None}
    assert io._geoparquet_extension_type_conflicts(ga.wkt(), wkt_spec) == []

    point_spec = {"encoding": "point", "crs": None}
    assert io._geoparquet_extension_type_conflicts(ga.point(), point_spec) == []

    # Wrong encoding label is detected for both native and serialized types
    assert any(
        "encoding mismatch" in c
        for c in io._geoparquet_extension_type_conflicts(ga.point(), wkb_spec)
    )
    assert any(
        "encoding mismatch" in c
        for c in io._geoparquet_extension_type_conflicts(ga.wkb(), point_spec)
    )

    # A missing "encoding" key simply disables the encoding check
    conflicts = io._geoparquet_extension_type_conflicts(
        ga.point(), {"crs": None}
    )
    assert not any("encoding mismatch" in c for c in conflicts)


def test_geoparquet_table_to_geoarrow_strict():
    item = pa.chunked_array([ga.as_geoarrow(["POINT (0 1)"])])
    tab = pa.table([item], names=["geometry"])

    # A consistent specification passes and preserves the extension type
    out = io._geoparquet_table_to_geoarrow(
        tab, {"geometry": {"encoding": "point", "crs": None}}, strict=True
    )
    assert out["geometry"].type == ga.point()

    # A conflicting specification raises before any columns are rewritten
    with pytest.raises(ValueError, match="encoding mismatch"):
        io._geoparquet_table_to_geoarrow(
            tab, {"geometry": {"encoding": "WKB", "crs": None}}, strict=True
        )

    # Without strict, the extension type is accepted regardless of the spec
    out = io._geoparquet_table_to_geoarrow(
        tab, {"geometry": {"encoding": "WKB", "crs": None}}
    )
    assert out["geometry"].type == ga.point()
