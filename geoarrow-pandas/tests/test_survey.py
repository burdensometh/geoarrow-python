"""Tests for survey-point consolidation and GeoParquet provenance.

The rules under test are:

* XY geometries are promoted to XYZ with NaN Z (never 0).
* Existing measured Z values are preserved exactly.
* Null geometries remain null features.
* Batches without any Z (and partially-Z batches) are recorded in provenance.
* The geometry column round-trips GeoParquet on a geoarrow extension dtype and
  remains usable for spatial analysis.
"""

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
from shapely import Point

import geoarrow.pyarrow as ga
import geoarrow.pandas as gapd
from geoarrow.pandas import survey
from geoarrow.types import Dimensions


pytest.importorskip("geopandas")


def _z_values(series):
    return [
        None if g is None else g.z
        for g in series.geoarrow.to_geopandas().geometry
    ]


def test_promotion_uses_nan_not_zero_and_keeps_measured_z():
    series, provenance = survey.consolidate_survey_batches(
        {
            "xy_batch": [Point(1, 2), Point(3, 4)],
            "xyz_batch": [Point(5, 6, 7)],
        }
    )

    # Uniform XYZ output on a geoarrow extension dtype
    assert isinstance(series.dtype, gapd.GeoArrowExtensionDtype)
    assert series.dtype.pyarrow_dtype.dimensions == Dimensions.XYZ

    z = _z_values(series)
    assert np.isnan(z[0]) and np.isnan(z[1])
    assert z[2] == 7.0
    # The hard requirement: no missing height was ever turned into zero.
    assert not any(v == 0.0 for v in z if v is not None)

    # XY and XYZ coordinates themselves survive untouched.
    geoms = list(series.geoarrow.to_geopandas().geometry)
    assert geoms[0].x == 1.0 and geoms[0].y == 2.0
    assert geoms[2].x == 5.0 and geoms[2].y == 6.0


def test_null_geometry_stays_null():
    series, _ = survey.consolidate_survey_batches(
        {"with_null": [Point(1, 2), None, Point(3, 4, 5)]}
    )

    assert series.isna().tolist() == [False, True, False]
    geoms = list(series.geoarrow.to_geopandas().geometry)
    assert geoms[1] is None
    z = _z_values(series)
    assert np.isnan(z[0]) and z[2] == 5.0 and z[1] is None


def test_provenance_records_z_missing_and_partial_batches():
    _, provenance = survey.consolidate_survey_batches(
        {
            "all_xy": [Point(0, 0)],
            "all_xyz": [Point(0, 0, 1)],
            "mixed": [Point(0, 0), Point(0, 0, 2)],
            "only_null": [None],
        }
    )

    # Batches that contributed no Z at all (including a null-only batch).
    assert provenance["z_missing_batches"] == ["all_xy", "only_null"]
    # A batch carrying both XY and XYZ rows is flagged as partial, not missing.
    assert provenance["z_partial_batches"] == ["mixed"]
    assert provenance["output_dimensions"] == "XYZ"
    assert "NaN" in provenance["z_missing_rule"]


def test_all_xy_input_is_not_forced_to_xyz():
    # Without any measured Z there is nothing to promote: keep the data XY and
    # record every batch as missing Z.
    series, provenance = survey.consolidate_survey_batches(
        {"b1": [Point(1, 2)], "b2": [Point(3, 4)]}
    )
    assert series.dtype.pyarrow_dtype.dimensions == Dimensions.XY
    assert provenance["z_missing_batches"] == ["b1", "b2"]
    assert provenance["output_dimensions"] == "XY"


def test_wkb_and_wkt_inputs_are_accepted():
    series, provenance = survey.consolidate_survey_batches(
        {
            "wkb_batch": [Point(1, 2).wkb],
            "wkt_batch": ["POINT (3 4 5)"],
        }
    )
    z = _z_values(series)
    assert np.isnan(z[0]) and z[1] == 5.0
    assert provenance["z_missing_batches"] == ["wkb_batch"]


def test_geoparquet_round_trip_keeps_extension_dtype_and_metadata(tmp_path):
    series, provenance = survey.consolidate_survey_batches(
        {
            "xy_batch": [Point(1, 2), None],
            "xyz_batch": [Point(3, 4, 5)],
        }
    )
    df = pd.DataFrame({"id": [0, 1, 2], "geometry": series})
    path = tmp_path / "survey.parquet"
    survey.write_survey_geoparquet(df, path, provenance)

    # The file is a standard GeoParquet: geo metadata is present.
    schema_meta = pa.parquet.read_schema(path).metadata
    assert b"geo" in schema_meta

    back_df, back_prov = survey.read_survey_geoparquet(path)

    # Geometry comes back on a geoarrow extension dtype, not plain binary.
    assert isinstance(back_df["geometry"].dtype, gapd.GeoArrowExtensionDtype)
    assert back_df["id"].tolist() == [0, 1, 2]

    # Spatial analysis works directly on the read-back column.
    bounds = back_df["geometry"].geoarrow.bounds()
    assert bounds.loc[0, "xmin"] == 1.0
    geoseries = back_df["geometry"].geoarrow.to_geopandas()
    assert geoseries[2].z == 5.0 and geoseries[1] is None

    # NaN-not-zero semantics and provenance survived the round trip.
    z = [None if g is None else g.z for g in geoseries.geometry]
    assert np.isnan(z[0]) and z[1] is None and z[2] == 5.0
    assert not any(v == 0.0 for v in z if v is not None)
    assert back_prov["z_missing_batches"] == ["xy_batch"]
    assert back_prov["output_dimensions"] == "XYZ"
    assert "NaN" in back_prov["z_missing_rule"]


def test_read_back_geometry_is_arrow_extension_array(tmp_path):
    series, provenance = survey.consolidate_survey_batches(
        {"b": [Point(1, 2), Point(3, 4, 5)]}
    )
    df = pd.DataFrame({"geometry": series})
    path = tmp_path / "survey.parquet"
    survey.write_survey_geoparquet(df, path, provenance)

    # Even reading with raw pyarrow, geometry is restored to an extension type.
    table = ga.io.read_geoparquet_table(path)
    assert isinstance(table["geometry"].type, ga.GeometryExtensionType)

    back_df, _ = survey.read_survey_geoparquet(path)
    arrow_array = pa.array(back_df["geometry"].array)
    assert isinstance(arrow_array.type, ga.GeometryExtensionType)


def test_native_pyarrow_parquet_path_round_trips(tmp_path):
    # Writing through raw pyarrow preserves the geoarrow extension type itself
    # (no WKB); the read-back column must still be an extension dtype.
    series, provenance = survey.consolidate_survey_batches(
        {"xy": [Point(1, 2)], "xyz": [Point(3, 4, 5)]}
    )
    df = pd.DataFrame({"geometry": series})
    path = tmp_path / "native.parquet"
    pa.parquet.write_table(pa.Table.from_pandas(df, preserve_index=False), path)

    raw = pa.parquet.read_table(path)
    assert isinstance(raw.schema.field("geometry").type, ga.GeometryExtensionType)

    back = pd.read_parquet(path)
    # ArrowDtype-backed geometry series feeds back into the same pipeline.
    re_consolidated, _ = survey.consolidate_survey_batches(
        {"reread": back["geometry"]}
    )
    z = _z_values(re_consolidated)
    assert np.isnan(z[0]) and z[1] == 5.0


def test_z_is_missing_mask(tmp_path):
    series, provenance = survey.consolidate_survey_batches(
        {
            "xy_batch": [Point(1, 2), None],
            "xyz_batch": [Point(3, 4, 5)],
            "mixed": [Point(6, 7), Point(8, 9, 10)],
        }
    )
    mask = survey.z_is_missing(series)
    # Rows 0 and 3 are promoted 2D points; null row 1 is not counted.
    assert mask.tolist() == [True, False, False, True, False]

    df = pd.DataFrame({"geometry": series})
    path = tmp_path / "survey.parquet"
    survey.write_survey_geoparquet(df, path, provenance)
    back_df, _ = survey.read_survey_geoparquet(path)
    # Same result through the WKB-encoded read-back column.
    assert survey.z_is_missing(back_df["geometry"]).tolist() == mask.tolist()


def test_z_is_missing_rejects_xy_series():
    series, _ = survey.consolidate_survey_batches({"b": [Point(1, 2)]})
    with pytest.raises(ValueError, match="no Z dimension"):
        survey.z_is_missing(series)


def test_write_rejects_plain_object_geometry_column(tmp_path):
    df = pd.DataFrame({"geometry": [Point(1, 2)]})
    with pytest.raises(TypeError, match="geoarrow extension dtype"):
        survey.write_survey_geoparquet(
            df, tmp_path / "x.parquet",
            {"z_missing_batches": [], "z_partial_batches": [],
             "z_missing_rule": "", "output_dimensions": "XY", "batch_order": []},
        )


def test_no_zero_z_anywhere_in_storage(tmp_path):
    # Belt-and-braces: inspect the physical Z buffer before and after write.
    series, _ = survey.consolidate_survey_batches(
        {"xy": [Point(1, 2)], "xyz": [Point(3, 4, 9)]}
    )
    xyz = ga.as_geoarrow(series.array._data, ga.point().with_dimensions(Dimensions.XYZ))
    chunks = xyz.chunks if hasattr(xyz, "chunks") else [xyz]
    z_buffer = []
    for chunk in chunks:
        z_buffer.extend(chunk.storage.flatten()[2].to_pylist())
    assert np.isnan(z_buffer[0]) and z_buffer[1] == 9.0
    assert 0.0 not in z_buffer
