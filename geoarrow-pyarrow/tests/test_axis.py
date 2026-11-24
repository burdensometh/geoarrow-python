import math
import struct

import geoarrow.pyarrow as ga
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pytest
from geoarrow import types
from geoarrow.pyarrow import _axis
from geoarrow.types import crs as crs_mod

EXT = (-80.0, 30.0, -70.0, 45.0)
EXT_LON_TUPLE = "POINT (-75 40)"


def statuses(status):
    return pc.struct_field(status, "status").to_pylist()


def wkts(obj):
    return ga.format_wkt(obj).to_pylist()


def _point_wkb(x, y):
    return b"\x01\x01\x00\x00\x00" + struct.pack("<dd", x, y)


# ---------------------------------------------------------------------------
# Enum constants and status type
# ---------------------------------------------------------------------------


def test_axis_order_status_constants():
    assert ga.AxisOrderStatus.AS_IS == "as_is"
    assert ga.AxisOrderStatus.SWAP == "swap"
    assert ga.AxisOrderStatus.AMBIGUOUS == "ambiguous"
    assert ga.AxisOrderStatus.OUTSIDE == "outside"
    assert ga.AxisOrderStatus.EMPTY == "empty"

    assert ga.AxisOrderVerdict.AS_IS == "as_is"
    assert ga.AxisOrderVerdict.SWAP == "swap"
    assert ga.AxisOrderVerdict.PENDING == "pending"


def test_axis_order_status_type():
    type_ = ga._axis.axis_order_status_type()
    assert type_.names == [
        "status",
        "original_in_extent",
        "swapped_in_extent",
        "pending",
    ]


# ---------------------------------------------------------------------------
# Extent normalization
# ---------------------------------------------------------------------------


def test_normalize_extent_sequence():
    assert _axis._normalize_extent((-80, 30, -70, 45)) == EXT
    assert _axis._normalize_extent([-80, 30, -70, 45]) == EXT
    assert _axis._normalize_extent(np.array([-80, 30, -70, 45])) == EXT

    extent = pa.array([-80.0, 30.0, -70.0, 45.0])
    assert _axis._normalize_extent(extent) == EXT


def test_normalize_extent_mapping():
    extent = {"xmin": -80, "ymin": 30, "xmax": -70, "ymax": 45}
    assert _axis._normalize_extent(extent) == EXT

    with pytest.raises(ValueError, match="missing"):
        _axis._normalize_extent({"xmin": -80})


def test_normalize_extent_invalid():
    with pytest.raises(ValueError, match="xmin <= xmax"):
        _axis._normalize_extent((-70, 30, -80, 45))

    with pytest.raises(ValueError, match="ymin <= ymax"):
        _axis._normalize_extent((-80, 45, -70, 30))

    with pytest.raises(ValueError, match="finite"):
        _axis._normalize_extent((float("nan"), 0, 1, 1))

    with pytest.raises(ValueError, match="finite"):
        _axis._normalize_extent((float("-inf"), 0, 1, 1))

    with pytest.raises(TypeError, match="sequence"):
        _axis._normalize_extent((0, 1, 2))

    with pytest.raises(TypeError, match="sequence"):
        _axis._normalize_extent("oops")

    with pytest.raises(TypeError, match="numeric"):
        _axis._normalize_extent(("a", 0, 1, 1))


# ---------------------------------------------------------------------------
# check_axis_order classification
# ---------------------------------------------------------------------------


def test_check_axis_order_as_is_and_swap():
    geometry = ga.as_geoarrow(["POINT (-75 40)", "POINT (40 -75)"])
    status = ga.check_axis_order(geometry, EXT)
    assert statuses(status) == ["as_is", "swap"]
    assert status.type == ga._axis.axis_order_status_type()

    record0 = status[0].as_py()
    assert record0 == {
        "status": "as_is",
        "original_in_extent": True,
        "swapped_in_extent": False,
        "pending": False,
    }

    record1 = status[1].as_py()
    assert record1 == {
        "status": "swap",
        "original_in_extent": False,
        "swapped_in_extent": True,
        "pending": False,
    }


def test_check_axis_order_ambiguous():
    # Both (40, 40) and its swap are inside this symmetric extent.
    geometry = ga.as_geoarrow(["POINT (40 40)"])
    status = ga.check_axis_order(geometry, (-50, -50, 50, 50))
    assert statuses(status) == ["ambiguous"]
    record = status[0].as_py()
    assert record["original_in_extent"] is True
    assert record["swapped_in_extent"] is True
    assert record["pending"] is True


def test_check_axis_order_outside():
    geometry = ga.as_geoarrow(["POINT (0 0)"])
    status = ga.check_axis_order(geometry, EXT)
    assert statuses(status) == ["outside"]
    assert status[0].as_py()["pending"] is True


def test_check_axis_order_empty_and_null():
    geometry = ga.as_geoarrow(pa.array(["POINT EMPTY", None, "POINT (-75 40)"], pa.utf8()))
    status = ga.check_axis_order(geometry, EXT)
    assert statuses(status) == ["empty", "empty", "as_is"]


def test_check_axis_order_null_geometry():
    # Null top-level values are "empty" and are not treated as evidence.
    storage = pa.StructArray.from_arrays(
        [pa.array([1.0, 2.0]), pa.array([3.0, 4.0])],
        names=["x", "y"],
        mask=pa.array([True, False]),
    )
    geometry = ga.point().wrap_array(storage)
    status = ga.check_axis_order(geometry, (1, 2.5, 2.5, 10))
    assert statuses(status) == ["empty", "as_is"]


def test_check_axis_order_non_finite_bounds():
    geometry = ga.as_geoarrow(["POINT (nan nan)", "POINT (inf 2)", "POINT (-75 40)"])
    status = ga.check_axis_order(geometry, EXT)
    assert statuses(status) == ["empty", "empty", "as_is"]


def test_check_axis_order_chunked():
    geometry = pa.chunked_array(
        [
            ga.as_geoarrow(["POINT (-75 40)"]),
            ga.as_geoarrow(["POINT (40 -75)", "POINT (0 0)"]),
        ]
    )
    status = ga.check_axis_order(geometry, EXT)
    assert isinstance(status, pa.ChunkedArray)
    assert statuses(status) == ["as_is", "swap", "outside"]


def test_check_axis_order_sliced():
    geometry = ga.as_geoarrow(["POINT (0 0)", "POINT (40 -75)", "POINT (-75 40)"])
    status = ga.check_axis_order(geometry[1:], EXT)
    assert statuses(status) == ["swap", "as_is"]


def test_check_axis_order_all_geometry_types():
    geometries = {
        "POINT (40 -75)": "POINT (-75 40)",
        "LINESTRING (40 -75, 41 -76)": "LINESTRING (-75 40, -76 41)",
        "POLYGON ((40 -75, 41 -75, 41 -76, 40 -75))": (
            "POLYGON ((-75 40, -75 41, -76 41, -75 40))"
        ),
        "MULTIPOINT (40 -75, 41 -76)": "MULTIPOINT (-75 40, -76 41)",
        "MULTILINESTRING ((40 -75, 41 -76))": (
            "MULTILINESTRING ((-75 40, -76 41))"
        ),
        "MULTIPOLYGON (((40 -75, 41 -75, 41 -76, 40 -75)))": (
            "MULTIPOLYGON (((-75 40, -75 41, -76 41, -75 40)))"
        ),
    }

    for wkt, _ in geometries.items():
        geometry = ga.as_geoarrow([wkt])
        status = ga.check_axis_order(geometry, EXT)
        assert statuses(status) == ["swap"], wkt


def test_check_axis_order_extra_dimensions():
    # z and m do not affect the x/y extent test.
    geometry = ga.as_geoarrow(["POINT Z (40 -75 100)", "POINT M (40 -75 7)"])
    status = ga.check_axis_order(geometry, EXT)
    assert statuses(status) == ["swap", "swap"]


def test_check_axis_order_wkb_and_wkt():
    for geometry in [ga.as_wkb(["POINT (40 -75)"]), ga.as_wkt(["POINT (40 -75)"])]:
        status = ga.check_axis_order(geometry, EXT)
        assert statuses(status) == ["swap"]


def test_check_axis_order_crs_validation():
    geometry = ga.with_crs(ga.as_geoarrow(["POINT (40 -75)"]), types.OGC_CRS84)
    assert statuses(ga.check_axis_order(geometry, EXT)) == ["swap"]
    assert statuses(
        ga.check_axis_order(geometry, EXT, expected_crs=types.OGC_CRS84)
    ) == ["swap"]

    with pytest.raises(ValueError, match="declare CRS"):
        ga.check_axis_order(geometry, EXT, expected_crs="EPSG:4326")

    with pytest.raises(ValueError, match="declare a CRS"):
        ga.check_axis_order(ga.as_geoarrow(["POINT (0 0)"]), EXT, require_crs=True)


def test_check_axis_order_table():
    geometry = ga.as_geoarrow(["POINT (40 -75)", "POINT (-75 40)"])
    table = pa.table({"id": pa.array([1, 2]), "geometry": geometry})
    status = ga.check_axis_order(table, EXT)
    assert statuses(status) == ["swap", "as_is"]

    with pytest.raises(ValueError, match="not found"):
        ga.check_axis_order(table, EXT, geometry_column="nope")


def test_table_geometry_column_selection():
    geometry = ga.as_geoarrow(["POINT (40 -75)"])
    other = ga.as_geoarrow(["POINT (0 0)"])

    table = pa.table({"id": [1], "geom": geometry})
    assert statuses(ga.check_axis_order(table, EXT)) == ["swap"]

    table_multi = pa.table({"id": [1], "geom": geometry, "other": other})
    with pytest.raises(ValueError, match="more than one geometry column"):
        ga.check_axis_order(table_multi, EXT)
    assert statuses(
        ga.check_axis_order(table_multi, EXT, geometry_column="geom")
    ) == ["swap"]

    table_none = pa.table({"id": [1]})
    with pytest.raises(ValueError, match="no geometry column"):
        ga.check_axis_order(table_none, EXT)


def test_table_geometry_column_prefers_geometry_then_geography():
    geometry = ga.as_geoarrow(["POINT (40 -75)"])
    table = pa.table({"id": [1], "geography": geometry})
    assert statuses(ga.check_axis_order(table, EXT)) == ["swap"]


# ---------------------------------------------------------------------------
# axis_order_evidence
# ---------------------------------------------------------------------------


def test_evidence_as_is():
    evidence = ga.axis_order_evidence(
        ga.as_geoarrow(["POINT (-75 40)", "POINT (-72 35)"]), EXT
    )
    assert evidence.verdict == ga.AxisOrderVerdict.AS_IS
    assert evidence.pending_reasons == ()
    assert not evidence.is_pending
    assert evidence.n_as_is == 2
    assert evidence.n_total == 2
    assert evidence.extent == EXT


def test_evidence_swap():
    evidence = ga.axis_order_evidence(
        ga.as_geoarrow(["POINT (40 -75)", "POINT (35 -72)"]), EXT
    )
    assert evidence.verdict == ga.AxisOrderVerdict.SWAP
    assert evidence.n_swap == 2
    assert evidence.pending_reasons == ()


def test_evidence_inconsistent():
    evidence = ga.axis_order_evidence(
        ga.as_geoarrow(["POINT (-75 40)", "POINT (40 -75)"]), EXT
    )
    assert evidence.verdict == ga.AxisOrderVerdict.PENDING
    assert evidence.pending_reasons == ("inconsistent",)
    assert evidence.n_as_is == 1
    assert evidence.n_swap == 1


def test_evidence_ambiguous():
    evidence = ga.axis_order_evidence(
        ga.as_geoarrow(["POINT (40 40)"]), (-50, -50, 50, 50)
    )
    assert evidence.verdict == ga.AxisOrderVerdict.PENDING
    assert "ambiguous" in evidence.pending_reasons
    assert evidence.n_ambiguous == 1
    assert evidence.n_pending == 1


def test_evidence_outside():
    evidence = ga.axis_order_evidence(ga.as_geoarrow(["POINT (0 0)"]), EXT)
    assert evidence.verdict == ga.AxisOrderVerdict.PENDING
    assert "outside" in evidence.pending_reasons
    assert evidence.n_outside == 1


def test_evidence_no_evaluable_features():
    empty = ga.wkt().wrap_array(pa.array([], pa.utf8()))
    evidence = ga.axis_order_evidence(empty, EXT)
    assert evidence.verdict == ga.AxisOrderVerdict.PENDING
    assert evidence.pending_reasons == ("no_evaluable_features",)
    assert evidence.n_total == 0

    all_empty = ga.as_geoarrow(
        pa.array(["POINT EMPTY", None], pa.utf8()), ga.point()
    )
    evidence = ga.axis_order_evidence(all_empty, EXT)
    assert evidence.verdict == ga.AxisOrderVerdict.PENDING
    assert evidence.pending_reasons == ("no_evaluable_features",)
    assert evidence.n_empty == 2


def test_evidence_crs_axis_order():
    geometry = ga.with_crs(ga.as_geoarrow(["POINT (40 -75)"]), types.OGC_CRS84)
    evidence = ga.axis_order_evidence(geometry, EXT)
    assert evidence.crs == types.OGC_CRS84
    assert evidence.crs_axis_order == ("Lon", "Lat")


# ---------------------------------------------------------------------------
# swap_axes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "wkt_input,wkt_expected",
    [
        ("POINT (1 2)", "POINT (2 1)"),
        ("POINT Z (1 2 3)", "POINT Z (2 1 3)"),
        ("POINT M (1 2 4)", "POINT M (2 1 4)"),
        ("POINT ZM (1 2 3 4)", "POINT ZM (2 1 3 4)"),
        ("LINESTRING (0 1, 2 3)", "LINESTRING (1 0, 3 2)"),
        ("POLYGON ((0 0, 1 0, 1 1, 0 0))", "POLYGON ((0 0, 0 1, 1 1, 0 0))"),
        ("MULTIPOINT (0 1, 2 3)", "MULTIPOINT (1 0, 3 2)"),
        (
            "MULTILINESTRING ((0 1, 2 3), (4 5, 6 7))",
            "MULTILINESTRING ((1 0, 3 2), (5 4, 7 6))",
        ),
        (
            "MULTIPOLYGON (((0 0, 1 0, 1 1, 0 0)))",
            "MULTIPOLYGON (((0 0, 0 1, 1 1, 0 0)))",
        ),
    ],
)
def test_swap_axes_native(wkt_input, wkt_expected):
    geometry = ga.as_geoarrow([wkt_input])
    assert wkts(ga.swap_axes(geometry)) == [wkt_expected]
    assert ga.swap_axes(geometry).type == geometry.type


def test_swap_axes_preserves_null():
    geometry = ga.as_geoarrow(pa.array(["POINT (1 2)", None], pa.utf8()))
    swapped = ga.swap_axes(geometry)
    assert wkts(swapped) == ["POINT (2 1)", None]
    assert swapped.is_null().to_pylist() == [False, True]


def test_swap_axes_interleaved():
    geometry = ga.as_geoarrow(
        ["LINESTRING (0 1, 2 3)", "LINESTRING EMPTY"],
        ga.linestring().with_coord_type(ga.CoordType.INTERLEAVED),
    )
    swapped = ga.swap_axes(geometry)
    assert swapped.type == geometry.type
    assert wkts(swapped) == ["LINESTRING (1 0, 3 2)", "LINESTRING EMPTY"]


def test_swap_axes_empty_geometries():
    point_empty = ga.point().wrap_array(
        pa.StructArray.from_arrays(
            [pa.array([float("nan")]), pa.array([float("nan")])], names=["x", "y"]
        )
    )
    assert wkts(ga.swap_axes(point_empty)) == ["POINT EMPTY"]

    line_empty = ga.as_geoarrow(
        pa.array(["LINESTRING EMPTY", "LINESTRING (0 1, 2 3)"], pa.utf8())
    )
    assert wkts(ga.swap_axes(line_empty)) == ["LINESTRING EMPTY", "LINESTRING (1 0, 3 2)"]


def test_swap_axes_wkb_and_wkt():
    assert wkts(ga.swap_axes(ga.as_wkb(["POINT (1 2)"]))) == ["POINT (2 1)"]
    assert wkts(ga.swap_axes(ga.as_wkt(["POINT (1 2)"]))) == ["POINT (2 1)"]
    assert wkts(ga.swap_axes(ga.as_wkb(["POINT (1 2)", None]))) == ["POINT (2 1)", None]


def test_swap_axes_wide_storage():
    wkb_bytes = _point_wkb(1, 2)

    large_wkb = ga.large_wkb().wrap_array(pa.array([wkb_bytes], pa.large_binary()))
    assert wkts(ga.swap_axes(large_wkb)) == ["POINT (2 1)"]
    assert ga.swap_axes(large_wkb).type.storage_type == pa.large_binary()

    wkb_view = ga.wkb_view().wrap_array(pa.array([wkb_bytes], pa.binary_view()))
    assert wkts(ga.swap_axes(wkb_view)) == ["POINT (2 1)"]
    assert ga.swap_axes(wkb_view).type.storage_type == pa.binary_view()

    large_wkt = ga.large_wkt().wrap_array(pa.array(["POINT (1 2)"], pa.large_utf8()))
    assert wkts(ga.swap_axes(large_wkt)) == ["POINT (2 1)"]
    assert ga.swap_axes(large_wkt).type.storage_type == pa.large_utf8()

    wkt_view = ga.wkt_view().wrap_array(pa.array(["POINT (1 2)"], pa.string_view()))
    assert wkts(ga.swap_axes(wkt_view)) == ["POINT (2 1)"]
    assert ga.swap_axes(wkt_view).type.storage_type == pa.string_view()


def test_swap_axes_ewkb_srid():
    # Extended WKB with an SRID (flag 0x20000000); the SRID must be preserved.
    ewkb = (
        struct.pack("<BII", 1, 0x20000001, 4326) + struct.pack("<dd", 11.0, 22.0)
    )
    swapped = ga.swap_axes(ga.array([ewkb]))
    swapped_bytes = swapped.storage[0].as_py()
    assert struct.unpack("<I", swapped_bytes[1:5])[0] == 0x20000001
    assert struct.unpack("<I", swapped_bytes[5:9])[0] == 4326
    assert struct.unpack("<dd", swapped_bytes[9:25]) == (22.0, 11.0)
    assert wkts(swapped) == ["POINT (22 11)"]


def test_swap_axes_geometrycollection_wkb():
    geometry = ga.as_wkb(
        ["GEOMETRYCOLLECTION (POINT (0 1), LINESTRING (2 3, 4 5))"]
    )
    assert wkts(ga.swap_axes(geometry)) == [
        "GEOMETRYCOLLECTION (POINT (1 0), LINESTRING (3 2, 5 4))"
    ]


def test_swap_axes_chunked():
    geometry = pa.chunked_array(
        [ga.as_geoarrow(["POINT (1 2)"]), ga.as_geoarrow(["POINT (3 4)"])]
    )
    swapped = ga.swap_axes(geometry)
    assert isinstance(swapped, pa.ChunkedArray)
    assert swapped.type == geometry.type
    assert wkts(swapped) == ["POINT (2 1)", "POINT (4 3)"]


def test_swap_axes_sliced():
    geometry = ga.as_geoarrow(["POINT (1 2)", "POINT (3 4)", "POINT (5 6)"])
    assert wkts(ga.swap_axes(geometry[1:])) == ["POINT (4 3)", "POINT (6 5)"]

    linestrings = ga.as_geoarrow(
        [
            "LINESTRING (0 1, 2 3)",
            "LINESTRING (4 5, 6 7)",
            "LINESTRING (8 9, 10 11)",
        ]
    )
    assert wkts(ga.swap_axes(linestrings[1:])) == [
        "LINESTRING (5 4, 7 6)",
        "LINESTRING (9 8, 11 10)",
    ]


def test_swap_axes_preserves_crs():
    geometry = ga.with_crs(
        ga.as_geoarrow(["POINT (1 2)", "POINT (3 4)"]), types.OGC_CRS84
    )
    swapped = ga.swap_axes(geometry)
    assert swapped.type.crs == types.OGC_CRS84
    assert swapped.type == geometry.type

    wkb = ga.with_crs(ga.as_wkb(["POINT (1 2)"]), types.OGC_CRS84)
    assert ga.swap_axes(wkb).type.crs == types.OGC_CRS84


def test_swap_axes_box():
    geometry = ga.as_geoarrow(["LINESTRING (0 1, 2 3)"])
    box = ga.box(geometry)
    swapped = ga.swap_axes(box)
    assert swapped.to_pylist() == [
        {"xmin": 1.0, "ymin": 0.0, "xmax": 3.0, "ymax": 2.0}
    ]


def test_swap_axes_box_with_z():
    from geoarrow.pyarrow import _type  # noqa: F401

    box_type = ga.extension_type(
        types.type_spec(
            types.Encoding.GEOARROW,
            geometry_type=types.GeometryType.BOX,
            dimensions=types.Dimensions.XYZ,
        )
    )
    storage = pa.StructArray.from_arrays(
        [
            pa.array([0.0]),
            pa.array([1.0]),
            pa.array([2.0]),
            pa.array([3.0]),
            pa.array([4.0]),
            pa.array([5.0]),
        ],
        fields=list(box_type.storage_type),
    )
    box = box_type.wrap_array(storage)
    assert ga.swap_axes(box).to_pylist() == [
        {
            "xmin": 1.0,
            "ymin": 0.0,
            "zmin": 2.0,
            "xmax": 4.0,
            "ymax": 3.0,
            "zmax": 5.0,
        }
    ]


def test_swap_axes_invalid_wkb():
    with pytest.raises(ValueError, match="byte order"):
        _axis._swap_wkb(b"\x05\x01\x00\x00\x00")


def test_swap_axes_unsupported_storage():
    storage = pa.array([1, 2], pa.int64())
    with pytest.raises(TypeError, match="storage type"):
        _axis._swap_storage(storage)


# ---------------------------------------------------------------------------
# fix_axis_order
# ---------------------------------------------------------------------------


def test_fix_axis_order_default_is_per_feature():
    # Most coordinates correct, one reversed: only the reversed one changes.
    geometry = ga.as_geoarrow(["POINT (-75 40)", "POINT (40 -75)"])
    result = ga.fix_axis_order(geometry, EXT)
    assert result.changed is True
    assert result.pending is False
    assert result.evidence.verdict == ga.AxisOrderVerdict.PENDING
    assert wkts(result.data) == ["POINT (-75 40)", "POINT (-75 40)"]


def test_fix_axis_order_batch_swap():
    geometry = ga.as_geoarrow(["POINT (40 -75)", "POINT (35 -72)"])
    result = ga.fix_axis_order(geometry, EXT, mode="batch")
    assert result.changed is True
    assert result.pending is False
    assert result.evidence.verdict == ga.AxisOrderVerdict.SWAP
    assert wkts(result.data) == ["POINT (-75 40)", "POINT (-72 35)"]
    # the input is untouched
    assert wkts(geometry) == ["POINT (40 -75)", "POINT (35 -72)"]


def test_fix_axis_order_batch_as_is():
    geometry = ga.as_geoarrow(["POINT (-75 40)", "POINT (-72 35)"])
    result = ga.fix_axis_order(geometry, EXT, mode="batch")
    assert result.changed is False
    assert result.pending is False
    assert result.evidence.verdict == ga.AxisOrderVerdict.AS_IS
    assert result.data is geometry


def test_fix_axis_order_batch_pending_does_not_change():
    geometry = ga.as_geoarrow(["POINT (-75 40)", "POINT (40 -75)"])
    result = ga.fix_axis_order(geometry, EXT, mode="batch")
    assert result.changed is False
    assert result.pending is True
    assert result.evidence.verdict == ga.AxisOrderVerdict.PENDING
    assert result.data is geometry


def test_fix_axis_order_batch_ambiguous_does_not_change():
    geometry = ga.as_geoarrow(["POINT (40 40)"])
    result = ga.fix_axis_order(geometry, (-50, -50, 50, 50), mode="batch")
    assert result.changed is False
    assert result.pending is True


def test_fix_axis_order_per_feature():
    geometry = ga.as_geoarrow(
        ["POINT (40 -75)", "POINT (0 0)", "POINT (-75 40)", None, "POINT EMPTY"]
    )
    result = ga.fix_axis_order(geometry, EXT, mode="per_feature")
    assert result.changed is True
    assert result.pending is True
    assert wkts(result.data) == [
        "POINT (-75 40)",
        "POINT (0 0)",
        "POINT (-75 40)",
        None,
        "POINT EMPTY",
    ]


def test_fix_axis_order_per_feature_all_resolved():
    geometry = ga.as_geoarrow(["POINT (40 -75)", "POINT (-75 40)"])
    result = ga.fix_axis_order(geometry, EXT, mode="per_feature")
    assert result.changed is True
    assert result.pending is False
    assert wkts(result.data) == ["POINT (-75 40)", "POINT (-75 40)"]


def test_fix_axis_order_per_feature_ambiguous_untouched():
    # extent (0, 5, 10, 15): (8, 2) only fits after a swap, (8, 8) fits both.
    geometry = ga.as_geoarrow(["POINT (8 8)", "POINT (8 2)"])
    result = ga.fix_axis_order(geometry, (0, 5, 10, 15), mode="per_feature")
    assert result.pending is True
    assert wkts(result.data) == ["POINT (8 8)", "POINT (2 8)"]
    assert statuses(result.status) == ["ambiguous", "swap"]


def test_fix_axis_order_per_feature_chunked():
    geometry = pa.chunked_array(
        [
            ga.as_geoarrow(["POINT (40 -75)"]),
            ga.as_geoarrow(["POINT (0 0)", "POINT (-75 40)"]),
        ]
    )
    result = ga.fix_axis_order(geometry, EXT, mode="per_feature")
    assert isinstance(result.data, pa.ChunkedArray)
    assert wkts(result.data) == [
        "POINT (-75 40)",
        "POINT (0 0)",
        "POINT (-75 40)",
    ]


def test_fix_axis_order_wkb_wkt():
    for geometry in [ga.as_wkb(["POINT (40 -75)", "POINT (35 -72)"]),
                     ga.as_wkt(["POINT (40 -75)", "POINT (35 -72)"])]:
        result = ga.fix_axis_order(geometry, EXT)
        assert result.changed is True
        assert wkts(result.data) == ["POINT (-75 40)", "POINT (-72 35)"]


def test_fix_axis_order_wide_storage():
    wkb = ga.large_wkb().wrap_array(
        pa.array([_point_wkb(40, -75), _point_wkb(35, -72)], pa.large_binary())
    )
    result = ga.fix_axis_order(wkb, EXT)
    assert result.changed is True
    assert wkts(result.data) == ["POINT (-75 40)", "POINT (-72 35)"]
    assert result.data.type.storage_type == pa.large_binary()


def test_fix_axis_order_view_storage_per_feature():
    # binary/string views have no if_else kernel; the selection widens and
    # narrows back, preserving the view storage type.
    wkb_view = ga.wkb_view().wrap_array(
        pa.array([_point_wkb(40, -75), _point_wkb(-75, 40)], pa.binary_view())
    )
    result = ga.fix_axis_order(wkb_view, EXT)
    assert wkts(result.data) == ["POINT (-75 40)", "POINT (-75 40)"]
    assert result.data.type.storage_type == pa.binary_view()

    wkt_view = ga.wkt_view().wrap_array(
        pa.array(["POINT (40 -75)", "POINT (-75 40)"], pa.string_view())
    )
    result = ga.fix_axis_order(wkt_view, EXT)
    assert wkts(result.data) == ["POINT (-75 40)", "POINT (-75 40)"]
    assert result.data.type.storage_type == pa.string_view()


def test_fix_axis_order_invalid_mode():
    geometry = ga.as_geoarrow(["POINT (40 -75)"])
    with pytest.raises(ValueError, match="Expected mode"):
        ga.fix_axis_order(geometry, EXT, mode="nope")


def test_fix_axis_order_no_evaluable_features():
    empty = ga.wkt().wrap_array(pa.array([], pa.utf8()))
    result = ga.fix_axis_order(empty, EXT)
    assert result.changed is False
    # No per-feature pending: an empty batch has nothing to confirm.
    assert result.pending is False
    assert result.evidence.pending_reasons == ("no_evaluable_features",)

    # Batch mode reports the indeterminate batch as pending.
    result = ga.fix_axis_order(empty, EXT, mode="batch")
    assert result.changed is False
    assert result.pending is True


# ---------------------------------------------------------------------------
# fix_axis_order table write-back
# ---------------------------------------------------------------------------


def test_fix_axis_order_table_write_back():
    geometry = ga.with_crs(
        ga.as_geoarrow(["POINT (40 -75)", "POINT (35 -72)"]), types.OGC_CRS84
    )
    field = pa.field("geometry", geometry.type, metadata={"fm": "mv"})
    table = pa.Table.from_arrays(
        [pa.array([1, 2]), geometry],
        schema=pa.schema(
            [pa.field("id", pa.int64()), field], metadata={"geo": "yes"}
        ),
    )

    result = ga.fix_axis_order(table, EXT)
    assert isinstance(result.data, pa.Table)
    assert result.changed is True
    assert result.data.column_names == ["id", "geometry"]
    assert wkts(result.data.column("geometry")) == ["POINT (-75 40)", "POINT (-72 35)"]
    assert result.data.schema.field("geometry").metadata == {b"fm": b"mv"}
    assert result.data.schema.metadata == {b"geo": b"yes"}
    assert result.data.schema.field("geometry").type.crs == types.OGC_CRS84


def test_fix_axis_order_table_unchanged_returns_table():
    geometry = ga.as_geoarrow(["POINT (-75 40)", "POINT (-72 35)"])
    table = pa.table({"id": pa.array([1, 2]), "geometry": geometry})
    result = ga.fix_axis_order(table, EXT)
    assert isinstance(result.data, pa.Table)
    assert result.changed is False
    assert result.data is table


def test_fix_axis_order_table_geometry_column():
    geometry = ga.as_geoarrow(["POINT (40 -75)"])
    other = ga.as_geoarrow(["POINT (0 0)"])
    table = pa.table({"id": [1], "geom": geometry, "other": other})

    with pytest.raises(ValueError, match="more than one geometry column"):
        ga.fix_axis_order(table, EXT)

    result = ga.fix_axis_order(table, EXT, geometry_column="geom")
    assert wkts(result.data.column("geom")) == ["POINT (-75 40)"]
    # the other geometry column is untouched
    assert wkts(result.data.column("other")) == ["POINT (0 0)"]


def test_fix_axis_order_does_not_mutate_input():
    geometry = ga.as_geoarrow(["POINT (40 -75)"])
    _ = ga.fix_axis_order(geometry, EXT, mode="per_feature")
    assert wkts(geometry) == ["POINT (40 -75)"]


# ---------------------------------------------------------------------------
# CRS axis order helper
# ---------------------------------------------------------------------------


def test_crs_axis_order():
    assert crs_mod.crs_axis_order(types.OGC_CRS84) == ("Lon", "Lat")
    assert crs_mod.crs_axis_order(types.OGC_CRS84.to_json()) == ("Lon", "Lat")
    assert crs_mod.crs_axis_order(None) is None
    assert crs_mod.crs_axis_order("EPSG:4326") == ("Lat", "Lon")
    assert crs_mod.crs_axis_order("EPSG:3857") == ("X", "Y")


def test_crs_axis_order_no_axis():
    assert crs_mod.crs_axis_order({"type": "GeographicCRS", "name": "x"}) is None
