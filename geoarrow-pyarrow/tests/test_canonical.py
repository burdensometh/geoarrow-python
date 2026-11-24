import struct

import geoarrow.pyarrow as ga
import pyarrow as pa
import pytest
from geoarrow import types
from geoarrow.pyarrow import canonical


def wkb(wkt):
    return ga.as_wkb([wkt])[0].as_py()


def big_endian_polygon(coords):
    out = b"\x00" + struct.pack(">I", 3) + struct.pack(">I", 1)
    out += struct.pack(">I", len(coords))
    for x, y in coords:
        out += struct.pack(">dd", x, y)
    return out


def signature(*wkts, mode="polygon"):
    return ga.geometry_signature(list(wkts), mode=mode).to_pylist()


# ---------------------------------------------------------------------------
# Signature stability
# ---------------------------------------------------------------------------


def test_signature_ring_start_vertex():
    a = "POLYGON ((0 0, 4 0, 4 4, 0 4, 0 0))"
    b = "POLYGON ((4 0, 4 4, 0 4, 0 0, 4 0))"
    c = "POLYGON ((0 4, 0 0, 4 0, 4 4, 0 4))"
    sa, sb, sc = signature(a, b, c)
    assert sa == sb == sc


def test_signature_ring_winding():
    forward = "POLYGON ((0 0, 4 0, 4 4, 0 4, 0 0))"
    reversed_ = "POLYGON ((0 0, 0 4, 4 4, 4 0, 0 0))"
    assert signature(forward) == signature(reversed_)


def test_signature_hole_order_and_winding():
    a = (
        "POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0), "
        "(1 1, 2 1, 2 2, 1 2, 1 1), "
        "(5 5, 6 5, 6 6, 5 6, 5 5))"
    )
    b = (
        "POLYGON ((0 0, 0 10, 10 10, 10 0, 0 0), "
        "(5 5, 5 6, 6 6, 6 5, 5 5), "
        "(1 1, 1 2, 2 2, 2 1, 1 1))"
    )
    assert signature(a) == signature(b)


def test_signature_polygon_vs_multipolygon():
    polygon = "POLYGON ((0 0, 4 0, 4 4, 0 4, 0 0))"
    multi = "MULTIPOLYGON (((0 0, 4 0, 4 4, 0 4, 0 0)))"
    assert signature(polygon) == signature(multi)


def test_signature_multipolygon_member_order():
    a = "MULTIPOLYGON (((0 0, 1 0, 1 1, 0 0)), ((10 10, 11 10, 11 11, 10 10)))"
    b = "MULTIPOLYGON (((11 10, 11 11, 10 10, 11 10)), ((0 0, 1 0, 1 1, 0 0)))"
    assert signature(a) == signature(b)


def test_signature_closing_vertex_and_duplicates():
    a = "POLYGON ((0 0, 4 0, 4 4, 0 4))"
    b = "POLYGON ((0 0, 0 0, 4 0, 4 4, 0 4, 0 0))"
    assert signature(a) == signature(b)


def test_signature_big_endian_wkb():
    wkb_le = wkb("POLYGON ((0 0, 4 0, 4 4, 0 4, 0 0))")
    wkb_be = big_endian_polygon([(0, 0), (4, 0), (4, 4), (0, 4), (0, 0)])
    as_wkb_le = ga.wkb().wrap_array(pa.array([wkb_le]))
    as_wkb_be = ga.wkb().wrap_array(pa.array([wkb_be]))
    assert (
        ga.geometry_signature(as_wkb_le).to_pylist()
        == ga.geometry_signature(as_wkb_be).to_pylist()
    )


def test_signature_different_geometry():
    a = "POLYGON ((0 0, 4 0, 4 4, 0 4, 0 0))"
    b = "POLYGON ((0 0, 4 0, 4 5, 0 4, 0 0))"
    assert signature(a) != signature(b)

    # A small coordinate change must be detected (no snapping or tolerance).
    c = "POLYGON ((0 0, 4 0, 4 4.000000001, 0 4, 0 0))"
    assert signature(a) != signature(c)


def test_signature_dimensions_do_not_collide():
    xy = "POLYGON ((0 0, 1 0, 1 1, 0 0))"
    xyz = "POLYGON Z ((0 0 5, 1 0 5, 1 1 5, 0 0 5))"
    assert signature(xy) != signature(xyz)


def test_signature_negative_zero():
    a = "POLYGON ((0 0, 1 0, 1 1, 0 0))"
    b = "POLYGON ((-0 -0, 1 0, 1 1, -0 -0))"
    assert signature(a) == signature(b)


def test_signature_empty_and_null():
    values = [None, "POLYGON EMPTY", "MULTIPOLYGON EMPTY"]
    sigs = ga.geometry_signature(ga.array(pa.array(values, type=pa.utf8())))

    # NULL stays NULL; empties get a real signature...
    assert sigs[0].as_py() is None
    assert sigs[1].as_py() is not None
    # ...and the two flavors of empty are equivalent.
    assert sigs[1].as_py() == sigs[2].as_py()
    # ...which differs from a non-empty geometry.
    assert sigs[1].as_py() != signature("POLYGON ((0 0, 1 0, 1 1, 0 0))")[0]


def test_signature_mode_is_part_of_domain():
    wkt = "POLYGON ((0 0, 1 0, 1 1, 0 0))"
    assert (
        ga.geometry_signature([wkt], mode="exact").to_pylist()
        != ga.geometry_signature([wkt], mode="polygon").to_pylist()
    )


def test_signature_exact_mode_start_vertex():
    a = "POLYGON ((0 0, 1 0, 1 1, 0 0))"
    b = "POLYGON ((1 0, 1 1, 0 0, 1 0))"
    assert signature(a, mode="exact") != signature(b, mode="exact")


def test_signature_exact_mode_endianness():
    wkt = "POLYGON ((0 0, 4 0, 4 4, 0 4, 0 0))"
    wkb_le = wkb(wkt)
    wkb_be = big_endian_polygon([(0, 0), (4, 0), (4, 4), (0, 4), (0, 0)])
    le = ga.wkb().wrap_array(pa.array([wkb_le]))
    be = ga.wkb().wrap_array(pa.array([wkb_be]))
    assert (
        ga.geometry_signature(le, mode="exact").to_pylist()
        == ga.geometry_signature(be, mode="exact").to_pylist()
    )


def test_signature_exact_mode_supports_non_polygons():
    values = ["POINT (0 1)", "LINESTRING (0 0, 1 1)", "GEOMETRYCOLLECTION EMPTY"]
    sigs = ga.geometry_signature(values, mode="exact").to_pylist()
    assert all(sig is not None for sig in sigs)
    assert len(set(sigs)) == len(sigs)


def test_signature_type_and_chunking():
    array = ga.geometry_signature(["POLYGON ((0 0, 1 0, 1 1, 0 0))"])
    assert isinstance(array, pa.FixedSizeBinaryArray)
    assert array.type == pa.binary(ga.SIGNATURE_SIZE_BYTES)

    chunked = pa.chunked_array(
        [
            ga.array(["POLYGON ((0 0, 1 0, 1 1, 0 0))"]),
            ga.array(["POLYGON ((2 2, 3 2, 3 3, 2 2))"]),
        ]
    )
    sigs = ga.geometry_signature(chunked)
    assert isinstance(sigs, pa.ChunkedArray)
    assert sigs.type == pa.binary(ga.SIGNATURE_SIZE_BYTES)
    assert len(sigs) == 2


def test_signature_empty_inputs():
    assert len(ga.geometry_signature([])) == 0
    assert len(ga.geometry_signature(pa.chunked_array([], type=pa.binary()))) == 0
    nulls = ga.geometry_signature([None, None])
    assert nulls.type == pa.binary(ga.SIGNATURE_SIZE_BYTES)
    assert nulls.to_pylist() == [None, None]


def test_signature_invalid_mode():
    with pytest.raises(ValueError, match="Expected mode"):
        ga.geometry_signature(["POLYGON EMPTY"], mode="not-a-mode")


# ---------------------------------------------------------------------------
# Canonical geometry (traceability)
# ---------------------------------------------------------------------------


def test_canonicalize_geometry_returns_wkb():
    canonical_out = ga.canonicalize_geometry(["POLYGON ((0 0, 1 0, 1 1, 0 0))"])
    assert canonical_out.type == ga.wkb()
    assert canonical_out.type.encoding == ga.Encoding.WKB


def test_canonicalize_geometry_preserves_crs():
    array = ga.with_crs(
        ga.array(["POLYGON ((0 0, 1 0, 1 1, 0 0))"]), types.OGC_CRS84
    )
    canonical_out = ga.canonicalize_geometry(array)
    assert canonical_out.type.crs == types.OGC_CRS84
    assert (
        ga.geometry_signature(array).to_pylist()
        == ga.geometry_signature(canonical_out).to_pylist()
    )


def test_canonicalize_geometry_does_not_modify_input():
    array = ga.array(["POLYGON ((0 0, 4 0, 4 4, 0 4, 0 0))"])
    before = ga.as_wkb(array).to_pylist()
    canonical_out = ga.canonicalize_geometry(array)
    assert ga.as_wkb(array).to_pylist() == before
    # The canonical value is a different object with different bytes.
    assert ga.as_wkb(canonical_out).to_pylist() != before


def test_canonicalize_geometry_is_idempotent():
    source = [
        "POLYGON ((4 0, 4 4, 0 4, 0 0, 4 0))",
        "MULTIPOLYGON (((10 10, 11 10, 11 11, 10 10)), ((0 0, 1 0, 1 1, 0 0)))",
    ]
    once = ga.canonicalize_geometry(source)
    twice = ga.canonicalize_geometry(once)
    assert ga.as_wkb(once).to_pylist() == ga.as_wkb(twice).to_pylist()


def test_canonicalize_geometry_mode_exact_keeps_vertex_order():
    wkt = "POLYGON ((4 0, 4 4, 0 4, 0 0, 4 0))"
    canonical_out = ga.canonicalize_geometry([wkt], mode="exact")
    assert ga.format_wkt(canonical_out).to_pylist() == [wkt]


def test_canonicalize_geometry_polygon_mode_rejects_points():
    with pytest.raises(ValueError, match="supports only POLYGON and MULTIPOLYGON"):
        ga.canonicalize_geometry(["POINT (0 1)"], mode="polygon")


def test_canonicalize_geometry_handles_big_endian():
    wkb_be = big_endian_polygon([(0, 0), (4, 0), (4, 4), (0, 4), (0, 0)])
    array = ga.wkb().wrap_array(pa.array([wkb_be]))
    canonical_out = ga.canonicalize_geometry(array)
    canonical_wkb = ga.as_wkb(canonical_out)[0].as_py()
    # Little-endian marker and ISO type code are restored.
    assert canonical_wkb[0] == 1
    assert struct.unpack("<I", canonical_wkb[1:5])[0] == 3


def test_canonicalize_geometry_rejects_spherical_edges():
    array = ga.with_edge_type(
        ga.array(["POLYGON ((0 0, 1 0, 1 1, 0 0))"]), ga.EdgeType.SPHERICAL
    )
    with pytest.raises(ValueError, match="requires planar edges"):
        ga.canonicalize_geometry(array)


def test_canonicalize_geometry_nulls_and_empty():
    values = ga.array(pa.array([None, "POLYGON EMPTY"], type=pa.utf8()))
    canonical_out = ga.canonicalize_geometry(values)
    assert canonical_out[0].as_py() is None
    assert canonical_out[1].as_py() is not None


def test_canonicalize_geometry_empty_inputs():
    out = ga.canonicalize_geometry(pa.chunked_array([], type=pa.binary()))
    assert out.type == ga.wkb()
    assert len(out) == 0

    nulls = ga.canonicalize_geometry([None, None])
    assert nulls.type == ga.wkb()
    assert nulls.to_pylist() == [None, None]


# ---------------------------------------------------------------------------
# Snapshot comparison
# ---------------------------------------------------------------------------


def _snapshot(ids, wkts):
    return pa.table({"id": pa.array(ids, type=pa.int64()), "geometry": ga.array(wkts)})


def test_compare_snapshots_statuses():
    before = _snapshot(
        [1, 2, 3, 4],
        [
            "POLYGON ((0 0, 1 0, 1 1, 0 0))",  # removed
            "POLYGON ((2 2, 3 2, 3 3, 2 2))",  # unchanged (vertex reorder)
            "POLYGON ((4 4, 5 4, 5 5, 4 4))",  # updated
            "POLYGON ((6 6, 7 6, 7 7, 6 6))",  # unchanged (identical)
        ],
    )
    after = _snapshot(
        [2, 3, 4, 5],
        [
            "POLYGON ((3 2, 3 3, 2 2, 3 2))",
            "POLYGON ((4 4, 5 4, 5 6, 4 4))",
            "POLYGON ((6 6, 7 6, 7 7, 6 6))",
            "POLYGON ((8 8, 9 8, 9 9, 8 8))",  # added
        ],
    )

    diff = ga.compare_snapshots(before, after, key="id")
    assert diff.select(["id", "status"]).to_pylist() == [
        {"id": 1, "status": "removed"},
        {"id": 2, "status": "unchanged"},
        {"id": 3, "status": "updated"},
        {"id": 4, "status": "unchanged"},
        {"id": 5, "status": "added"},
    ]


def test_compare_snapshots_keeps_original_geometry():
    before = _snapshot([1], ["POLYGON ((4 0, 4 4, 0 4, 0 0, 4 0))"])
    after = _snapshot([1], ["POLYGON ((0 0, 4 0, 4 4, 0 4, 0 0))"])

    diff = ga.compare_snapshots(before, after, key="id")
    row = diff.to_pylist()[0]
    assert row["status"] == "unchanged"
    # The original, non-canonical geometries are retained for traceability.
    assert ga.format_wkt(ga.array([row["geometry_before"]]))[0].as_py() == (
        "POLYGON ((4 0, 4 4, 0 4, 0 0, 4 0))"
    )
    assert ga.format_wkt(ga.array([row["geometry_after"]]))[0].as_py() == (
        "POLYGON ((0 0, 4 0, 4 4, 0 4, 0 0))"
    )
    assert row["signature_before"] == row["signature_after"]


def test_compare_snapshots_null_vs_empty():
    before = pa.table(
        {
            "id": [1, 2],
            "geometry": ga.array(pa.array([None, "POLYGON EMPTY"], type=pa.utf8())),
        }
    )
    after = pa.table(
        {
            "id": [1, 2],
            "geometry": ga.array(pa.array(["POLYGON EMPTY", None], type=pa.utf8())),
        }
    )

    diff = ga.compare_snapshots(before, after, key="id")
    assert diff.select(["id", "status"]).to_pylist() == [
        {"id": 1, "status": "updated"},
        {"id": 2, "status": "updated"},
    ]
    rows = diff.to_pylist()
    assert rows[0]["signature_before"] is None
    assert rows[0]["signature_after"] is not None
    assert rows[0]["geometry_before"] is None
    assert rows[1]["signature_before"] is not None
    assert rows[1]["signature_after"] is None


def test_compare_snapshots_all_null_geometry():
    table = pa.table(
        {"id": [1, 2], "geometry": pa.chunked_array([pa.array([None, None])])}
    )
    diff = ga.compare_snapshots(table, table, key="id")
    assert diff.column("status").to_pylist() == ["unchanged", "unchanged"]
    assert diff.column("geometry_before").null_count == 2


def test_compare_snapshots_schema_and_metadata():
    table = _snapshot([1], ["POLYGON ((0 0, 1 0, 1 1, 0 0))"])
    diff = ga.compare_snapshots(table, table, key="id")
    assert diff.schema.names == [
        "id",
        "status",
        "geometry_before",
        "geometry_after",
        "signature_before",
        "signature_after",
    ]
    assert diff.schema.field("signature_before").type == pa.binary(32)
    assert diff.schema.field("signature_after").type == pa.binary(32)

    import json

    metadata = json.loads(diff.schema.metadata[b"geoarrow.pyarrow.canonical"])
    assert metadata == {
        "version": 1,
        "mode": "polygon",
        "hash": "sha256",
        "signature_size": 32,
    }


def test_compare_snapshots_multi_key():
    before = pa.table(
        {
            "region": ["north", "south"],
            "parcel": [1, 2],
            "geometry": ga.array(
                [
                    "POLYGON ((0 0, 1 0, 1 1, 0 0))",
                    "POLYGON ((2 2, 3 2, 3 3, 2 2))",
                ]
            ),
        }
    )
    after = pa.table(
        {
            "region": ["north", "south"],
            "parcel": [1, 2],
            "geometry": ga.array(
                [
                    "POLYGON ((1 0, 1 1, 0 0, 1 0))",
                    "POLYGON ((2 2, 3 2, 3 4, 2 2))",
                ]
            ),
        }
    )

    diff = ga.compare_snapshots(before, after, key=["region", "parcel"])
    assert diff.select(["region", "parcel", "status"]).to_pylist() == [
        {"region": "north", "parcel": 1, "status": "unchanged"},
        {"region": "south", "parcel": 2, "status": "updated"},
    ]


def test_compare_snapshots_geometry_column_names():
    wkt = "POLYGON ((0 0, 1 0, 1 1, 0 0))"
    before = pa.table({"id": [1], "geom_before": ga.array([wkt])})
    after = pa.table({"id": [1], "geom_after": ga.array([wkt])})

    diff = ga.compare_snapshots(
        before,
        after,
        key="id",
        geometry_column="geom_before",
        geometry_column_after="geom_after",
    )
    assert diff.column("status").to_pylist() == ["unchanged"]


def test_compare_snapshots_empty_tables():
    before = pa.table(
        {"id": pa.array([], type=pa.int64()), "geometry": ga.array([], type=ga.wkt())}
    )
    diff = ga.compare_snapshots(before, before, key="id")
    assert diff.num_rows == 0
    assert diff.schema.names == [
        "id",
        "status",
        "geometry_before",
        "geometry_after",
        "signature_before",
        "signature_after",
    ]


def test_compare_snapshots_rejects_crs_mismatch():
    wkt = "POLYGON ((0 0, 1 0, 1 1, 0 0))"
    crs_wkt = ga.with_crs(ga.array([wkt]), types.OGC_CRS84)
    before = pa.table({"id": [1], "geometry": crs_wkt})
    after = pa.table({"id": [1], "geometry": ga.array([wkt])})

    with pytest.raises(ValueError, match="different coordinate reference systems"):
        ga.compare_snapshots(before, after, key="id")

    diff = ga.compare_snapshots(before, after, key="id", check_crs=False)
    assert diff.column("status").to_pylist() == ["unchanged"]


def test_compare_snapshots_rejects_edge_mismatch():
    wkt = "POLYGON ((0 0, 1 0, 1 1, 0 0))"
    spherical = ga.with_edge_type(ga.array([wkt]), ga.EdgeType.SPHERICAL)
    before = pa.table({"id": [1], "geometry": spherical})
    after = pa.table({"id": [1], "geometry": ga.array([wkt])})

    with pytest.raises(ValueError, match="different edge interpretations"):
        ga.compare_snapshots(before, after, key="id")


def test_compare_snapshots_rejects_duplicate_keys():
    table = _snapshot(
        [1, 1],
        ["POLYGON ((0 0, 1 0, 1 1, 0 0))", "POLYGON ((2 2, 3 2, 3 3, 2 2))"],
    )
    with pytest.raises(ValueError, match="not unique"):
        ga.compare_snapshots(table, table, key="id")


def test_compare_snapshots_rejects_null_keys():
    table = pa.table(
        {
            "id": pa.array([1, None], type=pa.int64()),
            "geometry": ga.array(
                [
                    "POLYGON ((0 0, 1 0, 1 1, 0 0))",
                    "POLYGON ((2 2, 3 2, 3 3, 2 2))",
                ]
            ),
        }
    )
    with pytest.raises(ValueError, match="contains nulls"):
        ga.compare_snapshots(table, table, key="id")


def test_compare_snapshots_rejects_key_type_mismatch():
    before = _snapshot([1], ["POLYGON ((0 0, 1 0, 1 1, 0 0))"])
    after = pa.table(
        {
            "id": pa.array(["1"]),
            "geometry": ga.array(["POLYGON ((0 0, 1 0, 1 1, 0 0))"]),
        }
    )
    with pytest.raises(ValueError, match="has type"):
        ga.compare_snapshots(before, after, key="id")


def test_compare_snapshots_rejects_missing_columns():
    table = _snapshot([1], ["POLYGON ((0 0, 1 0, 1 1, 0 0))"])
    with pytest.raises(ValueError, match="Geometry column 'nope' is missing"):
        ga.compare_snapshots(table, table, key="id", geometry_column="nope")

    with pytest.raises(ValueError, match="Key column 'nope' is missing"):
        ga.compare_snapshots(table, table, key="nope")


def test_compare_snapshots_rejects_reserved_key_names():
    table = _snapshot([1], ["POLYGON ((0 0, 1 0, 1 1, 0 0))"])
    with pytest.raises(ValueError, match="reserved names"):
        ga.compare_snapshots(table, table, key="status")


def test_compare_snapshots_rejects_duplicate_key_names():
    table = _snapshot([1], ["POLYGON ((0 0, 1 0, 1 1, 0 0))"])
    with pytest.raises(ValueError, match="duplicate column names"):
        ga.compare_snapshots(table, table, key=["id", "id"])


def test_compare_snapshots_mode_exact_detects_vertex_reorder():
    before = _snapshot([1], ["POLYGON ((0 0, 1 0, 1 1, 0 0))"])
    after = _snapshot([1], ["POLYGON ((1 0, 1 1, 0 0, 1 0))"])

    default = ga.compare_snapshots(before, after, key="id")
    assert default.column("status").to_pylist() == ["unchanged"]
    assert ga.compare_snapshots(
        before, after, key="id", mode="exact"
    ).column("status").to_pylist() == ["updated"]


def test_compare_snapshots_requires_tables():
    with pytest.raises(TypeError, match="pyarrow.Table"):
        ga.compare_snapshots([], [], key="id")


def test_compare_snapshots_invalid_mode():
    table = _snapshot([1], ["POLYGON ((0 0, 1 0, 1 1, 0 0))"])
    with pytest.raises(ValueError, match="Expected mode"):
        ga.compare_snapshots(table, table, key="id", mode="bogus")


def test_module_exports():
    assert canonical.CanonicalMode.POLYGON.value == "polygon"
    assert canonical.CanonicalMode.EXACT.value == "exact"
    assert ga.CanonicalMode is canonical.CanonicalMode
