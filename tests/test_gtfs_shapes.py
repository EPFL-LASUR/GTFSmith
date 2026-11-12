"""
Tests for GTFSShapeBuilder (fills missing shapes.txt from an OSM network).

How this file relates to conftest.py
------------------------------------
* Same convention as the rest of the suite: the module under test is
  imported with importlib in a fixture (`sb`), and the real `GTFS` class /
  `feed_a` fixtures from conftest.py are used for end-to-end tests.
* For unit tests of single methods, a real, schema-validated GTFS is more
  boilerplate than it is worth, so (like conftest's `BareGTFS`) a tiny
  `FakeGTFS` is used. It is `BareGTFS` plus `update_file`, which the
  builder calls when merging and which `BareGTFS` doesn't have. It also
  records the calls so the tests can check `validate=False`.
* The OSM network is replaced by `FakeNetwork`, which implements the two
  methods the builder uses (`get_geometry_for_stop_sequence` and
  `diagnose_stop_sequence`). One end-to-end test uses the REAL
  TransitRouteNetwork.

Adjust SHAPE_BUILDER_MODULE to wherever GTFSShapeBuilder lives.
"""

from __future__ import annotations

import importlib
import multiprocessing
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

SHAPE_BUILDER_MODULE = "gtfsmith.gtfs_shapes"


# ----------------------------------------------------------------------
# Fixtures / fakes
# ----------------------------------------------------------------------


@pytest.fixture(scope="session")
def sb():
    """The module that defines GTFSShapeBuilder and _ShapePattern."""
    return importlib.import_module(SHAPE_BUILDER_MODULE)


class FakeGTFS:
    """`.files`-only stand-in (like conftest.BareGTFS) with update_file."""

    def __init__(self, files):
        self.files = dict(files)
        self.updates: list[tuple[str, bool]] = []

    def update_file(self, name, df, validate=True):
        self.updates.append((name, validate))
        self.files[name] = df


# Handlers are module-level functions so that FakeNetwork stays picklable
# (needed by the multi-process test).


def echo_stops(leg, stops):
    return [[lat, lon] for lat, lon in stops]


def always_fail(leg, stops):
    raise ValueError("no OSM route")


def fail_for_bus(leg, stops):
    if leg["routeType"] == "BUS":
        raise ValueError("no OSM route for buses")
    return echo_stops(leg, stops)


def single_point(leg, stops):
    return [[stops[0][0], stops[0][1]]]


class FakeNetwork:
    DEFAULT_DIAGNOSIS = {
        "reason": "STOP_TOO_FAR",
        "fail_stop_index": 1,
        "fail_distance_m": 2000,
        "best_relation": "https://www.openstreetmap.org/relation/1",
    }

    def __init__(self, handler=None, diagnosis=None):
        self.handler = handler or echo_stops
        self.diagnosis = dict(
            self.DEFAULT_DIAGNOSIS if diagnosis is None else diagnosis
        )
        self.calls: list[dict] = []
        self.diagnose_calls: list[dict] = []

    def get_geometry_for_stop_sequence(
        self, leg, stops, max_stop_distance_m=750.0
    ):
        self.calls.append(
            {
                "leg": leg,
                "stops": stops,
                "max_stop_distance_m": max_stop_distance_m,
            }
        )
        return self.handler(leg, stops)

    def diagnose_stop_sequence(self, leg, stops, max_stop_distance_m=750.0):
        self.diagnose_calls.append(
            {
                "leg": leg,
                "stops": stops,
                "max_stop_distance_m": max_stop_distance_m,
            }
        )
        return dict(self.diagnosis)


def make_files():
    """
    Three stops on a parallel, two routes (RAIL "IC1", BUS "12") and three
    trips: T1 and T2 share one stop pattern, T3 is on the bus route.
    """
    return {
        "stops.txt": pd.DataFrame(
            {
                "stop_id": ["S1", "S2", "S3"],
                "stop_name": ["One", "Two", "Three"],
                "stop_lat": [46.0, 46.0, 46.0],
                "stop_lon": [7.0, 7.1, 7.2],
            }
        ),
        "routes.txt": pd.DataFrame(
            {
                "route_id": ["R1", "R2"],
                "route_short_name": ["IC1", "12"],
                "route_long_name": ["Long one", "Long two"],
                "route_type": [2, 3],
            }
        ),
        "trips.txt": pd.DataFrame(
            {
                "trip_id": ["T1", "T2", "T3"],
                "route_id": ["R1", "R1", "R2"],
                "service_id": ["SV", "SV", "SV"],
            }
        ),
        "stop_times.txt": pd.DataFrame(
            {
                "trip_id": ["T1"] * 3 + ["T2"] * 3 + ["T3"] * 2,
                "stop_id": ["S1", "S2", "S3"] * 2 + ["S1", "S2"],
                "stop_sequence": [1, 2, 3] * 2 + [1, 2],
            }
        ),
    }


def make_gtfs(overrides=None):
    files = make_files()
    files.update(overrides or {})
    return FakeGTFS(files)


def make_builder(sb, gtfs=None, network=None, **kwargs):
    return sb.GTFSShapeBuilder(
        gtfs if gtfs is not None else make_gtfs(),
        network if network is not None else FakeNetwork(),
        **kwargs,
    )


def make_pattern(sb, **kwargs):
    defaults = dict(
        route_id="R1",
        stop_ids=("S1", "S2", "S3"),
        trip_ids=["T1", "T2"],
        ref="IC1",
        route_type_label="RAIL",
        from_stop=(46.0, 7.0),
        to_stop=(46.0, 7.2),
        stop_coords=((46.0, 7.0), (46.0, 7.1), (46.0, 7.2)),
    )
    defaults.update(kwargs)
    return sb._ShapePattern(**defaults)


KEY_R1 = ("R1", ("S1", "S2", "S3"))
KEY_R2 = ("R2", ("S1", "S2"))


# ----------------------------------------------------------------------
# Construction
# ----------------------------------------------------------------------


def test_init_defaults(sb):
    builder = make_builder(sb)

    assert builder.max_stop_distance_m == 750.0
    assert builder.workers == 1
    assert builder.diagnose is True
    assert builder.report.empty
    assert builder._skipped == []
    assert builder._stop_names == {}


def test_init_stores_options(sb):
    network = FakeNetwork()
    gtfs = make_gtfs()

    builder = sb.GTFSShapeBuilder(
        gtfs, network, max_stop_distance_m=500.0, workers=4, diagnose=False
    )

    assert builder.gtfs is gtfs
    assert builder.network is network
    assert builder.max_stop_distance_m == 500.0
    assert builder.workers == 4
    assert builder.diagnose is False


@pytest.mark.parametrize(
    "route_type, label",
    [
        (0, "TRAM"),
        (1, "SUBWAY"),
        (2, "RAIL"),
        (3, "BUS"),
        (4, "FERRY"),
        (5, "CABLE_TRAM"),
        (6, "CABLE_CAR"),
        (7, "FUNICULAR"),
        (11, "TROLLEYBUS"),
        (12, "METRO"),
    ],
)
def test_route_type_labels(sb, route_type, label):
    builder = make_builder(sb)

    assert builder.GTFS_ROUTE_TYPE_LABELS[route_type] == label


def test_extended_route_types_are_not_labelled(sb):
    builder = make_builder(sb)

    assert 100 not in builder.GTFS_ROUTE_TYPE_LABELS


# ----------------------------------------------------------------------
# _pick_ref
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "short, long, route_id, expected",
    [
        ("IC1", "Long name", "R1", "IC1"),
        (None, "Long name", "R1", "Long name"),
        (np.nan, "Long name", "R1", "Long name"),
        ("   ", "Long name", "R1", "Long name"),
        (None, None, "R1", "R1"),
        (np.nan, np.nan, "R1", "R1"),
        ("", "  ", "R1", "R1"),
        (12, None, "R1", "12"),
    ],
)
def test_pick_ref(sb, short, long, route_id, expected):
    row = SimpleNamespace(
        route_short_name=short,
        route_long_name=long,
        route_id=route_id,
    )

    assert sb.GTFSShapeBuilder._pick_ref(row) == expected


# ----------------------------------------------------------------------
# _new_shape_id
# ----------------------------------------------------------------------


def test_new_shape_id_is_free_and_registered(sb):
    existing: set[str] = set()

    shape_id = sb.GTFSShapeBuilder._new_shape_id("R1", existing)

    assert shape_id == "osm_R1"
    assert existing == {"osm_R1"}


def test_new_shape_id_avoids_collisions(sb):
    existing = {"osm_R1", "osm_R1_1"}

    shape_id = sb.GTFSShapeBuilder._new_shape_id("R1", existing)

    assert shape_id == "osm_R1_2"
    assert "osm_R1_2" in existing


def test_new_shape_id_repeated_calls_are_unique(sb):
    existing: set[str] = set()

    ids = [sb.GTFSShapeBuilder._new_shape_id("R1", existing) for _ in range(3)]

    assert ids == ["osm_R1", "osm_R1_1", "osm_R1_2"]


# ----------------------------------------------------------------------
# _coords_to_shape_frame
# ----------------------------------------------------------------------


def test_coords_to_shape_frame_columns_and_values(sb):
    frame = sb.GTFSShapeBuilder._coords_to_shape_frame(
        "SH", [[46.0, 7.0], [47.0, 7.0], [47.0, 7.0]]
    )

    assert list(frame.columns) == [
        "shape_id",
        "shape_pt_lat",
        "shape_pt_lon",
        "shape_pt_sequence",
        "shape_dist_traveled",
    ]
    assert (frame["shape_id"] == "SH").all()
    assert frame["shape_pt_lat"].tolist() == [46.0, 47.0, 47.0]
    assert frame["shape_pt_lon"].tolist() == [7.0, 7.0, 7.0]
    assert frame["shape_pt_sequence"].tolist() == [0, 1, 2]


def test_coords_to_shape_frame_distance_is_cumulative_metres(sb):
    frame = sb.GTFSShapeBuilder._coords_to_shape_frame(
        "SH", [[46.0, 7.0], [47.0, 7.0], [47.0, 7.0]]
    )

    dist = frame["shape_dist_traveled"].to_numpy()

    assert dist[0] == 0.0
    # one degree of latitude is ~111.2 km
    assert dist[1] == pytest.approx(111_195, rel=1e-3)
    # a repeated point adds nothing
    assert dist[2] == pytest.approx(dist[1])


def test_coords_to_shape_frame_distance_is_non_decreasing(sb):
    coords = [[46.0 + 0.01 * i, 7.0 + 0.02 * i] for i in range(20)]

    frame = sb.GTFSShapeBuilder._coords_to_shape_frame("SH", coords)

    assert np.all(np.diff(frame["shape_dist_traveled"].to_numpy()) > 0)


# ----------------------------------------------------------------------
# _trips_missing_shapes
# ----------------------------------------------------------------------


def test_trips_missing_shapes_without_shape_id_column(sb):
    gtfs = make_gtfs()
    builder = make_builder(sb, gtfs)

    result = builder._trips_missing_shapes()

    assert result["trip_id"].tolist() == ["T1", "T2", "T3"]
    # a copy, not the feed's own table
    assert result is not gtfs.files["trips.txt"]


def test_trips_missing_shapes_null_and_unknown_shape_ids(sb):
    trips = make_files()["trips.txt"]
    trips["shape_id"] = ["SH1", None, "GONE"]
    shapes = pd.DataFrame(
        {
            "shape_id": ["SH1", "SH1"],
            "shape_pt_lat": [46.0, 46.1],
            "shape_pt_lon": [7.0, 7.1],
            "shape_pt_sequence": [0, 1],
        }
    )
    gtfs = make_gtfs({"trips.txt": trips, "shapes.txt": shapes})

    result = make_builder(sb, gtfs)._trips_missing_shapes()

    assert result["trip_id"].tolist() == ["T2", "T3"]


def test_trips_missing_shapes_without_shapes_file(sb):
    trips = make_files()["trips.txt"]
    trips["shape_id"] = ["SH1", None, "X"]
    gtfs = make_gtfs({"trips.txt": trips})

    result = make_builder(sb, gtfs)._trips_missing_shapes()

    # No shapes.txt at all: even trips with a shape_id can't resolve it.
    assert result["trip_id"].tolist() == ["T1", "T2", "T3"]


def test_trips_missing_shapes_none_missing(sb):
    trips = make_files()["trips.txt"]
    trips["shape_id"] = "SH1"
    shapes = pd.DataFrame({"shape_id": ["SH1"]})
    gtfs = make_gtfs({"trips.txt": trips, "shapes.txt": shapes})

    assert make_builder(sb, gtfs)._trips_missing_shapes().empty


# ----------------------------------------------------------------------
# _build_patterns
# ----------------------------------------------------------------------


def test_build_patterns_groups_trips_by_route_and_stop_sequence(sb):
    gtfs = make_gtfs()
    builder = make_builder(sb, gtfs)

    patterns = builder._build_patterns(gtfs.files["trips.txt"])

    assert set(patterns) == {KEY_R1, KEY_R2}

    p = patterns[KEY_R1]
    assert p.route_id == "R1"
    assert p.stop_ids == ("S1", "S2", "S3")
    assert p.trip_ids == ["T1", "T2"]
    assert p.ref == "IC1"
    assert p.route_type_label == "RAIL"
    assert p.from_stop == (46.0, 7.0)
    assert p.to_stop == (46.0, 7.2)
    assert p.stop_coords == ((46.0, 7.0), (46.0, 7.1), (46.0, 7.2))

    q = patterns[KEY_R2]
    assert q.trip_ids == ["T3"]
    assert q.ref == "12"
    assert q.route_type_label == "BUS"


def test_build_patterns_orders_stops_by_stop_sequence(sb):
    stop_times = pd.DataFrame(
        {
            "trip_id": ["T1", "T1", "T1"],
            "stop_id": ["S3", "S1", "S2"],
            "stop_sequence": [3, 1, 2],
        }
    )
    trips = make_files()["trips.txt"].iloc[[0]]
    gtfs = make_gtfs({"stop_times.txt": stop_times})
    builder = make_builder(sb, gtfs)

    patterns = builder._build_patterns(trips)

    assert list(patterns) == [("R1", ("S1", "S2", "S3"))]


def test_build_patterns_splits_different_stop_orders(sb):
    stop_times = pd.DataFrame(
        {
            "trip_id": ["T1"] * 3 + ["T2"] * 3,
            "stop_id": ["S1", "S2", "S3", "S1", "S3", "S2"],
            "stop_sequence": [1, 2, 3, 1, 2, 3],
        }
    )
    trips = make_files()["trips.txt"].iloc[[0, 1]]
    gtfs = make_gtfs({"stop_times.txt": stop_times})

    patterns = make_builder(sb, gtfs)._build_patterns(trips)

    assert set(patterns) == {
        ("R1", ("S1", "S2", "S3")),
        ("R1", ("S1", "S3", "S2")),
    }


def test_build_patterns_splits_same_stops_on_different_routes(sb):
    trips = pd.DataFrame(
        {
            "trip_id": ["T1", "T4"],
            "route_id": ["R1", "R2"],
            "service_id": ["SV", "SV"],
        }
    )
    stop_times = pd.DataFrame(
        {
            "trip_id": ["T1"] * 3 + ["T4"] * 3,
            "stop_id": ["S1", "S2", "S3"] * 2,
            "stop_sequence": [1, 2, 3] * 2,
        }
    )
    gtfs = make_gtfs({"trips.txt": trips, "stop_times.txt": stop_times})

    patterns = make_builder(sb, gtfs)._build_patterns(trips)

    assert set(patterns) == {
        ("R1", ("S1", "S2", "S3")),
        ("R2", ("S1", "S2", "S3")),
    }


def test_build_patterns_only_considers_given_trips(sb):
    gtfs = make_gtfs()
    only_t1 = gtfs.files["trips.txt"].iloc[[0]]

    patterns = make_builder(sb, gtfs)._build_patterns(only_t1)

    assert list(patterns) == [KEY_R1]
    assert patterns[KEY_R1].trip_ids == ["T1"]


def test_build_patterns_unknown_route_or_type_has_no_label(sb):
    trips = pd.DataFrame(
        {
            "trip_id": ["T1", "T2"],
            "route_id": ["RX", "R3"],
            "service_id": ["SV", "SV"],
        }
    )
    routes = pd.DataFrame(
        {
            "route_id": ["R3"],
            "route_short_name": ["X1"],
            "route_long_name": [None],
            "route_type": [100],  # extended type
        }
    )
    stop_times = pd.DataFrame(
        {
            "trip_id": ["T1"] * 2 + ["T2"] * 2,
            "stop_id": ["S1", "S2"] * 2,
            "stop_sequence": [1, 2] * 2,
        }
    )
    gtfs = make_gtfs(
        {"trips.txt": trips, "routes.txt": routes, "stop_times.txt": stop_times}
    )

    patterns = make_builder(sb, gtfs)._build_patterns(trips)

    unknown_route = patterns[("RX", ("S1", "S2"))]
    assert unknown_route.ref is None
    assert unknown_route.route_type_label is None

    extended = patterns[("R3", ("S1", "S2"))]
    assert extended.ref == "X1"
    assert extended.route_type_label is None


def test_build_patterns_skips_too_few_stops(sb):
    trips = pd.DataFrame(
        {
            "trip_id": ["T3", "T4"],
            "route_id": ["R2", "R2"],
            "service_id": ["SV", "SV"],
        }
    )
    stop_times = pd.DataFrame(
        {
            "trip_id": ["T3", "T4"],
            "stop_id": ["S1", "S1"],
            "stop_sequence": [1, 1],
        }
    )
    gtfs = make_gtfs({"trips.txt": trips, "stop_times.txt": stop_times})
    builder = make_builder(sb, gtfs)

    patterns = builder._build_patterns(trips)

    assert patterns == {}
    assert len(builder._skipped) == 1

    row = builder._skipped[0]
    assert row["reason"] == "TOO_FEW_STOPS"
    assert row["route_id"] == "R2"
    assert row["gtfs_ref"] == "12"
    assert row["route_type"] == "BUS"
    assert row["n_stops"] == 1
    assert row["n_trips"] == 2  # both trips share the skipped pattern
    assert row["first_stop"] == "One"
    assert row["example_trip_id"] == "T3"


def test_build_patterns_skips_stops_without_coordinates(sb):
    stops = pd.DataFrame(
        {
            "stop_id": ["S1", "S2", "S3"],
            "stop_name": ["One", "Two", "Three"],
            "stop_lat": [46.0, np.nan, 46.0],
            "stop_lon": [7.0, 7.1, 7.2],
        }
    )
    gtfs = make_gtfs({"stops.txt": stops})
    builder = make_builder(sb, gtfs)

    patterns = builder._build_patterns(gtfs.files["trips.txt"])

    # T3 (S1, S2) and T1/T2 (S1, S2, S3) all contain S2.
    assert patterns == {}

    reasons = {r["reason"] for r in builder._skipped}
    assert reasons == {"STOP_WITHOUT_COORDINATES"}

    by_route = {r["route_id"]: r for r in builder._skipped}
    assert by_route["R1"]["detail"] == "S2"
    assert by_route["R1"]["n_trips"] == 2


def test_build_patterns_unknown_stop_id_is_reported_with_id_as_name(sb):
    trips = make_files()["trips.txt"].iloc[[0]]
    stop_times = pd.DataFrame(
        {
            "trip_id": ["T1", "T1"],
            "stop_id": ["S1", "GHOST"],
            "stop_sequence": [1, 2],
        }
    )
    gtfs = make_gtfs({"stop_times.txt": stop_times})
    builder = make_builder(sb, gtfs)

    builder._build_patterns(trips)

    row = builder._skipped[0]
    assert row["reason"] == "STOP_WITHOUT_COORDINATES"
    assert row["detail"] == "GHOST"
    assert row["first_stop"] == "One"
    assert row["last_stop"] == "GHOST"  # no name known -> falls back to id


def test_build_patterns_detail_lists_at_most_five_missing_stops(sb):
    trips = make_files()["trips.txt"].iloc[[0]]
    ghosts = [f"G{i}" for i in range(8)]
    stop_times = pd.DataFrame(
        {
            "trip_id": ["T1"] * 8,
            "stop_id": ghosts,
            "stop_sequence": list(range(8)),
        }
    )
    gtfs = make_gtfs({"stop_times.txt": stop_times})
    builder = make_builder(sb, gtfs)

    builder._build_patterns(trips)

    assert builder._skipped[0]["detail"] == "G0, G1, G2, G3, G4"


def test_build_patterns_remembers_stop_names(sb):
    gtfs = make_gtfs()
    builder = make_builder(sb, gtfs)

    builder._build_patterns(gtfs.files["trips.txt"])

    assert builder._stop_names == {"S1": "One", "S2": "Two", "S3": "Three"}


def test_build_patterns_without_stop_name_column(sb):
    stops = make_files()["stops.txt"].drop(columns=["stop_name"])
    gtfs = make_gtfs({"stops.txt": stops})
    builder = make_builder(sb, gtfs)

    patterns = builder._build_patterns(gtfs.files["trips.txt"])

    assert set(patterns) == {KEY_R1, KEY_R2}
    assert builder._stop_names == {}


def test_build_patterns_resets_skipped_on_every_call(sb):
    stops = make_files()["stops.txt"]
    stops.loc[stops["stop_id"] == "S2", "stop_lat"] = np.nan
    gtfs = make_gtfs({"stops.txt": stops})
    builder = make_builder(sb, gtfs)

    builder._build_patterns(gtfs.files["trips.txt"])
    first = len(builder._skipped)
    builder._build_patterns(gtfs.files["trips.txt"])

    assert len(builder._skipped) == first


# ----------------------------------------------------------------------
# _solve_pattern
# ----------------------------------------------------------------------


def test_solve_pattern_builds_leg_and_uses_full_stop_sequence(sb):
    network = FakeNetwork()
    builder = make_builder(sb, network=network, max_stop_distance_m=500.0)
    pattern = make_pattern(sb)

    coords, info = builder._solve_pattern(pattern)

    assert info is None
    assert coords == [[46.0, 7.0], [46.0, 7.1], [46.0, 7.2]]

    (call,) = network.calls
    assert call["leg"] == {
        "route": "IC1",
        "routeType": "RAIL",
        "from": {"lat": 46.0, "lon": 7.0},
        "to": {"lat": 46.0, "lon": 7.2},
    }
    assert call["stops"] == [(46.0, 7.0), (46.0, 7.1), (46.0, 7.2)]
    assert call["max_stop_distance_m"] == 500.0


def test_solve_pattern_failure_with_diagnosis(sb):
    network = FakeNetwork(handler=always_fail)
    builder = make_builder(sb, network=network, diagnose=True)

    coords, info = builder._solve_pattern(make_pattern(sb))

    assert coords is None
    assert info["error"] == "no OSM route"
    # the diagnosis' reason replaces the generic NO_MATCH
    assert info["reason"] == "STOP_TOO_FAR"
    assert info["fail_stop_index"] == 1
    assert info["best_relation"].endswith("/relation/1")

    (call,) = network.diagnose_calls
    assert call["stops"] == [(46.0, 7.0), (46.0, 7.1), (46.0, 7.2)]
    assert call["max_stop_distance_m"] == 750.0
    assert call["leg"]["route"] == "IC1"


def test_solve_pattern_failure_without_diagnosis(sb):
    network = FakeNetwork(handler=always_fail)
    builder = make_builder(sb, network=network, diagnose=False)

    coords, info = builder._solve_pattern(make_pattern(sb))

    assert coords is None
    assert info == {"reason": "NO_MATCH", "error": "no OSM route"}
    assert network.diagnose_calls == []


def test_solve_pattern_only_catches_value_error(sb):
    def boom(leg, stops):
        raise RuntimeError("unexpected")

    builder = make_builder(sb, network=FakeNetwork(handler=boom))

    with pytest.raises(RuntimeError):
        builder._solve_pattern(make_pattern(sb))


# ----------------------------------------------------------------------
# _report_row
# ----------------------------------------------------------------------


def test_report_row_basic_fields(sb):
    builder = make_builder(sb)
    builder._stop_names = {"S1": "One", "S2": "Two", "S3": "Three"}

    row = builder._report_row(
        make_pattern(sb), "NO_MATCH", {"reason": "ignored", "error": "boom"}
    )

    assert row == {
        "route_id": "R1",
        "gtfs_ref": "IC1",
        "route_type": "RAIL",
        "n_stops": 3,
        "n_trips": 2,
        "first_stop": "One",
        "last_stop": "Three",
        "example_trip_id": "T1",
        "error": "boom",
        "reason": "NO_MATCH",
    }


def test_report_row_resolves_fail_stop_name(sb):
    builder = make_builder(sb)
    builder._stop_names = {"S1": "One", "S2": "Two", "S3": "Three"}

    row = builder._report_row(
        make_pattern(sb), "STOP_TOO_FAR", {"fail_stop_index": 1}
    )

    assert row["fail_stop"] == "Two"
    assert row["fail_stop_index"] == 1


def test_report_row_falls_back_to_stop_ids(sb):
    builder = make_builder(sb)  # no names known

    row = builder._report_row(
        make_pattern(sb), "STOP_TOO_FAR", {"fail_stop_index": 2}
    )

    assert row["first_stop"] == "S1"
    assert row["last_stop"] == "S3"
    assert row["fail_stop"] == "S3"


def test_report_row_without_fail_index_has_no_fail_stop(sb):
    row = make_builder(sb)._report_row(make_pattern(sb), "NO_MATCH", {})

    assert "fail_stop" not in row


# ----------------------------------------------------------------------
# _run_patterns
# ----------------------------------------------------------------------


def test_run_patterns_single_process_yields_in_order(sb):
    builder = make_builder(sb, workers=1)
    items = [
        (KEY_R1, make_pattern(sb)),
        (
            KEY_R2,
            make_pattern(
                sb,
                route_id="R2",
                stop_ids=("S1", "S2"),
                trip_ids=["T3"],
                ref="12",
                route_type_label="BUS",
                to_stop=(46.0, 7.1),
                stop_coords=((46.0, 7.0), (46.0, 7.1)),
            ),
        ),
    ]

    results = list(builder._run_patterns(items))

    assert len(results) == 2
    assert results[0][0] == [[46.0, 7.0], [46.0, 7.1], [46.0, 7.2]]
    assert results[1][0] == [[46.0, 7.0], [46.0, 7.1]]
    assert all(info is None for _, info in results)


# ----------------------------------------------------------------------
# _build_shapes
# ----------------------------------------------------------------------


def _patterns(sb, gtfs=None, builder=None):
    gtfs = gtfs or make_gtfs()
    builder = builder or make_builder(sb, gtfs)
    return builder, builder._build_patterns(gtfs.files["trips.txt"])


def test_build_shapes_success(sb):
    builder, patterns = _patterns(sb)

    frames, shape_ids = builder._build_shapes(patterns)

    assert shape_ids == {KEY_R1: "osm_R1", KEY_R2: "osm_R2"}
    assert len(frames) == 2
    assert frames[0]["shape_id"].iloc[0] == "osm_R1"
    assert len(frames[0]) == 3
    assert frames[1]["shape_id"].iloc[0] == "osm_R2"
    assert len(frames[1]) == 2
    assert builder.report.empty


def test_build_shapes_avoids_existing_shape_ids(sb):
    shapes = pd.DataFrame({"shape_id": ["osm_R1"]})
    gtfs = make_gtfs({"shapes.txt": shapes})
    builder, patterns = _patterns(sb, gtfs)

    _, shape_ids = builder._build_shapes(patterns)

    assert shape_ids[KEY_R1] == "osm_R1_1"
    assert shape_ids[KEY_R2] == "osm_R2"


def test_build_shapes_gives_each_pattern_of_a_route_its_own_id(sb):
    stop_times = pd.DataFrame(
        {
            "trip_id": ["T1"] * 3 + ["T2"] * 2,
            "stop_id": ["S1", "S2", "S3", "S1", "S3"],
            "stop_sequence": [1, 2, 3, 1, 2],
        }
    )
    trips = make_files()["trips.txt"].iloc[[0, 1]]
    gtfs = make_gtfs({"trips.txt": trips, "stop_times.txt": stop_times})
    builder, patterns = _patterns(sb, gtfs)

    _, shape_ids = builder._build_shapes(patterns)

    assert sorted(shape_ids.values()) == ["osm_R1", "osm_R1_1"]


def test_build_shapes_reports_unmatched_patterns(sb):
    network = FakeNetwork(handler=fail_for_bus)
    builder = make_builder(sb, network=network)
    patterns = builder._build_patterns(builder.gtfs.files["trips.txt"])

    frames, shape_ids = builder._build_shapes(patterns)

    assert list(shape_ids) == [KEY_R1]
    assert len(frames) == 1

    report = builder.report
    assert len(report) == 1
    row = report.iloc[0]
    assert row["route_id"] == "R2"
    assert row["gtfs_ref"] == "12"
    assert row["route_type"] == "BUS"
    assert row["n_trips"] == 1
    assert row["first_stop"] == "One"
    assert row["last_stop"] == "Two"
    # diagnose=True: the diagnosis' reason and details land in the report
    assert row["reason"] == "STOP_TOO_FAR"
    assert row["fail_stop"] == "Two"
    assert "error" in report.columns


def test_build_shapes_report_without_diagnosis(sb):
    network = FakeNetwork(handler=always_fail)
    builder = make_builder(sb, network=network, diagnose=False)
    patterns = builder._build_patterns(builder.gtfs.files["trips.txt"])

    frames, shape_ids = builder._build_shapes(patterns)

    assert frames == [] and shape_ids == {}
    assert set(builder.report["reason"]) == {"NO_MATCH"}
    assert len(builder.report) == 2


def test_build_shapes_unknown_route_type_is_reported_not_attempted(sb):
    network = FakeNetwork()
    builder = make_builder(sb, network=network)
    pattern = make_pattern(sb, route_type_label=None)

    frames, shape_ids = builder._build_shapes({KEY_R1: pattern})

    assert frames == [] and shape_ids == {}
    assert network.calls == []

    row = builder.report.iloc[0]
    assert row["reason"] == "UNKNOWN_ROUTE_TYPE"
    assert "GTFS_ROUTE_TYPE_LABELS" in row["detail"]


def test_build_shapes_geometry_too_short(sb):
    network = FakeNetwork(handler=single_point)
    builder = make_builder(sb, network=network)

    frames, shape_ids = builder._build_shapes({KEY_R1: make_pattern(sb)})

    assert frames == [] and shape_ids == {}
    assert builder.report.iloc[0]["reason"] == "GEOMETRY_TOO_SHORT"


def test_build_shapes_includes_patterns_skipped_earlier(sb):
    stop_times = make_files()["stop_times.txt"]
    # T3 only has one stop -> skipped by _build_patterns
    stop_times = stop_times[~((stop_times.trip_id == "T3") & (stop_times.stop_id == "S2"))]
    gtfs = make_gtfs({"stop_times.txt": stop_times})
    builder, patterns = _patterns(sb, gtfs)

    _, shape_ids = builder._build_shapes(patterns)

    assert list(shape_ids) == [KEY_R1]
    assert builder.report["reason"].tolist() == ["TOO_FEW_STOPS"]


def test_build_shapes_prints_summary_of_failures(sb, capsys):
    network = FakeNetwork(handler=always_fail)
    builder = make_builder(sb, network=network)
    patterns = builder._build_patterns(builder.gtfs.files["trips.txt"])

    builder._build_shapes(patterns)

    out = capsys.readouterr().out
    assert "STOP_TOO_FAR" in out
    assert "patterns" in out and "trips" in out


# ----------------------------------------------------------------------
# _merge_into_gtfs
# ----------------------------------------------------------------------


def _merge_setup(sb, gtfs):
    builder = make_builder(sb, gtfs)
    missing = builder._trips_missing_shapes()
    patterns = builder._build_patterns(missing)
    frame = sb.GTFSShapeBuilder._coords_to_shape_frame(
        "osm_R1", [[46.0, 7.0], [46.0, 7.1], [46.0, 7.2]]
    )
    return builder, patterns, frame


def test_merge_creates_shapes_file_and_assigns_trips(sb):
    gtfs = make_gtfs()
    builder, patterns, frame = _merge_setup(sb, gtfs)

    builder._merge_into_gtfs([frame], {KEY_R1: "osm_R1"}, patterns)

    assert len(gtfs.files["shapes.txt"]) == 3

    trips = gtfs.files["trips.txt"].set_index("trip_id")
    assert trips.loc["T1", "shape_id"] == "osm_R1"
    assert trips.loc["T2", "shape_id"] == "osm_R1"
    assert pd.isna(trips.loc["T3", "shape_id"])


def test_merge_updates_files_without_validation(sb):
    gtfs = make_gtfs()
    builder, patterns, frame = _merge_setup(sb, gtfs)

    builder._merge_into_gtfs([frame], {KEY_R1: "osm_R1"}, patterns)

    assert ("shapes.txt", False) in gtfs.updates
    assert ("trips.txt", False) in gtfs.updates


def test_merge_appends_to_existing_shapes_and_keeps_existing_assignments(sb):
    trips = make_files()["trips.txt"]
    trips["shape_id"] = ["SH0", None, None]
    shapes = pd.DataFrame(
        {
            "shape_id": ["SH0", "SH0"],
            "shape_pt_lat": [46.0, 46.1],
            "shape_pt_lon": [7.0, 7.1],
            "shape_pt_sequence": [0, 1],
        }
    )
    gtfs = make_gtfs({"trips.txt": trips, "shapes.txt": shapes})
    builder, patterns, frame = _merge_setup(sb, gtfs)

    # T1 already has SH0, so only T2 is in the pattern.
    assert patterns[KEY_R1].trip_ids == ["T2"]

    builder._merge_into_gtfs([frame], {KEY_R1: "osm_R1"}, patterns)

    result = gtfs.files["shapes.txt"]
    assert len(result) == 5
    assert result["shape_id"].tolist()[:2] == ["SH0", "SH0"]
    assert set(result["shape_id"]) == {"SH0", "osm_R1"}

    trips = gtfs.files["trips.txt"].set_index("trip_id")
    assert trips.loc["T1", "shape_id"] == "SH0"
    assert trips.loc["T2", "shape_id"] == "osm_R1"
    assert pd.isna(trips.loc["T3", "shape_id"])


def test_merge_with_no_frames_leaves_shapes_untouched(sb):
    gtfs = make_gtfs()
    builder, patterns, _ = _merge_setup(sb, gtfs)

    builder._merge_into_gtfs([], {}, patterns)

    assert "shapes.txt" not in gtfs.files
    assert ("shapes.txt", False) not in gtfs.updates


# ----------------------------------------------------------------------
# build()
# ----------------------------------------------------------------------


def test_build_returns_feed_unchanged_when_nothing_is_missing(sb, capsys):
    trips = make_files()["trips.txt"]
    trips["shape_id"] = "SH1"
    shapes = pd.DataFrame({"shape_id": ["SH1"]})
    gtfs = make_gtfs({"trips.txt": trips, "shapes.txt": shapes})
    network = FakeNetwork()

    result = make_builder(sb, gtfs, network).build()

    assert result is gtfs
    assert gtfs.updates == []
    assert network.calls == []
    assert "Found 0 trips" in capsys.readouterr().out


def test_build_returns_feed_unchanged_when_no_shape_can_be_built(sb, capsys):
    gtfs = make_gtfs()
    network = FakeNetwork(handler=always_fail)

    builder = make_builder(sb, gtfs, network)
    result = builder.build()

    assert result is gtfs
    assert "shapes.txt" not in gtfs.files
    assert gtfs.updates == []
    assert "shape_id" not in gtfs.files["trips.txt"].columns
    assert "No shapes could be generated" in capsys.readouterr().out
    # the report is still available for inspection
    assert len(builder.report) == 2


def test_build_happy_path(sb, capsys):
    gtfs = make_gtfs()

    result = make_builder(sb, gtfs).build()

    assert result is gtfs

    shapes = gtfs.files["shapes.txt"]
    assert set(shapes["shape_id"]) == {"osm_R1", "osm_R2"}
    assert len(shapes) == 5

    trips = gtfs.files["trips.txt"].set_index("trip_id")["shape_id"]
    assert trips.to_dict() == {"T1": "osm_R1", "T2": "osm_R1", "T3": "osm_R2"}

    out = capsys.readouterr().out
    assert "Found 3 trips" in out
    assert "Found 2 unique stop patterns" in out
    assert "Assigned shapes to 3 trips" in out


def test_build_partial_success_only_assigns_matched_patterns(sb):
    gtfs = make_gtfs()
    builder = make_builder(sb, gtfs, FakeNetwork(handler=fail_for_bus))

    builder.build()

    trips = gtfs.files["trips.txt"].set_index("trip_id")["shape_id"]
    assert trips["T1"] == "osm_R1"
    assert trips["T2"] == "osm_R1"
    assert pd.isna(trips["T3"])
    assert set(gtfs.files["shapes.txt"]["shape_id"]) == {"osm_R1"}
    assert builder.report["route_id"].tolist() == ["R2"]


def test_build_only_touches_trips_without_a_valid_shape(sb):
    trips = make_files()["trips.txt"]
    trips["shape_id"] = ["SH0", None, None]
    shapes = pd.DataFrame(
        {
            "shape_id": ["SH0", "SH0"],
            "shape_pt_lat": [46.0, 46.1],
            "shape_pt_lon": [7.0, 7.1],
            "shape_pt_sequence": [0, 1],
        }
    )
    gtfs = make_gtfs({"trips.txt": trips, "shapes.txt": shapes})
    network = FakeNetwork()

    make_builder(sb, gtfs, network).build()

    result = gtfs.files["trips.txt"].set_index("trip_id")["shape_id"]
    assert result["T1"] == "SH0"
    assert result["T2"] == "osm_R1"
    assert result["T3"] == "osm_R2"
    assert "SH0" in set(gtfs.files["shapes.txt"]["shape_id"])
    # T1 was not re-matched: only the T2 and T3 patterns hit the network
    assert len(network.calls) == 2


def test_build_passes_max_stop_distance_to_the_network(sb):
    network = FakeNetwork()

    make_builder(sb, network=network, max_stop_distance_m=123.0).build()

    assert {c["max_stop_distance_m"] for c in network.calls} == {123.0}


def test_build_solves_each_unique_pattern_once(sb):
    network = FakeNetwork()

    make_builder(sb, network=network).build()

    # T1 and T2 share a pattern -> 2 network calls for 3 trips
    assert len(network.calls) == 2


# ----------------------------------------------------------------------
# build() on the real GTFS class (fixtures from conftest.py)
# ----------------------------------------------------------------------


def test_build_on_real_gtfs_feed(feed_a, sb):
    builder = sb.GTFSShapeBuilder(feed_a, FakeNetwork())

    result = builder.build()

    assert result is feed_a

    shapes = feed_a.files["shapes.txt"]
    assert set(shapes["shape_id"]) == {"osm_R1", "osm_R0"}

    trips = feed_a.files["trips.txt"].set_index("trip_id")["shape_id"]
    assert trips["T1"] == "osm_R1"
    assert trips["T0"] == "osm_R0"


def test_build_on_real_gtfs_feed_with_real_network(feed_a, sb):
    pytest.importorskip("transitforge")

    from shapely.geometry import LineString
    from transitforge.dataclasses import TransitRoute, TransitStop
    from transitforge.network import TransitRouteNetwork

    # One OSM train route between the two platforms of feed A (STA_1 / STB_1).
    a = (46.5170, 6.6323)
    b = (46.2102, 6.1427)
    route = TransitRoute(
        relation_id=1,
        route_type="train",
        ref="IC1",
        stops=[
            TransitStop(1, a[0], a[1], "Lausanne"),
            TransitStop(2, b[0], b[1], "Geneve"),
        ],
        geometry=LineString([(a[1], a[0]), (b[1], b[0])]),
    )
    network = TransitRouteNetwork({1: route})

    builder = sb.GTFSShapeBuilder(feed_a, network, diagnose=True)
    builder.build()

    shapes = feed_a.files["shapes.txt"]

    # IC1 matches by ref, IC0 by the type fallback: both trips get a shape.
    trips = feed_a.files["trips.txt"].set_index("trip_id")["shape_id"]
    assert trips["T1"] == "osm_R1"
    assert trips["T0"] == "osm_R0"

    one = shapes[shapes["shape_id"] == "osm_R1"].sort_values("shape_pt_sequence")
    assert one["shape_pt_lat"].iloc[0] == pytest.approx(a[0])
    assert one["shape_pt_lon"].iloc[0] == pytest.approx(a[1])
    assert one["shape_pt_lat"].iloc[-1] == pytest.approx(b[0])
    assert one["shape_pt_lon"].iloc[-1] == pytest.approx(b[1])
    # Lausanne - Geneve is roughly 50 km as the crow flies
    assert 45_000 < one["shape_dist_traveled"].iloc[-1] < 60_000
    assert builder.report.empty


def test_real_network_failure_is_diagnosed_in_the_report(feed_a, sb):
    pytest.importorskip("transitforge")

    from transitforge.network import TransitRouteNetwork

    # An empty network: nothing can match, every pattern lands in the report.
    builder = sb.GTFSShapeBuilder(feed_a, TransitRouteNetwork({}), diagnose=True)

    builder.build()

    assert "shapes.txt" not in feed_a.files
    assert len(builder.report) == 2
    assert set(builder.report["reason"]) == {"NO_CANDIDATES"}


# ----------------------------------------------------------------------
# Multiple worker processes
# ----------------------------------------------------------------------


@pytest.mark.skipif(
    "fork" not in multiprocessing.get_all_start_methods(),
    reason="workers > 1 needs the 'fork' start method (Linux/macOS/WSL)",
)
def test_multiple_workers_give_the_same_result_as_one(sb):
    single = make_gtfs()
    make_builder(sb, single, FakeNetwork(), workers=1).build()

    multi = make_gtfs()
    make_builder(sb, multi, FakeNetwork(), workers=2).build()

    pd.testing.assert_frame_equal(
        single.files["shapes.txt"].reset_index(drop=True),
        multi.files["shapes.txt"].reset_index(drop=True),
    )
    pd.testing.assert_frame_equal(
        single.files["trips.txt"], multi.files["trips.txt"]
    )


@pytest.mark.skipif(
    "fork" not in multiprocessing.get_all_start_methods(),
    reason="workers > 1 needs the 'fork' start method (Linux/macOS/WSL)",
)
def test_multiple_workers_report_failures_like_a_single_process(sb):
    single = make_builder(sb, network=FakeNetwork(handler=fail_for_bus), workers=1)
    single.build()

    multi = make_builder(sb, network=FakeNetwork(handler=fail_for_bus), workers=2)
    multi.build()

    pd.testing.assert_frame_equal(single.report, multi.report)