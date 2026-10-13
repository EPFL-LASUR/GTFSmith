"""
Tests for gtfsmith.gtfs_operations:

    - create_intercity_transfers
    - filter_gtfs_per_mode
    - filter_gtfs_per_bbox

The file is self-contained (it does not use the conftest fixtures) and builds
small, hand-made feeds so that every expected value can be verified by reading
the test.

Conventions
-----------
* The "robustness" tests (optional files missing, input feed not mutated, no
  dangling parent_station, ...) are regression tests for gaps that have been
  fixed in gtfs_operations.py.
* Transfer rows are identified through unique ``min_transfer_time`` values
  (101, 102, ...) so assertions stay short and readable.

Adjust the two imports below if your package layout differs.
"""

import math
import zipfile
from pathlib import Path

import pandas as pd
import pytest

from gtfsmith import gtfs_operations as ops
from gtfsmith.gtfs_class import GTFS


# ===========================================================================
# Feed-building helpers
# ===========================================================================

COLUMNS = {
    "agency.txt": ["agency_id", "agency_name", "agency_url", "agency_timezone"],
    "routes.txt": ["route_id", "agency_id", "route_short_name", "route_type"],
    "trips.txt": ["route_id", "service_id", "trip_id"],
    "stop_times.txt": ["trip_id", "arrival_time", "departure_time", "stop_id", "stop_sequence"],
    "stops.txt": ["stop_id", "stop_name", "stop_lat", "stop_lon", "location_type", "parent_station"],
    "calendar.txt": [
        "service_id", "monday", "tuesday", "wednesday", "thursday", "friday",
        "saturday", "sunday", "start_date", "end_date",
    ],
    "calendar_dates.txt": ["service_id", "date", "exception_type"],
    "transfers.txt": [
        "from_stop_id", "to_stop_id", "from_route_id", "to_route_id",
        "from_trip_id", "to_trip_id", "transfer_type", "min_transfer_time",
    ],
    "frequencies.txt": ["trip_id", "start_time", "end_time", "headway_secs", "exact_times"],
}

CORE_FILES = ("agency.txt", "routes.txt", "trips.txt", "stop_times.txt", "stops.txt", "calendar.txt")


def df(name, rows):
    return pd.DataFrame(rows, columns=COLUMNS[name])


def make_gtfs(files, fill=CORE_FILES):
    """Build a GTFS object; core files that are not given are filled with empty tables."""
    full = {name: df(name, []) for name in fill}
    full.update(files)
    return GTFS(full)


def cal(service_id, days=(1, 1, 1, 1, 1, 0, 0)):
    return (service_id, *days, 20240101, 20241231)


def write_zip(feed, path):
    with zipfile.ZipFile(path, "w") as zf:
        for name, table in feed.files.items():
            zf.writestr(name, table.to_csv(index=False))
    return path


def ids(table, column):
    return set(table[column])


def assert_integrity(gtfs):
    """Every reference in the feed must point to an existing row."""
    f = gtfs.files
    routes = ids(f["routes.txt"], "route_id")
    trips = ids(f["trips.txt"], "trip_id")
    stops = ids(f["stops.txt"], "stop_id")

    assert ids(f["trips.txt"], "route_id") <= routes, "trip references a missing route"
    assert ids(f["stop_times.txt"], "trip_id") <= trips, "stop_time references a missing trip"
    assert ids(f["stop_times.txt"], "stop_id") <= stops, "stop_time references a missing stop"

    services = set()
    for name in ("calendar.txt", "calendar_dates.txt"):
        if name in f:
            services |= ids(f[name], "service_id")
    assert ids(f["trips.txt"], "service_id") <= services, "trip references a missing service"

    parents = {p for p in f["stops.txt"]["parent_station"].dropna() if p != ""}
    assert parents <= stops, "stop references a missing parent_station"


# ===========================================================================
# create_intercity_transfers
# ===========================================================================

def haversine_m(lat1, lon1, lat2, lon2):
    r = 6_371_000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = p2 - p1
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def expected_time(a, b, speed=20.0, penalty_min=10.0):
    """a, b: (lat, lon). Same formula as the docstring of create_intercity_transfers."""
    return haversine_m(*a, *b) / 1000.0 / speed * 3600.0 + penalty_min * 60.0


def station_feed(stations, standalone=()):
    """
    stations:   {station_id: (lat, lon, [platform_id, ...])}
    standalone: [(stop_id, lat, lon), ...]  location_type 0 stops without parent
    """
    rows = []
    for sid, (lat, lon, platforms) in stations.items():
        rows.append((sid, sid, lat, lon, 1, ""))
        for p in platforms:
            rows.append((p, p, lat, lon, 0, sid))
    for sid, lat, lon in standalone:
        rows.append((sid, sid, lat, lon, 0, ""))
    return make_gtfs({"stops.txt": df("stops.txt", rows)})


def cluster(prefix, lat0, lon0, dlats, platforms=1):
    """Stations on a N-S line: {prefix_A: (lat0 + dlat, lon0, [prefix_A_1, ...])}"""
    out = {}
    for letter, dlat in zip("ABCDEFG", dlats):
        sid = f"{prefix}_{letter}"
        out[sid] = (lat0 + dlat, lon0, [f"{sid}_{i + 1}" for i in range(platforms)])
    return out


# Three stations ~4.4 km apart in a row (A-C ~8.9 km): one blob, all within 10 km
PARIS = cluster("P", 48.85, 2.35, (0.0, 0.04, 0.08))
LYON = cluster("L", 45.76, 4.84, (0.0, 0.04, 0.08))


def pairs(transfers):
    return set(zip(transfers["from_stop_id"], transfers["to_stop_id"]))


def time_of(transfers, a, b):
    row = transfers[(transfers["from_stop_id"] == a) & (transfers["to_stop_id"] == b)]
    assert len(row) == 1, f"expected exactly one transfer {a} -> {b}, got {len(row)}"
    return int(row.iloc[0]["min_transfer_time"])


EXPECTED_TRANSFER_COLUMNS = ["from_stop_id", "to_stop_id", "transfer_type", "min_transfer_time"]

CAP_S = 3600       # create_intercity_transfers drops transfers longer than this
MARGIN_S = 150     # projection/rounding slack around the cap


class TestCreateIntercityTransfers:

    def test_returns_dataframe_with_expected_columns(self):
        result = ops.create_intercity_transfers(station_feed(PARIS))

        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == EXPECTED_TRANSFER_COLUMNS

    def test_every_ordered_pair_in_a_cluster_is_connected(self):
        result = ops.create_intercity_transfers(station_feed(PARIS))

        platforms = ["P_A_1", "P_B_1", "P_C_1"]
        expected = {(a, b) for a in platforms for b in platforms if a != b}
        assert pairs(result) == expected

    def test_transfer_type_is_minimum_time_transfer(self):
        result = ops.create_intercity_transfers(station_feed(PARIS))

        assert (result["transfer_type"] == 2).all()

    def test_transfer_times_are_integers(self):
        result = ops.create_intercity_transfers(station_feed(PARIS))

        assert pd.api.types.is_integer_dtype(result["min_transfer_time"])

    def test_transfers_are_symmetric(self):
        result = ops.create_intercity_transfers(station_feed(PARIS))

        for a, b in pairs(result):
            assert time_of(result, a, b) == time_of(result, b, a)

    def test_no_duplicate_rows(self):
        result = ops.create_intercity_transfers(station_feed(PARIS, ))

        assert not result.duplicated(subset=["from_stop_id", "to_stop_id"]).any()

    def test_no_self_transfers(self):
        result = ops.create_intercity_transfers(station_feed(PARIS))

        assert (result["from_stop_id"] != result["to_stop_id"]).all()

    def test_transfers_connect_platforms_not_parent_stations(self):
        result = ops.create_intercity_transfers(station_feed(PARIS))

        stations = set(PARIS)
        assert not (set(result["from_stop_id"]) | set(result["to_stop_id"])) & stations

    # ---- transfer-time formula -------------------------------------------------

    @pytest.mark.parametrize(
        "speed, penalty",
        [(20.0, 10.0), (10.0, 10.0), (40.0, 5.0), (20.0, 0.0), (15.0, 25.0)],
    )
    def test_transfer_time_follows_documented_formula(self, speed, penalty):
        result = ops.create_intercity_transfers(
            station_feed(PARIS), mean_tp_speed=speed, interchange_penalty_min=penalty
        )

        checked = 0
        for sa, sb in [("P_A", "P_B"), ("P_B", "P_C"), ("P_A", "P_C")]:
            a, b = PARIS[sa][:2], PARIS[sb][:2]
            expected = expected_time(a, b, speed, penalty)
            pair = (f"{sa}_1", f"{sb}_1")

            if expected <= CAP_S - MARGIN_S:
                # 3 % tolerance: the function measures in EPSG:3035, not on the sphere
                assert time_of(result, *pair) == pytest.approx(expected, rel=0.03)
                checked += 1
            elif expected >= CAP_S + MARGIN_S:
                # over the one-hour cap -> must have been dropped
                assert pair not in pairs(result)
            # inside the margin around the cap either outcome is legitimate

        assert checked > 0

    def test_penalty_is_added_in_minutes(self):
        low = ops.create_intercity_transfers(station_feed(PARIS), interchange_penalty_min=10.0)
        high = ops.create_intercity_transfers(station_feed(PARIS), interchange_penalty_min=20.0)

        assert time_of(high, "P_A_1", "P_B_1") - time_of(low, "P_A_1", "P_B_1") == 600

    def test_doubling_speed_halves_travel_component(self):
        slow = ops.create_intercity_transfers(
            station_feed(PARIS), mean_tp_speed=20.0, interchange_penalty_min=0.0
        )
        fast = ops.create_intercity_transfers(
            station_feed(PARIS), mean_tp_speed=40.0, interchange_penalty_min=0.0
        )

        assert time_of(slow, "P_A_1", "P_C_1") == pytest.approx(2 * time_of(fast, "P_A_1", "P_C_1"), abs=1)

    def test_transfers_longer_than_one_hour_are_dropped(self):
        # penalty alone (61 min) already exceeds the 3600 s cap
        result = ops.create_intercity_transfers(station_feed(PARIS), interchange_penalty_min=61.0)

        assert result.empty
        assert list(result.columns) == EXPECTED_TRANSFER_COLUMNS

    def test_all_returned_transfers_are_at_most_one_hour(self):
        result = ops.create_intercity_transfers(
            station_feed(PARIS), interchange_penalty_min=45.0, mean_tp_speed=5.0, max_dist_km=10.0
        )

        assert (result["min_transfer_time"] <= 3600).all()

    # ---- blobs / thresholds ------------------------------------------------------

    def test_blob_below_minimum_station_count_yields_nothing(self):
        two = {k: PARIS[k] for k in ("P_A", "P_B")}

        result = ops.create_intercity_transfers(station_feed(two))  # default min = 3

        assert result.empty
        assert list(result.columns) == EXPECTED_TRANSFER_COLUMNS

    def test_lowering_min_stations_per_blob_enables_small_blobs(self):
        two = {k: PARIS[k] for k in ("P_A", "P_B")}

        result = ops.create_intercity_transfers(station_feed(two), min_stations_per_blob=2)

        assert pairs(result) == {("P_A_1", "P_B_1"), ("P_B_1", "P_A_1")}

    def test_single_station_yields_nothing(self):
        one = {"P_A": PARIS["P_A"]}

        result = ops.create_intercity_transfers(station_feed(one), min_stations_per_blob=1)

        assert result.empty

    def test_max_dist_excludes_far_pairs_inside_the_same_blob(self):
        # A-B ~8 km, B-C ~8 km (one blob), but A-C ~16 km > max_dist 10 km
        chain = cluster("P", 48.85, 2.35, (0.0, 0.072, 0.144))

        result = ops.create_intercity_transfers(station_feed(chain), max_dist_km=10.0)

        assert pairs(result) == {
            ("P_A_1", "P_B_1"), ("P_B_1", "P_A_1"),
            ("P_B_1", "P_C_1"), ("P_C_1", "P_B_1"),
        }

    def test_raising_max_dist_connects_far_pairs_in_the_same_blob(self):
        chain = cluster("P", 48.85, 2.35, (0.0, 0.072, 0.144))

        result = ops.create_intercity_transfers(station_feed(chain), max_dist_km=20.0)

        assert ("P_A_1", "P_C_1") in pairs(result)
        assert ("P_C_1", "P_A_1") in pairs(result)

    def test_small_blob_buffer_keeps_stations_in_separate_blobs(self):
        # two stations ~15 km apart: buffers of 5 km do not touch -> separate blobs
        far = {
            "P_A": (48.85, 2.35, ["P_A_1"]),
            "P_B": (48.985, 2.35, ["P_B_1"]),
        }

        result = ops.create_intercity_transfers(
            station_feed(far), blob_buffer_km=5.0, min_stations_per_blob=1, max_dist_km=20.0
        )

        assert result.empty

    def test_large_blob_buffer_merges_stations_into_one_blob(self):
        far = {
            "P_A": (48.85, 2.35, ["P_A_1"]),
            "P_B": (48.985, 2.35, ["P_B_1"]),
        }

        result = ops.create_intercity_transfers(
            station_feed(far), blob_buffer_km=8.0, min_stations_per_blob=2, max_dist_km=20.0
        )

        assert pairs(result) == {("P_A_1", "P_B_1"), ("P_B_1", "P_A_1")}

    def test_distant_cities_are_never_linked(self):
        result = ops.create_intercity_transfers(
            station_feed({**PARIS, **LYON}), max_dist_km=1000.0
        )

        paris_platforms = {p for _, _, plats in PARIS.values() for p in plats}
        lyon_platforms = {p for _, _, plats in LYON.values() for p in plats}
        for a, b in pairs(result):
            assert (a in paris_platforms) == (b in paris_platforms)
            assert (a in lyon_platforms) == (b in lyon_platforms)

    def test_each_city_is_connected_internally(self):
        result = ops.create_intercity_transfers(station_feed({**PARIS, **LYON}))

        # 3 platforms per city -> 6 ordered pairs per city
        assert len(result) == 12

    # ---- platform expansion ----------------------------------------------------

    def test_station_pair_is_expanded_to_cross_product_of_platforms(self):
        stations = {
            "P_A": (48.85, 2.35, ["A1", "A2"]),
            "P_B": (48.89, 2.35, ["B1", "B2", "B3"]),
            "P_C": (48.93, 2.35, []),  # childless: counts for the blob, yields no rows
        }

        result = ops.create_intercity_transfers(station_feed(stations))

        a_to_b = {(a, b) for a in ("A1", "A2") for b in ("B1", "B2", "B3")}
        b_to_a = {(b, a) for a, b in a_to_b}
        assert pairs(result) == a_to_b | b_to_a
        assert len(result) == 12

    def test_childless_station_counts_towards_blob_size(self):
        stations = {
            "P_A": (48.85, 2.35, ["A1"]),
            "P_B": (48.89, 2.35, ["B1"]),
            "P_C": (48.93, 2.35, []),
        }

        result = ops.create_intercity_transfers(station_feed(stations))  # min=3 satisfied by P_C

        assert pairs(result) == {("A1", "B1"), ("B1", "A1")}

    def test_station_without_platforms_on_both_sides_yields_nothing(self):
        stations = {k: (v[0], v[1], []) for k, v in PARIS.items()}

        result = ops.create_intercity_transfers(station_feed(stations))

        assert result.empty
        assert list(result.columns) == EXPECTED_TRANSFER_COLUMNS

    def test_standalone_stops_are_ignored(self):
        feed = station_feed(PARIS, standalone=[("BUS1", 48.86, 2.35), ("BUS2", 48.87, 2.35)])

        result = ops.create_intercity_transfers(feed)

        assert not {"BUS1", "BUS2"} & (set(result["from_stop_id"]) | set(result["to_stop_id"]))
        assert len(result) == 6

    def test_all_platforms_of_a_station_share_the_station_transfer_time(self):
        stations = {
            "P_A": (48.85, 2.35, ["A1", "A2"]),
            "P_B": (48.89, 2.35, ["B1", "B2"]),
            "P_C": (48.93, 2.35, ["C1"]),
        }

        result = ops.create_intercity_transfers(station_feed(stations))

        times = {time_of(result, a, b) for a in ("A1", "A2") for b in ("B1", "B2")}
        assert len(times) == 1

    # ---- robustness / API ------------------------------------------------------

    def test_input_feed_is_not_mutated(self):
        feed = station_feed(PARIS)
        before = feed.files["stops.txt"].copy()

        ops.create_intercity_transfers(feed)

        pd.testing.assert_frame_equal(feed.files["stops.txt"], before)

    def test_feed_without_stations_returns_empty_frame(self):
        feed = station_feed({}, standalone=[("BUS1", 48.86, 2.35)])

        result = ops.create_intercity_transfers(feed)

        assert result.empty
        assert list(result.columns) == EXPECTED_TRANSFER_COLUMNS

    @pytest.mark.parametrize("as_type", [str, Path])
    def test_accepts_a_path_to_a_zip(self, tmp_path, as_type):
        zip_path = write_zip(station_feed(PARIS), tmp_path / "feed.zip")

        from_path = ops.create_intercity_transfers(as_type(zip_path))
        from_obj = ops.create_intercity_transfers(station_feed(PARIS))

        assert pairs(from_path) == pairs(from_obj)

    def test_is_deterministic(self):
        first = ops.create_intercity_transfers(station_feed({**PARIS, **LYON}))
        second = ops.create_intercity_transfers(station_feed({**PARIS, **LYON}))

        pd.testing.assert_frame_equal(
            first.sort_values(["from_stop_id", "to_stop_id"]).reset_index(drop=True),
            second.sort_values(["from_stop_id", "to_stop_id"]).reset_index(drop=True),
        )


# ===========================================================================
# Shared feeds for the two filter functions
# ===========================================================================

def mode_feed(drop=()):
    """
    Three modes on separate stops:
        rail (type 2): trip TR1  X1 -> Y1        service SV1   (stations ST_X, ST_Y)
        bus  (type 3): trips TB1 B1 -> B2, TB2 B2 -> B1   service SV2
        tram (type 0): trip TT1  T1 -> T2        service SV3
    Transfers carry unique min_transfer_time ids (101..107).
    """
    files = {
        "agency.txt": df("agency.txt", [("A1", "Agency", "https://example.org", "Europe/Zurich")]),
        "routes.txt": df("routes.txt", [
            ("RB", "A1", "B", 3), ("RT", "A1", "T", 0), ("RR", "A1", "R", 2),
        ]),
        "trips.txt": df("trips.txt", [
            ("RR", "SV1", "TR1"),
            ("RB", "SV2", "TB1"),
            ("RB", "SV2", "TB2"),
            ("RT", "SV3", "TT1"),
        ]),
        "stops.txt": df("stops.txt", [
            ("ST_X", "Station X", 46.21, 6.14, 1, ""),
            ("X1", "X platform 1", 46.21, 6.14, 0, "ST_X"),
            ("X2", "X platform 2", 46.21, 6.14, 0, "ST_X"),   # never served
            ("ST_Y", "Station Y", 46.30, 6.20, 1, ""),
            ("Y1", "Y platform 1", 46.30, 6.20, 0, "ST_Y"),
            ("B1", "Bus 1", 46.22, 6.15, 0, ""),
            ("B2", "Bus 2", 46.23, 6.16, 0, ""),
            ("T1", "Tram 1", 46.24, 6.17, 0, ""),
            ("T2", "Tram 2", 46.25, 6.18, 0, ""),
        ]),
        "stop_times.txt": df("stop_times.txt", [
            ("TR1", "08:00:00", "08:00:00", "X1", 1), ("TR1", "09:00:00", "09:00:00", "Y1", 2),
            ("TB1", "08:00:00", "08:00:00", "B1", 1), ("TB1", "08:10:00", "08:10:00", "B2", 2),
            ("TB2", "09:00:00", "09:00:00", "B2", 1), ("TB2", "09:10:00", "09:10:00", "B1", 2),
            ("TT1", "07:00:00", "07:00:00", "T1", 1), ("TT1", "07:05:00", "07:05:00", "T2", 2),
        ]),
        "calendar.txt": df("calendar.txt", [cal("SV1"), cal("SV2"), cal("SV3")]),
        "calendar_dates.txt": df("calendar_dates.txt", [
            ("SV1", "20240501", 2), ("SV2", "20240501", 2), ("SV3", "20240501", 2),
        ]),
        "frequencies.txt": df("frequencies.txt", [
            ("TR1", "06:00:00", "10:00:00", 600, 0),
            ("TB1", "06:00:00", "10:00:00", 300, 0),
            ("TT1", "06:00:00", "10:00:00", 900, 0),
        ]),
        "transfers.txt": df("transfers.txt", [
            # from_stop, to_stop, from_route, to_route, from_trip, to_trip, type, id
            ("X1", "Y1", "", "", "", "", 2, 101),        # rail only
            ("B1", "B2", "", "", "", "", 2, 102),        # bus only
            ("X1", "B1", "", "", "", "", 2, 103),        # rail <-> bus
            ("B2", "B1", "", "", "TB1", "TB2", 1, 104),  # bus, trip-specific
            ("B1", "B2", "", "", "TR1", "", 2, 105),     # bus stops but a rail trip
            ("B2", "B1", "RB", "RB", "", "", 2, 106),    # bus, route-specific
            ("B1", "B2", "RB", "RT", "", "", 2, 107),    # bus stops, bus -> tram route
        ]),
    }
    for name in drop:
        files.pop(name, None)
    return make_gtfs(files, fill=())


MODE_EXPECTED = {
    3: dict(
        routes={"RB"}, trips={"TB1", "TB2"}, stops={"B1", "B2"},
        services={"SV2"}, transfers={102, 104, 106}, freq_trips={"TB1"},
    ),
    2: dict(
        routes={"RR"}, trips={"TR1"}, stops={"ST_X", "X1", "ST_Y", "Y1"},
        services={"SV1"}, transfers={101}, freq_trips={"TR1"},
    ),
    0: dict(
        routes={"RT"}, trips={"TT1"}, stops={"T1", "T2"},
        services={"SV3"}, transfers=set(), freq_trips={"TT1"},
    ),
}


# ===========================================================================
# filter_gtfs_per_mode
# ===========================================================================

class TestFilterPerMode:

    def test_returns_a_gtfs_object(self):
        assert isinstance(ops.filter_gtfs_per_mode(mode_feed(), 3), GTFS)

    @pytest.mark.parametrize("route_type", sorted(MODE_EXPECTED))
    def test_routes_are_filtered_by_type(self, route_type):
        result = ops.filter_gtfs_per_mode(mode_feed(), route_type)

        assert ids(result.files["routes.txt"], "route_id") == MODE_EXPECTED[route_type]["routes"]
        assert (result.files["routes.txt"]["route_type"] == route_type).all()

    @pytest.mark.parametrize("route_type", sorted(MODE_EXPECTED))
    def test_trips_and_stop_times_follow_the_routes(self, route_type):
        result = ops.filter_gtfs_per_mode(mode_feed(), route_type)

        expected = MODE_EXPECTED[route_type]["trips"]
        assert ids(result.files["trips.txt"], "trip_id") == expected
        assert ids(result.files["stop_times.txt"], "trip_id") == expected

    @pytest.mark.parametrize("route_type", sorted(MODE_EXPECTED))
    def test_only_used_stops_and_their_parents_are_kept(self, route_type):
        result = ops.filter_gtfs_per_mode(mode_feed(), route_type)

        assert ids(result.files["stops.txt"], "stop_id") == MODE_EXPECTED[route_type]["stops"]

    def test_unserved_platform_of_a_kept_station_is_dropped(self):
        result = ops.filter_gtfs_per_mode(mode_feed(), 2)

        stops = ids(result.files["stops.txt"], "stop_id")
        assert "ST_X" in stops and "X1" in stops
        assert "X2" not in stops

    @pytest.mark.parametrize("route_type", sorted(MODE_EXPECTED))
    def test_calendar_and_calendar_dates_keep_only_used_services(self, route_type):
        result = ops.filter_gtfs_per_mode(mode_feed(), route_type)

        expected = MODE_EXPECTED[route_type]["services"]
        assert ids(result.files["calendar.txt"], "service_id") == expected
        assert ids(result.files["calendar_dates.txt"], "service_id") == expected

    @pytest.mark.parametrize("route_type", sorted(MODE_EXPECTED))
    def test_transfers_require_every_reference_to_survive(self, route_type):
        result = ops.filter_gtfs_per_mode(mode_feed(), route_type)

        kept = set(result.files["transfers.txt"]["min_transfer_time"])
        assert kept == MODE_EXPECTED[route_type]["transfers"]

    def test_transfer_with_a_trip_from_another_mode_is_dropped(self):
        result = ops.filter_gtfs_per_mode(mode_feed(), 3)

        # id 105: bus stops, but from_trip_id is a rail trip
        assert 105 not in set(result.files["transfers.txt"]["min_transfer_time"])

    def test_transfer_with_a_route_from_another_mode_is_dropped(self):
        result = ops.filter_gtfs_per_mode(mode_feed(), 3)

        # id 107: bus stops, but to_route_id is a tram route
        assert 107 not in set(result.files["transfers.txt"]["min_transfer_time"])

    @pytest.mark.parametrize("route_type", sorted(MODE_EXPECTED))
    def test_frequencies_follow_the_kept_trips(self, route_type):
        result = ops.filter_gtfs_per_mode(mode_feed(), route_type)

        assert ids(result.files["frequencies.txt"], "trip_id") == MODE_EXPECTED[route_type]["freq_trips"]

    def test_agency_table_is_kept_whole(self):
        original = mode_feed().files["agency.txt"]

        result = ops.filter_gtfs_per_mode(mode_feed(), 3)

        pd.testing.assert_frame_equal(result.files["agency.txt"].reset_index(drop=True), original)

    @pytest.mark.parametrize("route_type", sorted(MODE_EXPECTED))
    def test_result_is_referentially_consistent(self, route_type):
        assert_integrity(ops.filter_gtfs_per_mode(mode_feed(), route_type))

    def test_row_content_of_kept_rows_is_unchanged(self):
        original = mode_feed().files["stop_times.txt"]

        result = ops.filter_gtfs_per_mode(mode_feed(), 3)

        expected = original[original["trip_id"].isin({"TB1", "TB2"})].reset_index(drop=True)
        pd.testing.assert_frame_equal(
            result.files["stop_times.txt"].reset_index(drop=True), expected
        )

    def test_unknown_route_type_gives_empty_tables(self):
        result = ops.filter_gtfs_per_mode(mode_feed(), 11)

        for name in ("routes.txt", "trips.txt", "stop_times.txt", "stops.txt"):
            assert result.files[name].empty, name

    def test_optional_calendar_txt_may_be_absent(self):
        result = ops.filter_gtfs_per_mode(mode_feed(drop=("calendar.txt",)), 3)

        assert "calendar.txt" not in result.files
        assert ids(result.files["calendar_dates.txt"], "service_id") == {"SV2"}

    def test_optional_frequencies_txt_may_be_absent(self):
        result = ops.filter_gtfs_per_mode(mode_feed(drop=("frequencies.txt",)), 3)

        assert "frequencies.txt" not in result.files
        assert ids(result.files["trips.txt"], "trip_id") == {"TB1", "TB2"}

    @pytest.mark.parametrize("as_type", [str, Path])
    def test_accepts_a_path_to_a_zip(self, tmp_path, as_type):
        zip_path = write_zip(mode_feed(), tmp_path / "feed.zip")

        from_path = ops.filter_gtfs_per_mode(as_type(zip_path), 3)
        from_obj = ops.filter_gtfs_per_mode(mode_feed(), 3)

        assert ids(from_path.files["trips.txt"], "trip_id") == ids(from_obj.files["trips.txt"], "trip_id")
        assert ids(from_path.files["stops.txt"], "stop_id") == ids(from_obj.files["stops.txt"], "stop_id")

    def test_output_path_writes_the_filtered_feed(self, tmp_path):
        out = tmp_path / "bus_only.zip"

        ops.filter_gtfs_per_mode(mode_feed(), 3, output_path=str(out))

        assert out.exists()
        if zipfile.is_zipfile(out):
            with zipfile.ZipFile(out) as zf:
                assert "trips.txt" in zf.namelist()

    def test_no_output_is_written_without_output_path(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)

        ops.filter_gtfs_per_mode(mode_feed(), 3)

        assert list(tmp_path.iterdir()) == []

    # ---- robustness / regression tests ---------------------------------------------

    def test_input_feed_is_not_mutated(self):
        feed = mode_feed()
        before = set(feed.files)

        ops.filter_gtfs_per_mode(feed, 3)

        assert set(feed.files) == before

    def test_feed_without_transfers_txt_is_supported(self):
        result = ops.filter_gtfs_per_mode(mode_feed(drop=("transfers.txt",)), 3)

        assert ids(result.files["trips.txt"], "trip_id") == {"TB1", "TB2"}
        assert "transfers.txt" not in result.files

    def test_feed_without_calendar_dates_txt_is_supported(self):
        result = ops.filter_gtfs_per_mode(mode_feed(drop=("calendar_dates.txt",)), 3)

        assert ids(result.files["calendar.txt"], "service_id") == {"SV2"}
        assert "calendar_dates.txt" not in result.files

    def test_feed_without_parent_station_column_is_supported(self):
        feed = mode_feed()
        feed.files["stops.txt"] = feed.files["stops.txt"].drop(columns="parent_station")

        result = ops.filter_gtfs_per_mode(feed, 3)

        assert ids(result.files["stops.txt"], "stop_id") == {"B1", "B2"}


# ===========================================================================
# filter_gtfs_per_bbox
# ===========================================================================

GENEVA_BBOX = (6.0, 46.0, 6.3, 46.4)       # (min_lon, min_lat, max_lon, max_lat)
ZURICH_BBOX = (8.4, 47.3, 8.7, 47.5)
WORLD_BBOX = (-180.0, -90.0, 180.0, 90.0)


def bbox_feed(drop=()):
    """
    Geneva: station GEP (platforms GE1, GE2), standalone EDGE exactly on the
            GENEVA_BBOX corner (min_lon / max_lat).
    Zurich: station ZHP (platforms ZH1, ZH2).

        T_GE  (RG, SV1)  GE1 -> GE2
        T_ZH  (RZ, SV2)  ZH1 -> ZH2
        T_IC  (RI, SV3)  GE1 -> ZH1      (crosses the bounding box)
    """
    files = {
        "agency.txt": df("agency.txt", [("A1", "Agency", "https://example.org", "Europe/Zurich")]),
        "routes.txt": df("routes.txt", [
            ("RG", "A1", "G", 3), ("RZ", "A1", "Z", 3), ("RI", "A1", "I", 2),
        ]),
        "trips.txt": df("trips.txt", [
            ("RG", "SV1", "T_GE"), ("RZ", "SV2", "T_ZH"), ("RI", "SV3", "T_IC"),
        ]),
        "stops.txt": df("stops.txt", [
            ("GEP", "Geneva station", 46.21, 6.14, 1, ""),
            ("GE1", "Geneva platform 1", 46.2101, 6.1401, 0, "GEP"),
            ("GE2", "Geneva platform 2", 46.22, 6.15, 0, "GEP"),
            ("EDGE", "Edge stop", 46.4, 6.0, 0, ""),
            ("ZHP", "Zurich station", 47.378, 8.540, 1, ""),
            ("ZH1", "Zurich platform 1", 47.3781, 8.5401, 0, "ZHP"),
            ("ZH2", "Zurich platform 2", 47.38, 8.55, 0, "ZHP"),
        ]),
        "stop_times.txt": df("stop_times.txt", [
            ("T_GE", "08:00:00", "08:00:00", "GE1", 1), ("T_GE", "08:10:00", "08:10:00", "GE2", 2),
            ("T_ZH", "08:00:00", "08:00:00", "ZH1", 1), ("T_ZH", "08:10:00", "08:10:00", "ZH2", 2),
            ("T_IC", "10:00:00", "10:00:00", "GE1", 1), ("T_IC", "12:00:00", "12:00:00", "ZH1", 2),
        ]),
        "calendar.txt": df("calendar.txt", [cal("SV1"), cal("SV2"), cal("SV3")]),
        "calendar_dates.txt": df("calendar_dates.txt", [
            ("SV1", "20240501", 2), ("SV2", "20240501", 2), ("SV3", "20240501", 2),
        ]),
        "frequencies.txt": df("frequencies.txt", [
            ("T_GE", "06:00:00", "10:00:00", 600, 0),
            ("T_ZH", "06:00:00", "10:00:00", 600, 0),
        ]),
        "transfers.txt": df("transfers.txt", [
            ("GE1", "GE2", "", "", "", "", 2, 201),            # inside Geneva
            ("ZH1", "ZH2", "", "", "", "", 2, 202),            # inside Zurich
            ("GE1", "ZH1", "", "", "", "", 2, 203),            # crosses the box
            ("GE1", "GE2", "", "", "T_GE", "T_IC", 1, 204),    # trips both kept
            ("GE1", "GE2", "", "", "T_ZH", "", 2, 205),        # trip dropped
        ]),
    }
    for name in drop:
        files.pop(name, None)
    return make_gtfs(files, fill=())


def _pairs(table, a, b):
    return set(zip(table[a], table[b]))


class TestFilterPerBbox:

    def test_returns_a_gtfs_object(self):
        assert isinstance(ops.filter_gtfs_per_bbox(bbox_feed(), GENEVA_BBOX), GTFS)

    def test_only_stops_inside_the_box_are_kept(self):
        result = ops.filter_gtfs_per_bbox(bbox_feed(), GENEVA_BBOX)

        assert ids(result.files["stops.txt"], "stop_id") == {"GEP", "GE1", "GE2", "EDGE"}

    def test_box_edges_are_inclusive(self):
        # EDGE sits exactly on min_lon (6.0) and max_lat (46.4)
        result = ops.filter_gtfs_per_bbox(bbox_feed(), GENEVA_BBOX)

        assert "EDGE" in ids(result.files["stops.txt"], "stop_id")

    def test_trips_with_at_least_one_stop_inside_are_kept(self):
        result = ops.filter_gtfs_per_bbox(bbox_feed(), GENEVA_BBOX)

        assert ids(result.files["trips.txt"], "trip_id") == {"T_GE", "T_IC"}

    def test_stop_times_outside_the_box_are_dropped(self):
        result = ops.filter_gtfs_per_bbox(bbox_feed(), GENEVA_BBOX)

        assert _pairs(result.files["stop_times.txt"], "trip_id", "stop_id") == {
            ("T_GE", "GE1"), ("T_GE", "GE2"), ("T_IC", "GE1"),
        }

    def test_routes_and_services_follow_the_kept_trips(self):
        result = ops.filter_gtfs_per_bbox(bbox_feed(), GENEVA_BBOX)

        assert ids(result.files["routes.txt"], "route_id") == {"RG", "RI"}
        assert ids(result.files["calendar.txt"], "service_id") == {"SV1", "SV3"}
        assert ids(result.files["calendar_dates.txt"], "service_id") == {"SV1", "SV3"}

    def test_transfers_require_stops_and_trips_to_survive(self):
        result = ops.filter_gtfs_per_bbox(bbox_feed(), GENEVA_BBOX)

        assert set(result.files["transfers.txt"]["min_transfer_time"]) == {201, 204}

    def test_frequencies_follow_the_kept_trips(self):
        result = ops.filter_gtfs_per_bbox(bbox_feed(), GENEVA_BBOX)

        assert ids(result.files["frequencies.txt"], "trip_id") == {"T_GE"}

    def test_agency_table_is_kept_whole(self):
        original = bbox_feed().files["agency.txt"]

        result = ops.filter_gtfs_per_bbox(bbox_feed(), GENEVA_BBOX)

        pd.testing.assert_frame_equal(result.files["agency.txt"].reset_index(drop=True), original)

    def test_second_region(self):
        result = ops.filter_gtfs_per_bbox(bbox_feed(), ZURICH_BBOX)

        assert ids(result.files["stops.txt"], "stop_id") == {"ZHP", "ZH1", "ZH2"}
        assert ids(result.files["trips.txt"], "trip_id") == {"T_ZH", "T_IC"}
        assert ids(result.files["routes.txt"], "route_id") == {"RZ", "RI"}
        assert ids(result.files["calendar.txt"], "service_id") == {"SV2", "SV3"}
        assert set(result.files["transfers.txt"]["min_transfer_time"]) == {202}

    @pytest.mark.parametrize("bbox", [GENEVA_BBOX, ZURICH_BBOX, WORLD_BBOX])
    def test_result_is_referentially_consistent(self, bbox):
        assert_integrity(ops.filter_gtfs_per_bbox(bbox_feed(), bbox))

    def test_box_covering_everything_keeps_everything(self):
        original = bbox_feed()

        result = ops.filter_gtfs_per_bbox(bbox_feed(), WORLD_BBOX)

        for name in ("stops.txt", "stop_times.txt", "trips.txt", "routes.txt"):
            assert len(result.files[name]) == len(original.files[name]), name
        assert set(result.files["transfers.txt"]["min_transfer_time"]) == {201, 202, 203, 204, 205}

    def test_box_without_any_stop_raises(self):
        with pytest.raises(ValueError, match="does not contain any GTFS stop"):
            ops.filter_gtfs_per_bbox(bbox_feed(), (0.0, 0.0, 1.0, 1.0))

    def test_bbox_order_is_lon_lat_not_lat_lon(self):
        # swapping the axes puts the Geneva box in the wrong hemisphere of the plane
        swapped = (GENEVA_BBOX[1], GENEVA_BBOX[0], GENEVA_BBOX[3], GENEVA_BBOX[2])

        with pytest.raises(ValueError):
            ops.filter_gtfs_per_bbox(bbox_feed(), swapped)

    def test_optional_calendar_txt_may_be_absent(self):
        result = ops.filter_gtfs_per_bbox(bbox_feed(drop=("calendar.txt",)), GENEVA_BBOX)

        assert "calendar.txt" not in result.files
        assert ids(result.files["calendar_dates.txt"], "service_id") == {"SV1", "SV3"}

    def test_optional_frequencies_txt_may_be_absent(self):
        result = ops.filter_gtfs_per_bbox(bbox_feed(drop=("frequencies.txt",)), GENEVA_BBOX)

        assert "frequencies.txt" not in result.files

    def test_feed_without_parent_station_column_is_supported(self):
        feed = bbox_feed()
        feed.files["stops.txt"] = feed.files["stops.txt"].drop(columns="parent_station")

        result = ops.filter_gtfs_per_bbox(feed, GENEVA_BBOX)

        assert ids(result.files["stops.txt"], "stop_id") == {"GEP", "GE1", "GE2", "EDGE"}

    @pytest.mark.parametrize("as_type", [str, Path])
    def test_accepts_a_path_to_a_zip(self, tmp_path, as_type):
        zip_path = write_zip(bbox_feed(), tmp_path / "feed.zip")

        from_path = ops.filter_gtfs_per_bbox(as_type(zip_path), GENEVA_BBOX)
        from_obj = ops.filter_gtfs_per_bbox(bbox_feed(), GENEVA_BBOX)

        assert ids(from_path.files["trips.txt"], "trip_id") == ids(from_obj.files["trips.txt"], "trip_id")
        assert ids(from_path.files["stops.txt"], "stop_id") == ids(from_obj.files["stops.txt"], "stop_id")

    def test_output_path_writes_the_filtered_feed(self, tmp_path):
        out = tmp_path / "geneva.zip"

        ops.filter_gtfs_per_bbox(bbox_feed(), GENEVA_BBOX, output_path=str(out))

        assert out.exists()
        if zipfile.is_zipfile(out):
            with zipfile.ZipFile(out) as zf:
                assert "stops.txt" in zf.namelist()

    def test_no_output_is_written_without_output_path(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)

        ops.filter_gtfs_per_bbox(bbox_feed(), GENEVA_BBOX)

        assert list(tmp_path.iterdir()) == []

    # ---- robustness / regression tests ---------------------------------------------

    def test_input_feed_is_not_mutated(self):
        feed = bbox_feed()
        before = set(feed.files)

        ops.filter_gtfs_per_bbox(feed, GENEVA_BBOX)

        assert set(feed.files) == before

    def test_feed_without_transfers_txt_is_supported(self):
        result = ops.filter_gtfs_per_bbox(bbox_feed(drop=("transfers.txt",)), GENEVA_BBOX)

        assert ids(result.files["trips.txt"], "trip_id") == {"T_GE", "T_IC"}
        assert "transfers.txt" not in result.files

    def test_feed_without_calendar_dates_txt_is_supported(self):
        result = ops.filter_gtfs_per_bbox(bbox_feed(drop=("calendar_dates.txt",)), GENEVA_BBOX)

        assert ids(result.files["calendar.txt"], "service_id") == {"SV1", "SV3"}
        assert "calendar_dates.txt" not in result.files

    def test_parent_station_outside_the_box_does_not_leave_dangling_reference(self):
        feed = bbox_feed()
        feed.files["stops.txt"] = pd.concat([
            feed.files["stops.txt"],
            df("stops.txt", [
                ("PX", "Parent outside", 46.2, 6.5, 1, ""),
                ("CX", "Child inside", 46.2, 6.2, 0, "PX"),
            ]),
        ], ignore_index=True)
        feed.files["trips.txt"] = pd.concat(
            [feed.files["trips.txt"], df("trips.txt", [("RG", "SV1", "T_X")])], ignore_index=True
        )
        feed.files["stop_times.txt"] = pd.concat(
            [feed.files["stop_times.txt"], df("stop_times.txt", [("T_X", "09:00:00", "09:00:00", "CX", 1)])],
            ignore_index=True,
        )

        result = ops.filter_gtfs_per_bbox(feed, GENEVA_BBOX)

        assert_integrity(result)
        assert "PX" in ids(result.files["stops.txt"], "stop_id")


# ===========================================================================
# Extra robustness: stop hierarchies deeper than one level, sparse transfers
# ===========================================================================

def add_rows(feed, name, rows):
    feed.files[name] = pd.concat([feed.files[name], df(name, rows)], ignore_index=True)


class TestFilterRobustness:

    def test_mode_filter_keeps_the_whole_ancestor_chain(self):
        # boarding area (type 4) -> platform -> station, only the boarding area is served
        feed = mode_feed()
        add_rows(feed, "stops.txt", [
            ("ST_Z", "Z station", 46.4, 6.4, 1, ""),
            ("PL_Z", "Z platform", 46.4, 6.4, 0, "ST_Z"),
            ("BA_Z", "Z boarding area", 46.4, 6.4, 4, "PL_Z"),
        ])
        add_rows(feed, "trips.txt", [("RB", "SV2", "TB3")])
        add_rows(feed, "stop_times.txt", [("TB3", "10:00:00", "10:00:00", "BA_Z", 1)])

        result = ops.filter_gtfs_per_mode(feed, 3)

        assert {"BA_Z", "PL_Z", "ST_Z"} <= ids(result.files["stops.txt"], "stop_id")
        assert_integrity(result)

    def test_bbox_filter_keeps_the_whole_ancestor_chain(self):
        feed = bbox_feed()
        add_rows(feed, "stops.txt", [
            ("ST_O", "Outside station", 46.2, 6.6, 1, ""),
            ("PL_O", "Outside platform", 46.2, 6.6, 0, "ST_O"),
            ("BA", "Inside boarding area", 46.2, 6.2, 4, "PL_O"),
        ])
        add_rows(feed, "trips.txt", [("RG", "SV1", "T_BA")])
        add_rows(feed, "stop_times.txt", [("T_BA", "09:00:00", "09:00:00", "BA", 1)])

        result = ops.filter_gtfs_per_bbox(feed, GENEVA_BBOX)

        assert {"BA", "PL_O", "ST_O"} <= ids(result.files["stops.txt"], "stop_id")
        assert_integrity(result)

    def test_mode_filter_handles_transfers_with_only_the_required_columns(self):
        feed = mode_feed()
        feed.files["transfers.txt"] = pd.DataFrame({
            "from_stop_id": ["B1", "X1"],
            "to_stop_id": ["B2", "Y1"],
            "transfer_type": [2, 2],
            "min_transfer_time": [101, 102],
        })

        result = ops.filter_gtfs_per_mode(feed, 3)

        assert list(result.files["transfers.txt"]["min_transfer_time"]) == [101]

    def test_bbox_filter_handles_transfers_with_only_the_required_columns(self):
        feed = bbox_feed()
        feed.files["transfers.txt"] = pd.DataFrame({
            "from_stop_id": ["GE1", "ZH1"],
            "to_stop_id": ["GE2", "ZH2"],
            "transfer_type": [2, 2],
            "min_transfer_time": [201, 202],
        })

        result = ops.filter_gtfs_per_bbox(feed, GENEVA_BBOX)

        assert list(result.files["transfers.txt"]["min_transfer_time"]) == [201]

    def test_filtered_result_does_not_share_memory_with_the_input(self):
        feed = mode_feed()

        result = ops.filter_gtfs_per_mode(feed, 3)
        result.files["agency.txt"].loc[0, "agency_name"] = "CHANGED"
        result.files["trips.txt"].loc[result.files["trips.txt"].index[0], "service_id"] = "CHANGED"

        assert feed.files["agency.txt"].loc[0, "agency_name"] == "Agency"
        assert "CHANGED" not in set(feed.files["trips.txt"]["service_id"])

    def test_filtering_twice_gives_the_same_result(self):
        feed = bbox_feed()

        first = ops.filter_gtfs_per_bbox(feed, GENEVA_BBOX)
        second = ops.filter_gtfs_per_bbox(feed, GENEVA_BBOX)

        for name in first.files:
            pd.testing.assert_frame_equal(first.files[name], second.files[name])