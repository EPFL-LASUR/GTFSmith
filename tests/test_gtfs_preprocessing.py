"""
Tests for the stop-cleanup module:

    - merge_duplicated_physical_stops
    - prefix_stop_ids
    - create_parent_stations
    - add_missing_parent_station

Self-contained (does not use the conftest fixtures): small hand-made feeds
make every expected value checkable by reading the test.

Tests marked ``xfail(strict=True)`` document REAL bugs found by reading the
code. They are strict on purpose: once a bug is fixed the test XPASSes, pytest
reports that as a failure, and you simply delete the marker.

Adjust the two imports below to your package layout (the module name in
particular: I did not know what this file is called).
"""

import zipfile
from pathlib import Path

import pandas as pd
import pytest

from gtfsmith import gtfs_preprocessing as ops
from gtfsmith.gtfs_class import GTFS


# ===========================================================================
# Helpers
# ===========================================================================

STOP_COLUMNS = ["stop_id", "stop_name", "stop_lat", "stop_lon", "location_type", "parent_station"]
STOP_TIME_COLUMNS = ["trip_id", "arrival_time", "departure_time", "stop_id", "stop_sequence"]
TRANSFER_COLUMNS = ["from_stop_id", "to_stop_id", "transfer_type", "min_transfer_time"]
TRIP_COLUMNS = ["route_id", "service_id", "trip_id"]


def stops_df(rows):
    return pd.DataFrame(rows, columns=STOP_COLUMNS)



def make_feed(stops, stop_times=(), transfers=None, trips=None):
    files = {
        "agency.txt": pd.DataFrame([
            {
                "agency_id": "AGENCY",
                "agency_name": "Test Agency",
                "agency_url": "https://example.com",
                "agency_timezone": "Europe/Paris",
            }
        ]),
        "routes.txt": pd.DataFrame([
            {
                "route_id": "R1",
                "agency_id": "AGENCY",
                "route_short_name": "R1",
                "route_long_name": "Test Route",
                "route_type": 3,
            }
        ]),
        "stops.txt": stops,
        "stop_times.txt": pd.DataFrame(list(stop_times), columns=STOP_TIME_COLUMNS),
    }

    if trips is None:
        trip_ids = (files["stop_times.txt"]["trip_id"].dropna().unique())
        trips = [("R1", "SV1", trip_id) for trip_id in trip_ids]

    files["trips.txt"] = pd.DataFrame(list(trips), columns=TRIP_COLUMNS)

    if transfers is not None:
        files["transfers.txt"] = pd.DataFrame(list(transfers), columns=TRANSFER_COLUMNS)

    return GTFS(files)

def write_zip(feed, path):
    with zipfile.ZipFile(path, "w") as zf:
        for name, table in feed.files.items():
            zf.writestr(name, table.to_csv(index=False))
    return path


def snapshot(feed):
    return {name: table.copy() for name, table in feed.files.items()}


def assert_feed_unchanged(feed, before):
    assert set(feed.files) == set(before)
    for name, table in before.items():
        pd.testing.assert_frame_equal(feed.files[name], table)


def stops_of(feed):
    return feed.files["stops.txt"]


def non_empty(series):
    return {v for v in series.dropna() if v != ""}


def parent_map(stops):
    """{stop_id: parent_station} for stops that have a parent."""
    pairs = zip(stops["stop_id"], stops["parent_station"])
    return {s: p for s, p in pairs if pd.notna(p) and p != ""}


def assert_valid_hierarchy(feed):
    """Unique stop_ids and no reference to a missing stop anywhere."""
    stops = stops_of(feed)
    stop_ids = set(stops["stop_id"])

    assert not stops["stop_id"].duplicated().any(), "duplicate stop_ids"
    assert non_empty(stops["parent_station"]) <= stop_ids, "dangling parent_station"

    if "stop_times.txt" in feed.files:
        assert set(feed.files["stop_times.txt"]["stop_id"]) <= stop_ids, "stop_time -> missing stop"

    if "transfers.txt" in feed.files:
        transfers = feed.files["transfers.txt"]
        for column in ("from_stop_id", "to_stop_id"):
            assert non_empty(transfers[column]) <= stop_ids, f"transfer {column} -> missing stop"


def hier_feed():
    """Two stations with platforms plus one standalone stop, trips and transfers."""
    stops = stops_df([
        ("S1", "Station 1", 46.20, 6.14, 1, ""),
        ("S1a", "Platform 1a", 46.20, 6.14, 0, "S1"),
        ("S1b", "Platform 1b", 46.20, 6.14, 0, "S1"),
        ("S2", "Station 2", 46.30, 6.20, 1, ""),
        ("S2a", "Platform 2a", 46.30, 6.20, 0, "S2"),
        ("LONE", "Lone stop", 46.40, 6.30, 0, ""),
    ])
    stop_times = [
        ("T1", "08:00:00", "08:00:00", "S1a", 1),
        ("T1", "09:00:00", "09:00:00", "S2a", 2),
        ("T2", "10:00:00", "10:00:00", "LONE", 1),
    ]
    transfers = [
        ("S1a", "S2a", 2, 300),
        ("S1", "S2", 2, 600),
    ]
    trips = [("R1", "SV1", "T1"), ("R1", "SV1", "T2")]
    return make_feed(stops, stop_times, transfers, trips)


# ===========================================================================
# prefix_stop_ids
# ===========================================================================

class TestPrefixStopIds:

    def test_returns_a_gtfs_object(self):
        assert isinstance(ops.prefix_stop_ids(hier_feed(), "it"), GTFS)

    def test_every_stop_id_is_prefixed(self):
        original = set(stops_of(hier_feed())["stop_id"])

        result = ops.prefix_stop_ids(hier_feed(), "it")

        assert set(stops_of(result)["stop_id"]) == {f"it:{s}" for s in original}

    def test_row_count_and_other_columns_are_preserved(self):
        original = stops_of(hier_feed())

        result = stops_of(ops.prefix_stop_ids(hier_feed(), "it"))

        assert len(result) == len(original)
        for column in ("stop_name", "stop_lat", "stop_lon", "location_type"):
            assert list(result[column]) == list(original[column])

    def test_parent_station_references_follow_the_new_ids(self):
        original = parent_map(stops_of(hier_feed()))

        result = parent_map(stops_of(ops.prefix_stop_ids(hier_feed(), "it")))

        assert result == {f"it:{c}": f"it:{p}" for c, p in original.items()}

    def test_empty_parent_station_is_not_prefixed(self):
        result = stops_of(ops.prefix_stop_ids(hier_feed(), "it"))

        values = set(result["parent_station"].dropna())
        assert "it:" not in values
        assert "" in values  # parentless stops keep an empty value

    def test_missing_parent_station_stays_missing(self):
        feed = hier_feed()
        feed.files["stops.txt"].loc[feed.files["stops.txt"]["stop_id"] == "LONE", "parent_station"] = None

        result = stops_of(ops.prefix_stop_ids(feed, "it"))

        lone = result[result["stop_id"] == "it:LONE"].iloc[0]
        assert pd.isna(lone["parent_station"])

    def test_stop_times_are_remapped(self):
        result = ops.prefix_stop_ids(hier_feed(), "it")

        stop_times = result.files["stop_times.txt"]
        assert list(stop_times["stop_id"]) == ["it:S1a", "it:S2a", "it:LONE"]
        assert list(stop_times["trip_id"]) == ["T1", "T1", "T2"]
        assert list(stop_times["stop_sequence"]) == [1, 2, 1]

    def test_transfers_are_remapped(self):
        result = ops.prefix_stop_ids(hier_feed(), "it")

        transfers = result.files["transfers.txt"]
        assert list(transfers["from_stop_id"]) == ["it:S1a", "it:S1"]
        assert list(transfers["to_stop_id"]) == ["it:S2a", "it:S2"]
        assert list(transfers["min_transfer_time"]) == [300, 600]

    def test_feed_without_transfers_is_supported(self):
        feed = hier_feed()
        feed.files.pop("transfers.txt")

        result = ops.prefix_stop_ids(feed, "it")

        assert "transfers.txt" not in result.files
        assert set(stops_of(result)["stop_id"]) == {f"it:{s}" for s in
                                                    ["S1", "S1a", "S1b", "S2", "S2a", "LONE"]}

    def test_feed_without_parent_station_column_is_supported(self):
        feed = make_feed(
            stops_df([("A", "Alpha", 46.2, 6.1, 0, "")]).drop(columns="parent_station"),
            [("T1", "08:00:00", "08:00:00", "A", 1)],
        )

        result = ops.prefix_stop_ids(feed, "it")

        assert list(stops_of(result)["stop_id"]) == ["it:A"]
        assert list(result.files["stop_times.txt"]["stop_id"]) == ["it:A"]

    def test_integer_stop_ids_become_prefixed_strings(self):
        feed = make_feed(
            stops_df([(1, "One", 46.2, 6.1, 0, ""), (2, "Two", 46.3, 6.2, 0, "")])
            .drop(columns="parent_station"),
            [("T1", "08:00:00", "08:00:00", 1, 1), ("T1", "09:00:00", "09:00:00", 2, 2)],
        )

        result = ops.prefix_stop_ids(feed, "it")

        assert list(stops_of(result)["stop_id"]) == ["it:1", "it:2"]
        assert list(result.files["stop_times.txt"]["stop_id"]) == ["it:1", "it:2"]

    @pytest.mark.parametrize(
        "prefix, expected",
        [("it", "it:"), ("it:", "it:"), ("it::", "it:"), ("a:b", "a:b:"), (7, "7:")],
    )
    def test_prefix_is_normalised_to_end_with_exactly_one_colon(self, prefix, expected):
        result = stops_of(ops.prefix_stop_ids(hier_feed(), prefix))

        assert all(s.startswith(expected) for s in result["stop_id"])
        assert not any(s.startswith(expected + ":") for s in result["stop_id"])

    def test_result_is_referentially_consistent(self):
        assert_valid_hierarchy(ops.prefix_stop_ids(hier_feed(), "it"))

    def test_unrelated_tables_are_untouched(self):
        original_trips = hier_feed().files["trips.txt"]

        result = ops.prefix_stop_ids(hier_feed(), "it")

        pd.testing.assert_frame_equal(result.files["trips.txt"], original_trips)

    def test_prefixing_twice_composes(self):
        once = ops.prefix_stop_ids(hier_feed(), "a")
        twice = ops.prefix_stop_ids(once, "b")

        assert set(stops_of(twice)["stop_id"]) == {
            f"b:a:{s}" for s in ["S1", "S1a", "S1b", "S2", "S2a", "LONE"]
        }
        assert_valid_hierarchy(twice)

    def test_prefixed_feeds_from_different_sources_cannot_collide(self):
        a = ops.prefix_stop_ids(hier_feed(), "a")
        b = ops.prefix_stop_ids(hier_feed(), "b")

        assert not set(stops_of(a)["stop_id"]) & set(stops_of(b)["stop_id"])

    @pytest.mark.parametrize("as_type", [str, Path])
    def test_accepts_a_path_to_a_zip(self, tmp_path, as_type):
        zip_path = write_zip(hier_feed(), tmp_path / "feed.zip")

        result = ops.prefix_stop_ids(as_type(zip_path), "it")

        assert set(stops_of(result)["stop_id"]) == {f"it:{s}" for s in
                                                    ["S1", "S1a", "S1b", "S2", "S2a", "LONE"]}

    def test_output_path_writes_the_feed(self, tmp_path):
        out = tmp_path / "prefixed.zip"

        ops.prefix_stop_ids(hier_feed(), "it", output_path=out)

        assert out.exists()
        if zipfile.is_zipfile(out):
            with zipfile.ZipFile(out) as zf:
                assert "stops.txt" in zf.namelist()

    def test_no_output_is_written_without_output_path(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)

        ops.prefix_stop_ids(hier_feed(), "it")

        assert list(tmp_path.iterdir()) == []



# ===========================================================================
# create_parent_stations
# ===========================================================================

def flat_feed(location_type=0, parent_station=""):
    """A flat feed (no hierarchy), as produced by e.g. Spanish / Italian operators."""
    stops = stops_df([
        ("A", "Alpha", 46.20, 6.10, location_type, parent_station),
        ("B", "Beta", 46.30, 6.20, location_type, parent_station),
        ("C", "Gamma", 46.40, 6.30, location_type, parent_station),
    ])
    stops["wheelchair_boarding"] = [1, 2, 0]
    stop_times = [
        ("T1", "08:00:00", "08:00:00", "A", 1),
        ("T1", "09:00:00", "09:00:00", "B", 2),
    ]
    transfers = [("A", "B", 2, 120)]
    return make_feed(stops, stop_times, transfers)


@pytest.fixture(params=[(0, ""), (None, None)], ids=["explicit-zeros", "missing-values"])
def flat(request):
    return flat_feed(*request.param)


class TestCreateParentStations:

    def test_returns_a_gtfs_object(self, flat):
        assert isinstance(ops.create_parent_stations(flat), GTFS)

    def test_one_parent_is_created_per_stop(self, flat):
        result = stops_of(ops.create_parent_stations(flat))

        assert len(result) == 6
        assert (result["location_type"] == 1).sum() == 3
        assert (result["location_type"] == 0).sum() == 3

    def test_parent_id_is_Par_followed_by_the_child_id(self, flat):
        # NB: the docstring says 'Par:' + id, the code produces 'Par' + id.
        result = stops_of(ops.create_parent_stations(flat))

        parents = result[result["location_type"] == 1]
        assert set(parents["stop_id"]) == {"ParA", "ParB", "ParC"}

    def test_children_keep_their_ids_and_point_to_their_parent(self, flat):
        result = stops_of(ops.create_parent_stations(flat))

        children = result[result["location_type"] == 0]
        assert parent_map(children) == {"A": "ParA", "B": "ParB", "C": "ParC"}

    def test_parents_have_no_parent_themselves(self, flat):
        result = stops_of(ops.create_parent_stations(flat))

        parents = result[result["location_type"] == 1]
        assert all(p == "" or pd.isna(p) for p in parents["parent_station"])

    def test_parent_copies_name_coordinates_and_other_attributes(self, flat):
        result = stops_of(ops.create_parent_stations(flat)).set_index("stop_id")

        for child, parent in [("A", "ParA"), ("B", "ParB"), ("C", "ParC")]:
            for column in ("stop_name", "stop_lat", "stop_lon", "wheelchair_boarding"):
                assert result.loc[parent, column] == result.loc[child, column], (child, column)

    def test_child_ids_names_and_coordinates_are_preserved(self, flat):
        original = stops_of(flat).set_index("stop_id")

        result = stops_of(ops.create_parent_stations(flat)).set_index("stop_id")

        for child in ("A", "B", "C"):
            for column in ("stop_name", "stop_lat", "stop_lon"):
                assert result.loc[child, column] == original.loc[child, column]

    def test_stop_times_and_transfers_still_resolve(self, flat):
        result = ops.create_parent_stations(flat)

        assert_valid_hierarchy(result)

    def test_stop_times_and_transfers_are_unchanged(self, flat):
        before = snapshot(flat)

        result = ops.create_parent_stations(flat)

        pd.testing.assert_frame_equal(result.files["stop_times.txt"], before["stop_times.txt"])
        pd.testing.assert_frame_equal(result.files["transfers.txt"], before["transfers.txt"])

    def test_result_has_a_clean_unique_index(self, flat):
        result = stops_of(ops.create_parent_stations(flat))

        assert list(result.index) == list(range(len(result)))
        assert not result["stop_id"].duplicated().any()

    def test_feed_without_hierarchy_columns_is_supported(self):
        feed = flat_feed()
        feed.files["stops.txt"] = feed.files["stops.txt"].drop(columns=["location_type", "parent_station"])

        result = stops_of(ops.create_parent_stations(feed))

        assert (result["location_type"] == 1).sum() == 3
        assert parent_map(result[result["location_type"] == 0]) == {"A": "ParA", "B": "ParB", "C": "ParC"}

    def test_result_works_with_parent_based_functions(self, flat):
        # the whole point of the function: downstream code needs a 2-level hierarchy
        result = ops.create_parent_stations(flat)

        stops = stops_of(result)
        parents = set(stops.loc[stops["location_type"] == 1, "stop_id"])
        assert set(parent_map(stops).values()) == parents

    @pytest.mark.parametrize("as_type", [str, Path])
    def test_accepts_a_path_to_a_zip(self, tmp_path, as_type):
        zip_path = write_zip(flat_feed(), tmp_path / "feed.zip")

        result = ops.create_parent_stations(as_type(zip_path))

        assert len(stops_of(result)) == 6

    def test_output_file_writes_the_feed(self, tmp_path, flat):
        out = tmp_path / "with_parents.zip"

        ops.create_parent_stations(flat, output_path=str(out))

        assert out.exists()

    def test_no_output_is_written_without_output_file(self, tmp_path, monkeypatch, flat):
        monkeypatch.chdir(tmp_path)

        ops.create_parent_stations(flat)

        assert list(tmp_path.iterdir()) == []



# ===========================================================================
# add_missing_parent_station
# ===========================================================================

def austrian_feed():
    """
    Three platforms of two physical stations without a parent (location_type and
    parent_station missing) + one complete station/platform pair that must be left alone.
    """
    stops = stops_df([
        ("at:41:3351:0:1", "Hall 1", 47.0000, 15.0000, None, None),
        ("at:41:3351:0:4", "Hall 4", 47.0002, 15.0004, None, None),
        ("at:42:7:0:2", "Other", 48.0000, 16.0000, None, None),
        ("DE:1", "Existing station", 50.0, 8.0, 1, None),
        ("DE:1:a", "Existing platform", 50.0, 8.0, 0, "DE:1"),
    ])
    stops["wheelchair_boarding"] = [1, 1, 2, 0, 0]
    stop_times = [
        ("T1", "08:00:00", "08:00:00", "at:41:3351:0:1", 1),
        ("T1", "09:00:00", "09:00:00", "at:42:7:0:2", 2),
    ]
    return make_feed(stops, stop_times)


def by_id(feed):
    return stops_of(feed).set_index("stop_id")


class TestAddMissingParentStation:

    def test_returns_a_gtfs_object_with_stops(self):
        result = ops.add_missing_parent_station(austrian_feed())

        assert isinstance(result, GTFS)
        assert "stops.txt" in result.files

    def test_one_parent_per_derived_group_is_added(self):
        result = stops_of(ops.add_missing_parent_station(austrian_feed()))

        assert len(result) == 5 + 2
        assert {"at:41:3351", "at:42:7"} <= set(result["stop_id"])

    def test_parent_id_is_derived_by_dropping_the_last_two_id_segments(self):
        result = by_id(ops.add_missing_parent_station(austrian_feed()))

        assert result.loc["at:41:3351:0:1", "parent_station"] == "at:41:3351"
        assert result.loc["at:41:3351:0:4", "parent_station"] == "at:41:3351"
        assert result.loc["at:42:7:0:2", "parent_station"] == "at:42:7"

    def test_children_become_regular_stops(self):
        result = by_id(ops.add_missing_parent_station(austrian_feed()))

        for child in ("at:41:3351:0:1", "at:41:3351:0:4", "at:42:7:0:2"):
            assert result.loc[child, "location_type"] == 0

    def test_new_parents_are_stations_without_a_parent(self):
        result = by_id(ops.add_missing_parent_station(austrian_feed()))

        for parent in ("at:41:3351", "at:42:7"):
            assert result.loc[parent, "location_type"] == 1
            assert pd.isna(result.loc[parent, "parent_station"])

    def test_parent_name_is_taken_from_the_first_child(self):
        result = by_id(ops.add_missing_parent_station(austrian_feed()))

        assert result.loc["at:41:3351", "stop_name"] == "Hall 1"
        assert result.loc["at:42:7", "stop_name"] == "Other"

    def test_parent_position_is_the_mean_of_its_children(self):
        result = by_id(ops.add_missing_parent_station(austrian_feed()))

        assert result.loc["at:41:3351", "stop_lat"] == pytest.approx(47.0001)
        assert result.loc["at:41:3351", "stop_lon"] == pytest.approx(15.0002)
        assert result.loc["at:42:7", "stop_lat"] == pytest.approx(48.0)

    def test_other_columns_of_new_parents_are_missing(self):
        result = by_id(ops.add_missing_parent_station(austrian_feed()))

        assert pd.isna(result.loc["at:41:3351", "wheelchair_boarding"])

    def test_column_set_and_order_are_preserved(self):
        original = list(stops_of(austrian_feed()).columns)

        result = stops_of(ops.add_missing_parent_station(austrian_feed()))

        assert list(result.columns) == original

    def test_rows_that_already_have_a_hierarchy_are_untouched(self):
        result = by_id(ops.add_missing_parent_station(austrian_feed()))

        assert result.loc["DE:1", "location_type"] == 1
        assert pd.isna(result.loc["DE:1", "parent_station"])
        assert result.loc["DE:1:a", "location_type"] == 0
        assert result.loc["DE:1:a", "parent_station"] == "DE:1"
        assert result.loc["DE:1", "stop_name"] == "Existing station"

    def test_child_ids_names_and_coordinates_are_preserved(self):
        original = by_id(austrian_feed())

        result = by_id(ops.add_missing_parent_station(austrian_feed()))

        for child in ("at:41:3351:0:1", "at:41:3351:0:4", "at:42:7:0:2"):
            for column in ("stop_name", "stop_lat", "stop_lon"):
                assert result.loc[child, column] == original.loc[child, column]

    def test_stop_times_still_resolve(self):
        assert_valid_hierarchy(ops.add_missing_parent_station(austrian_feed()))

    def test_stop_times_are_unchanged(self):
        before = austrian_feed().files["stop_times.txt"]

        result = ops.add_missing_parent_station(austrian_feed())

        pd.testing.assert_frame_equal(result.files["stop_times.txt"], before)

    def test_result_has_a_clean_unique_index(self):
        result = stops_of(ops.add_missing_parent_station(austrian_feed()))

        assert list(result.index) == list(range(len(result)))

    def test_running_twice_changes_nothing_the_second_time(self):
        once = ops.add_missing_parent_station(austrian_feed())
        before = stops_of(once).copy()

        twice = ops.add_missing_parent_station(once)

        pd.testing.assert_frame_equal(stops_of(twice), before)

    @pytest.mark.parametrize("as_type", [str, Path])
    def test_accepts_a_path_to_a_zip(self, tmp_path, as_type):
        zip_path = write_zip(austrian_feed(), tmp_path / "feed.zip")

        result = ops.add_missing_parent_station(as_type(zip_path))

        assert {"at:41:3351", "at:42:7"} <= set(stops_of(result)["stop_id"])

    def test_output_file_writes_the_feed(self, tmp_path):
        out = tmp_path / "with_parents.zip"

        ops.add_missing_parent_station(austrian_feed(), output_path=str(out))

        assert out.exists()

    def test_feed_that_needs_no_change_is_returned_intact(self):
        feed = make_feed(stops_df([
            ("S1", "Station", 46.2, 6.1, 1, None),
            ("S1a", "Platform", 46.2, 6.1, 0, "S1"),
        ]))
        original = stops_of(feed).copy()

        result = ops.add_missing_parent_station(feed)

        assert isinstance(result, GTFS)
        pd.testing.assert_frame_equal(stops_of(result), original)

    # ---- known gaps ---------------------------------------------------------

    def test_stop_id_without_separators_does_not_create_duplicate_ids(self):
        feed = make_feed(stops_df([("123", "Plain", 47.0, 15.0, None, None)]))

        result = ops.add_missing_parent_station(feed)

        assert not stops_of(result)["stop_id"].duplicated().any()


    def test_existing_station_with_the_derived_id_is_reused(self):
        feed = make_feed(stops_df([
            ("at:41:3351", "Existing parent", 47.0, 15.0, 1, None),
            ("at:41:3351:0:1", "Hall 1", 47.0, 15.0, None, None),
        ]))

        result = ops.add_missing_parent_station(feed)

        assert not stops_of(result)["stop_id"].duplicated().any()
        assert parent_map(stops_of(result)) == {"at:41:3351:0:1": "at:41:3351"}



# ===========================================================================
# merge_duplicated_physical_stops
# ===========================================================================

# 0.0009 degrees of latitude ~ 100 m
DLAT_100M = 0.0009


def dup_feed(order=("P1", "P2", "P3"), p2_offset=DLAT_100M, extra_stops=()):
    """
    P1 "Paris Montparnasse Hall 1 - 2"   (children P1_a, P1_b)
    P2 "Paris Montparnasse Vaugirard"    p2_offset north of P1  (child P2_a)
    P3 "Lyon Part-Dieu"                  hundreds of km away     (child P3_a)
    """
    lat = 48.8400
    stations = {
        "P1": ("Paris Montparnasse Hall 1 - 2", lat, 2.3190, ["P1_a", "P1_b"]),
        "P2": ("Paris Montparnasse Vaugirard", lat + p2_offset, 2.3190, ["P2_a"]),
        "P3": ("Lyon Part-Dieu", 45.7606, 4.8590, ["P3_a"]),
    }
    rows = []
    for sid in order:
        name, la, lo, children = stations[sid]
        rows.append((sid, name, la, lo, 1, ""))
        for child in children:
            rows.append((child, f"{name} platform", la, lo, 0, sid))
    rows.extend(extra_stops)

    stop_times = [
        ("T1", "08:00:00", "08:00:00", "P1_a", 1),
        ("T1", "09:00:00", "09:00:00", "P2_a", 2),
        ("T2", "10:00:00", "10:00:00", "P3_a", 1),
    ]
    transfers = [
        ("P2", "P3", 2, 600),     # references the (future) losing parent
        ("P1_a", "P3_a", 2, 900),
    ]
    return make_feed(stops_df(rows), stop_times, transfers)


def parents_of(feed):
    stops = stops_of(feed)
    return set(stops.loc[stops["location_type"] == 1, "stop_id"])


class TestMergeDuplicatedPhysicalStops:

    # ---- validation -----------------------------------------------------------

    def test_missing_stops_txt_raises(self):
        feed = dup_feed()
        feed.files.pop("stops.txt")

        with pytest.raises(ValueError, match="stops.txt"):
            ops.merge_duplicated_physical_stops(feed)

    @pytest.mark.parametrize("column", ["stop_id", "location_type", "parent_station", "stop_lat", "stop_lon"])
    def test_missing_required_column_raises(self, column):
        feed = dup_feed()
        feed.files["stops.txt"] = feed.files["stops.txt"].drop(columns=column)

        with pytest.raises(ValueError, match="missing required columns"):
            ops.merge_duplicated_physical_stops(feed)

    @pytest.mark.parametrize("radius", [0, 0.0, -1, -300.0])
    def test_non_positive_radius_raises(self, radius):
        with pytest.raises(ValueError, match="greater than zero"):
            ops.merge_duplicated_physical_stops(dup_feed(), radius_m=radius)

    # ---- nothing to merge -----------------------------------------------------

    def test_distant_stations_are_left_alone(self):
        feed = dup_feed(p2_offset=0.0045)  # ~500 m, default radius 300 m
        before = stops_of(feed).copy()

        result = ops.merge_duplicated_physical_stops(feed)

        pd.testing.assert_frame_equal(stops_of(result), before)

    def test_returns_a_gtfs_object_even_when_nothing_is_merged(self):
        assert isinstance(ops.merge_duplicated_physical_stops(dup_feed(p2_offset=0.0045)), GTFS)

    # ---- merging --------------------------------------------------------------

    def test_returns_a_gtfs_object(self):
        assert isinstance(ops.merge_duplicated_physical_stops(dup_feed()), GTFS)

    def test_first_parent_in_stops_txt_wins(self):
        result = ops.merge_duplicated_physical_stops(dup_feed())

        assert parents_of(result) == {"P1", "P3"}

    def test_order_in_stops_txt_decides_the_winner(self):
        result = ops.merge_duplicated_physical_stops(dup_feed(order=("P2", "P1", "P3")))

        assert parents_of(result) == {"P2", "P3"}

    def test_losing_parent_is_removed(self):
        result = ops.merge_duplicated_physical_stops(dup_feed())

        assert "P2" not in set(stops_of(result)["stop_id"])

    def test_children_of_the_losing_parent_are_reparented_to_the_winner(self):
        result = ops.merge_duplicated_physical_stops(dup_feed())

        assert parent_map(stops_of(result))["P2_a"] == "P1"

    def test_child_stop_ids_are_preserved(self):
        original_children = {"P1_a", "P1_b", "P2_a", "P3_a"}

        result = ops.merge_duplicated_physical_stops(dup_feed())

        assert original_children <= set(stops_of(result)["stop_id"])

    def test_children_of_the_winner_and_of_other_stations_are_untouched(self):
        result = parent_map(stops_of(ops.merge_duplicated_physical_stops(dup_feed())))

        assert result["P1_a"] == "P1"
        assert result["P1_b"] == "P1"
        assert result["P3_a"] == "P3"

    def test_winner_keeps_its_own_attributes(self):
        result = stops_of(ops.merge_duplicated_physical_stops(dup_feed())).set_index("stop_id")

        assert result.loc["P1", "stop_name"] == "Paris Montparnasse Hall 1 - 2"

    def test_transfers_referencing_the_losing_parent_are_remapped(self):
        result = ops.merge_duplicated_physical_stops(dup_feed())

        transfers = result.files["transfers.txt"]
        assert "P2" not in set(transfers["from_stop_id"]) | set(transfers["to_stop_id"])
        assert ("P1", "P3") in set(zip(transfers["from_stop_id"], transfers["to_stop_id"]))

    def test_stop_times_are_unchanged_because_child_ids_survive(self):
        before = dup_feed().files["stop_times.txt"]

        result = ops.merge_duplicated_physical_stops(dup_feed())

        pd.testing.assert_frame_equal(result.files["stop_times.txt"], before)

    def test_result_is_referentially_consistent(self):
        assert_valid_hierarchy(ops.merge_duplicated_physical_stops(dup_feed()))

    def test_result_has_a_clean_index(self):
        result = stops_of(ops.merge_duplicated_physical_stops(dup_feed()))

        assert list(result.index) == list(range(len(result)))

    def test_row_count_drops_by_exactly_the_number_of_removed_parents(self):
        original = len(stops_of(dup_feed()))

        result = stops_of(ops.merge_duplicated_physical_stops(dup_feed()))

        assert len(result) == original - 1

    # ---- radius ---------------------------------------------------------------

    def test_radius_decides_whether_stations_are_merged(self):
        feed_kwargs = dict(p2_offset=0.0018)  # ~200 m

        small = ops.merge_duplicated_physical_stops(dup_feed(**feed_kwargs), radius_m=100.0)
        large = ops.merge_duplicated_physical_stops(dup_feed(**feed_kwargs), radius_m=300.0)

        assert parents_of(small) == {"P1", "P2", "P3"}
        assert parents_of(large) == {"P1", "P3"}

    def test_stations_just_outside_a_tight_radius_are_not_merged(self):
        result = ops.merge_duplicated_physical_stops(dup_feed(), radius_m=50.0)  # P2 is ~100 m away

        assert parents_of(result) == {"P1", "P2", "P3"}

    def test_three_close_stations_collapse_into_the_first(self):
        stops = stops_df([
            ("P1", "One", 48.84000, 2.3190, 1, ""),
            ("P1_a", "One a", 48.84000, 2.3190, 0, "P1"),
            ("P2", "Two", 48.84050, 2.3190, 1, ""),
            ("P2_a", "Two a", 48.84050, 2.3190, 0, "P2"),
            ("P3", "Three", 48.84100, 2.3190, 1, ""),
            ("P3_a", "Three a", 48.84100, 2.3190, 0, "P3"),
        ])
        feed = make_feed(stops, [("T1", "08:00:00", "08:00:00", "P1_a", 1)])

        result = ops.merge_duplicated_physical_stops(feed, radius_m=300.0)

        assert parents_of(result) == {"P1"}
        assert set(parent_map(stops_of(result)).values()) == {"P1"}
        assert {"P1_a", "P2_a", "P3_a"} <= set(stops_of(result)["stop_id"])

    # ---- what must not be merged ----------------------------------------------

    def test_standalone_stops_next_to_a_station_are_not_merged(self):
        feed = dup_feed(extra_stops=[("BUS", "Bus stop", 48.8401, 2.3190, 0, "")])

        result = ops.merge_duplicated_physical_stops(feed)

        stops = stops_of(result).set_index("stop_id")
        assert "BUS" in stops.index
        assert stops.loc["BUS", "location_type"] == 0

    def test_children_are_never_merged_with_each_other(self):
        # P1_a and P1_b are at the same position but are distinct platforms
        result = ops.merge_duplicated_physical_stops(dup_feed())

        assert {"P1_a", "P1_b"} <= set(stops_of(result)["stop_id"])

    # ---- API ------------------------------------------------------------------

    def test_is_idempotent(self):
        once = ops.merge_duplicated_physical_stops(dup_feed())
        before = stops_of(once).copy()

        twice = ops.merge_duplicated_physical_stops(once)

        pd.testing.assert_frame_equal(stops_of(twice), before)

    @pytest.mark.parametrize("as_type", [str, Path])
    def test_accepts_a_path_to_a_zip(self, tmp_path, as_type):
        zip_path = write_zip(dup_feed(), tmp_path / "feed.zip")

        result = ops.merge_duplicated_physical_stops(as_type(zip_path))

        assert parents_of(result) == {"P1", "P3"}