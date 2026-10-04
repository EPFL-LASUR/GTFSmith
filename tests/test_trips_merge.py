import pandas as pd

from conftest import BareGTFS


def _translated_child_to_parent(gm, other, parent_mapping):
    """Mirrors the translation merge() applies before calling _match_trips
    other's child->parent mapping, but with the parent id translated into 
    self's id-space via parent_mapping."""
    raw = gm.gtfs_utils._build_child_to_parent_mapping(other.files["stops.txt"])
    return {child: parent_mapping.get(parent, parent) for child, parent in raw.items()}


def test_frequencies_to_trips_expands_headway_based_service(gm):
    feed = BareGTFS({
        "trips.txt": pd.DataFrame([
            {"trip_id": "REP", "route_id": "R1", "service_id": "SV1"},
        ]),
        "stop_times.txt": pd.DataFrame([
            {"trip_id": "REP", "stop_sequence": 1, "stop_id": "S1", "arrival_time": "08:00:00", "departure_time": "08:00:00"},
            {"trip_id": "REP", "stop_sequence": 2, "stop_id": "S2", "arrival_time": "08:10:00", "departure_time": "08:10:00"},
        ]),
        "frequencies.txt": pd.DataFrame([
            {"trip_id": "REP", "start_time": "08:00:00", "end_time": "09:00:00",
             "headway_secs": 1800, "exact_times": 0},
        ]),
    })

    result = gm.frequencies_to_trips(feed)

    # frequencies.txt is fully consumed.
    assert "frequencies.txt" not in result.files

    trips = result.files["trips.txt"]
    stop_times = result.files["stop_times.txt"]

    # Representative trip "REP" is gone, replaced by 2 generated instances
    # (08:00 and 08:30 departures within a 1-hour window on a 30 min headway).
    assert "REP" not in set(trips["trip_id"])
    generated_ids = sorted(trips["trip_id"])
    assert len(generated_ids) == 2

    first_trip_times = (
        stop_times[stop_times["trip_id"] == generated_ids[0]]
        .sort_values("stop_sequence")["departure_time"].tolist()
    )
    second_trip_times = (
        stop_times[stop_times["trip_id"] == generated_ids[1]]
        .sort_values("stop_sequence")["departure_time"].tolist()
    )

    assert first_trip_times == ["08:00:00", "08:10:00"]
    assert second_trip_times == ["08:30:00", "08:40:00"]


def test_frequencies_to_trips_is_a_noop_without_a_frequencies_file(gm):
    trips = pd.DataFrame([{"trip_id": "T1", "route_id": "R1", "service_id": "SV1"}])
    feed = BareGTFS({"trips.txt": trips, "stop_times.txt": pd.DataFrame(columns=["trip_id", "stop_sequence", "stop_id", "arrival_time", "departure_time"])})

    result = gm.frequencies_to_trips(feed)

    assert result.files["trips.txt"] is trips  # untouched


def test_match_trips_finds_the_same_physical_trip_across_feeds(gm, feed_a, feed_b):
    parent_mapping = gm._match_parent_stations(feed_a, feed_b)
    child_to_parent_self = gm.gtfs_utils._build_child_to_parent_mapping(feed_a.files["stops.txt"])
    child_to_parent_other = _translated_child_to_parent(gm, feed_b, parent_mapping)

    matches = gm._match_trips(feed_a, feed_b, child_to_parent_self, child_to_parent_other)

    assert len(matches) == 1
    row = matches.iloc[0]
    assert row["first_trip_id"] == "T1"
    assert row["second_trip_id"] == "T1b"
    assert row["max_departure_difference"] <= 60

    # T0 (self-only) and T2b (other-only, different stops) must not match
    # anything.
    assert "T0" not in set(matches["first_trip_id"])
    assert "T2b" not in set(matches["second_trip_id"])


def test_match_trips_rejects_pairs_outside_the_time_tolerance(gm, feed_a, feed_b):
    # Push T1b's arrival far outside the default 60s tolerance.
    stop_times = feed_b.files["stop_times.txt"].copy()
    stop_times.loc[stop_times["stop_id"] == "STBb_1", ["arrival_time", "departure_time"]] = "09:10:00"
    feed_b.files["stop_times.txt"] = stop_times

    parent_mapping = gm._match_parent_stations(feed_a, feed_b)
    child_to_parent_self = gm.gtfs_utils._build_child_to_parent_mapping(feed_a.files["stops.txt"])
    child_to_parent_other = _translated_child_to_parent(gm, feed_b, parent_mapping)

    matches = gm._match_trips(feed_a, feed_b, child_to_parent_self, child_to_parent_other)

    assert matches.empty


def test_merge_trip_data_keeps_only_unmatched_other_trips_and_remaps_stops(gm, feed_a, feed_b):
    parent_mapping = gm._match_parent_stations(feed_a, feed_b)
    child_mapping = gm._match_child_stops(feed_a, feed_b, parent_mapping)
    _, _, stop_id_mapping = gm._merge_stop_hierarchy(feed_a, feed_b, parent_mapping, child_mapping)

    matches = pd.DataFrame([{"first_trip_id": "T1", "second_trip_id": "T1b"}])

    merged_trips, merged_stop_times = gm._merge_trip_data(
        feed_a, feed_b, matches, stop_id_mapping=stop_id_mapping,
    )

    trip_ids = set(merged_trips["trip_id"])
    # Self's trips are always kept; only other's unmatched trip (T2b) is added.
    assert trip_ids == {"T1", "T0", "T2b"}

    # T2b's stop_times should keep referencing other's surviving (unmatched)
    # stop ids -- STAb_3 was never matched so it is not remapped.
    t2b_stops = set(merged_stop_times.loc[merged_stop_times["trip_id"] == "T2b", "stop_id"])
    assert t2b_stops == {"STAb_3", "STC_1"}