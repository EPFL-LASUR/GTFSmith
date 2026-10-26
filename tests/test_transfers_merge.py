import pandas as pd

from conftest import BareGTFS


def test_merge_transfers_returns_none_when_neither_feed_has_transfers(gm, feed_a, feed_b):
    for feed in (feed_a, feed_b):
        feed.files.pop("transfers.txt", None)

    result = gm.merge_transfers(
        feed_a, feed_b,
        parent_mapping={}, other_child_to_parent={}, child_mapping={},
    )

    assert result is None


def test_merge_transfers_propagates_internal_pattern_onto_new_platform(gm, feed_a, feed_b):
    parent_mapping = gm._match_parent_stations(feed_a, feed_b)
    child_mapping = gm._match_child_stops(feed_a, feed_b, parent_mapping)
    other_child_to_parent = gm.gtfs_utils._build_child_to_parent_mapping(feed_b.files["stops.txt"])
    trip_matches = pd.DataFrame([{"first_trip_id": "T1", "second_trip_id": "T1b"}])

    merged = gm.merge_transfers(
        feed_a, feed_b,
        parent_mapping=parent_mapping,
        other_child_to_parent=other_child_to_parent,
        child_mapping=child_mapping,
        trip_matches=trip_matches,
    )

    pairs = set(zip(merged["from_stop_id"], merged["to_stop_id"]))

    # Self's own internal transfer pattern at STA survives.
    assert ("STA_1", "STA_2") in pairs
    assert ("STA_2", "STA_1") in pairs

    # STAb_3 is a brand-new platform under the matched STA/STAb station and
    # has no counterpart in self -- it should inherit STA's existing
    # platform<->platform transfer pattern against every existing platform,
    # in both directions.
    assert ("STAb_3", "STA_1") in pairs
    assert ("STA_1", "STAb_3") in pairs
    assert ("STAb_3", "STA_2") in pairs
    assert ("STA_2", "STAb_3") in pairs

    assert len(merged) == 6
    assert not merged.duplicated(subset=[
        "from_stop_id", "to_stop_id", "from_route_id", "to_route_id", "from_trip_id", "to_trip_id",
    ]).any()


def test_merge_transfers_remaps_others_surviving_transfers_onto_merged_ids(gm, feed_a, feed_b):
    # Give feed_b its own transfer between two stops that both get matched
    # into feed_a's ids -- it must be rewritten onto self's ids, not kept
    # pointing at other's now-defunct ids.
    feed_b.files["transfers.txt"] = pd.DataFrame([
        {"from_stop_id": "STAb_1", "to_stop_id": "STBb_1", "from_route_id": "", "to_route_id": "",
         "from_trip_id": "", "to_trip_id": "", "transfer_type": "2", "min_transfer_time": 600},
    ])

    parent_mapping = gm._match_parent_stations(feed_a, feed_b)
    child_mapping = gm._match_child_stops(feed_a, feed_b, parent_mapping)
    other_child_to_parent = gm.gtfs_utils._build_child_to_parent_mapping(feed_b.files["stops.txt"])

    merged = gm.merge_transfers(
        feed_a, feed_b,
        parent_mapping=parent_mapping,
        other_child_to_parent=other_child_to_parent,
        child_mapping=child_mapping,
    )

    pairs = set(zip(merged["from_stop_id"], merged["to_stop_id"]))
    assert ("STA_1", "STB_1") in pairs
    assert ("STAb_1", "STBb_1") not in pairs