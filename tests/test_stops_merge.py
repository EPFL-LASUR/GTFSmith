import pandas as pd

from conftest import BareGTFS, _stops_rows


def test_match_parent_stations_matches_close_similarly_named_stations(gm, feed_a, feed_b):
    mapping = gm._match_parent_stations(feed_a, feed_b)

    assert isinstance(mapping, dict)
    assert mapping == {"STAb": "STA", "STBb": "STB"}
    # STC (Zurich) is far from anything in feed_a -> must not be matched.
    assert "STC" not in mapping


def test_match_parent_stations_returns_empty_dict_when_no_candidates_in_range(gm):
    close_pair = BareGTFS({"stops.txt": _stops_rows([
        ("P1", "Station One", 46.5, 6.6, 1, "", ""),
    ])})
    far_away = BareGTFS({"stops.txt": _stops_rows([
        ("P2", "Station Two", -33.8, 151.2, 1, "", ""),  # Sydney
    ])})

    mapping = gm._match_parent_stations(close_pair, far_away)

    # Regression for bug #1: must be an empty dict, never a DataFrame.
    assert mapping == {}
    assert isinstance(mapping, dict)


def test_match_parent_stations_returns_empty_dict_when_a_feed_has_no_parent_stations(gm):
    with_stations = BareGTFS({"stops.txt": _stops_rows([
        ("P1", "Station One", 46.5, 6.6, 1, "", ""),
    ])})
    without_stations = BareGTFS({"stops.txt": _stops_rows([
        ("S1", "Flat stop", 46.5, 6.6, 0, "", ""),
    ])})

    mapping = gm._match_parent_stations(with_stations, without_stations)

    assert mapping == {}
    assert isinstance(mapping, dict)


def test_match_child_stops_matches_on_platform_code_under_matched_parents(gm, feed_a, feed_b):
    parent_mapping = gm._match_parent_stations(feed_a, feed_b)

    child_mapping = gm._match_child_stops(feed_a, feed_b, parent_mapping)

    assert child_mapping == {"STAb_1": "STA_1", "STBb_1": "STB_1"}
    # STAb_3 has no platform-code counterpart under STA -> must stay unmatched.
    assert "STAb_3" not in child_mapping
    # STC_1's parent (STC) was never matched -> must stay unmatched.
    assert "STC_1" not in child_mapping


def test_merge_stops_combines_matched_and_survives_unmatched(gm, feed_a, feed_b):
    merged_stops, parent_mapping, child_mapping, child_to_parent, stop_id_mapping = gm.merge_stops(
        feed_a, feed_b, return_matches=True,
    )

    # 5 of self's own stops + 3 survivors from other (STAb_3, STC, STC_1).
    assert len(merged_stops) == 8
    assert not merged_stops["stop_id"].duplicated().any()

    survivors = set(merged_stops["stop_id"])
    assert {"STA", "STA_1", "STA_2", "STB", "STB_1", "STAb_3", "STC", "STC_1"} == survivors
    assert {"STAb", "STAb_1", "STBb", "STBb_1"}.isdisjoint(survivors)

    # STAb_3 is a new child of a matched station -> must be re-parented
    # onto self's station id, not left pointing at other's original id.
    reparented = merged_stops.loc[merged_stops["stop_id"] == "STAb_3", "parent_station"].iloc[0]
    assert reparented == "STA"

    assert stop_id_mapping["STAb"] == "STA"
    assert stop_id_mapping["STAb_1"] == "STA_1"