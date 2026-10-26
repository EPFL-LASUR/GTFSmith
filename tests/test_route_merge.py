"""
Tests for the ROUTES part of gtfs_merge:
    _match_routes, _merge_route_data, merge_routes
plus end-to-end checks on routes through gm.merge().
"""

import pandas as pd
import pytest

from conftest import BareGTFS


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def trips_df(rows):
    """rows: (trip_id, route_id, service_id)"""
    return pd.DataFrame(rows, columns=["trip_id", "route_id", "service_id"])


def matches_df(pairs):
    """pairs: (first_trip_id, second_trip_id)"""
    return pd.DataFrame(pairs, columns=["first_trip_id", "second_trip_id"])


def routes_df(rows):
    """rows: (route_id, route_long_name)"""
    return pd.DataFrame(rows, columns=["route_id", "route_long_name"])


def route_matches_df(pairs):
    """pairs: (first_route_id, second_route_id)"""
    return pd.DataFrame(pairs, columns=["first_route_id", "second_route_id"])


# --------------------------------------------------------------------------
# _match_routes
# --------------------------------------------------------------------------

class TestMatchRoutes:

    def test_empty_trip_matches_returns_empty_frame_with_columns(self, gm):
        first = BareGTFS({"trips.txt": trips_df([("T1", "R1", "S1")])})
        second = BareGTFS({"trips.txt": trips_df([("T1b", "R1b", "S1b")])})

        result = gm._match_routes(first, second, matches_df([]))

        assert result.empty
        assert list(result.columns) == ["first_route_id", "second_route_id", "matched_trip_count"]

    def test_missing_columns_raises(self, gm):
        first = BareGTFS({"trips.txt": trips_df([("T1", "R1", "S1")])})
        second = BareGTFS({"trips.txt": trips_df([("T1b", "R1b", "S1b")])})

        with pytest.raises(ValueError, match="missing columns"):
            gm._match_routes(first, second, pd.DataFrame({"first_trip_id": ["T1"]}))

    def test_single_matched_trip_matches_its_routes(self, gm):
        first = BareGTFS({"trips.txt": trips_df([("T1", "R1", "S1")])})
        second = BareGTFS({"trips.txt": trips_df([("T1b", "R1b", "S1b")])})

        result = gm._match_routes(first, second, matches_df([("T1", "T1b")]))

        assert result.to_dict("records") == [
            {"first_route_id": "R1", "second_route_id": "R1b", "matched_trip_count": 1}
        ]

    def test_matched_trip_count_is_aggregated_per_route_pair(self, gm):
        first = BareGTFS({"trips.txt": trips_df([
            ("T1", "R1", "S1"), ("T2", "R1", "S1"), ("T3", "R2", "S1"),
        ])})
        second = BareGTFS({"trips.txt": trips_df([
            ("T1b", "R1b", "S1"), ("T2b", "R1b", "S1"), ("T3b", "R2b", "S1"),
        ])})
        trip_matches = matches_df([("T1", "T1b"), ("T2", "T2b"), ("T3", "T3b")])

        result = gm._match_routes(first, second, trip_matches)

        records = sorted(result.to_dict("records"), key=lambda r: r["first_route_id"])
        assert records == [
            {"first_route_id": "R1", "second_route_id": "R1b", "matched_trip_count": 2},
            {"first_route_id": "R2", "second_route_id": "R2b", "matched_trip_count": 1},
        ]

    def test_unmatched_trips_do_not_create_route_matches(self, gm):
        first = BareGTFS({"trips.txt": trips_df([("T1", "R1", "S1"), ("T9", "R9", "S1")])})
        second = BareGTFS({"trips.txt": trips_df([("T1b", "R1b", "S1"), ("T8b", "R8b", "S1")])})

        result = gm._match_routes(first, second, matches_df([("T1", "T1b")]))

        assert set(result["first_route_id"]) == {"R1"}
        assert set(result["second_route_id"]) == {"R1b"}

    def test_extra_columns_in_trip_matches_are_ignored(self, gm):
        first = BareGTFS({"trips.txt": trips_df([("T1", "R1", "S1")])})
        second = BareGTFS({"trips.txt": trips_df([("T1b", "R1b", "S1")])})
        trip_matches = matches_df([("T1", "T1b")]).assign(stop_count=2, max_arrival_difference=30)

        result = gm._match_routes(first, second, trip_matches)

        assert list(result.columns) == ["first_route_id", "second_route_id", "matched_trip_count"]

    def test_self_route_mapping_to_two_other_routes_raises(self, gm):
        first = BareGTFS({"trips.txt": trips_df([("T1", "R1", "S1"), ("T2", "R1", "S1")])})
        second = BareGTFS({"trips.txt": trips_df([("T1b", "RA", "S1"), ("T2b", "RB", "S1")])})

        with pytest.raises(ValueError, match="maps to multiple routes in other"):
            gm._match_routes(first, second, matches_df([("T1", "T1b"), ("T2", "T2b")]))

    def test_other_route_mapping_to_two_self_routes_raises(self, gm):
        first = BareGTFS({"trips.txt": trips_df([("T1", "R1", "S1"), ("T2", "R2", "S1")])})
        second = BareGTFS({"trips.txt": trips_df([("T1b", "RA", "S1"), ("T2b", "RA", "S1")])})

        with pytest.raises(ValueError, match="maps to multiple routes in self"):
            gm._match_routes(first, second, matches_df([("T1", "T1b"), ("T2", "T2b")]))

    def test_matched_trip_missing_from_self_trips_raises(self, gm):
        first = BareGTFS({"trips.txt": trips_df([("T1", "R1", "S1")])})
        second = BareGTFS({"trips.txt": trips_df([("T1b", "R1b", "S1")])})

        with pytest.raises(ValueError, match="no route_id"):
            gm._match_routes(first, second, matches_df([("GHOST", "T1b")]))

    def test_matched_trip_missing_from_other_trips_raises(self, gm):
        first = BareGTFS({"trips.txt": trips_df([("T1", "R1", "S1")])})
        second = BareGTFS({"trips.txt": trips_df([("T1b", "R1b", "S1")])})

        with pytest.raises(ValueError, match="no route_id"):
            gm._match_routes(first, second, matches_df([("T1", "GHOST")]))

    def test_inputs_are_not_mutated(self, gm):
        first_trips = trips_df([("T1", "R1", "S1")])
        second_trips = trips_df([("T1b", "R1b", "S1")])
        trip_matches = matches_df([("T1", "T1b")])
        snapshots = [df.copy() for df in (first_trips, second_trips, trip_matches)]

        gm._match_routes(
            BareGTFS({"trips.txt": first_trips}), BareGTFS({"trips.txt": second_trips}), trip_matches
        )

        for original, snapshot in zip((first_trips, second_trips, trip_matches), snapshots):
            pd.testing.assert_frame_equal(original, snapshot)


# --------------------------------------------------------------------------
# _merge_route_data
# --------------------------------------------------------------------------

class TestMergeRouteData:

    def test_no_matches_concatenates_everything_self_first(self, gm):
        first = BareGTFS({"routes.txt": routes_df([("R1", "One"), ("R0", "Zero")])})
        second = BareGTFS({"routes.txt": routes_df([("R9", "Nine")])})

        result = gm._merge_route_data(first, second, route_matches_df([]))

        assert list(result["route_id"]) == ["R1", "R0", "R9"]
        assert list(result.index) == [0, 1, 2]

    def test_matched_route_from_other_is_dropped(self, gm):
        first = BareGTFS({"routes.txt": routes_df([("R1", "Self name")])})
        second = BareGTFS({"routes.txt": routes_df([("R1b", "Other name"), ("R2b", "Unmatched")])})

        result = gm._merge_route_data(first, second, route_matches_df([("R1", "R1b")]))

        assert set(result["route_id"]) == {"R1", "R2b"}

    def test_self_is_authoritative_for_matched_route_attributes(self, gm):
        first = BareGTFS({"routes.txt": routes_df([("R1", "Self name")])})
        second = BareGTFS({"routes.txt": routes_df([("R1b", "Other name")])})

        result = gm._merge_route_data(first, second, route_matches_df([("R1", "R1b")]))

        assert result.loc[result["route_id"] == "R1", "route_long_name"].iloc[0] == "Self name"
        assert "Other name" not in set(result["route_long_name"])

    def test_unmatched_routes_from_both_feeds_survive(self, gm):
        first = BareGTFS({"routes.txt": routes_df([("R1", "A"), ("R0", "B")])})
        second = BareGTFS({"routes.txt": routes_df([("R1b", "C"), ("R2b", "D")])})

        result = gm._merge_route_data(first, second, route_matches_df([("R1", "R1b")]))

        assert list(result["route_id"]) == ["R1", "R0", "R2b"]

    def test_columns_present_in_only_one_feed_are_kept(self, gm):
        first = BareGTFS({"routes.txt": routes_df([("R1", "A")])})
        second = BareGTFS({"routes.txt": routes_df([("R2b", "B")]).assign(route_color="FF0000")})

        result = gm._merge_route_data(first, second, route_matches_df([]))

        assert "route_color" in result.columns
        assert pd.isna(result.loc[result["route_id"] == "R1", "route_color"].iloc[0])
        assert result.loc[result["route_id"] == "R2b", "route_color"].iloc[0] == "FF0000"

    def test_missing_columns_raises(self, gm):
        first = BareGTFS({"routes.txt": routes_df([("R1", "A")])})
        second = BareGTFS({"routes.txt": routes_df([("R1b", "B")])})

        with pytest.raises(ValueError, match="missing columns"):
            gm._merge_route_data(first, second, pd.DataFrame({"first_route_id": ["R1"]}))

    def test_duplicate_first_route_id_raises(self, gm):
        first = BareGTFS({"routes.txt": routes_df([("R1", "A")])})
        second = BareGTFS({"routes.txt": routes_df([("RA", "B"), ("RB", "C")])})

        with pytest.raises(ValueError, match="A route from self matches multiple routes from other"):
            gm._merge_route_data(first, second, route_matches_df([("R1", "RA"), ("R1", "RB")]))

    def test_duplicate_second_route_id_raises(self, gm):
        first = BareGTFS({"routes.txt": routes_df([("R1", "A"), ("R2", "B")])})
        second = BareGTFS({"routes.txt": routes_df([("RA", "C")])})

        with pytest.raises(ValueError, match="A route from other matches multiple routes from self"):
            gm._merge_route_data(first, second, route_matches_df([("R1", "RA"), ("R2", "RA")]))

    def test_inputs_are_not_mutated(self, gm):
        first_routes = routes_df([("R1", "A")])
        second_routes = routes_df([("R1b", "B"), ("R2b", "C")])
        snapshots = [first_routes.copy(), second_routes.copy()]

        gm._merge_route_data(
            BareGTFS({"routes.txt": first_routes}),
            BareGTFS({"routes.txt": second_routes}),
            route_matches_df([("R1", "R1b")]),
        )

        pd.testing.assert_frame_equal(first_routes, snapshots[0])
        pd.testing.assert_frame_equal(second_routes, snapshots[1])

    def test_colliding_unmatched_route_ids_do_not_create_duplicates(self, gm):
        first = BareGTFS({"routes.txt": routes_df([("R1", "Self route")])})
        second = BareGTFS({"routes.txt": routes_df([("R1", "Totally different route")])})

        result = gm._merge_route_data(first, second, route_matches_df([]))

        assert not result["route_id"].duplicated().any()


# --------------------------------------------------------------------------
# merge_routes
# --------------------------------------------------------------------------

class TestMergeRoutes:

    def test_returns_only_routes_by_default(self, gm):
        first = BareGTFS({
            "trips.txt": trips_df([("T1", "R1", "S1")]),
            "routes.txt": routes_df([("R1", "A")]),
        })
        second = BareGTFS({
            "trips.txt": trips_df([("T1b", "R1b", "S1")]),
            "routes.txt": routes_df([("R1b", "B"), ("R2b", "C")]),
        })

        result = gm.merge_routes(first, second, matches_df([("T1", "T1b")]))

        assert isinstance(result, pd.DataFrame)
        assert list(result["route_id"]) == ["R1", "R2b"]

    def test_return_matches_returns_tuple(self, gm):
        first = BareGTFS({
            "trips.txt": trips_df([("T1", "R1", "S1")]),
            "routes.txt": routes_df([("R1", "A")]),
        })
        second = BareGTFS({
            "trips.txt": trips_df([("T1b", "R1b", "S1")]),
            "routes.txt": routes_df([("R1b", "B")]),
        })

        merged, route_matches = gm.merge_routes(
            first, second, matches_df([("T1", "T1b")]), return_matches=True
        )

        assert list(merged["route_id"]) == ["R1"]
        assert route_matches.to_dict("records") == [
            {"first_route_id": "R1", "second_route_id": "R1b", "matched_trip_count": 1}
        ]

    def test_no_trip_matches_keeps_all_routes(self, gm):
        first = BareGTFS({
            "trips.txt": trips_df([("T1", "R1", "S1")]),
            "routes.txt": routes_df([("R1", "A")]),
        })
        second = BareGTFS({
            "trips.txt": trips_df([("T1b", "R1b", "S1")]),
            "routes.txt": routes_df([("R1b", "B")]),
        })

        result = gm.merge_routes(first, second, matches_df([]))

        assert list(result["route_id"]) == ["R1", "R1b"]

    def test_unmatched_trip_on_a_matched_route_keeps_valid_route_reference(self, gm):
        stop_times_cols = ["trip_id", "stop_sequence", "stop_id"]
        first = BareGTFS({
            "trips.txt": trips_df([("T1", "R1", "S1")]),
            "stop_times.txt": pd.DataFrame([("T1", 1, "A")], columns=stop_times_cols),
            "routes.txt": routes_df([("R1", "Self route")]),
        })
        second = BareGTFS({
            # T1b is matched with T1 (so R1b <-> R1 are matched);
            # T4b runs on the same route in `other` but is NOT matched.
            "trips.txt": trips_df([("T1b", "R1b", "S1"), ("T4b", "R1b", "S1")]),
            "stop_times.txt": pd.DataFrame(
                [("T1b", 1, "A"), ("T4b", 1, "A")], columns=stop_times_cols
            ),
            "routes.txt": routes_df([("R1b", "Other route")]),
        })
        trip_matches = matches_df([("T1", "T1b")])

        merged_trips, _ = gm.merge_trips(first, second, trip_matches)
        merged_routes = gm.merge_routes(first, second, trip_matches)

        assert set(merged_trips["route_id"]) <= set(merged_routes["route_id"])


# --------------------------------------------------------------------------
# End to end through gm.merge()
# --------------------------------------------------------------------------

class TestRoutesEndToEnd:

    def test_merge_keeps_self_route_and_drops_matched_other_route(self, gm, feed_a, feed_b):
        merged = gm.merge(feed_a, feed_b, create_transfers=False)

        routes = merged.files["routes.txt"]
        # R1 (self) <-> R1b (other) are matched via T1/T1b; R0 and R2b are unmatched.
        assert set(routes["route_id"]) == {"R1", "R0", "R2b"}
        assert not routes["route_id"].duplicated().any()

    def test_merge_preserves_self_route_attributes(self, gm, feed_a, feed_b):
        merged = gm.merge(feed_a, feed_b, create_transfers=False)

        routes = merged.files["routes.txt"].set_index("route_id")
        assert routes.loc["R1", "route_long_name"] == "Lausanne - Geneve"
        assert routes.loc["R1", "agency_id"] == "AG1"

    def test_merge_remaps_agency_of_surviving_other_routes(self, gm, feed_a, feed_b):
        merged = gm.merge(feed_a, feed_b, create_transfers=False)

        routes = merged.files["routes.txt"].set_index("route_id")
        # R2b's agency AG2 conflicts with self's unrelated AG2 -> renamed.
        assert routes.loc["R2b", "agency_id"] == "other_AG2"
        # Self's own routes are untouched.
        assert routes.loc["R0", "agency_id"] == "AG2"

    def test_every_merged_trip_references_an_existing_route(self, gm, feed_a, feed_b):
        merged = gm.merge(feed_a, feed_b, create_transfers=False)

        trip_routes = set(merged.files["trips.txt"]["route_id"])
        assert trip_routes <= set(merged.files["routes.txt"]["route_id"])