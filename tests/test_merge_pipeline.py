def test_merge_produces_a_consistent_combined_feed(gm, feed_a, feed_b):
    merged = gm.merge(feed_a, feed_b)

    stops = merged.files["stops.txt"]
    trips = merged.files["trips.txt"]
    stop_times = merged.files["stop_times.txt"]
    routes = merged.files["routes.txt"]
    agency = merged.files["agency.txt"]
    calendar = merged.files["calendar.txt"]
    transfers = merged.files["transfers.txt"]

    # --- stops: matched stations/platforms collapsed, unmatched ones kept ---
    assert len(stops) == 8
    assert not stops["stop_id"].duplicated().any()
    assert set(stops["stop_id"]) == {
        "STA", "STA_1", "STA_2", "STB", "STB_1", "STAb_3", "STC", "STC_1",
    }

    # --- trips: the shared physical trip (T1/T1b) collapses to self's T1 ---
    assert set(trips["trip_id"]) == {"T1", "T0", "T2b"}
    assert not trips["trip_id"].duplicated().any()
    # stop_times stay consistent with the surviving trips and merged stops.
    assert set(stop_times["trip_id"]) == {"T1", "T0", "T2b"}
    assert set(stop_times["stop_id"]).issubset(set(stops["stop_id"]))

    # --- routes: R1/R1b collapse to self's R1; R2b (unmatched) survives ---
    assert set(routes["route_id"]) == {"R1", "R0", "R2b"}
    assert not routes["route_id"].duplicated().any()

    # --- agencies: AG1 collapses; the unrelated-but-same-id AG2 from the
    # other feed is renamed rather than merged into self's AG2 ---
    assert set(agency["agency_id"]) == {"AG1", "AG2", "other_AG2"}
    assert not agency["agency_id"].duplicated().any()
    # R2b's agency must follow the rename (it was "AG2" in feed B).
    r2b_agency = routes.loc[routes["route_id"] == "R2b", "agency_id"].iloc[0]
    assert r2b_agency == "other_AG2"

    # --- services: SV1b collapses into SV1 (self authoritative); SV2b survives ---
    assert set(calendar["service_id"]) == {"SV1", "SV2b"}
    assert set(trips["service_id"]) == {"SV1", "SV2b"}

    # --- transfers: self's pair + propagation onto the new STAb_3 platform ---
    assert len(transfers) == 6
    pairs = set(zip(transfers["from_stop_id"], transfers["to_stop_id"]))
    assert ("STA_1", "STA_2") in pairs
    assert ("STAb_3", "STA_1") in pairs
    assert ("STAb_3", "STA_2") in pairs


def test_merge_does_not_mutate_the_original_feed_objects_files_dict_entries(gm, feed_a, feed_b):
    # merge_stops/merge_trips build new DataFrames rather than editing the
    # inputs' tables in place (frequencies_to_trips is an accepted,
    # documented exception, since it operates on trips/stop_times/
    # frequencies.txt directly on the passed-in GTFS object).
    original_routes_a = feed_a.files["routes.txt"].copy(deep=True)
    original_agency_a = feed_a.files["agency.txt"].copy(deep=True)

    gm.merge(feed_a, feed_b)

    pd_testing_equal = original_routes_a.equals(feed_a.files["routes.txt"])
    assert pd_testing_equal
    assert original_agency_a.equals(feed_a.files["agency.txt"])