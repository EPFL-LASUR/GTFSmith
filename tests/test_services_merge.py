import pandas as pd

from conftest import BareGTFS


def test_service_dates_combines_calendar_and_calendar_dates(gm):
    feed = BareGTFS({
        "calendar.txt": pd.DataFrame([
            {"service_id": "SV1", "monday": 1, "tuesday": 1, "wednesday": 1, "thursday": 1,
             "friday": 1, "saturday": 0, "sunday": 0, "start_date": 20240101, "end_date": 20240105},
        ]),
        "calendar_dates.txt": pd.DataFrame([
            # Add a Saturday (exception_type=1) and remove the Wednesday (exception_type=2).
            {"service_id": "SV1", "date": "20240106", "exception_type": 1},
            {"service_id": "SV1", "date": "20240103", "exception_type": 2},
        ]),
    })

    dates = gm._service_dates(feed)

    assert dates["SV1"] == {"20240101", "20240102", "20240104", "20240105", "20240106"}


def test_match_services_requires_identical_effective_dates(gm, feed_a, feed_b):
    trip_matches = pd.DataFrame([{"first_trip_id": "T1", "second_trip_id": "T1b"}])

    service_matches, self_dates, other_dates = gm._match_services(feed_a, feed_b, trip_matches)

    assert len(service_matches) == 1
    row = service_matches.iloc[0]
    assert row["self_service_id"] == "SV1"
    assert row["other_service_id"] == "SV1b"


def test_match_services_rejects_differing_date_sets(gm):
    self_feed = BareGTFS({
        "trips.txt": pd.DataFrame([{"trip_id": "TA", "route_id": "RA", "service_id": "SV1"}]),
        "calendar.txt": pd.DataFrame([
            {"service_id": "SV1", "monday": 1, "tuesday": 0, "wednesday": 0, "thursday": 0,
             "friday": 0, "saturday": 0, "sunday": 0, "start_date": 20240101, "end_date": 20240101},
        ]),
        "calendar_dates.txt": pd.DataFrame(columns=["service_id", "date", "exception_type"]),
    })
    other_feed = BareGTFS({
        "trips.txt": pd.DataFrame([{"trip_id": "TB", "route_id": "RB", "service_id": "SV1X"}]),
        "calendar.txt": pd.DataFrame([
            {"service_id": "SV1X", "monday": 0, "tuesday": 1, "wednesday": 0, "thursday": 0,
             "friday": 0, "saturday": 0, "sunday": 0, "start_date": 20240102, "end_date": 20240102},
        ]),
        "calendar_dates.txt": pd.DataFrame(columns=["service_id", "date", "exception_type"]),
    })
    trip_matches = pd.DataFrame([{"first_trip_id": "TA", "second_trip_id": "TB"}])

    service_matches, _, _ = gm._match_services(self_feed, other_feed, trip_matches)

    assert service_matches.empty


def test_dates_to_calendar_round_trips_through_service_dates(gm):
    # An irregular pattern spanning two weeks, including a Saturday and a
    # skipped weekday -- the actual invariant that matters is that
    # calendar + calendar_dates together reproduce exactly this date set,
    # regardless of how the weekly/exception split happens to land.
    dates = {"20240101", "20240102", "20240104", "20240105", "20240106", "20240110"}

    calendar_row, exceptions = gm._dates_to_calendar("SV1", dates)

    feed = BareGTFS({
        "calendar.txt": pd.DataFrame([calendar_row]),
        "calendar_dates.txt": pd.DataFrame(exceptions, columns=["service_id", "date", "exception_type"]),
    })

    reconstructed = gm._service_dates(feed)

    assert reconstructed["SV1"] == dates


def test_merge_services_keeps_self_authoritative_for_matched_services(gm, feed_a, feed_b):
    trip_matches = pd.DataFrame([{"first_trip_id": "T1", "second_trip_id": "T1b"}])
    service_matches, self_dates, other_dates = gm._match_services(feed_a, feed_b, trip_matches)
    service_mapping = gm._service_mapping(feed_a, service_matches)

    merged_trips = pd.DataFrame([
        {"trip_id": "T1", "route_id": "R1", "service_id": "SV1"},
        {"trip_id": "T0", "route_id": "R0", "service_id": "SV1"},
        {"trip_id": "T2b", "route_id": "R2b", "service_id": "SV2b"},
    ])

    merged_calendar, merged_calendar_dates = gm.merge_services(
        self_dates=self_dates, other_dates=other_dates,
        merged_trips=merged_trips, service_mapping=service_mapping,
    )

    service_ids = set(merged_calendar["service_id"])
    # SV1b collapsed into SV1 (self authoritative); SV2b is unmatched and survives.
    assert service_ids == {"SV1", "SV2b"}