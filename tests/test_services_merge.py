import pandas as pd
import pytest

from conftest import BareGTFS


WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
MON_FRI = ["monday", "tuesday", "wednesday", "thursday", "friday"]
WEEK = {"20240101", "20240102", "20240103", "20240104", "20240105"}
WEEKEND = {"20240106", "20240107"}




def calendar_row(service_id, days, start, end):
    """days: iterable of weekday names on which the service runs."""
    row = {"service_id": service_id}
    row.update({d: int(d in days) for d in WEEKDAYS})
    row["start_date"] = start
    row["end_date"] = end
    return row


def trips_df(rows):
    """rows: (trip_id, route_id, service_id)"""
    return pd.DataFrame(rows, columns=["trip_id", "route_id", "service_id"])


def matches_df(pairs):
    """pairs: (first_trip_id, second_trip_id)"""
    return pd.DataFrame(pairs, columns=["first_trip_id", "second_trip_id"])


def service_feed(trips, calendar_rows=None, calendar_dates=None):
    files = {"trips.txt": trips_df(trips)}
    if calendar_rows is not None:
        files["calendar.txt"] = pd.DataFrame(calendar_rows)
    if calendar_dates is not None:
        files["calendar_dates.txt"] = pd.DataFrame(calendar_dates)
    return BareGTFS(files)


def expand_calendar(row, exceptions):
    """Independently re-expand a calendar row + exception rows into a date set."""
    start = pd.to_datetime(str(row["start_date"]), format="%Y%m%d")
    end = pd.to_datetime(str(row["end_date"]), format="%Y%m%d")
    dates = {
        d.strftime("%Y%m%d")
        for d in pd.date_range(start, end, freq="D")
        if row[WEEKDAYS[d.weekday()]] == 1
    }
    for exc in exceptions:
        if exc["exception_type"] == 1:
            dates.add(exc["date"])
        elif exc["exception_type"] == 2:
            dates.discard(exc["date"])
    return dates


def effective_dates(gm, calendar, calendar_dates, service_id):
    """Effective dates of a service in a merged calendar/calendar_dates pair."""
    feed = BareGTFS({"calendar.txt": calendar, "calendar_dates.txt": calendar_dates})
    return gm._service_dates(feed)[service_id]


# --------------------------------------------------------------------------
# _service_dates
# --------------------------------------------------------------------------

class TestServiceDates:

    def test_service_dates_combines_calendar_and_calendar_dates(self, gm):
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


    def test_match_services_requires_identical_effective_dates(self, gm, feed_a, feed_b):
        trip_matches = pd.DataFrame([{"first_trip_id": "T1", "second_trip_id": "T1b"}])

        service_matches, self_dates, other_dates = gm._match_services(feed_a, feed_b, trip_matches)

        assert len(service_matches) == 1
        row = service_matches.iloc[0]
        assert row["self_service_id"] == "SV1"
        assert row["other_service_id"] == "SV1b"


    def test_match_services_rejects_differing_date_sets(self, gm):
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


    def test_dates_to_calendar_round_trips_through_service_dates(self, gm):
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


    def test_merge_services_keeps_self_authoritative_for_matched_services(self, gm, feed_a, feed_b):
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

    def test_no_calendar_files_returns_empty_dict(self, gm):
        assert gm._service_dates(BareGTFS({})) == {}

    def test_calendar_only_weekday_service(self, gm):
        feed = BareGTFS({"calendar.txt": pd.DataFrame([
            calendar_row("SV1", MON_FRI, 20240101, 20240107),
        ])})

        assert gm._service_dates(feed) == {"SV1": WEEK}

    def test_calendar_only_weekend_service(self, gm):
        feed = BareGTFS({"calendar.txt": pd.DataFrame([
            calendar_row("WE", ["saturday", "sunday"], 20240101, 20240114),
        ])})

        assert gm._service_dates(feed)["WE"] == {"20240106", "20240107", "20240113", "20240114"}

    def test_start_and_end_dates_are_inclusive(self, gm):
        feed = BareGTFS({"calendar.txt": pd.DataFrame([
            calendar_row("SV1", WEEKDAYS, 20240110, 20240112),
        ])})

        assert gm._service_dates(feed)["SV1"] == {"20240110", "20240111", "20240112"}

    def test_service_with_no_active_weekday_has_no_dates(self, gm):
        feed = BareGTFS({"calendar.txt": pd.DataFrame([
            calendar_row("NONE", [], 20240101, 20240107),
        ])})

        assert gm._service_dates(feed)["NONE"] == set()

    def test_multiple_services_are_independent(self, gm):
        feed = BareGTFS({"calendar.txt": pd.DataFrame([
            calendar_row("WK", MON_FRI, 20240101, 20240105),
            calendar_row("WE", ["saturday", "sunday"], 20240106, 20240107),
        ])})

        dates = gm._service_dates(feed)

        assert dates["WK"] == WEEK
        assert dates["WE"] == WEEKEND

    def test_exception_type_1_adds_a_date(self, gm):
        feed = BareGTFS({
            "calendar.txt": pd.DataFrame([calendar_row("SV1", MON_FRI, 20240101, 20240105)]),
            "calendar_dates.txt": pd.DataFrame([
                {"service_id": "SV1", "date": "20240106", "exception_type": 1},
            ]),
        })

        assert gm._service_dates(feed)["SV1"] == WEEK | {"20240106"}

    def test_exception_type_2_removes_a_date(self, gm):
        feed = BareGTFS({
            "calendar.txt": pd.DataFrame([calendar_row("SV1", MON_FRI, 20240101, 20240105)]),
            "calendar_dates.txt": pd.DataFrame([
                {"service_id": "SV1", "date": "20240103", "exception_type": 2},
            ]),
        })

        assert gm._service_dates(feed)["SV1"] == WEEK - {"20240103"}

    def test_exceptions_only_affect_their_own_service(self, gm):
        feed = BareGTFS({
            "calendar.txt": pd.DataFrame([
                calendar_row("A", MON_FRI, 20240101, 20240105),
                calendar_row("B", MON_FRI, 20240101, 20240105),
            ]),
            "calendar_dates.txt": pd.DataFrame([
                {"service_id": "A", "date": "20240102", "exception_type": 2},
            ]),
        })

        dates = gm._service_dates(feed)

        assert "20240102" not in dates["A"]
        assert "20240102" in dates["B"]

    def test_calendar_dates_only_feed(self, gm):
        feed = BareGTFS({"calendar_dates.txt": pd.DataFrame([
            {"service_id": "SV1", "date": "20240101", "exception_type": 1},
            {"service_id": "SV1", "date": "20240105", "exception_type": 1},
            {"service_id": "SV2", "date": "20240110", "exception_type": 1},
        ])})

        assert gm._service_dates(feed) == {
            "SV1": {"20240101", "20240105"},
            "SV2": {"20240110"},
        }

    def test_calendar_dates_only_with_removal(self, gm):
        feed = BareGTFS({"calendar_dates.txt": pd.DataFrame([
            {"service_id": "SV1", "date": "20240101", "exception_type": 1},
            {"service_id": "SV1", "date": "20240102", "exception_type": 1},
            {"service_id": "SV1", "date": "20240101", "exception_type": 2},
        ])})

        assert gm._service_dates(feed)["SV1"] == {"20240102"}

    def test_empty_calendar_dates_frame_is_ignored(self, gm):
        feed = BareGTFS({
            "calendar.txt": pd.DataFrame([calendar_row("SV1", MON_FRI, 20240101, 20240105)]),
            "calendar_dates.txt": pd.DataFrame(columns=["service_id", "date", "exception_type"]),
        })

        assert gm._service_dates(feed)["SV1"] == WEEK

    def test_integer_service_ids_and_dates_are_stringified(self, gm):
        feed = BareGTFS({
            "calendar.txt": pd.DataFrame([calendar_row(101, MON_FRI, 20240101, 20240105)]),
            "calendar_dates.txt": pd.DataFrame([
                {"service_id": 101, "date": 20240106, "exception_type": 1},
            ]),
        })

        dates = gm._service_dates(feed)

        assert "20240106" in dates["101"]

    def test_inputs_are_not_mutated(self, gm):
        calendar = pd.DataFrame([calendar_row("SV1", MON_FRI, 20240101, 20240105)])
        calendar_dates = pd.DataFrame([
            {"service_id": "SV1", "date": "20240102", "exception_type": 2},
        ])
        snapshots = (calendar.copy(), calendar_dates.copy())

        gm._service_dates(BareGTFS({"calendar.txt": calendar, "calendar_dates.txt": calendar_dates}))

        pd.testing.assert_frame_equal(calendar, snapshots[0])
        pd.testing.assert_frame_equal(calendar_dates, snapshots[1])


# --------------------------------------------------------------------------
# _match_services / _service_mapping
# --------------------------------------------------------------------------

class TestMatchServices:

    def test_empty_trip_matches_returns_empty_result_and_date_dicts(self, gm):
        first = service_feed([("T1", "R1", "SV1")], [calendar_row("SV1", MON_FRI, 20240101, 20240105)])
        second = service_feed([("T1b", "R1b", "SV1b")], [calendar_row("SV1b", MON_FRI, 20240101, 20240105)])

        result, self_dates, other_dates = gm._match_services(first, second, matches_df([]))

        assert result.empty
        assert list(result.columns) == ["self_service_id", "other_service_id", "matched_trip_count"]
        assert set(self_dates) == {"SV1"}
        assert set(other_dates) == {"SV1b"}

    def test_missing_columns_raises(self, gm):
        first = service_feed([("T1", "R1", "SV1")], [calendar_row("SV1", MON_FRI, 20240101, 20240105)])
        second = service_feed([("T1b", "R1b", "SV1b")], [calendar_row("SV1b", MON_FRI, 20240101, 20240105)])

        with pytest.raises(ValueError, match="missing columns"):
            gm._match_services(first, second, pd.DataFrame({"first_trip_id": ["T1"]}))

    def test_same_effective_dates_via_different_representations_match(self, gm):
        # self: weekly calendar; other: the very same days as explicit calendar_dates
        first = service_feed([("T1", "R1", "SV1")], [calendar_row("SV1", MON_FRI, 20240101, 20240105)])
        second = service_feed(
            [("T1b", "R1b", "SV1b")],
            calendar_dates=[
                {"service_id": "SV1b", "date": d, "exception_type": 1} for d in sorted(WEEK)
            ],
        )

        result, _, _ = gm._match_services(first, second, matches_df([("T1", "T1b")]))

        assert len(result) == 1
        assert result.iloc[0]["self_service_id"] == "SV1"
        assert result.iloc[0]["other_service_id"] == "SV1b"

    def test_matched_trip_count_is_aggregated(self, gm):
        first = service_feed(
            [("T1", "R1", "SV1"), ("T2", "R1", "SV1")],
            [calendar_row("SV1", MON_FRI, 20240101, 20240105)],
        )
        second = service_feed(
            [("T1b", "R1b", "SV1b"), ("T2b", "R1b", "SV1b")],
            [calendar_row("SV1b", MON_FRI, 20240101, 20240105)],
        )

        result, _, _ = gm._match_services(first, second, matches_df([("T1", "T1b"), ("T2", "T2b")]))

        assert result.iloc[0]["matched_trip_count"] == 2

    def test_only_matching_pairs_are_kept_when_some_differ(self, gm):
        first = service_feed(
            [("T1", "R1", "SV1"), ("T2", "R1", "SV2")],
            [
                calendar_row("SV1", MON_FRI, 20240101, 20240105),
                calendar_row("SV2", ["saturday"], 20240106, 20240106),
            ],
        )
        second = service_feed(
            [("T1b", "R1b", "SVa"), ("T2b", "R1b", "SVb")],
            [
                calendar_row("SVa", MON_FRI, 20240101, 20240105),     # same as SV1
                calendar_row("SVb", ["sunday"], 20240107, 20240107),  # differs from SV2
            ],
        )

        result, _, _ = gm._match_services(first, second, matches_df([("T1", "T1b"), ("T2", "T2b")]))

        assert result.to_dict("records") == [
            {"self_service_id": "SV1", "other_service_id": "SVa", "matched_trip_count": 1}
        ]

    def test_matched_trip_missing_from_self_raises(self, gm):
        first = service_feed([("T1", "R1", "SV1")], [calendar_row("SV1", MON_FRI, 20240101, 20240105)])
        second = service_feed([("T1b", "R1b", "SV1b")], [calendar_row("SV1b", MON_FRI, 20240101, 20240105)])

        with pytest.raises(ValueError, match="matched trip from self has no service_id"):
            gm._match_services(first, second, matches_df([("GHOST", "T1b")]))

    def test_matched_trip_missing_from_other_raises(self, gm):
        first = service_feed([("T1", "R1", "SV1")], [calendar_row("SV1", MON_FRI, 20240101, 20240105)])
        second = service_feed([("T1b", "R1b", "SV1b")], [calendar_row("SV1b", MON_FRI, 20240101, 20240105)])

        with pytest.raises(ValueError, match="matched trip from other has no service_id"):
            gm._match_services(first, second, matches_df([("T1", "GHOST")]))

    def test_self_service_matching_two_other_services_raises(self, gm):
        # Two distinct `other` services with the exact same dates both match SV1.
        # Current behaviour: the one-to-one constraint raises. (Arguably both
        # could safely be mapped onto SV1 -- if you change that, update this test.)
        first = service_feed(
            [("T1", "R1", "SV1"), ("T2", "R1", "SV1")],
            [calendar_row("SV1", MON_FRI, 20240101, 20240105)],
        )
        second = service_feed(
            [("T1b", "R1b", "SVa"), ("T2b", "R1b", "SVb")],
            [
                calendar_row("SVa", MON_FRI, 20240101, 20240105),
                calendar_row("SVb", MON_FRI, 20240101, 20240105),
            ],
        )

        with pytest.raises(ValueError, match="A service from self matches multiple services from other"):
            gm._match_services(first, second, matches_df([("T1", "T1b"), ("T2", "T2b")]))

    def test_other_service_matching_two_self_services_raises(self, gm):
        first = service_feed(
            [("T1", "R1", "SVa"), ("T2", "R1", "SVb")],
            [
                calendar_row("SVa", MON_FRI, 20240101, 20240105),
                calendar_row("SVb", MON_FRI, 20240101, 20240105),
            ],
        )
        second = service_feed(
            [("T1b", "R1b", "SV1"), ("T2b", "R1b", "SV1")],
            [calendar_row("SV1", MON_FRI, 20240101, 20240105)],
        )

        with pytest.raises(ValueError, match="A service from other matches multiple services from self"):
            gm._match_services(first, second, matches_df([("T1", "T1b"), ("T2", "T2b")]))

    def test_integer_service_ids_match(self, gm):
        first = service_feed([("T1", "R1", 1)], [calendar_row(1, MON_FRI, 20240101, 20240105)])
        second = service_feed([("T1b", "R1b", 7)], [calendar_row(7, MON_FRI, 20240101, 20240105)])

        result, _, _ = gm._match_services(first, second, matches_df([("T1", "T1b")]))

        assert len(result) == 1
        assert int(result.iloc[0]["self_service_id"]) == 1
        assert int(result.iloc[0]["other_service_id"]) == 7


class TestServiceMapping:

    def test_empty_matches_return_empty_dict(self, gm):
        empty = pd.DataFrame(columns=["self_service_id", "other_service_id", "matched_trip_count"])

        assert gm._service_mapping(BareGTFS({}), empty) == {}

    def test_maps_other_to_self(self, gm):
        matches = pd.DataFrame([
            {"self_service_id": "SV1", "other_service_id": "SV1b", "matched_trip_count": 3},
            {"self_service_id": "SV2", "other_service_id": "SV2b", "matched_trip_count": 1},
        ])

        assert gm._service_mapping(BareGTFS({}), matches) == {"SV1b": "SV1", "SV2b": "SV2"}


# --------------------------------------------------------------------------
# _dates_to_calendar
# --------------------------------------------------------------------------

class TestDatesToCalendar:

    def test_empty_dates_raises(self, gm):
        with pytest.raises(ValueError, match="no operating dates"):
            gm._dates_to_calendar("SV1", set())

    def test_full_weekday_week_becomes_pure_weekly_service(self, gm):
        row, exceptions = gm._dates_to_calendar("SV1", WEEK)

        assert row["service_id"] == "SV1"
        assert [row[d] for d in WEEKDAYS] == [1, 1, 1, 1, 1, 0, 0]
        assert row["start_date"] == 20240101
        assert row["end_date"] == 20240105
        assert exceptions == []

    def test_start_and_end_dates_are_ints(self, gm):
        row, _ = gm._dates_to_calendar("SV1", {"20240101", "20240105"})

        assert isinstance(row["start_date"], int)
        assert isinstance(row["end_date"], int)

    def test_single_date_sets_only_its_weekday(self, gm):
        row, exceptions = gm._dates_to_calendar("SV1", {"20240103"})  # a Wednesday

        assert row["wednesday"] == 1
        assert sum(row[d] for d in WEEKDAYS) == 1
        assert row["start_date"] == row["end_date"] == 20240103
        assert exceptions == []

    def test_every_day_for_two_weeks_sets_all_weekdays_no_exceptions(self, gm):
        dates = {d.strftime("%Y%m%d") for d in pd.date_range("2024-01-01", "2024-01-14")}

        row, exceptions = gm._dates_to_calendar("SV1", dates)

        assert all(row[d] == 1 for d in WEEKDAYS)
        assert exceptions == []

    def test_weekday_that_only_partially_operates_is_not_set(self, gm):
        # Mondays in range: Jan 1, 8, 15 -> service runs on only two of them.
        row, exceptions = gm._dates_to_calendar("SV1", {"20240101", "20240115"})

        assert row["monday"] == 0
        assert sorted(exceptions, key=lambda e: e["date"]) == [
            {"service_id": "SV1", "date": "20240101", "exception_type": 1},
            {"service_id": "SV1", "date": "20240115", "exception_type": 1},
        ]

    def test_holiday_inside_a_weekday_pattern(self, gm):
        # Mon-Fri for two weeks, except Wed Jan 10. Wednesday then operates on only
        # 1 of 2 occurrences -> wednesday=0 plus one explicit addition (Jan 3).
        dates = {
            "20240101", "20240102", "20240103", "20240104", "20240105",
            "20240108", "20240109", "20240111", "20240112",
        }

        row, exceptions = gm._dates_to_calendar("SV1", dates)

        assert [row[d] for d in ("monday", "tuesday", "thursday", "friday")] == [1, 1, 1, 1]
        assert row["wednesday"] == 0
        assert exceptions == [{"service_id": "SV1", "date": "20240103", "exception_type": 1}]

    def test_exception_rows_have_expected_keys(self, gm):
        _, exceptions = gm._dates_to_calendar("SV1", {"20240101", "20240115"})

        assert exceptions
        for exc in exceptions:
            assert set(exc) == {"service_id", "date", "exception_type"}
            assert exc["exception_type"] in (1, 2)
            assert isinstance(exc["date"], str)

    @pytest.mark.parametrize(
        "dates",
        [
            WEEK,
            {"20240103"},                                                         # one day
            {"20240101", "20240115"},                                             # sparse Mondays
            {"20240106", "20240107", "20240113", "20240114", "20240120"},         # weekends + stray Saturday
            {"20231230", "20240102"},                                             # across a year boundary
            {
                "20240101", "20240102", "20240103", "20240104", "20240105",
                "20240108", "20240109", "20240111", "20240112",
            },                                                                    # holiday in the pattern
            {d.strftime("%Y%m%d") for d in pd.date_range("2024-02-01", "2024-03-15")},  # 6 weeks, daily
        ],
    )
    def test_round_trip_preserves_exact_dates(self, gm, dates):
        row, exceptions = gm._dates_to_calendar("SV1", dates)

        assert expand_calendar(row, exceptions) == dates

    def test_accepts_unsorted_input(self, gm):
        row_a, exc_a = gm._dates_to_calendar("SV1", {"20240105", "20240101", "20240103"})
        row_b, exc_b = gm._dates_to_calendar("SV1", {"20240101", "20240103", "20240105"})

        assert row_a == row_b
        assert exc_a == exc_b


# --------------------------------------------------------------------------
# merge_services
# --------------------------------------------------------------------------

class TestMergeServices:

    def test_returns_calendar_and_calendar_dates_frames(self, gm):
        result = gm.merge_services(
            self_dates={"SV1": WEEK},
            other_dates={},
            merged_trips=pd.DataFrame({"service_id": ["SV1"]}),
            service_mapping={},
        )

        assert len(result) == 2
        calendar, calendar_dates = result
        assert isinstance(calendar, pd.DataFrame)
        assert isinstance(calendar_dates, pd.DataFrame)

    def test_calendar_dates_has_schema_columns_even_when_empty(self, gm):
        _, calendar_dates = gm.merge_services(
            self_dates={"SV1": WEEK},
            other_dates={},
            merged_trips=pd.DataFrame({"service_id": ["SV1"]}),
            service_mapping={},
        )

        assert calendar_dates.empty
        assert list(calendar_dates.columns) == ["service_id", "date", "exception_type"]

    def test_self_services_used_by_trips_are_kept(self, gm):
        calendar, calendar_dates = gm.merge_services(
            self_dates={"SV1": WEEK, "SV2": WEEKEND},
            other_dates={},
            merged_trips=pd.DataFrame({"service_id": ["SV1", "SV2"]}),
            service_mapping={},
        )

        assert set(calendar["service_id"]) == {"SV1", "SV2"}
        assert effective_dates(gm, calendar, calendar_dates, "SV1") == WEEK
        assert effective_dates(gm, calendar, calendar_dates, "SV2") == WEEKEND

    def test_unused_self_service_is_dropped(self, gm):
        calendar, _ = gm.merge_services(
            self_dates={"SV1": WEEK, "UNUSED": WEEKEND},
            other_dates={},
            merged_trips=pd.DataFrame({"service_id": ["SV1"]}),
            service_mapping={},
        )

        assert set(calendar["service_id"]) == {"SV1"}

    def test_unused_other_service_is_dropped(self, gm):
        calendar, _ = gm.merge_services(
            self_dates={"SV1": WEEK},
            other_dates={"UNUSEDb": WEEKEND},
            merged_trips=pd.DataFrame({"service_id": ["SV1"]}),
            service_mapping={},
        )

        assert set(calendar["service_id"]) == {"SV1"}

    def test_unmatched_other_service_survives_with_its_own_dates(self, gm):
        calendar, calendar_dates = gm.merge_services(
            self_dates={"SV1": WEEK},
            other_dates={"SV2b": WEEKEND},
            merged_trips=pd.DataFrame({"service_id": ["SV1", "SV2b"]}),
            service_mapping={},
        )

        assert set(calendar["service_id"]) == {"SV1", "SV2b"}
        assert effective_dates(gm, calendar, calendar_dates, "SV2b") == WEEKEND

    def test_matched_other_service_uses_self_dates_and_gets_no_row(self, gm):
        # SV1b was remapped to SV1 in the merged trips. It must not get its own row,
        # and SV1 keeps self's dates even though other's dates differ.
        calendar, calendar_dates = gm.merge_services(
            self_dates={"SV1": WEEK},
            other_dates={"SV1b": WEEKEND},
            merged_trips=pd.DataFrame({"service_id": ["SV1", "SV1"]}),
            service_mapping={"SV1b": "SV1"},
        )

        assert list(calendar["service_id"]) == ["SV1"]
        assert effective_dates(gm, calendar, calendar_dates, "SV1") == WEEK

    def test_mixed_scenario(self, gm):
        calendar, calendar_dates = gm.merge_services(
            self_dates={"SV1": WEEK, "OLD": WEEKEND},
            other_dates={"SV1b": WEEK, "SV2b": WEEKEND},
            merged_trips=pd.DataFrame({"service_id": ["SV1", "SV1", "SV2b"]}),
            service_mapping={"SV1b": "SV1"},
        )

        assert set(calendar["service_id"]) == {"SV1", "SV2b"}
        assert effective_dates(gm, calendar, calendar_dates, "SV1") == WEEK
        assert effective_dates(gm, calendar, calendar_dates, "SV2b") == WEEKEND

    def test_service_without_known_dates_raises(self, gm):
        with pytest.raises(ValueError, match="Could not determine operating dates"):
            gm.merge_services(
                self_dates={"SV1": WEEK},
                other_dates={},
                merged_trips=pd.DataFrame({"service_id": ["SV1", "GHOST"]}),
                service_mapping={},
            )

    def test_irregular_service_roundtrips_through_calendar_and_exceptions(self, gm):
        irregular = {"20240101", "20240115", "20240120"}

        calendar, calendar_dates = gm.merge_services(
            self_dates={"SV1": irregular},
            other_dates={},
            merged_trips=pd.DataFrame({"service_id": ["SV1"]}),
            service_mapping={},
        )

        assert effective_dates(gm, calendar, calendar_dates, "SV1") == irregular

    def test_integer_mapping_keys_are_handled(self, gm):
        # _service_dates stringifies ids, while the mapping keeps trips.txt's native dtype.
        calendar, _ = gm.merge_services(
            self_dates={"1": WEEK},
            other_dates={"7": WEEK},
            merged_trips=pd.DataFrame({"service_id": [1, 1]}),
            service_mapping={7: 1},
        )

        assert set(calendar["service_id"].astype(str)) == {"1"}

    def test_does_not_mutate_input_date_sets(self, gm):
        self_dates = {"SV1": set(WEEK)}
        other_dates = {"SV2b": set(WEEKEND)}

        gm.merge_services(
            self_dates=self_dates,
            other_dates=other_dates,
            merged_trips=pd.DataFrame({"service_id": ["SV1", "SV2b"]}),
            service_mapping={},
        )

        assert self_dates == {"SV1": WEEK}
        assert other_dates == {"SV2b": WEEKEND}

    @pytest.mark.xfail(
        strict=False,
        reason="If an UNMATCHED service_id from `other` is identical to a service_id in `self` "
               "(very common: both feeds use '1', 'WD', ...), other's dates silently overwrite "
               "self's, and other's trips are never renamed. Self's trips then run on the wrong "
               "days. A fix would rename the colliding other-ids (in trips too), as is done for "
               "agency_ids.",
    )
    def test_colliding_unmatched_service_ids_do_not_overwrite_self(self, gm):
        calendar, calendar_dates = gm.merge_services(
            self_dates={"SV1": WEEK},
            other_dates={"SV1": WEEKEND},   # same id, different days, not matched
            merged_trips=pd.DataFrame({"service_id": ["SV1", "SV1"]}),
            service_mapping={},
        )

        assert effective_dates(gm, calendar, calendar_dates, "SV1") == WEEK


# --------------------------------------------------------------------------
# End to end through gm.merge()
# --------------------------------------------------------------------------

def _add_trip(feed, trip_id, route_id, service_id, stops):
    """Append a trip to a (real) GTFS feed. stops: [(stop_id, 'HH:MM:SS'), ...]"""
    feed.files["trips.txt"] = pd.concat(
        [
            feed.files["trips.txt"],
            pd.DataFrame([{"trip_id": trip_id, "route_id": route_id, "service_id": service_id}]),
        ],
        ignore_index=True,
    )
    feed.files["stop_times.txt"] = pd.concat(
        [
            feed.files["stop_times.txt"],
            pd.DataFrame([
                {
                    "trip_id": trip_id, "stop_sequence": i + 1, "stop_id": stop_id,
                    "arrival_time": time, "departure_time": time,
                }
                for i, (stop_id, time) in enumerate(stops)
            ]),
        ],
        ignore_index=True,
    )


class TestCalendarsEndToEnd:

    def test_merged_calendar_contains_exactly_the_services_in_use(self, gm, feed_a, feed_b):
        merged = gm.merge(feed_a, feed_b, create_transfers=False)

        calendar = merged.files["calendar.txt"]
        # SV1 (self) <-> SV1b (other) matched; SV2b unmatched survives.
        assert set(calendar["service_id"]) == {"SV1", "SV2b"}
        assert not calendar["service_id"].duplicated().any()

    def test_merged_calendar_rows_are_correct(self, gm, feed_a, feed_b):
        merged = gm.merge(feed_a, feed_b, create_transfers=False)

        calendar = merged.files["calendar.txt"].set_index("service_id")
        sv1 = calendar.loc["SV1"]
        assert [sv1[d] for d in WEEKDAYS] == [1, 1, 1, 1, 1, 0, 0]
        assert (sv1["start_date"], sv1["end_date"]) == (20240101, 20240105)

        sv2b = calendar.loc["SV2b"]
        assert [sv2b[d] for d in WEEKDAYS] == [0, 0, 0, 0, 0, 1, 1]
        assert (sv2b["start_date"], sv2b["end_date"]) == (20240106, 20240107)

    def test_merged_calendar_dates_is_empty_with_correct_columns(self, gm, feed_a, feed_b):
        merged = gm.merge(feed_a, feed_b, create_transfers=False)

        calendar_dates = merged.files["calendar_dates.txt"]
        assert calendar_dates.empty
        assert list(calendar_dates.columns) == ["service_id", "date", "exception_type"]

    def test_effective_dates_are_preserved_for_every_surviving_service(self, gm, feed_a, feed_b):
        self_dates = gm._service_dates(feed_a)
        other_dates = gm._service_dates(feed_b)

        merged = gm.merge(feed_a, feed_b, create_transfers=False)

        merged_dates = gm._service_dates(merged)
        assert merged_dates["SV1"] == self_dates["SV1"]
        assert merged_dates["SV2b"] == other_dates["SV2b"]

    def test_every_merged_trip_references_an_existing_service(self, gm, feed_a, feed_b):
        merged = gm.merge(feed_a, feed_b, create_transfers=False)

        trip_services = set(merged.files["trips.txt"]["service_id"])
        defined = set(merged.files["calendar.txt"]["service_id"]) | set(
            merged.files["calendar_dates.txt"]["service_id"]
        )
        assert trip_services <= defined

    def test_unmatched_other_trip_on_a_matched_service_is_remapped_to_self_service(
        self, gm, feed_a, feed_b
    ):
        # T3b is NOT a duplicate of any self trip (different time), but it runs on SV1b,
        # which has the same dates as self's SV1 -> it must be re-pointed to SV1.
        _add_trip(feed_b, "T3b", "R2b", "SV1b", [("STAb_1", "10:00:00"), ("STBb_1", "10:50:00")])

        merged = gm.merge(feed_a, feed_b, create_transfers=False)

        trips = merged.files["trips.txt"].set_index("trip_id")
        assert trips.loc["T3b", "service_id"] == "SV1"
        assert "SV1b" not in set(merged.files["calendar.txt"]["service_id"])

    def test_input_calendar_files_are_not_mutated(self, gm, feed_a, feed_b):
        calendar_a = feed_a.files["calendar.txt"].copy()
        calendar_b = feed_b.files["calendar.txt"].copy()

        gm.merge(feed_a, feed_b, create_transfers=False)

        pd.testing.assert_frame_equal(feed_a.files["calendar.txt"], calendar_a)
        pd.testing.assert_frame_equal(feed_b.files["calendar.txt"], calendar_b)