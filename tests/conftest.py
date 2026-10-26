"""
Shared fixtures for the whole gtfs_toolkit test suite.

Earlier revisions of this suite (for gtfs_merge.py alone) ran against
hand-written fakes of gtfs_class/gtfs_utils/gtfs_schema/gtfs_operations,
since those modules didn't exist yet. They do now, so this file imports
the real ones directly -- no more shimming sys.modules. If a test fails
here, it's exercising real behaviour, not a stand-in's approximation of it.

Two kinds of GTFS-like object are used across the suite:

  * `BareGTFS` -- a minimal `.files` dict wrapper with NO schema
    validation, for unit tests that exercise a single function and only
    need a couple of specific files/columns (e.g. just stops.txt to test
    _match_parent_stations). Forcing every such test to build a fully
    valid 5-file feed would be a lot of unrelated boilerplate.
  * The real `GTFS` class -- used for anything that needs genuine
    end-to-end validation, and for the `feed_a`/`feed_b` fixtures below,
    since gtfs_merge.merge() constructs a real `GTFS(...)` internally and
    the full-pipeline tests need that to succeed.
"""

from __future__ import annotations

import importlib

import pandas as pd
import pytest




class BareGTFS:
    """`.files`-only stand-in, no schema validation. See module docstring."""

    def __init__(self, files):
        self.files = dict(files)


@pytest.fixture(scope="session")
def gm():
    """The real gtfs_merge module."""
    return importlib.import_module(f"gtfsmith.gtfs_merge")


@pytest.fixture(scope="session")
def gtfs_utils(gm):
    return gm.gtfs_utils


@pytest.fixture(scope="session")
def GTFS_Schema(gm):
    return gm.GTFS_Schema


@pytest.fixture(scope="session")
def GTFS(gm):
    return gm.GTFS


@pytest.fixture(scope="session")
def gtfs_operations():
    return importlib.import_module("gtfsmith.gtfs_operations")


@pytest.fixture(scope="session")
def gtfs_class():
    return importlib.import_module("gtfsmith.gtfs_class")


@pytest.fixture(scope="session")
def gtfs_schema():
    return importlib.import_module("gtfsmith.gtfs_schema")


# --------------------------------------------------------------------------
# Sample feeds, built as real (schema-validated) GTFS objects.
#
# Feed A (self, authoritative) and Feed B (other) share two physical
# stations (Lausanne, Geneve) that should be matched by geography + name,
# plus one trip each that describes the same physical journey and should
# be matched across feeds. Each feed also carries data designed to
# exercise a specific merge path:
#   - STAb_3 (feed B): a brand-new platform at an otherwise-matched
#     station -> exercises transfer propagation.
#   - STC / STC_1 (feed B): an entirely unmatched station -> survives
#     untouched.
#   - AG2 / "Agency Two" (both feeds): same agency_id, unrelated agency,
#     not reachable via any matched route -> exercises the agency_id
#     rename-on-conflict path.
#   - T0 / T2b: one unmatched trip per feed, each on its own route.
# --------------------------------------------------------------------------

def _stops_rows(rows):
    return pd.DataFrame(rows, columns=[
        "stop_id", "stop_name", "stop_lat", "stop_lon",
        "location_type", "parent_station", "platform_code",
    ])


def make_feed_a(GTFS):
    stops = _stops_rows([
        ("STA", "Lausanne Gare", 46.5170, 6.6323, 1, "", ""),
        ("STA_1", "Lausanne Gare - Platform 1", 46.5170, 6.6323, 0, "STA", "1"),
        ("STA_2", "Lausanne Gare - Platform 2", 46.5171, 6.6324, 0, "STA", "2"),
        ("STB", "Geneve Gare", 46.2102, 6.1427, 1, "", ""),
        ("STB_1", "Geneve Gare - Platform 1", 46.2102, 6.1427, 0, "STB", "1"),
    ])

    trips = pd.DataFrame([
        {"trip_id": "T1", "route_id": "R1", "service_id": "SV1"},
        {"trip_id": "T0", "route_id": "R0", "service_id": "SV1"},
    ])

    stop_times = pd.DataFrame([
        {"trip_id": "T1", "stop_sequence": 1, "stop_id": "STA_1", "arrival_time": "08:00:00", "departure_time": "08:00:00"},
        {"trip_id": "T1", "stop_sequence": 2, "stop_id": "STB_1", "arrival_time": "08:50:00", "departure_time": "08:50:00"},
        {"trip_id": "T0", "stop_sequence": 1, "stop_id": "STA_1", "arrival_time": "14:00:00", "departure_time": "14:00:00"},
        {"trip_id": "T0", "stop_sequence": 2, "stop_id": "STB_1", "arrival_time": "14:50:00", "departure_time": "14:50:00"},
    ])

    routes = pd.DataFrame([
        {"route_id": "R1", "agency_id": "AG1", "route_short_name": "IC1", "route_long_name": "Lausanne - Geneve", "route_type": 2},
        {"route_id": "R0", "agency_id": "AG2", "route_short_name": "IC0", "route_long_name": "Lausanne - Geneve (off-peak)", "route_type": 2},
    ])

    agency = pd.DataFrame([
        {"agency_id": "AG1", "agency_name": "Agency One", "agency_url": "http://a1.example", "agency_timezone": "Europe/Zurich"},
        {"agency_id": "AG2", "agency_name": "Agency Two (Self)", "agency_url": "http://a2.example", "agency_timezone": "Europe/Zurich"},
    ])

    calendar = pd.DataFrame([
        {"service_id": "SV1", "monday": 1, "tuesday": 1, "wednesday": 1, "thursday": 1,
         "friday": 1, "saturday": 0, "sunday": 0, "start_date": 20240101, "end_date": 20240105},
    ])

    calendar_dates = pd.DataFrame(columns=["service_id", "date", "exception_type"])

    transfers = pd.DataFrame([
        {"from_stop_id": "STA_1", "to_stop_id": "STA_2", "from_route_id": "", "to_route_id": "",
         "from_trip_id": "", "to_trip_id": "", "transfer_type": "2", "min_transfer_time": 120},
        {"from_stop_id": "STA_2", "to_stop_id": "STA_1", "from_route_id": "", "to_route_id": "",
         "from_trip_id": "", "to_trip_id": "", "transfer_type": "2", "min_transfer_time": 120},
    ])

    return GTFS({
        "stops.txt": stops,
        "trips.txt": trips,
        "stop_times.txt": stop_times,
        "routes.txt": routes,
        "agency.txt": agency,
        "calendar.txt": calendar,
        "calendar_dates.txt": calendar_dates,
        "transfers.txt": transfers,
    })


def make_feed_b(GTFS):
    stops = _stops_rows([
        ("STAb", "Lausanne Gare", 46.5171, 6.6322, 1, "", ""),        # matches STA
        ("STAb_1", "Lausanne - Platform 1", 46.5171, 6.6322, 0, "STAb", "1"),  # matches STA_1
        ("STAb_3", "Lausanne - Platform 3", 46.5171, 6.6322, 0, "STAb", "3"),  # NEW, no match
        ("STBb", "Geneve Gare", 46.2103, 6.1428, 1, "", ""),          # matches STB
        ("STBb_1", "Geneve - Platform 1", 46.2103, 6.1428, 0, "STBb", "1"),    # matches STB_1
        ("STC", "Zurich HB", 47.3769, 8.5417, 1, "", ""),              # unmatched (far away)
        ("STC_1", "Zurich HB - Platform 1", 47.3769, 8.5417, 0, "STC", "1"),
    ])

    trips = pd.DataFrame([
        {"trip_id": "T1b", "route_id": "R1b", "service_id": "SV1b"},
        {"trip_id": "T2b", "route_id": "R2b", "service_id": "SV2b"},
    ])

    stop_times = pd.DataFrame([
        {"trip_id": "T1b", "stop_sequence": 1, "stop_id": "STAb_1", "arrival_time": "08:00:00", "departure_time": "08:00:00"},
        {"trip_id": "T1b", "stop_sequence": 2, "stop_id": "STBb_1", "arrival_time": "08:50:30", "departure_time": "08:50:30"},
        {"trip_id": "T2b", "stop_sequence": 1, "stop_id": "STAb_3", "arrival_time": "09:00:00", "departure_time": "09:00:00"},
        {"trip_id": "T2b", "stop_sequence": 2, "stop_id": "STC_1", "arrival_time": "11:00:00", "departure_time": "11:00:00"},
    ])

    routes = pd.DataFrame([
        {"route_id": "R1b", "agency_id": "AG1", "route_short_name": "IC1", "route_long_name": "Lausanne - Geneve", "route_type": 2},
        {"route_id": "R2b", "agency_id": "AG2", "route_short_name": "RE2", "route_long_name": "Lausanne - Zurich", "route_type": 2},
    ])

    agency = pd.DataFrame([
        # Same agency_id "AG1" as feed A -- reached via a matched route
        # (R1b/T1b), so this row should be dropped in favour of self's AG1.
        {"agency_id": "AG1", "agency_name": "Agency One (Other)", "agency_url": "http://a1-other.example", "agency_timezone": "Europe/Zurich"},
        # Same agency_id "AG2" as feed A but an UNRELATED agency (only
        # reachable via R2b, which never matches) -> id conflict, must be
        # renamed rather than silently merged into self's AG2.
        {"agency_id": "AG2", "agency_name": "Agency Two (Other, unrelated)", "agency_url": "http://a2-other.example", "agency_timezone": "Europe/Zurich"},
    ])

    calendar = pd.DataFrame([
        # Identical effective dates to feed A's SV1 -> should match.
        {"service_id": "SV1b", "monday": 1, "tuesday": 1, "wednesday": 1, "thursday": 1,
         "friday": 1, "saturday": 0, "sunday": 0, "start_date": 20240101, "end_date": 20240105},
        # Disjoint dates -> stays unmatched, survives as its own service.
        {"service_id": "SV2b", "monday": 0, "tuesday": 0, "wednesday": 0, "thursday": 0,
         "friday": 0, "saturday": 1, "sunday": 1, "start_date": 20240106, "end_date": 20240107},
    ])

    calendar_dates = pd.DataFrame(columns=["service_id", "date", "exception_type"])

    return GTFS({
        "stops.txt": stops,
        "trips.txt": trips,
        "stop_times.txt": stop_times,
        "routes.txt": routes,
        "agency.txt": agency,
        "calendar.txt": calendar,
        "calendar_dates.txt": calendar_dates,
        # Deliberately no transfers.txt for feed B.
    })


@pytest.fixture
def feed_a(GTFS):
    """Function-scoped: merge() mutates its inputs in place (e.g. via
    frequencies_to_trips), so every test gets a fresh feed."""
    return make_feed_a(GTFS)


@pytest.fixture
def feed_b(GTFS):
    return make_feed_b(GTFS)

@pytest.fixture
def minimal_files():
    return {
        "agency.txt": pd.DataFrame([
            {"agency_id": "AG1", "agency_name": "A", "agency_url": "http://a.example", "agency_timezone": "Europe/Zurich"},
        ]),
        "stops.txt": pd.DataFrame([
            {"stop_id": "S1", "stop_name": "Stop One", "stop_lat": 46.5, "stop_lon": 6.6},
        ]),
        "routes.txt": pd.DataFrame([
            {"route_id": "R1", "route_type": 3},
        ]),
        "trips.txt": pd.DataFrame([
            {"route_id": "R1", "service_id": "SV1", "trip_id": "T1"},
        ]),
        "stop_times.txt": pd.DataFrame([
            {"trip_id": "T1", "arrival_time": "08:00:00", "departure_time": "08:00:00", "stop_id": "S1", "stop_sequence": 1},
        ]),
    }