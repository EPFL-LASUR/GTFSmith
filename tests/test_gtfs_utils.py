from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest

from conftest import _stops_rows


# convert_to_seconds / to_hhmmss / _vectorized_to_seconds
def test_convert_to_seconds_handles_times_past_midnight(gtfs_utils):
    assert gtfs_utils.convert_to_seconds("25:30:15") == 25 * 3600 + 30 * 60 + 15


def test_convert_to_seconds_rejects_non_string_input(gtfs_utils):
    with pytest.raises(TypeError):
        gtfs_utils.convert_to_seconds(28800)


def test_to_hhmmss_round_trips_with_convert_to_seconds(gtfs_utils):
    for value in ["00:00:00", "08:05:09", "25:30:15"]:
        assert gtfs_utils.to_hhmmss(gtfs_utils.convert_to_seconds(value)) == value


def test_vectorized_to_seconds_matches_scalar_version(gtfs_utils):
    times = pd.Series(["08:00:00", "23:59:59", "25:00:00"])
    expected = [gtfs_utils.convert_to_seconds(t) for t in times]

    result = gtfs_utils._vectorized_to_seconds(times)

    assert list(result) == expected


# _normalize_stop_name
def test_normalize_stop_name_is_case_insensitive(gtfs_utils):
    """Regression: the function never lowercased its output, so two
    names differing only by case were treated as dissimilar by the
    (case-sensitive) fuzzy ratio used for stop matching."""
    assert gtfs_utils._normalize_stop_name("LAUSANNE GARE") == gtfs_utils._normalize_stop_name("Lausanne Gare")


def test_normalize_stop_name_folds_punctuation_to_spaces(gtfs_utils):
    """Regression: the docstring always claimed punctuation was replaced
    with spaces, but the code never did it -- "Paris-Montparnasse" and
    "Paris Montparnasse" normalized to different strings."""
    assert gtfs_utils._normalize_stop_name("Paris-Montparnasse") == gtfs_utils._normalize_stop_name("Paris Montparnasse")


def test_normalize_stop_name_strips_accents(gtfs_utils):
    assert gtfs_utils._normalize_stop_name("Genève") == gtfs_utils._normalize_stop_name("Geneve")


def test_normalize_stop_name_handles_missing_values(gtfs_utils):
    assert gtfs_utils._normalize_stop_name(None) == ""
    assert gtfs_utils._normalize_stop_name(float("nan")) == ""


# _normalize_platform_code
def test_normalize_platform_code_strips_and_uppercases(gtfs_utils):
    assert gtfs_utils._normalize_platform_code(" 3b ") == "3B"


def test_normalize_platform_code_empty_or_missing_is_none(gtfs_utils):
    assert gtfs_utils._normalize_platform_code("") is None
    assert gtfs_utils._normalize_platform_code("   ") is None
    assert gtfs_utils._normalize_platform_code(None) is None



# _get_parent_stations / _prepare_coordinates
def test_get_parent_stations_filters_location_type_1_only(gtfs_utils):
    stops = _stops_rows([
        ("P1", "Parent", 46.5, 6.6, 1, "", ""),
        ("C1", "Child", 46.5, 6.6, 0, "P1", ""),
    ])

    result = gtfs_utils._get_parent_stations(stops)

    assert list(result["stop_id"]) == ["P1"]


def test_get_parent_stations_handles_blank_location_type_without_raising(gtfs_utils):
    """location_type is Optional per the GTFS spec; a feed that leaves it
    blank for flat stops must not crash _get_parent_stations."""
    stops = pd.DataFrame([
        {"stop_id": "S1", "stop_name": "Flat stop", "stop_lat": 46.5, "stop_lon": 6.6, "location_type": None},
    ])

    result = gtfs_utils._get_parent_stations(stops)

    assert result.empty


def test_prepare_coordinates_drops_rows_with_missing_coords(gtfs_utils):
    stops = pd.DataFrame([
        {"stop_id": "S1", "stop_lat": 46.5, "stop_lon": 6.6},
        {"stop_id": "S2", "stop_lat": None, "stop_lon": 6.6},
    ])

    result = gtfs_utils._prepare_coordinates(stops)

    assert list(result["stop_id"]) == ["S1"]


# _build_child_to_parent_mapping -- the critical fix
def test_build_child_to_parent_mapping_maps_platforms_to_their_station(gtfs_utils):
    stops = _stops_rows([
        ("STA", "Station", 46.5, 6.6, 1, "", ""),
        ("STA_1", "Platform 1", 46.5, 6.6, 0, "STA", "1"),
    ])

    mapping = gtfs_utils._build_child_to_parent_mapping(stops)

    assert mapping == {"STA": "STA", "STA_1": "STA"}


def test_build_child_to_parent_mapping_maps_flat_stops_to_themselves(gtfs_utils):
    """Regression (critical): flat stops with no parent_station and no
    station hierarchy (location_type blank/0) used to be silently
    DROPPED from the mapping entirely, instead of resolving to
    themselves the way a parent station does. Downstream, gtfs_merge.py's
    trip-signature matching requires every stop referenced by
    stop_times.txt to resolve to *some* station id, so this broke the
    whole merge pipeline for any feed without a full station hierarchy
    -- i.e. most bus-only / non-hierarchical real-world feeds."""
    stops = _stops_rows([
        ("S1", "Flat stop, no hierarchy", 46.5, 6.6, 0, "", ""),
    ])

    mapping = gtfs_utils._build_child_to_parent_mapping(stops)

    assert mapping == {"S1": "S1"}


def test_build_child_to_parent_mapping_handles_missing_location_type_column_values(gtfs_utils):
    stops = pd.DataFrame([
        {"stop_id": "S1", "stop_name": "Flat", "parent_station": None},
    ])

    mapping = gtfs_utils._build_child_to_parent_mapping(stops)

    assert mapping == {"S1": "S1"}


def test_build_child_to_parent_mapping_every_stop_gets_a_key(gtfs_utils):
    stops = _stops_rows([
        ("STA", "Station A", 46.5, 6.6, 1, "", ""),
        ("STA_1", "Platform", 46.5, 6.6, 0, "STA", "1"),
        ("FLAT1", "Flat stop", 46.6, 6.7, 0, "", ""),
        ("FLAT2", "Another flat stop", 46.7, 6.8, None, None, ""),
    ])

    mapping = gtfs_utils._build_child_to_parent_mapping(stops)

    assert set(mapping.keys()) == {"STA", "STA_1", "FLAT1", "FLAT2"}
    assert mapping["FLAT1"] == "FLAT1"
    assert mapping["FLAT2"] == "FLAT2"



# _latlon_to_xy
def test_latlon_to_xy_distance_matches_a_known_short_offset(gtfs_utils):
    # Roughly 111.2m per 0.001 degree of latitude.
    stops = pd.DataFrame([
        {"stop_lat": 46.500, "stop_lon": 6.600},
        {"stop_lat": 46.501, "stop_lon": 6.600},
    ])

    xy = gtfs_utils._latlon_to_xy(stops, reference_lat=46.5)
    distance = float(np.linalg.norm(xy[0] - xy[1]))

    assert distance == pytest.approx(111.2, rel=0.01)



# _find_nearby_parent_stations / _build_parent_merge_mapping
def test_find_nearby_parent_stations_returns_each_pair_once(gtfs_utils):
    stops = _stops_rows([
        ("P1", "Station A", 46.5000, 6.6000, 1, "", ""),
        ("P2", "Station A bis", 46.5001, 6.6001, 1, "", ""),  # ~15m away
        ("P3", "Far away", 10.0, 10.0, 1, "", ""),
    ])

    result = gtfs_utils._find_nearby_parent_stations(stops, radius_m=300.0)

    assert len(result) == 1
    row = result.iloc[0]
    assert {row["first_stop_id"], row["second_stop_id"]} == {"P1", "P2"}



def test_build_parent_merge_mapping_picks_the_earliest_stop_as_winner(gtfs_utils):
    stops = _stops_rows([
        ("P1", "First in file", 46.5000, 6.6000, 1, "", ""),
        ("P2", "Second in file", 46.5001, 6.6001, 1, "", ""),
    ])
    nearby = pd.DataFrame([{"first_stop_id": "P2", "second_stop_id": "P1", "distance_m": 15.0,}])

    mapping = gtfs_utils._build_parent_merge_mapping(stops, nearby)

    # Should produce losing_id -> winning_id
    assert mapping == {"P2": "P1"}


def test_remap_parent_references_is_a_noop_with_empty_mapping(gtfs_utils):
    from conftest import BareGTFS

    stops = _stops_rows([("P1", "Station", 46.5, 6.6, 1, "", "")])
    gtfs = BareGTFS({"stops.txt": stops})

    result = gtfs_utils._remap_parent_references(gtfs, {})

    assert result is gtfs



# valid_id
def test_valid_id_accepts_blank_or_known_ids(gtfs_utils):
    column = pd.Series(["R1", "", None, "R404"])
    valid_ids = {"R1"}

    result = gtfs_utils.valid_id(column, valid_ids)

    assert list(result) == [True, True, True, False]



# get_coordinates (network call -- mocked)
def test_get_coordinates_returns_lat_lon_on_success(gtfs_utils):
    fake_location = MagicMock(latitude=46.5, longitude=6.6)
    with patch("gtfsmith.gtfs_utils.Nominatim") as mock_nominatim:
        mock_nominatim.return_value.geocode.return_value = fake_location
        result = gtfs_utils.get_coordinates("Lausanne")

    assert result == (46.5, 6.6)


def test_get_coordinates_returns_none_none_when_not_found(gtfs_utils):
    with patch("gtfsmith.gtfs_utils.Nominatim") as mock_nominatim:
        mock_nominatim.return_value.geocode.return_value = None
        result = gtfs_utils.get_coordinates("Nowhereville")

    assert result == (None, None)


def test_get_coordinates_returns_none_none_on_exception(gtfs_utils):
    with patch("gtfsmith.gtfs_utils.Nominatim") as mock_nominatim:
        mock_nominatim.return_value.geocode.side_effect = RuntimeError("network down")
        result = gtfs_utils.get_coordinates("Lausanne")

    assert result == (None, None)



# normalize_route_type
@pytest.mark.parametrize("extended,basic", [
    (100, 2), (199, 2),    # rail
    (401, 1),              # metro
    (700, 3), (202, 3), (1500, 3),  # bus
    (900, 0),              # tram
    (1000, 4),             # ferry
    (1300, 6), (1303, 6),  # gondola
    (1400, 7),             # funicular
])
def test_normalize_route_type_maps_extended_types_to_basic(gtfs_utils, extended, basic):
    assert gtfs_utils.normalize_route_type(extended) == basic


@pytest.mark.parametrize("already_basic", [0, 1, 2, 3, 4, 5, 6, 7])
def test_normalize_route_type_passes_through_already_basic_types(gtfs_utils, already_basic):
    assert gtfs_utils.normalize_route_type(already_basic) == already_basic