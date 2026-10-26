import warnings

import pandas as pd
import pytest


def test_construction_succeeds_with_only_the_required_files(GTFS, minimal_files):
    gtfs = GTFS(minimal_files)
    assert set(gtfs.files.keys()) == {"agency.txt", "stops.txt", "routes.txt", "trips.txt", "stop_times.txt"}


def test_missing_required_file_raises(GTFS, minimal_files):
    files = minimal_files
    del files["stop_times.txt"]

    with pytest.raises(ValueError, match="stop_times.txt"):
        GTFS(files)


def test_unexpected_file_is_kept_but_warns(GTFS, minimal_files):
    files = minimal_files
    files["my_custom_extension.txt"] = pd.DataFrame([{"a": 1}])

    with pytest.warns(UserWarning, match="my_custom_extension.txt"):
        gtfs = GTFS(files)

    assert "my_custom_extension.txt" in gtfs.files
    # Unrecognised files pass through untouched -- no schema to validate against.
    assert gtfs.files["my_custom_extension.txt"].equals(files["my_custom_extension.txt"])


def test_known_file_is_coerced_to_its_schema_dtypes(GTFS, minimal_files):
    files = minimal_files
    # route_type arrives as a string, as it would straight out of a CSV.
    files["routes.txt"] = pd.DataFrame([{"route_id": "R1", "route_type": "3"}])

    gtfs = GTFS(files)

    assert pd.api.types.is_integer_dtype(gtfs.files["routes.txt"]["route_type"])
    assert gtfs.files["routes.txt"]["route_type"].iloc[0] == 3


def test_invalid_value_for_a_constrained_field_is_rejected(GTFS, minimal_files):
    files = minimal_files
    # location_type only allows 0-4.
    files["stops.txt"]["location_type"] = [9]

    with pytest.raises(Exception):
        GTFS(files)


def test_export_and_reimport_round_trips_through_a_zip(GTFS, tmp_path, minimal_files):
    gtfs = GTFS(minimal_files)
    zip_path = tmp_path / "feed.zip"

    gtfs.export_gtfs(zip_path)
    assert zip_path.exists()

    reloaded = GTFS.from_zip(zip_path)

    assert set(reloaded.files.keys()) == set(gtfs.files.keys())
    assert reloaded.files["stops.txt"]["stop_id"].tolist() == ["S1"]
    assert reloaded.files["trips.txt"]["trip_id"].tolist() == ["T1"]


def test_from_zip_rejects_a_missing_file(GTFS, tmp_path):
    with pytest.raises(FileNotFoundError):
        GTFS.from_zip(tmp_path / "does_not_exist.zip")


def test_from_zip_rejects_a_non_zip_file(GTFS, tmp_path):
    not_a_zip = tmp_path / "feed.txt"
    not_a_zip.write_text("not a zip")

    with pytest.raises(ValueError, match="ZIP"):
        GTFS.from_zip(not_a_zip)


def test_from_zip_strips_column_whitespace_and_blanks_become_na(GTFS, tmp_path):
    import zipfile

    zip_path = tmp_path / "feed.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("agency.txt", "agency_id,agency_name ,agency_url,agency_timezone\nAG1,A,http://a.example,Europe/Zurich\n")
        zf.writestr("stops.txt", "stop_id,stop_name,stop_lat,stop_lon,stop_code\nS1,Stop One,46.5,6.6,   \n")
        zf.writestr("routes.txt", "route_id,route_type\nR1,3\n")
        zf.writestr("trips.txt", "route_id,service_id,trip_id\nR1,SV1,T1\n")
        zf.writestr("stop_times.txt", "trip_id,arrival_time,departure_time,stop_id,stop_sequence\nT1,08:00:00,08:00:00,S1,1\n")

    gtfs = GTFS.from_zip(zip_path)

    # The trailing space in " agency_name " must have been stripped so the
    # schema's required "agency_name" column is actually found.
    assert "agency_name" in gtfs.files["agency.txt"].columns
    # A whitespace-only stop_code must have become missing, not a literal
    # string of spaces.
    assert pd.isna(gtfs.files["stops.txt"]["stop_code"].iloc[0])