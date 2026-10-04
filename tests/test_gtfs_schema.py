import pandas as pd
import pytest


def test_calendar_weekday_columns_coerce_from_strings(gtfs_schema):
    """Regression: the Calendar model's monday..sunday fields were
    missing coerce=True (every other int field in the schema has it),
    so a weekday column that wasn't already exactly `int` dtype -- e.g.
    strings straight out of a CSV -- failed validation instead of being
    coerced."""
    df = pd.DataFrame([
        {"service_id": "SV1", "monday": "1", "tuesday": "0", "wednesday": "1",
         "thursday": "0", "friday": "1", "saturday": "0", "sunday": "0",
         "start_date": 20240101, "end_date": 20240107},
    ])

    validated = gtfs_schema.GTFS_Schema.Calendar.validate(df)

    assert pd.api.types.is_integer_dtype(validated["monday"])
    assert validated["monday"].iloc[0] == 1


def test_calendar_rejects_a_weekday_value_outside_0_1(gtfs_schema):
    df = pd.DataFrame([
        {"service_id": "SV1", "monday": 2, "tuesday": 0, "wednesday": 0,
         "thursday": 0, "friday": 0, "saturday": 0, "sunday": 0,
         "start_date": 20240101, "end_date": 20240107},
    ])

    with pytest.raises(Exception):
        gtfs_schema.GTFS_Schema.Calendar.validate(df)


def test_routes_route_type_is_required_and_rejects_missing_values(gtfs_schema):
    """Regression: route_type was declared as a required (non-Optional)
    Series[int] field but also marked nullable=True, letting a missing
    route_type -- which the GTFS spec requires -- silently through. A
    downstream `routes["route_type"] == X` comparison on a column that
    can contain NaN can then raise instead of filtering cleanly."""
    df = pd.DataFrame([{"route_id": "R1", "route_type": None}])

    with pytest.raises(Exception):
        gtfs_schema.GTFS_Schema.Routes.validate(df)


def test_routes_route_type_coerces_from_string(gtfs_schema):
    df = pd.DataFrame([{"route_id": "R1", "route_type": "3"}])

    validated = gtfs_schema.GTFS_Schema.Routes.validate(df)

    assert validated["route_type"].iloc[0] == 3


def test_transfers_transfer_type_is_restricted_to_valid_codes(gtfs_schema):
    df = pd.DataFrame([{"from_stop_id": "A", "to_stop_id": "B", "transfer_type": 9}])

    with pytest.raises(Exception):
        gtfs_schema.GTFS_Schema.Transfers.validate(df)


def test_required_files_and_schemas_are_consistent(gtfs_schema):
    schema = gtfs_schema.GTFS_Schema
    # Every required file must actually have a schema registered for it.
    assert set(schema.required_files).issubset(set(schema.SCHEMAS))


def test_columns_lists_match_their_model_fields(gtfs_schema):
    schema = gtfs_schema.GTFS_Schema
    assert schema.transfers_columns == list(schema.Transfers.__annotations__.keys())
    assert schema.calendar_columns == list(schema.Calendar.__annotations__.keys())
    assert schema.stops_columns == list(schema.Stops.__annotations__.keys())