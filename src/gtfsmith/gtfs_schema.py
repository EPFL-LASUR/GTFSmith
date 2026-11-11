from platform import platform
from numpy import float64
from pandera.typing import Series
from typing import Optional
import pandera.pandas as pa
import pandas as pd


STANDARD_ROUTE_TYPES: set[int] = {
    0, 1, 2, 3, 4, 5, 6, 7, 11, 12
}

EXTENDED_ROUTE_TYPES: set[int] = {
    100, 101, 102, 103, 104, 105, 106, 107, 108, 109,
    110, 111, 112, 113, 114, 115, 116, 117,
    200, 201, 202, 203, 204, 205, 206, 207, 208, 209,
    400, 401, 402, 403, 404, 405,
    700, 701, 702, 703, 704, 705, 706, 707, 708, 709,
    710, 711, 712, 713, 714, 715, 716,
    800,
    900, 901, 902, 903, 904, 905, 906,
    1000,
    1100,
    1200,
    1300, 1301, 1302, 1303, 1304, 1305, 1306, 1307,
    1400,
    1500, 1501, 1502, 1503, 1504, 1505, 1506, 1507,
    1700, 1702,
}

ACCEPTED_ROUTE_TYPES: set[int] = (
    STANDARD_ROUTE_TYPES | EXTENDED_ROUTE_TYPES
)


class GTFS_Schema(object):
    class Agency(pa.DataFrameModel):
        agency_id: Series[str] = pa.Field(coerce=True)
        agency_name: Series[str] = pa.Field(coerce=True)
        agency_url: Series[str] = pa.Field(coerce=True)
        agency_timezone: Series[str] = pa.Field(coerce=True)
        agency_lang: Optional[Series[str]] = pa.Field(coerce=True, nullable=True)
        agency_phone: Optional[Series[str]] = pa.Field(coerce=True, nullable=True)
        agency_fare_url: Optional[Series[str]] = pa.Field(coerce=True, nullable=True)
        agency_email: Optional[Series[str]] = pa.Field(coerce=True, nullable=True)

    class Stops(pa.DataFrameModel):
        stop_id: Series[str] = pa.Field(coerce=True)
        stop_code: Optional[Series[str]] = pa.Field(coerce=True, nullable=True)
        stop_name: Series[str] = pa.Field(coerce=True)
        stop_desc: Optional[Series[str]] = pa.Field(coerce=True, nullable=True)
        stop_lat: Series[float64] = pa.Field(coerce=True, nullable=True)
        stop_lon: Series[float64] = pa.Field(coerce=True, nullable=True)
        zone_id: Optional[Series[str]] = pa.Field(coerce=True, nullable=True)
        stop_url: Optional[Series[str]] = pa.Field(coerce=True, nullable=True)
        location_type: Optional[Series[pd.Int64Dtype]] = pa.Field(
            coerce=True, nullable=True, isin=[0, 1, 2, 3, 4]
        )
        parent_station: Optional[Series[str]] = pa.Field(coerce=True, nullable=True)
        stop_timezone: Optional[Series[str]] = pa.Field(coerce=True, nullable=True)
        wheelchair_boarding: Optional[Series[pd.Int64Dtype]] = pa.Field(
            coerce=True, nullable=True, isin=[0, 1, 2]
        )
        level_id: Optional[Series[str]] = pa.Field(coerce=True, nullable=True)
        platform_code: Optional[Series[str]] = pa.Field(coerce=True, nullable=True)

    class Routes(pa.DataFrameModel):
        route_id: Series[str] = pa.Field(coerce=True)
        agency_id: Optional[Series[str]] = pa.Field(coerce=True, nullable=True)
        route_short_name: Optional[Series[str]] = pa.Field(coerce=True, nullable=True)
        route_long_name: Optional[Series[str]] = pa.Field(coerce=True, nullable=True)
        route_desc: Optional[Series[str]] = pa.Field(coerce=True, nullable=True)
        route_type: Series[int] = pa.Field(coerce=True, isin=ACCEPTED_ROUTE_TYPES)
        route_url: Optional[Series[str]] = pa.Field(coerce=True, nullable=True)
        route_color: Optional[Series[str]] = pa.Field(coerce=True, nullable=True)
        route_text_color: Optional[Series[str]] = pa.Field(coerce=True, nullable=True)
        route_sort_order: Optional[Series[pd.Int64Dtype]] = pa.Field(
            coerce=True, nullable=True
        )
        continuous_pickup: Optional[Series[pd.Int64Dtype]] = pa.Field(
            coerce=True, nullable=True, isin=[0, 1, 2, 3]
        )
        continuous_drop_off: Optional[Series[pd.Int64Dtype]] = pa.Field(
            coerce=True, nullable=True, isin=[0, 1, 2, 3]
        )

    class Trips(pa.DataFrameModel):
        route_id: Series[str] = pa.Field(coerce=True)
        service_id: Series[str] = pa.Field(coerce=True)
        trip_id: Series[str] = pa.Field(coerce=True)
        trip_headsign: Optional[Series[str]] = pa.Field(coerce=True, nullable=True)
        trip_short_name: Optional[Series[str]] = pa.Field(coerce=True, nullable=True)
        direction_id: Optional[Series[pd.Int64Dtype]] = pa.Field(
            coerce=True, nullable=True, isin=[0, 1]
        )
        block_id: Optional[Series[str]] = pa.Field(coerce=True, nullable=True)
        shape_id: Optional[Series[str]] = pa.Field(coerce=True, nullable=True)
        wheelchair_accessible: Optional[Series[pd.Int64Dtype]] = pa.Field(
            coerce=True, nullable=True, isin=[0, 1, 2]
        )
        bikes_allowed: Optional[Series[pd.Int64Dtype]] = pa.Field(
            coerce=True, nullable=True, isin=[0, 1, 2]
        )

    class Stop_Times(pa.DataFrameModel):
        trip_id: Series[str] = pa.Field(coerce=True)
        arrival_time: Series[str] = pa.Field(coerce=True, nullable=True)
        departure_time: Series[str] = pa.Field(coerce=True, nullable=True)
        stop_id: Series[str] = pa.Field(coerce=True)
        stop_sequence: Series[int] = pa.Field(coerce=True)
        stop_headsign: Optional[Series[str]] = pa.Field(coerce=True, nullable=True)
        pickup_type: Optional[Series[pd.Int64Dtype]] = pa.Field(
            coerce=True, nullable=True, isin=[0, 1, 2, 3]
        )
        drop_off_type: Optional[Series[pd.Int64Dtype]] = pa.Field(
            coerce=True, nullable=True, isin=[0, 1, 2, 3]
        )
        # continuous_pickup: Optional[Series[pd.Int64Dtype]] = pa.Field(
        #     coerce=True, nullable=True, isin=[0, 1, 2, 3]
        # )
        # continuous_drop_off: Optional[Series[pd.Int64Dtype]] = pa.Field(
        #     coerce=True, nullable=True, isin=[0, 1, 2, 3]
        # )
        shape_dist_traveled: Optional[Series[float64]] = pa.Field(
            coerce=True, nullable=True, ge=0
        )
        timepoint: Optional[Series[pd.Int64Dtype]] = pa.Field(
            coerce=True, nullable=True, isin=[0, 1]
        )

    class Calendar(pa.DataFrameModel):
        service_id: Series[str] = pa.Field(coerce=True)
        monday: Series[int] = pa.Field(isin=[0, 1], coerce=True)
        tuesday: Series[int] = pa.Field(isin=[0, 1], coerce=True)
        wednesday: Series[int] = pa.Field(isin=[0, 1], coerce=True)
        thursday: Series[int] = pa.Field(isin=[0, 1], coerce=True)
        friday: Series[int] = pa.Field(isin=[0, 1], coerce=True)
        saturday: Series[int] = pa.Field(isin=[0, 1], coerce=True)
        sunday: Series[int] = pa.Field(isin=[0, 1], coerce=True)
        start_date: Series[pd.Int64Dtype] = pa.Field(coerce=True)
        end_date: Series[pd.Int64Dtype] = pa.Field(coerce=True)

    class Calendar_Dates(pa.DataFrameModel):
        service_id: Series[str] = pa.Field(coerce=True)
        date: Series[str] = pa.Field(coerce=True)
        exception_type: Series[int] = pa.Field(coerce=True, isin=[1, 2])

    class Shapes(pa.DataFrameModel):
        shape_id: Series[str] = pa.Field(coerce=True)
        shape_pt_lat: Series[float64] = pa.Field(coerce=True)
        shape_pt_lon: Series[float64] = pa.Field(coerce=True)
        shape_pt_sequence: Series[int] = pa.Field(coerce=True)
        shape_dist_traveled: Optional[Series[float64]] = pa.Field(
            coerce=True, nullable=True
        )

    class Feed_Info(pa.DataFrameModel):
        feed_publisher_name: Series[str] = pa.Field(coerce=True)
        feed_publisher_url: Series[str] = pa.Field(coerce=True)
        feed_lang: Series[str] = pa.Field(coerce=True)
        feed_start_date: Optional[Series[str]] = pa.Field(coerce=True, nullable=True)
        feed_end_date: Optional[Series[str]] = pa.Field(coerce=True, nullable=True)
        feed_version: Optional[Series[str]] = pa.Field(coerce=True, nullable=True)
        feed_contact_email: Optional[Series[str]] = pa.Field(coerce=True, nullable=True)
        feed_contact_url: Optional[Series[str]] = pa.Field(coerce=True, nullable=True)


    class Transfers(pa.DataFrameModel):
        from_stop_id: Series[str] = pa.Field(coerce=True)
        to_stop_id: Series[str] = pa.Field(coerce=True)
        transfer_type: Series[int] = pa.Field(
            coerce=True,
            isin=[0, 1, 2, 3, 4, 5]
        )
        min_transfer_time: Optional[Series[pd.Int64Dtype]] = pa.Field(
            coerce=True,
            nullable=True,
            ge=0
        )
        from_route_id: Optional[Series[str]] = pa.Field(
            coerce=True,
            nullable=True
        )
        to_route_id: Optional[Series[str]] = pa.Field(
            coerce=True,
            nullable=True
        )
        from_trip_id: Optional[Series[str]] = pa.Field(
            coerce=True,
            nullable=True
        )
        to_trip_id: Optional[Series[str]] = pa.Field(
            coerce=True,
            nullable=True
        )

    class Frequencies(pa.DataFrameModel):
        trip_id: Series[str] = pa.Field(coerce=True)
        start_time: Series[str] = pa.Field(coerce=True)
        end_time: Series[str] = pa.Field(coerce=True)
        headway_secs: Series[int] = pa.Field(coerce=True, ge=1)
        exact_times: Optional[Series[pd.Int64Dtype]] = pa.Field(
            coerce=True,
            nullable=True,
            isin=[0, 1]
        )

    class Attributions(pa.DataFrameModel):
        attribution_id: Series[str] = pa.Field(coerce=True)
        organization_name: Series[str] = pa.Field(coerce=True)
        is_producer: Optional[Series[pd.Int64Dtype]] = pa.Field(
            coerce=True,
            nullable=True,
            isin=[0, 1]
        )
        attribution_url: Optional[Series[str]] = pa.Field(
            coerce=True,
            nullable=True
        )
        attribution_email: Optional[Series[str]] = pa.Field(
            coerce=True,
            nullable=True
        )

    class Translations(pa.DataFrameModel):
        table_name: Series[str] = pa.Field(coerce=True)
        field_name: Series[str] = pa.Field(coerce=True)
        record_id: Optional[Series[str]] = pa.Field(
            coerce=True,
            nullable=True
        )
        record_sub_id: Optional[Series[str]] = pa.Field(
            coerce=True,
            nullable=True
        )
        field_value: Optional[Series[str]] = pa.Field(coerce=True)
        language: Series[str] = pa.Field(coerce=True)
        translation: Series[str] = pa.Field(coerce=True)

    class Levels(pa.DataFrameModel):
        level_id: Series[str] = pa.Field(coerce=True)
        level_index: Series[float64] = pa.Field(coerce=True)
        level_name: Optional[Series[str]] = pa.Field(
            coerce=True,
            nullable=True
        )

    class Pathways(pa.DataFrameModel):
        pathway_id: Series[str] = pa.Field(coerce=True)
        from_stop_id: Series[str] = pa.Field(coerce=True)
        to_stop_id: Series[str] = pa.Field(coerce=True)
        pathway_mode: Series[int] = pa.Field(
            coerce=True,
            isin=[1, 2, 3, 4, 5, 6, 7]
        )
        is_bidirectional: Series[int] = pa.Field(
            coerce=True,
            isin=[0, 1]
        )
        traversal_time: Optional[Series[pd.Int64Dtype]] = pa.Field(
            coerce=True,
            nullable=True,
            ge=0
        )

    class Station_Tracks(pa.DataFrameModel):
        stop_id: Series[str] = pa.Field(coerce=True)
        track_ref: Series[str] = pa.Field(coerce=True)
        lat: Series[float64] = pa.Field(coerce=True)
        lon: Series[float64] = pa.Field(coerce=True)
        bearing: Optional[Series[float64]] = pa.Field(
            coerce=True,
            nullable=True
        )

    required_files = [
        "agency.txt",
        "stops.txt",
        "routes.txt",
        "trips.txt",
        "stop_times.txt",
    ]

    agency_columns = list(Agency.__annotations__.keys())
    stops_columns = list(Stops.__annotations__.keys())
    routes_columns = list(Routes.__annotations__.keys())
    trips_columns = list(Trips.__annotations__.keys())
    stop_times_columns = list(Stop_Times.__annotations__.keys())
    calendar_columns = list(Calendar.__annotations__.keys())
    calendar_dates_columns = list(Calendar_Dates.__annotations__.keys())
    shapes_columns = list(Shapes.__annotations__.keys())
    feed_info_columns = list(Feed_Info.__annotations__.keys())
    transfers_columns = list(Transfers.__annotations__.keys())
    frequencies_columns = list(Frequencies.__annotations__.keys())
    attributions_columns = list(Attributions.__annotations__.keys())
    translations_columns = list(Translations.__annotations__.keys())
    levels_columns = list(Levels.__annotations__.keys())
    pathways_columns = list(Pathways.__annotations__.keys())
    station_tracks_columns = list(Station_Tracks.__annotations__.keys())


    SCHEMAS = {
        "agency.txt": Agency,
        "stops.txt": Stops,
        "routes.txt": Routes,
        "trips.txt": Trips,
        "stop_times.txt": Stop_Times,
        "calendar.txt": Calendar,
        "calendar_dates.txt": Calendar_Dates,
        "shapes.txt": Shapes,
        "feed_info.txt": Feed_Info,
        "transfers.txt": Transfers,
        "frequencies.txt": Frequencies,
        "attributions.txt": Attributions,
        "translations.txt": Translations,
        "levels.txt": Levels,
        "pathways.txt": Pathways,
        "station_tracks.txt": Station_Tracks,
    }