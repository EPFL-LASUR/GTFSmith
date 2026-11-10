import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np
import pandas as pd
from tqdm import tqdm

from .gtfs_class import GTFS
from transitforge import TransitRouteNetwork


# set before forking worker processes
_BUILDER = None 


@dataclass
class _ShapePattern:
    """
    One unique:

        route + ordered stop sequence

    combination.

    All trips sharing the exact same route and stop order share one
    generated shape.
    """

    route_id: str
    stop_ids: tuple[str, ...]
    trip_ids: list[str]

    ref: str | None
    route_type_label: str | None

    from_stop: tuple[float, float]
    to_stop: tuple[float, float]

    # All stop coordinates in GTFS order.
    stop_coords: tuple[tuple[float, float], ...]


class GTFSShapeBuilder:
    """
    Fills missing shapes.txt geometry for a GTFS feed using an
    OSM TransitRouteNetwork.

    Shape reconstruction is performed using the COMPLETE ordered
    GTFS stop sequence.

    Example:

        GTFS:

            A -> B -> C -> D -> E

        The builder will:

            1. Match A -> B -> C -> D -> E against one OSM route.
            2. Extract OSM geometry for:
                   A -> B
                   B -> C
                   C -> D
                   D -> E
            3. Concatenate the geometry.
            4. Write one GTFS shape containing the entire route.

    workers:
        1     -> run in this process (default)
        None  -> use all cores (requires the "fork" start method,
                 i.e. Linux/macOS/WSL)
        n > 1 -> use n worker processes

    diagnose:
        When True, every pattern without a shape is analysed and listed
        in `builder.report` (a DataFrame) after build(), with the ref used
        and the reason it failed.
    """

    def __init__(
        self,
        gtfs: GTFS,
        network: TransitRouteNetwork,
        max_stop_distance_m: float = 750.0,
        workers: int | None = 1,
        diagnose: bool = True,
    ):

        self.GTFS_ROUTE_TYPE_LABELS: dict[int, str] = {
            0: "TRAM",
            1: "SUBWAY",
            2: "RAIL",
            3: "BUS",
            4: "FERRY",
            5: "CABLE_TRAM",
            6: "CABLE_CAR",
            7: "FUNICULAR",
            11: "TROLLEYBUS",
            12: "METRO",
        }

        self.gtfs = gtfs
        self.network = network
        self.max_stop_distance_m = max_stop_distance_m
        self.workers = workers
        self.diagnose = diagnose 

        # Filled during build()
        self.report = pd.DataFrame()
        self._skipped: list[dict] = []
        self._stop_names: dict = {}

    # ==================================================================
    # MAIN BUILD
    # ==================================================================

    def build(self) -> "GTFS":

        print("1/4 Finding trips without shapes...")

        trips_needing_shapes = self._trips_missing_shapes()

        print(f"Found {len(trips_needing_shapes)} trips without a valid shape")

        if trips_needing_shapes.empty:
            return self.gtfs

        print("2/4 Grouping trips by stop pattern...")

        patterns = self._build_patterns(trips_needing_shapes)

        print(f"Found {len(patterns)} unique stop patterns")

        print("3/4 Matching patterns to OSM routes...")

        new_shape_frames, pattern_shape_ids = self._build_shapes(patterns)

        unmatched = len(patterns) - len(pattern_shape_ids)

        print(f"Built {len(pattern_shape_ids)} shapes ({unmatched} unmatched)")

        if not pattern_shape_ids:
            print("No shapes could be generated - returning feed unchanged.")

            return self.gtfs

        print("4/4 Merging shapes into the GTFS feed...")

        self._merge_into_gtfs(new_shape_frames, pattern_shape_ids, patterns)

        assigned_trip_count = sum(len(patterns[key].trip_ids) for key in pattern_shape_ids)

        print(f"Assigned shapes to {assigned_trip_count} trips")

        return self.gtfs

    # ==================================================================
    # FIND TRIPS MISSING SHAPES
    # ==================================================================

    def _trips_missing_shapes(self) -> pd.DataFrame:
        """
        Return trips that don't have a resolvable shape_id.

        A shape is considered missing if:

            - shape_id column doesn't exist;
            - shape_id is null;
            - shape_id doesn't exist in shapes.txt.
        """

        trips = self.gtfs.files["trips.txt"]

        if "shape_id" not in trips.columns:
            return trips.copy()

        if "shapes.txt" in self.gtfs.files:
            existing_shape_ids = set(self.gtfs.files["shapes.txt"]["shape_id"])
        else:
            existing_shape_ids = set()

        missing_mask = (
            trips["shape_id"].isna()
            | ~trips["shape_id"].isin(existing_shape_ids)
        )

        return trips.loc[missing_mask].copy()

    # ==================================================================
    # BUILD STOP PATTERNS
    # ==================================================================

    def _build_patterns(
        self,
        trips: pd.DataFrame,
    ) -> dict[tuple, _ShapePattern]:
        """
        Group trips by:

            (route_id, ordered stop sequence)

        This ensures that identical trips share one generated shape.
        Stop coordinates are only looked up for the first trip of each
        pattern.

        Patterns that cannot even be attempted (fewer than 2 stops, a stop
        without coordinates) are collected in self._skipped so that they
        show up in the final report instead of disappearing silently.
        """

        stop_times = self.gtfs.files["stop_times.txt"]
        stops = self.gtfs.files["stops.txt"]
        routes = self.gtfs.files["routes.txt"]

        # --------------------------------------------------------------
        # GTFS stop coordinates and names
        # --------------------------------------------------------------

        valid = stops.dropna(subset=["stop_lat", "stop_lon"])

        stop_coords_by_id = dict(
            zip(
                valid["stop_id"],
                zip(
                    valid["stop_lat"].astype(float),
                    valid["stop_lon"].astype(float),
                ),
            )
        )

        names = (
            dict(zip(stops["stop_id"], stops["stop_name"]))
            if "stop_name" in stops.columns
            else {}
        )
        self._stop_names = names

        # --------------------------------------------------------------
        # Route metadata
        # --------------------------------------------------------------

        route_info = {
            row.route_id: (
                self._pick_ref(row),
                self.GTFS_ROUTE_TYPE_LABELS.get(int(row.route_type)),
            )
            for row in routes.itertuples()
        }

        # --------------------------------------------------------------
        # Ordered stop sequence per trip (only the needed columns)
        # --------------------------------------------------------------

        wanted = stop_times["trip_id"].isin(trips["trip_id"])

        ordered = stop_times.loc[
            wanted,
            ["trip_id", "stop_id", "stop_sequence"],
        ].sort_values(["trip_id", "stop_sequence"])

        trip_stop_sequences = (
            ordered
            .groupby("trip_id", sort=False)["stop_id"]
            .agg(tuple)
        )

        del ordered

        trip_route = dict(zip(trips["trip_id"], trips["route_id"]))

        patterns: dict[tuple, _ShapePattern] = {}
        skipped: dict[tuple, dict] = {}

        # --------------------------------------------------------------
        # Build patterns
        # --------------------------------------------------------------

        for trip_id, stop_ids in trip_stop_sequences.items():

            route_id = trip_route.get(trip_id)

            if route_id is None:
                continue

            key = (route_id, stop_ids)

            pattern = patterns.get(key)

            if pattern is not None:
                pattern.trip_ids.append(trip_id)
                continue

            row = skipped.get(key)

            if row is not None:
                row["n_trips"] += 1
                continue

            ref, label = route_info.get(route_id, (None, None))

            base = {
                "route_id": route_id,
                "gtfs_ref": ref,
                "route_type": label,
                "n_stops": len(stop_ids),
                "n_trips": 1,
                "first_stop": names.get(stop_ids[0], stop_ids[0]),
                "last_stop": names.get(stop_ids[-1], stop_ids[-1]),
                "example_trip_id": trip_id,
            }

            if len(stop_ids) < 2:
                skipped[key] = {**base, "reason": "TOO_FEW_STOPS"}
                continue

            # Every GTFS stop must have coordinates.
            missing = [s for s in stop_ids if s not in stop_coords_by_id]

            if missing:
                skipped[key] = {
                    **base,
                    "reason": "STOP_WITHOUT_COORDINATES",
                    "detail": ", ".join(map(str, missing[:5])),
                }
                continue

            coords = tuple(stop_coords_by_id[s] for s in stop_ids)

            patterns[key] = _ShapePattern(
                route_id=route_id,
                stop_ids=stop_ids,
                trip_ids=[trip_id],
                ref=ref,
                route_type_label=label,
                from_stop=coords[0],
                to_stop=coords[-1],
                stop_coords=coords,
            )

        self._skipped = list(skipped.values())

        return patterns

    @staticmethod
    def _pick_ref(row) -> str:
        """
        Pick the identifier used to match a GTFS route to OSM.

        Preference:

            route_short_name
            route_long_name
            route_id
        """

        for value in (
            row.route_short_name,
            row.route_long_name,
            row.route_id,
        ):

            if (
                pd.notna(value)
                and str(value).strip()
            ):
                return str(value)

        return str(row.route_id)

    # ==================================================================
    # MATCH PATTERNS AND BUILD SHAPES
    # ==================================================================

    def _run_patterns(self, items):
        """
        Yield (coords, info) for every pattern, in order, either in this
        process or in forked worker processes.
        """

        if self.workers == 1:
            for _, pattern in items:
                yield self._solve_pattern(pattern)
            return

        global _BUILDER
        _BUILDER = self

        ctx = multiprocessing.get_context("fork")

        with ProcessPoolExecutor(
            max_workers=self.workers,
            mp_context=ctx,
        ) as pool:
            yield from pool.map(
                self._solve,
                [pattern for _, pattern in items],
                chunksize=32,
            )

    def _solve_pattern(self, pattern: "_ShapePattern"):
        """
        Reconstruct the geometry of one stop pattern.

        Returns (coords, None) on success or (None, info_dict) on failure,
        where info_dict explains why no route was found.
        """

        # The leg object identifies the OSM route/ref/type. Geometry
        # reconstruction itself uses the COMPLETE stop sequence.
        leg = {
            "route": pattern.ref,
            "routeType": pattern.route_type_label,
            "from": {
                "lat": pattern.from_stop[0],
                "lon": pattern.from_stop[1],
            },
            "to": {
                "lat": pattern.to_stop[0],
                "lon": pattern.to_stop[1],
            },
        }

        try:
            coords = self.network.get_geometry_for_stop_sequence(
                leg=leg,
                stops=list(pattern.stop_coords),
                max_stop_distance_m=self.max_stop_distance_m,
            )
            return coords, None

        except ValueError as exc:
            info = {"reason": "NO_MATCH", "error": str(exc)}

            if self.diagnose:
                info.update(self.network.diagnose_stop_sequence(
                    leg,
                    list(pattern.stop_coords),
                    self.max_stop_distance_m
                    ))

            return None, info


    def _solve(self, pattern: "_ShapePattern"):
        return self._solve_pattern(_BUILDER, pattern)


    def _report_row(
        self,
        pattern: _ShapePattern,
        reason: str,
        info: dict,
    ) -> dict:
        """
        One row of the "no shape" report.
        """

        names = self._stop_names

        row = {
            "route_id": pattern.route_id,
            "gtfs_ref": pattern.ref,
            "route_type": pattern.route_type_label,
            "n_stops": len(pattern.stop_ids),
            "n_trips": len(pattern.trip_ids),
            "first_stop": names.get(pattern.stop_ids[0], pattern.stop_ids[0]),
            "last_stop": names.get(pattern.stop_ids[-1], pattern.stop_ids[-1]),
            "example_trip_id": pattern.trip_ids[0],
        }

        row.update({k: v for k, v in info.items() if k != "reason"})
        row["reason"] = reason

        fail = info.get("fail_stop_index")

        if fail is not None:
            sid = pattern.stop_ids[fail]
            row["fail_stop"] = names.get(sid, sid)

        return row

    def _build_shapes(
        self,
        patterns: dict[tuple, _ShapePattern],
    ) -> tuple[
        list[pd.DataFrame],
        dict[tuple, str],
    ]:

        existing_shape_ids: set[str] = set()

        if "shapes.txt" in self.gtfs.files:
            existing_shape_ids = set(self.gtfs.files["shapes.txt"]["shape_id"])

        new_shape_frames: list[pd.DataFrame] = []
        pattern_shape_ids: dict[tuple, str] = {}

        # Rows for the "no shape" report: patterns skipped earlier,
        # plus everything that fails below.
        rows: list[dict] = list(self._skipped)

        items = []

        for key, pattern in patterns.items():

            if pattern.route_type_label is None:
                rows.append(
                    self._report_row(
                        pattern,
                        "UNKNOWN_ROUTE_TYPE",
                        {
                            "detail": ("route_type not in GTFS_ROUTE_TYPE_LABELS")
                        },
                    )
                )
                continue

            items.append((key, pattern))

        # --------------------------------------------------------------
        # Process every unique stop pattern
        # --------------------------------------------------------------

        for (key, pattern), (coords, info) in tqdm(
            zip(items, self._run_patterns(items)),
            total=len(items),
            desc="Matching stop patterns",
            unit="pattern",
        ):

            if coords is None:
                rows.append(self._report_row(pattern, info["reason"], info))
                continue

            if len(coords) < 2:
                rows.append(
                    self._report_row(
                        pattern,
                        "GEOMETRY_TOO_SHORT",
                        {
                            "detail": ("route matched but produced fewer than 2 points")
                        },
                    )
                )
                continue

            shape_id = self._new_shape_id(pattern.route_id, existing_shape_ids)

            new_shape_frames.append(
                self._coords_to_shape_frame(shape_id, coords)
            )

            pattern_shape_ids[key] = shape_id

        self.report = pd.DataFrame(rows)

        if rows:
            print(
                self.report.groupby("reason")["n_trips"]
                .agg(patterns="size", trips="sum")
                .sort_values("patterns", ascending=False)
            )

        return new_shape_frames, pattern_shape_ids

    # ==================================================================
    # SHAPE ID
    # ==================================================================

    @staticmethod
    def _new_shape_id(
        route_id: str,
        existing_shape_ids: set[str],
    ) -> str:
        """
        Generate a collision-free shape_id.
        """

        base = f"osm_{route_id}"

        shape_id = base
        suffix = 1

        while shape_id in existing_shape_ids:
            shape_id = f"{base}_{suffix}"
            suffix += 1

        existing_shape_ids.add(shape_id)

        return shape_id

    # ==================================================================
    # COORDINATES -> SHAPES.TXT
    # ==================================================================

    @staticmethod
    def _coords_to_shape_frame(
        shape_id: str,
        coords: list[list[float]],
    ) -> pd.DataFrame:
        """
        Convert ordered [lat, lon] coordinates into GTFS shapes.txt rows.

        shape_dist_traveled is cumulative distance in meters
        (vectorised haversine).
        """

        arr = np.asarray(coords, dtype=float)

        lat = np.radians(arr[:, 0])
        lon = np.radians(arr[:, 1])

        a = (
            np.sin(np.diff(lat) / 2) ** 2
            + np.cos(lat[:-1])
            * np.cos(lat[1:])
            * np.sin(np.diff(lon) / 2) ** 2
        )

        seg = 2 * 6_371_000.0 * np.arcsin(np.sqrt(np.minimum(a, 1.0)))

        return pd.DataFrame(
            {
                "shape_id": shape_id,
                "shape_pt_lat": arr[:, 0],
                "shape_pt_lon": arr[:, 1],
                "shape_pt_sequence": np.arange(len(arr)),
                "shape_dist_traveled": np.concatenate(
                    ([0.0], np.cumsum(seg))
                ),
            }
        )

    # ==================================================================
    # MERGE BACK INTO GTFS
    # ==================================================================

    def _merge_into_gtfs(
        self,
        new_shape_frames: list[pd.DataFrame],
        pattern_shape_ids: dict[tuple, str],
        patterns: dict[tuple, _ShapePattern],
    ) -> None:

        # --------------------------------------------------------------
        # Merge shapes.txt in ONE concat operation
        # --------------------------------------------------------------

        print("  Merging: starting")
        print(f"  New shape frames: {len(new_shape_frames):,}")


        frames = []

        if "shapes.txt" in self.gtfs.files:
            frames.append(self.gtfs.files["shapes.txt"])

        frames.extend(new_shape_frames)

        if frames:
            shapes = pd.concat(frames, ignore_index=True, copy=False)

            self.gtfs.update_file(
                "shapes.txt",
                shapes,
                validate=False,
            )

        del frames
        del new_shape_frames

        # --------------------------------------------------------------
        # Assign shape_id to trips.txt
        # --------------------------------------------------------------

        trips = self.gtfs.files["trips.txt"]

        if "shape_id" not in trips.columns:
            trips["shape_id"] = pd.NA

        trip_to_shape = {
            trip_id: shape_id
            for key, shape_id in pattern_shape_ids.items()
            for trip_id in patterns[key].trip_ids
        }       

        mask = trips["trip_id"].isin(trip_to_shape)


        trips.loc[mask, "shape_id"] = (
            trips.loc[mask, "trip_id"].map(trip_to_shape)
        )

        self.gtfs.update_file(
            "trips.txt",
            trips,
            validate=False,
        )