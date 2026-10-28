import pandas as pd
import zipfile
from pathlib import Path
import warnings

from .gtfs_schema import GTFS_Schema

class GTFS:


    def __init__(self, files: dict[str, pd.DataFrame]):
        self.files = self._validate_files(files)

    @classmethod
    def _warn_extended_route_types(cls, routes: pd.DataFrame) -> None:
        """Warn when routes use extended GTFS route types."""
        extended = sorted(
            set(routes["route_type"].dropna().unique())
            & cls.EXTENDED_ROUTE_TYPES
        )

        if extended:
            warnings.warn(f"This GTFS feed uses extended route types: f{extended}. These values are defined in Google's "
                "extended GTFS route types reference and are accepted by this package.",
                UserWarning, stacklevel=3,
            )


    @classmethod
    def _validate_files(cls, files: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
        """
        Verifies that the given feed has the required files and columns

        Unexpected files are added with a warning
        """

        required = set(GTFS_Schema.required_files)
        expected = set(GTFS_Schema.SCHEMAS)
        supplied = set(files)

        missing = required - supplied
        if missing:
            raise ValueError(f"Missing required files: {sorted(missing)}")

        unexpected = supplied - expected
        if unexpected:
            warnings.warn(f"Unexpected files in GTFS feed: {sorted(unexpected)}", UserWarning, stacklevel=2)

        feed_dict = {}

        for file_name, df in files.items():
            if file_name in expected:
                df = GTFS_Schema.SCHEMAS[file_name].validate(df)

                if file_name == "routes.txt":
                    cls._warn_extended_route_types(df)

            feed_dict[file_name] = df

        return feed_dict
    

    @classmethod
    def _read_zip(cls, path: str | Path) -> dict[str, pd.DataFrame]:
        """
        Returns a dictionary with the name of the GTFS files and their content
        """

        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(path)

        if path.suffix.lower() != ".zip":
            raise ValueError(f"Expected a ZIP file: {path}")

        files = {}

        with zipfile.ZipFile(path) as zip_file:

            for member in zip_file.namelist():

                if member.endswith("/") or not member.lower().endswith(".txt"):
                    continue

                file_name = Path(member).name

                with zip_file.open(member) as f:
                    files[file_name] = pd.read_csv(f, dtype_backend="pyarrow")
                    files[file_name].columns = files[file_name].columns.str.strip() # Normalize column name
                    files[file_name] = files[file_name].replace(r"^\s*$", pd.NA, regex=True) # Convert empty / whitespace-only strings to missing values

        return files

    @classmethod
    def from_zip(cls, path: str | Path) -> "GTFS":
        files = cls._read_zip(path)
        return cls(files)

    
    def export_gtfs(self, path: str | Path) -> None:
        """
        Exports a GTFS object to a zip file
        """

        path = Path(path)

        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:

            for file_name, df in self.files.items():
                zf.writestr(file_name, df.to_csv(index=False))