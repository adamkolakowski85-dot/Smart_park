import os
from datetime import datetime, timedelta
from typing import List, Tuple, Optional

import pandas as pd


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _safe_read_csv(path: str, columns: List[str]) -> pd.DataFrame:
    """
    Read CSV safely. If missing/empty/bad, return an empty DF with given columns.
    Auto-normalizes missing columns.
    """
    if not os.path.exists(path):
        return pd.DataFrame(columns=columns)

    try:
        df = pd.read_csv(path)
        if df is None:
            return pd.DataFrame(columns=columns)

        for c in columns:
            if c not in df.columns:
                df[c] = None

        return df[columns].copy()
    except Exception:
        return pd.DataFrame(columns=columns)


def _safe_write_csv(df: pd.DataFrame, path: str) -> None:
    """
    Best-effort safe write (atomic-ish): write to temp then replace.
    """
    tmp_path = f"{path}.tmp"
    df.to_csv(tmp_path, index=False)
    os.replace(tmp_path, path)


def _norm(x) -> str:
    return str(x).strip().lower()


class LocationManager:
    """
    Manages parking locations using a CSV file.
    NOW SUPPORTS COORDINATES: lat/lon
    """

    COLS = [
        "location",
        "notes",
        "is_active",
        "created_at",
        "created_by",
        "verified_count",
        "flagged_count",
        "trust_score",
        "lat",
        "lon",
    ]

    def __init__(self, file_path: str = "locations.csv"):
        self.file_path = file_path
        self._ensure_file()

    def _ensure_file(self) -> None:
        if not os.path.exists(self.file_path):
            df = pd.DataFrame(columns=self.COLS)
            _safe_write_csv(df, self.file_path)
            return

        try:
            df = pd.read_csv(self.file_path)
            if df is None:
                df = pd.DataFrame(columns=self.COLS)
        except Exception:
            df = pd.DataFrame(columns=self.COLS)

        modified = False
        for col in self.COLS:
            if col not in df.columns:
                if col == "is_active":
                    df[col] = True
                elif col in ("verified_count", "flagged_count"):
                    df[col] = 0
                elif col == "trust_score":
                    df[col] = 50.0
                elif col in ("lat", "lon"):
                    df[col] = None
                else:
                    df[col] = ""
                modified = True

        if modified:
            _safe_write_csv(df[self.COLS], self.file_path)

    def _compute_trust_score(self, row) -> float:
        try:
            verified = float(row.get("verified_count", 0) or 0)
        except Exception:
            verified = 0.0
        try:
            flagged = float(row.get("flagged_count", 0) or 0)
        except Exception:
            flagged = 0.0

        score = 50.0 + 5.0 * verified - 12.0 * flagged
        return max(0.0, min(100.0, score))

    def get_all_locations_df(self) -> pd.DataFrame:
        self._ensure_file()
        df = _safe_read_csv(self.file_path, self.COLS)

        if df.empty:
            return df

        df["verified_count"] = pd.to_numeric(df["verified_count"], errors="coerce").fillna(0).astype(int)
        df["flagged_count"] = pd.to_numeric(df["flagged_count"], errors="coerce").fillna(0).astype(int)
        df["is_active"] = df["is_active"].astype(str).str.lower().isin(["true", "1", "yes"])

        df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
        df["lon"] = pd.to_numeric(df["lon"], errors="coerce")

        df["trust_score"] = df.apply(self._compute_trust_score, axis=1)
        return df

    def get_active_locations(self) -> List[str]:
        df = self.get_all_locations_df()
        if df is None or df.empty:
            return []
        active = df[df["is_active"] == True]["location"].dropna().astype(str).tolist()
        return [x.strip() for x in active if x and x.strip()]

    def get_locations_ranked(self) -> List[str]:
        df = self.get_all_locations_df()
        if df is None or df.empty:
            return []
        df = df[df["is_active"] == True].copy()
        if df.empty:
            return []
        df = df.sort_values(by=["trust_score", "verified_count"], ascending=[False, False])
        active = df["location"].dropna().astype(str).tolist()
        return [x.strip() for x in active if x and x.strip()]

    def get_locations_with_coords_df(self) -> pd.DataFrame:
        """
        Returns active locations that have valid lat/lon.
        Used by the map.
        """
        df = self.get_all_locations_df()
        if df is None or df.empty:
            return pd.DataFrame(columns=["location", "lat", "lon"])

        df = df[df["is_active"] == True].copy()
        df = df[df["lat"].notna() & df["lon"].notna()].copy()
        if df.empty:
            return pd.DataFrame(columns=["location", "lat", "lon"])

        return df[["location", "lat", "lon"]].copy()

    def add_location(
        self,
        location: str,
        notes: str = "",
        created_by: str = "anonymous",
        lat: Optional[float] = None,
        lon: Optional[float] = None,
    ) -> bool:
        """
        Adds a new location (case-insensitive). Optionally store lat/lon.
        """
        self._ensure_file()
        df = _safe_read_csv(self.file_path, self.COLS)

        location_clean = (location or "").strip()
        if not location_clean:
            return False

        existing = df["location"].fillna("").astype(str).str.strip().str.lower()
        if location_clean.lower() in existing.values:
            return False

        # Clean coords
        try:
            lat_val = float(lat) if lat is not None else None
        except Exception:
            lat_val = None
        try:
            lon_val = float(lon) if lon is not None else None
        except Exception:
            lon_val = None

        new_row = {
            "location": location_clean,
            "notes": (notes or "").strip(),
            "is_active": True,
            "created_at": _now_iso(),
            "created_by": (created_by or "anonymous").strip(),
            "verified_count": 0,
            "flagged_count": 0,
            "trust_score": 50.0,
            "lat": lat_val,
            "lon": lon_val,
        }

        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        _safe_write_csv(df[self.COLS], self.file_path)
        return True

    def update_location_coords(self, location: str, lat: Optional[float], lon: Optional[float]) -> bool:
        """
        Update lat/lon for an existing location. Returns True if updated.
        """
        self._ensure_file()
        df = _safe_read_csv(self.file_path, self.COLS)
        if df.empty:
            return False

        target = _norm(location)
        loc_series = df["location"].fillna("").astype(str).map(_norm)
        mask = loc_series == target
        if not mask.any():
            return False

        try:
            lat_val = float(lat) if lat is not None else None
        except Exception:
            lat_val = None
        try:
            lon_val = float(lon) if lon is not None else None
        except Exception:
            lon_val = None

        df.loc[mask, "lat"] = lat_val
        df.loc[mask, "lon"] = lon_val
        _safe_write_csv(df[self.COLS], self.file_path)
        return True

    def deactivate_location(self, location: str) -> None:
        self._ensure_file()
        df = _safe_read_csv(self.file_path, self.COLS)
        if df.empty:
            return

        target = _norm(location)
        loc_series = df["location"].fillna("").astype(str).map(_norm)
        df.loc[loc_series == target, "is_active"] = False
        _safe_write_csv(df[self.COLS], self.file_path)

    def increment_verified(self, location: str) -> None:
        self._ensure_file()
        df = _safe_read_csv(self.file_path, self.COLS)
        if df.empty:
            return

        target = _norm(location)
        loc_series = df["location"].fillna("").astype(str).map(_norm)
        mask = loc_series == target
        if not mask.any():
            return

        df.loc[mask, "verified_count"] = pd.to_numeric(df.loc[mask, "verified_count"], errors="coerce").fillna(0).astype(int) + 1
        for idx in df[mask].index:
            df.loc[idx, "trust_score"] = self._compute_trust_score(df.loc[idx])
        _safe_write_csv(df[self.COLS], self.file_path)

    def increment_flagged(self, location: str) -> None:
        self._ensure_file()
        df = _safe_read_csv(self.file_path, self.COLS)
        if df.empty:
            return

        target = _norm(location)
        loc_series = df["location"].fillna("").astype(str).map(_norm)
        mask = loc_series == target
        if not mask.any():
            return

        df.loc[mask, "flagged_count"] = pd.to_numeric(df.loc[mask, "flagged_count"], errors="coerce").fillna(0).astype(int) + 1
        for idx in df[mask].index:
            df.loc[idx, "trust_score"] = self._compute_trust_score(df.loc[idx])
        _safe_write_csv(df[self.COLS], self.file_path)


class EventLogger:
    """
    Logs user and system events to a CSV file.
    """

    COLS = [
        "timestamp",
        "event_type",
        "location",
        "day_of_week",
        "hour",
        "found_parking",
        "user",
        "details",
        "spot_type",
    ]

    def __init__(self, file_path: str = "events.csv"):
        self.file_path = file_path
        self._ensure_file()

    def _ensure_file(self) -> None:
        if not os.path.exists(self.file_path):
            df = pd.DataFrame(columns=self.COLS)
            _safe_write_csv(df, self.file_path)
            return

        try:
            df = pd.read_csv(self.file_path)
            if df is None:
                df = pd.DataFrame(columns=self.COLS)
        except Exception:
            df = pd.DataFrame(columns=self.COLS)

        modified = False
        for c in self.COLS:
            if c not in df.columns:
                df[c] = ""
                modified = True

        if modified:
            _safe_write_csv(df[self.COLS], self.file_path)

    def log(
        self,
        event_type: str,
        location=None,
        day_of_week=None,
        hour=None,
        found_parking=None,
        user: str = "anonymous",
        details: str = "",
        spot_type: str = "",
    ) -> None:
        self._ensure_file()
        df = _safe_read_csv(self.file_path, self.COLS)

        new_row = {
            "timestamp": _now_iso(),
            "event_type": str(event_type).strip() if event_type else "unknown",
            "location": location,
            "day_of_week": day_of_week,
            "hour": hour,
            "found_parking": found_parking,
            "user": str(user).strip() if user else "anonymous",
            "details": str(details).strip(),
            "spot_type": str(spot_type).strip() if spot_type else "",
        }

        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        _safe_write_csv(df[self.COLS], self.file_path)


def can_submit_report(
    user: str,
    location: str,
    found_parking: int,
    now: Optional[datetime] = None
) -> Tuple[bool, str]:
    """
    Rate limit and duplicate detection.

    Rules:
    - Per user+location: max 1 report every 2 minutes
    - Per user globally: max 10 reports per hour
    - Duplicate: same user, same location, same result within 3 minutes
    """
    if now is None:
        now = datetime.now()

    if not os.path.exists("events.csv"):
        return True, ""

    try:
        df = pd.read_csv("events.csv")
    except Exception:
        return True, ""

    if df is None or len(df) == 0:
        return True, ""

    for col in ["timestamp", "event_type", "user", "location", "found_parking"]:
        if col not in df.columns:
            return True, ""

    try:
        df = df[df["event_type"] == "report_submit"].copy()
        df["timestamp_dt"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df[df["timestamp_dt"].notna()].copy()
    except Exception:
        return True, ""

    if len(df) == 0:
        return True, ""

    u = _norm(user)
    loc = _norm(location)

    user_df = df[df["user"].astype(str).map(_norm) == u].copy()
    if len(user_df) == 0:
        return True, ""

    location_df = user_df[user_df["location"].astype(str).map(_norm) == loc].copy()
    if len(location_df) > 0:
        last_report = location_df["timestamp_dt"].max()
        minutes_ago = (now - last_report).total_seconds() / 60.0
        if minutes_ago < 2.0:
            wait_m = int(2 - minutes_ago) + 1
            return False, f"Please wait {wait_m} minute(s) before submitting another report for this location."

    one_hour_ago = now - timedelta(hours=1)
    if len(user_df[user_df["timestamp_dt"] >= one_hour_ago]) >= 10:
        return False, "You have submitted 10 reports in the last hour. Please wait before submitting more."

    three_min_ago = now - timedelta(minutes=3)
    recent_location = location_df[location_df["timestamp_dt"] >= three_min_ago]
    if len(recent_location) > 0:
        for _, row in recent_location.iterrows():
            try:
                prev_result = int(row["found_parking"])
                if prev_result == int(found_parking):
                    return False, "Duplicate report detected. You already submitted this result for this location recently."
            except Exception:
                continue

    return True, ""


def get_app_stats(parking_data_file: str = "parking_data.csv") -> dict:
    base = {
        "total_reports": 0,
        "success_rate": 0.0,
        "overall_success_rate": 0.0,
        "num_locations": 0,
        "locations_tracked": 0,
        "last_update": None,
        "last_updated": None,
    }

    if not os.path.exists(parking_data_file):
        return base

    try:
        df = pd.read_csv(parking_data_file)
    except Exception:
        return base

    if df is None or len(df) == 0:
        return base

    total_reports = int(len(df))

    success_rate = 0.0
    if "found_parking" in df.columns:
        try:
            success_rate = float(pd.to_numeric(df["found_parking"], errors="coerce").mean())
            if success_rate != success_rate:
                success_rate = 0.0
        except Exception:
            success_rate = 0.0

    num_locations = 0
    if "location" in df.columns:
        try:
            num_locations = int(df["location"].nunique())
        except Exception:
            num_locations = 0

    last_update = None
    if "timestamp" in df.columns:
        try:
            last_update = df["timestamp"].max()
        except Exception:
            last_update = None

    return {
        "total_reports": total_reports,
        "success_rate": success_rate,
        "num_locations": num_locations,
        "last_update": last_update,
        "overall_success_rate": success_rate,
        "locations_tracked": num_locations,
        "last_updated": last_update,
    }
