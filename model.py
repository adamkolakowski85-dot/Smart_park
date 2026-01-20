"""
Parking availability prediction model.

- Uses logistic regression (scikit-learn) if available
- Falls back to statistics + heuristic if sklearn isn't installed or data is small
- Supports spot_type: "general" vs "accessible"
"""

import os
import warnings
from datetime import datetime

import pandas as pd

warnings.filterwarnings("ignore")

# Optional dependencies (make app not crash if missing)
SKLEARN_OK = True
try:
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import LabelEncoder
except Exception:
    SKLEARN_OK = False
    np = None
    LogisticRegression = None
    LabelEncoder = None


class ParkingPredictor:
    """
    Predicts probability of finding parking (0..1).

    If scikit-learn is available and there is enough data, trains logistic regression.
    Otherwise uses intelligent statistical fallbacks + time heuristic.
    """

    def __init__(self, data_file="parking_data.csv", location_manager=None):
        self.data_file = data_file
        self.location_manager = location_manager

        # Encoders/model (only if sklearn ok)
        self.model = None
        self.location_encoder = LabelEncoder() if SKLEARN_OK else None
        self.day_encoder = LabelEncoder() if SKLEARN_OK else None
        self.spot_type_encoder = LabelEncoder() if SKLEARN_OK else None

        self.locations = []
        self.days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        self.spot_types = ["general", "accessible"]

        # stats
        self.location_stats = {}
        self.day_stats = {}
        self.hour_stats = {}
        self.spot_type_stats = {}
        self.overall_avg = 0.5

        self._load_and_train()

    def _set_defaults(self):
        # get locations from location manager if possible
        if self.location_manager:
            try:
                self.locations = self.location_manager.get_active_locations()
            except Exception:
                self.locations = []

        if not self.locations:
            self.locations = ["Downtown", "Mall", "Airport", "University", "Stadium"]

        self.overall_avg = 0.5
        self.model = None

        # fit encoders if sklearn present
        if SKLEARN_OK:
            try:
                self.location_encoder.fit(self.locations)
                self.day_encoder.fit(self.days)
                self.spot_type_encoder.fit(self.spot_types)
            except Exception:
                self.model = None

    def _calculate_statistics(self, df: pd.DataFrame):
        # overall avg
        try:
            self.overall_avg = float(pd.to_numeric(df["found_parking"], errors="coerce").mean())
            if self.overall_avg != self.overall_avg:  # NaN
                self.overall_avg = 0.5
        except Exception:
            self.overall_avg = 0.5

        # group stats
        def _stats(group_col):
            try:
                return (
                    df.groupby(group_col)["found_parking"]
                    .agg(["mean", "count"])
                    .to_dict("index")
                )
            except Exception:
                return {}

        self.location_stats = _stats("location")
        self.day_stats = _stats("day_of_week")
        self.hour_stats = _stats("hour")
        self.spot_type_stats = _stats("spot_type")

    def _load_and_train(self):
        if not os.path.exists(self.data_file):
            self._set_defaults()
            return

        try:
            df = pd.read_csv(self.data_file)
        except Exception:
            self._set_defaults()
            return

        if df is None or len(df) == 0:
            self._set_defaults()
            return

        # auto-migrate
        if "spot_type" not in df.columns:
            df["spot_type"] = "general"

        # normalize
        df["spot_type"] = df["spot_type"].fillna("general").astype(str).str.lower().str.strip()
        df.loc[~df["spot_type"].isin(["general", "accessible"]), "spot_type"] = "general"

        df["hour"] = pd.to_numeric(df.get("hour", 0), errors="coerce").fillna(0).astype(int).clip(0, 23)
        df["found_parking"] = pd.to_numeric(df.get("found_parking", 0), errors="coerce").fillna(0).astype(int).clip(0, 1)

        # locations present
        try:
            self.locations = sorted(df["location"].dropna().astype(str).unique().tolist())
        except Exception:
            self.locations = []

        self._calculate_statistics(df)

        # If sklearn missing, stop here (use stats/heuristics)
        if not SKLEARN_OK:
            self.model = None
            return

        # Need enough data to train
        if len(df) < 10:
            self.model = None
            try:
                self.location_encoder.fit(self.locations if self.locations else ["Downtown"])
                self.day_encoder.fit(self.days)
                self.spot_type_encoder.fit(self.spot_types)
            except Exception:
                self.model = None
            return

        # Ensure required cols exist
        for col in ["location", "day_of_week", "hour", "spot_type"]:
            if col not in df.columns:
                self.model = None
                return

        # Prepare features
        X = df[["location", "day_of_week", "hour", "spot_type"]].copy()
        y = df["found_parking"].values

        try:
            X["location_encoded"] = self.location_encoder.fit_transform(X["location"].astype(str))
            X["day_encoded"] = self.day_encoder.fit_transform(X["day_of_week"].astype(str))
            X["spot_type_encoded"] = self.spot_type_encoder.fit_transform(X["spot_type"].astype(str))
        except Exception:
            self.model = None
            return

        X_train = X[["location_encoded", "day_encoded", "hour", "spot_type_encoded"]].values

        self.model = LogisticRegression(
            random_state=42,
            max_iter=1000,
            C=1.0,
            solver="lbfgs",
        )

        try:
            self.model.fit(X_train, y)
        except Exception:
            self.model = None

    def predict(self, location: str, day_of_week: str, hour: int, spot_type: str = "general") -> float:
        # normalize
        location = str(location or "").strip()
        day_of_week = str(day_of_week or "").strip()
        try:
            hour = int(hour)
        except Exception:
            hour = datetime.now().hour
        hour = max(0, min(23, hour))

        spot_type = str(spot_type or "general").strip().lower()
        if spot_type not in ["general", "accessible"]:
            spot_type = "general"

        # Try ML model
        if self.model is not None and SKLEARN_OK:
            try:
                if (
                    hasattr(self.location_encoder, "classes_")
                    and location in self.location_encoder.classes_
                    and spot_type in self.spot_type_encoder.classes_
                    and day_of_week in self.day_encoder.classes_
                ):
                    loc_enc = self.location_encoder.transform([location])[0]
                    day_enc = self.day_encoder.transform([day_of_week])[0]
                    spot_enc = self.spot_type_encoder.transform([spot_type])[0]
                    X_pred = np.array([[loc_enc, day_enc, hour, spot_enc]])
                    proba = self.model.predict_proba(X_pred)[0][1]
                    return float(proba)
            except Exception:
                pass

        # Statistics fallback
        return float(self._statistical_prediction(location, day_of_week, hour, spot_type))

    def _statistical_prediction(self, location: str, day_of_week: str, hour: int, spot_type: str) -> float:
        """
        Multi-level fallback:
        1) exact (loc+day+hour+spot) if >=3
        2) loc+day+spot if >=5
        3) loc+spot if >=5
        4) spot overall if >=10
        5) loc overall if >=5
        6) day overall if >=5
        7) hour overall if >=5
        8) overall avg
        9) heuristic
        """
        try:
            if not os.path.exists(self.data_file):
                return self._time_heuristic(hour, spot_type)

            df = pd.read_csv(self.data_file)
            if df is None or len(df) == 0:
                return self._time_heuristic(hour, spot_type)

            if "spot_type" not in df.columns:
                df["spot_type"] = "general"
            df["spot_type"] = df["spot_type"].fillna("general").astype(str).str.lower().str.strip()
            df.loc[~df["spot_type"].isin(["general", "accessible"]), "spot_type"] = "general"

            df["hour"] = pd.to_numeric(df.get("hour", 0), errors="coerce").fillna(0).astype(int).clip(0, 23)
            df["found_parking"] = pd.to_numeric(df.get("found_parking", 0), errors="coerce").fillna(0).astype(int).clip(0, 1)

            exact = df[
                (df["location"] == location)
                & (df["day_of_week"] == day_of_week)
                & (df["hour"] == hour)
                & (df["spot_type"] == spot_type)
            ]
            if len(exact) >= 3:
                return float(exact["found_parking"].mean())

            loc_day_spot = df[
                (df["location"] == location)
                & (df["day_of_week"] == day_of_week)
                & (df["spot_type"] == spot_type)
            ]
            if len(loc_day_spot) >= 5:
                return float(loc_day_spot["found_parking"].mean())

            loc_spot = df[
                (df["location"] == location)
                & (df["spot_type"] == spot_type)
            ]
            if len(loc_spot) >= 5:
                return float(loc_spot["found_parking"].mean())

            # use cached stats if they exist
            if spot_type in self.spot_type_stats and self.spot_type_stats[spot_type].get("count", 0) >= 10:
                return float(self.spot_type_stats[spot_type]["mean"])

            if location in self.location_stats and self.location_stats[location].get("count", 0) >= 5:
                return float(self.location_stats[location]["mean"])

            if day_of_week in self.day_stats and self.day_stats[day_of_week].get("count", 0) >= 5:
                return float(self.day_stats[day_of_week]["mean"])

            if hour in self.hour_stats and self.hour_stats[hour].get("count", 0) >= 5:
                return float(self.hour_stats[hour]["mean"])

            if self.overall_avg > 0:
                return float(self.overall_avg)

        except Exception:
            pass

        return float(self._time_heuristic(hour, spot_type))

    def _time_heuristic(self, hour: int, spot_type: str) -> float:
        if hour in [0, 1, 2, 3, 4, 5, 6]:
            base = 0.85
        elif hour in [7, 8, 9]:
            base = 0.30
        elif hour in [10, 11, 12, 13, 14, 15, 16]:
            base = 0.60
        elif hour in [17, 18, 19]:
            base = 0.35
        else:
            base = 0.70

        if spot_type == "accessible":
            return min(1.0, base + 0.10)
        return base

    def get_locations(self):
        if self.location_manager:
            try:
                return self.location_manager.get_active_locations()
            except Exception:
                pass
        return self.locations if self.locations else ["Downtown", "Mall", "Airport"]
