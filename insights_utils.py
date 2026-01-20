"""
Insights utilities for SmartPark Tier 2 features.

Provides:
- 7-day trend analysis
- Top/worst locations ranking
- Anomaly detection
"""

import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any


def compute_7_day_trend(parking_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute success rate trend for last 7 days.

    Returns:
        dict with:
            - dates: list of date strings
            - success_rates: list of success rates (0-1)
            - has_data: bool
    """
    result = {
        'dates': [],
        'success_rates': [],
        'has_data': False
    }

    if parking_df is None or len(parking_df) == 0:
        return result

    try:
        df = parking_df.copy()
        df['timestamp_dt'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df = df[df['timestamp_dt'].notna()].copy()

        if len(df) == 0:
            return result

        # Get last 7 days
        now = datetime.now()
        seven_days_ago = now - timedelta(days=7)
        df = df[df['timestamp_dt'] >= seven_days_ago].copy()

        if len(df) == 0:
            return result

        # Group by date
        df['date'] = df['timestamp_dt'].dt.date
        daily = df.groupby('date')['found_parking'].mean().reset_index()
        daily = daily.sort_values('date')

        result['dates'] = [str(d) for d in daily['date'].tolist()]
        result['success_rates'] = daily['found_parking'].tolist()
        result['has_data'] = True

    except Exception:
        pass

    return result


def compute_top_locations(parking_df: pd.DataFrame, days: int = 7, min_reports: int = 5) -> Dict[str, List[Dict]]:
    """
    Compute top 3 best and worst locations for last N days.

    Args:
        parking_df: DataFrame with parking reports
        days: number of days to look back
        min_reports: minimum reports needed to be included

    Returns:
        dict with:
            - best: list of {location, success_rate, report_count}
            - worst: list of {location, success_rate, report_count}
    """
    result = {
        'best': [],
        'worst': []
    }

    if parking_df is None or len(parking_df) == 0:
        return result

    try:
        df = parking_df.copy()
        df['timestamp_dt'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df = df[df['timestamp_dt'].notna()].copy()

        if len(df) == 0:
            return result

        # Filter last N days
        now = datetime.now()
        cutoff = now - timedelta(days=days)
        df = df[df['timestamp_dt'] >= cutoff].copy()

        if len(df) == 0:
            return result

        # Group by location
        location_stats = df.groupby('location').agg({
            'found_parking': ['mean', 'count']
        }).reset_index()
        location_stats.columns = ['location', 'success_rate', 'report_count']

        # Filter by min_reports
        location_stats = location_stats[location_stats['report_count'] >= min_reports].copy()

        if len(location_stats) == 0:
            return result

        # Sort by success rate
        location_stats = location_stats.sort_values('success_rate', ascending=False)

        # Top 3 best
        best = location_stats.head(3)
        result['best'] = [
            {
                'location': row['location'],
                'success_rate': float(row['success_rate']),
                'report_count': int(row['report_count'])
            }
            for _, row in best.iterrows()
        ]

        # Top 3 worst
        worst = location_stats.tail(3).sort_values('success_rate')
        result['worst'] = [
            {
                'location': row['location'],
                'success_rate': float(row['success_rate']),
                'report_count': int(row['report_count'])
            }
            for _, row in worst.iterrows()
        ]

    except Exception:
        pass

    return result


def compute_anomalies(parking_df: pd.DataFrame, threshold: float = 0.30) -> List[Dict[str, Any]]:
    """
    Detect anomalies: locations where last 24h success rate dropped >=30pp vs previous 7-day avg.

    Args:
        parking_df: DataFrame with parking reports
        threshold: minimum drop in percentage points (0.30 = 30pp)

    Returns:
        list of anomaly dicts with:
            - location: str
            - recent_rate: float (last 24h)
            - baseline_rate: float (previous 7 days)
            - drop: float (percentage point drop)
    """
    anomalies = []

    if parking_df is None or len(parking_df) == 0:
        return anomalies

    try:
        df = parking_df.copy()
        df['timestamp_dt'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df = df[df['timestamp_dt'].notna()].copy()

        if len(df) == 0:
            return anomalies

        now = datetime.now()
        last_24h = now - timedelta(hours=24)
        last_7d = now - timedelta(days=7)
        baseline_start = now - timedelta(days=8)

        # Get locations
        locations = df['location'].unique()

        for location in locations:
            loc_df = df[df['location'] == location].copy()

            # Last 24h data
            recent_df = loc_df[loc_df['timestamp_dt'] >= last_24h]
            if len(recent_df) < 3:  # Need at least 3 reports
                continue

            # Previous 7-day baseline (exclude last 24h)
            baseline_df = loc_df[
                (loc_df['timestamp_dt'] >= baseline_start) &
                (loc_df['timestamp_dt'] < last_24h)
            ]
            if len(baseline_df) < 5:  # Need at least 5 reports for baseline
                continue

            recent_rate = float(recent_df['found_parking'].mean())
            baseline_rate = float(baseline_df['found_parking'].mean())

            drop = baseline_rate - recent_rate

            if drop >= threshold:
                anomalies.append({
                    'location': location,
                    'recent_rate': recent_rate,
                    'baseline_rate': baseline_rate,
                    'drop': drop
                })

    except Exception:
        pass

    # Sort by drop magnitude (highest first)
    anomalies.sort(key=lambda x: x['drop'], reverse=True)

    return anomalies