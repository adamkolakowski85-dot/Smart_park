"""
Live utilities for SmartPark Tier 1 features.

Provides:
- Live status computation with recency weighting
- Prediction confidence scoring
- Explainable prediction text generation
NOW SUPPORTS: spot_type (general vs accessible)
"""

import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any

# ------------------------------------------------------------
# CONFIGURATION CONSTANTS
# ------------------------------------------------------------

# Time window for considering reports "recent" (in minutes)
RECENCY_WINDOW_MINUTES = 60

# Auto-refresh interval for live board (in seconds)
AUTO_REFRESH_SECONDS = 15

# Exponential decay factor for weighting recent reports
DECAY_FACTOR = 0.5

# Minimum reports needed for different confidence levels
MIN_REPORTS_HIGH_CONFIDENCE = 15
MIN_REPORTS_MEDIUM_CONFIDENCE = 5
MIN_REPORTS_LOW_CONFIDENCE = 1


# ------------------------------------------------------------
# TIER 1 FEATURE 1: LIVE PARKING BOARD
# ------------------------------------------------------------

def compute_live_status(
    location: str,
    parking_df: pd.DataFrame,
    predictor,
    recency_minutes: int = RECENCY_WINDOW_MINUTES,
    spot_type: str = "general"
) -> Dict[str, Any]:
    """
    Compute live parking status for a specific location and spot type.
    
    Uses recent reports with exponential time decay weighting.
    Falls back to historical prediction if no recent data.
    
    Args:
        location: Location name
        parking_df: DataFrame with parking reports
        predictor: ParkingPredictor instance for fallback
        recency_minutes: Time window for "recent" reports
        spot_type: "general" or "accessible"
    
    Returns:
        dict with:
            - availability: float (0-1 probability)
            - status_text: str ("High", "Medium", "Low")
            - icon: str (emoji)
            - is_live: bool (whether based on recent data)
            - last_report_text: str (time ago)
            - report_count: int
            - trend_icons: str (✅❌ pattern showing last 5 reports)
    """
    now = datetime.now()
    cutoff_time = now - timedelta(minutes=recency_minutes)
    
    # Default response structure
    result = {
        'availability': 0.5,
        'status_text': 'Unknown',
        'icon': '⚪',
        'is_live': False,
        'last_report_text': 'No reports',
        'report_count': 0,
        'trend_icons': ''
    }
    
    # Safety check: empty dataframe
    if parking_df is None or len(parking_df) == 0:
        result['availability'] = predictor.predict(location, now.strftime('%A'), now.hour, spot_type)
        result['status_text'], result['icon'] = _get_status_from_probability(result['availability'])
        return result
    
    # Auto-migrate spot_type column
    df = parking_df.copy()
    if 'spot_type' not in df.columns:
        df['spot_type'] = 'general'
    
    # Filter for this location and spot type
    try:
        location_df = df[
            (df['location'] == location) & 
            (df['spot_type'] == spot_type)
        ].copy()
    except Exception:
        result['availability'] = predictor.predict(location, now.strftime('%A'), now.hour, spot_type)
        result['status_text'], result['icon'] = _get_status_from_probability(result['availability'])
        return result
    
    if len(location_df) == 0:
        result['availability'] = predictor.predict(location, now.strftime('%A'), now.hour, spot_type)
        result['status_text'], result['icon'] = _get_status_from_probability(result['availability'])
        return result
    
    # Parse timestamps safely
    try:
        location_df['timestamp_dt'] = pd.to_datetime(location_df['timestamp'], errors='coerce')
        location_df = location_df[location_df['timestamp_dt'].notna()].copy()
    except Exception:
        result['availability'] = predictor.predict(location, now.strftime('%A'), now.hour, spot_type)
        result['status_text'], result['icon'] = _get_status_from_probability(result['availability'])
        return result
    
    if len(location_df) == 0:
        result['availability'] = predictor.predict(location, now.strftime('%A'), now.hour, spot_type)
        result['status_text'], result['icon'] = _get_status_from_probability(result['availability'])
        return result
    
    # Get most recent report time
    most_recent = location_df['timestamp_dt'].max()
    minutes_ago = (now - most_recent).total_seconds() / 60.0
    
    if minutes_ago < 60:
        result['last_report_text'] = f"{int(minutes_ago)}m ago"
    elif minutes_ago < 1440:
        result['last_report_text'] = f"{int(minutes_ago/60)}h ago"
    else:
        result['last_report_text'] = f"{int(minutes_ago/1440)}d ago"
    
    # Filter for recent reports
    recent_df = location_df[location_df['timestamp_dt'] >= cutoff_time].copy()
    result['report_count'] = len(recent_df)
    
    # If we have recent reports, compute weighted availability
    if len(recent_df) > 0:
        result['is_live'] = True
        
        # Calculate time-weighted availability using exponential decay
        recent_df['minutes_ago'] = (now - recent_df['timestamp_dt']).dt.total_seconds() / 60.0
        recent_df['weight'] = _exponential_decay(recent_df['minutes_ago'].values, DECAY_FACTOR)
        
        # Weighted average
        try:
            weighted_found = (recent_df['found_parking'] * recent_df['weight']).sum()
            total_weight = recent_df['weight'].sum()
            
            if total_weight > 0:
                result['availability'] = weighted_found / total_weight
            else:
                result['availability'] = float(recent_df['found_parking'].mean())
        except Exception:
            result['availability'] = float(recent_df['found_parking'].mean())
        
        # Generate trend icons from last 5 reports
        last_5 = recent_df.nlargest(5, 'timestamp_dt')['found_parking'].values
        result['trend_icons'] = ''.join(['✅' if x == 1 else '❌' for x in last_5])
    
    else:
        # No recent reports - use historical prediction
        result['is_live'] = False
        result['availability'] = predictor.predict(location, now.strftime('%A'), now.hour, spot_type)
        result['trend_icons'] = ''
    
    # Determine status text and icon
    result['status_text'], result['icon'] = _get_status_from_probability(result['availability'])
    
    return result


def _exponential_decay(minutes_ago_array, decay_factor: float):
    """Apply exponential decay to weights based on time."""
    import numpy as np
    return np.exp(-decay_factor * minutes_ago_array / 60.0)


def _get_status_from_probability(prob: float) -> tuple:
    """Convert probability to status text and icon."""
    if prob >= 0.65:
        return ("High", "🟢")
    elif prob >= 0.35:
        return ("Medium", "🟡")
    else:
        return ("Low", "🔴")


# ------------------------------------------------------------
# TIER 1 FEATURE 2: PREDICTION CONFIDENCE INDICATOR
# ------------------------------------------------------------

def compute_prediction_confidence(
    location: str,
    day_of_week: str,
    hour: int,
    parking_df: pd.DataFrame,
    spot_type: str = "general"
) -> Dict[str, Any]:
    """
    Compute confidence score for a prediction.
    
    Confidence is based on:
    1. Number of relevant reports (exact matches and similar contexts)
    2. Recency of reports
    
    Args:
        location: Location name
        day_of_week: Day of week
        hour: Hour of day (0-23)
        parking_df: DataFrame with parking reports
        spot_type: "general" or "accessible"
    
    Returns:
        dict with:
            - level: str ("Low", "Medium", "High")
            - stars: str (visual star rating)
            - report_count: int (number of relevant reports)
            - explanation: str (brief description)
    """
    result = {
        'level': 'Low',
        'stars': '★☆☆☆☆',
        'report_count': 0,
        'explanation': 'Insufficient data'
    }
    
    # Safety check
    if parking_df is None or len(parking_df) == 0:
        return result
    
    try:
        # Auto-migrate
        df = parking_df.copy()
        if 'spot_type' not in df.columns:
            df['spot_type'] = 'general'
        
        # Level 1: Exact matches (location + day + hour + spot_type)
        exact_matches = df[
            (df['location'] == location) &
            (df['day_of_week'] == day_of_week) &
            (df['hour'] == hour) &
            (df['spot_type'] == spot_type)
        ]
        exact_count = len(exact_matches)
        
        # Level 2: Location + day + spot_type matches
        location_day_matches = df[
            (df['location'] == location) &
            (df['day_of_week'] == day_of_week) &
            (df['spot_type'] == spot_type)
        ]
        location_day_count = len(location_day_matches)
        
        # Level 3: Location + spot_type matches
        location_spot_matches = df[
            (df['location'] == location) &
            (df['spot_type'] == spot_type)
        ]
        location_spot_count = len(location_spot_matches)
        
        # Determine confidence based on best available data
        if exact_count >= MIN_REPORTS_HIGH_CONFIDENCE:
            result['level'] = 'High'
            result['stars'] = '★★★★★'
            result['report_count'] = exact_count
            result['explanation'] = f'{exact_count} exact matches found'
        elif exact_count >= MIN_REPORTS_MEDIUM_CONFIDENCE:
            result['level'] = 'Medium'
            result['stars'] = '★★★★☆'
            result['report_count'] = exact_count
            result['explanation'] = f'{exact_count} exact matches found'
        elif location_day_count >= MIN_REPORTS_HIGH_CONFIDENCE:
            result['level'] = 'Medium'
            result['stars'] = '★★★☆☆'
            result['report_count'] = location_day_count
            result['explanation'] = f'{location_day_count} reports for this location/type on {day_of_week}s'
        elif location_day_count >= MIN_REPORTS_MEDIUM_CONFIDENCE:
            result['level'] = 'Low-Medium'
            result['stars'] = '★★☆☆☆'
            result['report_count'] = location_day_count
            result['explanation'] = f'{location_day_count} reports for this location/type on {day_of_week}s'
        elif location_spot_count >= MIN_REPORTS_MEDIUM_CONFIDENCE:
            result['level'] = 'Low'
            result['stars'] = '★☆☆☆☆'
            result['report_count'] = location_spot_count
            result['explanation'] = f'{location_spot_count} reports for this location/type (various times)'
        else:
            result['level'] = 'Very Low'
            result['stars'] = '☆☆☆☆☆'
            result['report_count'] = location_spot_count
            result['explanation'] = 'Limited data - using general patterns'
    
    except Exception:
        pass
    
    return result


# ------------------------------------------------------------
# TIER 1 FEATURE 3: EXPLAINABLE PREDICTION TEXT
# ------------------------------------------------------------

def generate_prediction_explanation(
    location: str,
    day_of_week: str,
    hour: int,
    parking_df: pd.DataFrame,
    probability: float,
    spot_type: str = "general"
) -> Dict[str, str]:
    """
    Generate human-readable explanation of how prediction was made.
    
    Explanation is data-driven and never reveals raw model coefficients.
    
    Args:
        location: Location name
        day_of_week: Day of week
        hour: Hour of day
        parking_df: DataFrame with parking reports
        probability: Predicted probability
        spot_type: "general" or "accessible"
    
    Returns:
        dict with:
            - text: str (main explanation)
            - details: str (additional context)
    """
    spot_label = "accessible" if spot_type == "accessible" else "general"
    
    # Default explanation for no data
    default_explanation = {
        'text': f"This prediction is based on typical {spot_label} parking patterns for {hour:02d}:00. "
                f"No specific historical data is available for {location} on {day_of_week}s at this time.",
        'details': "Help improve predictions by submitting parking reports!"
    }
    
    # Safety check
    if parking_df is None or len(parking_df) == 0:
        return default_explanation
    
    try:
        # Auto-migrate
        df = parking_df.copy()
        if 'spot_type' not in df.columns:
            df['spot_type'] = 'general'
        
        # Try exact matches
        exact_matches = df[
            (df['location'] == location) &
            (df['day_of_week'] == day_of_week) &
            (df['hour'] == hour) &
            (df['spot_type'] == spot_type)
        ]
        
        if len(exact_matches) >= 3:
            success_rate = exact_matches['found_parking'].mean()
            count = len(exact_matches)
            success_pct = int(success_rate * 100)
            
            return {
                'text': f"Based on {count} {spot_label} parking report(s) for {location} on {day_of_week}s "
                        f"at {hour:02d}:00, parking was available {success_pct}% of the time.",
                'details': f"This is a data-driven prediction with {count} historical observations."
            }
        
        # Try location + day + spot_type (broader time)
        hour_min = max(0, hour - 2)
        hour_max = min(23, hour + 2)
        
        location_day_matches = df[
            (df['location'] == location) &
            (df['day_of_week'] == day_of_week) &
            (df['spot_type'] == spot_type) &
            (df['hour'] >= hour_min) &
            (df['hour'] <= hour_max)
        ]
        
        if len(location_day_matches) >= 5:
            success_rate = location_day_matches['found_parking'].mean()
            count = len(location_day_matches)
            success_pct = int(success_rate * 100)
            
            return {
                'text': f"Based on {count} {spot_label} parking report(s) for {location} on {day_of_week}s "
                        f"between {hour_min:02d}:00–{hour_max:02d}:00, "
                        f"parking was available {success_pct}% of the time.",
                'details': f"This prediction uses data from similar times on {day_of_week}s."
            }
        
        # Try location + spot_type (any day, similar time)
        location_spot_matches = df[
            (df['location'] == location) &
            (df['spot_type'] == spot_type) &
            (df['hour'] >= hour_min) &
            (df['hour'] <= hour_max)
        ]
        
        if len(location_spot_matches) >= 5:
            success_rate = location_spot_matches['found_parking'].mean()
            count = len(location_spot_matches)
            success_pct = int(success_rate * 100)
            
            return {
                'text': f"Based on {count} {spot_label} parking report(s) for {location} "
                        f"between {hour_min:02d}:00–{hour_max:02d}:00 (any day), "
                        f"parking was available {success_pct}% of the time.",
                'details': f"This prediction uses historical data from {location} at similar times."
            }
        
        # Limited location data for this spot type
        all_location_spot = df[
            (df['location'] == location) &
            (df['spot_type'] == spot_type)
        ]
        if len(all_location_spot) > 0:
            success_rate = all_location_spot['found_parking'].mean()
            count = len(all_location_spot)
            success_pct = int(success_rate * 100)
            
            return {
                'text': f"Based on {count} {spot_label} parking report(s) for {location} (all times), "
                        f"parking was available {success_pct}% of the time. "
                        f"The time-specific prediction also considers typical patterns for {hour:02d}:00.",
                'details': f"Limited data for this specific time. More reports will improve accuracy."
            }
        
        # No data for this spot type at this location
        return {
            'text': f"This prediction uses general {spot_label} parking patterns for {hour:02d}:00. "
                    f"No historical data is available yet for {spot_label} parking at {location}.",
            'details': "Submit parking reports to build location-specific predictions!"
        }
    
    except Exception:
        return default_explanation