# app.py — SmartPark v2.3 (single-map Folium + scaled bubbles + click-to-add)
# NOTE:
# - Uses ONE map (Folium via streamlit-folium)
# - Bubble size scales by TOTAL reports for that location + selected spot_type
# - Click anywhere on the map to capture lat/lon, then add a location
# - Keeps the rest of your app (Live Board / Check / Submit / Manage / Analytics) working

import os
import time
import math
from datetime import datetime

import folium
from streamlit_folium import st_folium
import pandas as pd
import streamlit as st

from live_utils import (
    AUTO_REFRESH_SECONDS,
    RECENCY_WINDOW_MINUTES,
    compute_live_status,
    compute_prediction_confidence,
    generate_prediction_explanation,
)
from model import ParkingPredictor
from utils import EventLogger, LocationManager, get_app_stats

# ============================================================
# PAGE CONFIG + STYLES
# ============================================================
st.set_page_config(
    page_title="SmartPark - Parking Intelligence",
    page_icon="🅿️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main > div { padding-top: 2rem; }
    .stMetric { background-color: #f0f2f6; padding: 1rem; border-radius: 0.5rem; }
    .live-status-card {
        background: white;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .trend-icons { font-size: 0.95rem; letter-spacing: 2px; }
    .confidence-stars { font-size: 1.1rem; }
    .map-selection-box {
        background: #f0f8ff;
        border: 2px solid #4CAF50;
        border-radius: 8px;
        padding: 1rem;
        margin-top: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# CONSTANTS
# ============================================================
PARKING_DATA_FILE = "parking_data.csv"
EXPECTED_COLS = ["timestamp", "location", "day_of_week", "hour", "found_parking", "spot_type"]

# East Rockaway, NY (default center)
DEFAULT_CENTER_LAT = 40.6423
DEFAULT_CENTER_LON = -73.6696


# ============================================================
# SMALL HELPERS
# ============================================================
def spot_label(v: str) -> str:
    v = (v or "").strip().lower()
    return "Accessible / Handicap" if v == "accessible" else "General parking"


def make_folium_map(center_lat: float, center_lon: float, zoom: int = 14) -> folium.Map:
    m = folium.Map(location=[center_lat, center_lon], zoom_start=zoom, control_scale=True)
    folium.TileLayer("OpenStreetMap", name="OSM", control=False).add_to(m)
    return m


def compute_time_ago_label(last_updated_value) -> str:
    if not last_updated_value:
        return "Never"
    try:
        last_dt = pd.to_datetime(last_updated_value, errors="coerce")
        if pd.isna(last_dt):
            return "Unknown"
        minutes = (datetime.now() - last_dt.to_pydatetime()).total_seconds() / 60.0
        if minutes < 60:
            return f"{int(minutes)}m ago"
        if minutes < 1440:
            return f"{int(minutes / 60)}h ago"
        return f"{int(minutes / 1440)}d ago"
    except Exception:
        return "Unknown"


def _get_status_and_icon(prob: float) -> tuple[str, str]:
    """Local safe copy of status thresholds (so we don't rely on importing private helpers)."""
    try:
        p = float(prob)
    except Exception:
        p = 0.5
    if p >= 0.65:
        return "High", "🟢"
    if p >= 0.35:
        return "Medium", "🟡"
    return "Low", "🔴"


def scale_radius(total_reports: int) -> int:
    """
    Convert total report count -> bubble radius for the map (for our internal scale).
    Targets (approx):
      0 -> ~80
      5 -> ~136
      20 -> ~192
      50 -> ~257
    Clamped to [70, 300].
    """
    try:
        n = int(total_reports)
    except Exception:
        n = 0
    if n < 0:
        n = 0

    r = 80 + 25 * math.sqrt(n)
    r = max(70, min(300, r))
    return int(r)


def safe_read_parking_data(path: str = PARKING_DATA_FILE) -> pd.DataFrame:
    """Read parking data safely. Always returns a DF with EXPECTED_COLS."""
    if not os.path.exists(path):
        return pd.DataFrame(columns=EXPECTED_COLS)

    try:
        df = pd.read_csv(path)

        for c in EXPECTED_COLS:
            if c not in df.columns:
                df[c] = None

        df["spot_type"] = df["spot_type"].fillna("general")
        df["spot_type"] = df["spot_type"].astype(str).str.strip().str.lower()
        df.loc[~df["spot_type"].isin(["general", "accessible"]), "spot_type"] = "general"

        df["hour"] = pd.to_numeric(df["hour"], errors="coerce").fillna(0).astype(int).clip(0, 23)
        df["found_parking"] = pd.to_numeric(df["found_parking"], errors="coerce").fillna(0).astype(int).clip(0, 1)

        return df[EXPECTED_COLS].copy()
    except Exception:
        return pd.DataFrame(columns=EXPECTED_COLS)


def append_parking_report(new_report: dict, path: str = PARKING_DATA_FILE) -> None:
    df = safe_read_parking_data(path)
    if "spot_type" not in new_report or not str(new_report.get("spot_type", "")).strip():
        new_report["spot_type"] = "general"
    df = pd.concat([df, pd.DataFrame([new_report])], ignore_index=True)
    df.to_csv(path, index=False)


def safe_locations_with_coords(location_manager: LocationManager) -> pd.DataFrame:
    """
    Works whether or not your utils.py has get_locations_with_coords_df().
    Returns active locations with lat/lon if present.
    """
    if hasattr(location_manager, "get_locations_with_coords_df"):
        try:
            df = location_manager.get_locations_with_coords_df()
            if df is None:
                return pd.DataFrame(columns=["location", "lat", "lon"])
            return df.copy()
        except Exception:
            pass

    try:
        df = location_manager.get_all_locations_df()
        if df is None or len(df) == 0:
            return pd.DataFrame(columns=["location", "lat", "lon"])

        if "lat" not in df.columns:
            df["lat"] = None
        if "lon" not in df.columns:
            df["lon"] = None

        if "is_active" in df.columns:
            df["is_active"] = df["is_active"].astype(str).str.lower().isin(["true", "1", "yes"])
            df = df[df["is_active"] == True].copy()

        df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
        df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
        df = df[df["lat"].notna() & df["lon"].notna()].copy()

        return df[["location", "lat", "lon"]].copy()
    except Exception:
        return pd.DataFrame(columns=["location", "lat", "lon"])


def add_bubble_markers(m: folium.Map, df: pd.DataFrame):
    """
    Adds colored circle markers to a folium map from map_df.
    Expects columns:
      lat, lon, color (RGBA list), radius (int),
      location, availability_pct, status_text, total_reports, recent_reports, last_report_text, icon.
    """
    if df is None or len(df) == 0:
        return

    for _, row in df.iterrows():
        try:
            lat = float(row.get("lat"))
            lon = float(row.get("lon"))
        except Exception:
            continue

        rgba = row.get("color", [120, 120, 120, 200])
        try:
            r, g, b, a = int(rgba[0]), int(rgba[1]), int(rgba[2]), int(rgba[3])
        except Exception:
            r, g, b, a = 120, 120, 120, 200

        hex_color = f"#{r:02x}{g:02x}{b:02x}"
        opacity = max(0.25, min(1.0, a / 255.0))

        loc = str(row.get("location", ""))
        status_text = str(row.get("status_text", "Unknown"))
        icon = str(row.get("icon", "⚪"))
        avail = str(row.get("availability_pct", ""))
        total_reports = int(row.get("total_reports", 0) or 0)
        recent_reports = int(row.get("recent_reports", 0) or 0)
        last_report_text = str(row.get("last_report_text", "No reports"))

        popup_html = f"""
        <div style="min-width:240px;">
          <b>{loc}</b><br/>
          {icon} {status_text}<br/>
          Availability: {avail}<br/>
          Last report: {last_report_text}<br/>
          Recent ({RECENCY_WINDOW_MINUTES}m): {recent_reports}<br/>
          Total reports: {total_reports}
        </div>
        """

        try:
            raw_r = int(row.get("radius", 120) or 120)
        except Exception:
            raw_r = 120

        # Folium CircleMarker radius is "pixel-ish". Convert & clamp for a nice look.
        folium_radius = max(6, min(30, raw_r / 9.0))

        folium.CircleMarker(
            location=[lat, lon],
            radius=folium_radius,
            color=hex_color,
            fill=True,
            fill_color=hex_color,
            fill_opacity=opacity,
            opacity=opacity,
            popup=folium.Popup(popup_html, max_width=340),
            tooltip=f"{loc} • {avail} • total reports: {total_reports}",
        ).add_to(m)


def safe_compute_live_status(location: str, parking_df: pd.DataFrame, predictor, recency_minutes: int, spot_type: str):
    """
    compute_live_status() might accept spot_type in your version… or it might not.
    This wrapper makes it safe either way.
    """
    try:
        return compute_live_status(
            location=location,
            parking_df=parking_df,
            predictor=predictor,
            recency_minutes=recency_minutes,
            spot_type=spot_type,
        )
    except TypeError:
        # Fallback: older signature; our df is already spot-filtered.
        return compute_live_status(
            location=location,
            parking_df=parking_df,
            predictor=predictor,
            recency_minutes=recency_minutes,
        )


# ============================================================
# CACHED INITIALIZATION
# ============================================================
@st.cache_resource
def init_managers():
    loc_manager = LocationManager()
    event_logger = EventLogger()
    predictor = ParkingPredictor(data_file=PARKING_DATA_FILE, location_manager=loc_manager)
    return loc_manager, event_logger, predictor


location_manager, event_logger, predictor = init_managers()

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.title("🅿️ SmartPark")
    st.markdown("### Parking Intelligence System")
    st.markdown("---")

    user_nickname = st.text_input(
        "Your Nickname (optional)",
        value="",
        max_chars=20,
        help="Used for activity logging. Leave blank for 'Anonymous'",
        key="user_nickname",
    )
    if not user_nickname.strip():
        user_nickname = "Anonymous"

    st.markdown("---")
    st.markdown("### 📊 Quick Stats")
    stats = get_app_stats(PARKING_DATA_FILE)
    st.metric("Total Reports", f"{int(stats.get('total_reports', 0)):,}")
    st.metric("Locations", int(stats.get("num_locations", stats.get("locations_tracked", 0))))

    st.markdown("---")
    with st.expander("ℹ️ About SmartPark"):
        st.markdown(
            """
            SmartPark uses machine learning to predict parking availability based on:
            - **Location**
            - **Day of Week**
            - **Time of Day**
            - **Spot Type** (General vs Accessible)

            Submit reports to improve predictions for everyone.
            """
        )

# ============================================================
# STATUS HEADER
# ============================================================
stats = get_app_stats(PARKING_DATA_FILE)

st.markdown("### 📈 System Status")
header_col1, header_col2, header_col3, header_col4 = st.columns(4)

with header_col1:
    st.metric("Total Reports", f"{int(stats.get('total_reports', 0)):,}")

with header_col2:
    sr = stats.get("overall_success_rate", stats.get("success_rate", 0.0))
    try:
        sr = float(sr)
    except Exception:
        sr = 0.0
    st.metric("Success Rate", f"{sr:.1%}")

with header_col3:
    st.metric("Active Locations", int(stats.get("num_locations", stats.get("locations_tracked", 0))))

with header_col4:
    last_val = stats.get("last_updated", stats.get("last_update", None))
    st.metric("Last Updated", compute_time_ago_label(last_val))

st.markdown("---")

# ============================================================
# MAIN TABS
# ============================================================
tab_map, tab_live, tab_check, tab_submit, tab_manage, tab_analytics = st.tabs(
    ["🗺️ Map", "🟢 Live Board", "🔍 Check Availability", "📝 Submit Report", "📍 Manage Locations", "📊 Analytics"]
)

# ============================================================
# TAB: MAP (SINGLE MAP: Folium bubbles + click-to-add)
# ============================================================
with tab_map:
    st.header("🗺️ Interactive Parking Map")
    st.markdown(f"One map • Click anywhere to add locations • Updates every {AUTO_REFRESH_SECONDS}s")

    map_spot_type = st.radio(
        "♿ Show availability for:",
        options=["general", "accessible"],
        format_func=spot_label,
        horizontal=True,
        key="map_spot_type",
    )
    # Fullscreen toggle
    if "map_fullscreen" not in st.session_state:
        st.session_state["map_fullscreen"] = False

        fs_col1, fs_col2 = st.columns([1, 3])
    with fs_col1:
        st.session_state["map_fullscreen"] = st.toggle("🖥️ Full screen map", value=st.session_state["map_fullscreen"])
    with fs_col2:
        st.caption("Fullscreen hides the sidebar and expands the map height.")

        apply_fullscreen_css(st.session_state["map_fullscreen"])

map_height = 900 if st.session_state["map_fullscreen"] else 520

    # Auto-refresh timer
    if "map_last_refresh" not in st.session_state:
        st.session_state.map_last_refresh = datetime.now()
    seconds_since_map_refresh = (datetime.now() - st.session_state.map_last_refresh).total_seconds()
    next_map_refresh = max(0, AUTO_REFRESH_SECONDS - int(seconds_since_map_refresh))
    st.caption(f"🔄 Next refresh in {next_map_refresh}s")

    # Load locations w/ coords
    loc_df = safe_locations_with_coords(location_manager)

    # Load parking data once
    parking_df = safe_read_parking_data(PARKING_DATA_FILE)
    parking_df["location_norm"] = parking_df["location"].fillna("").astype(str).str.strip().str.lower()
    parking_df["spot_type_norm"] = parking_df["spot_type"].fillna("general").astype(str).str.strip().str.lower()

    # Filter by selected spot type (used for live + totals)
    parking_df_spot = parking_df[parking_df["spot_type_norm"] == map_spot_type].copy()

    map_rows = []
    if loc_df is not None and len(loc_df) > 0:
        loc_df = loc_df.copy()
        loc_df["location_norm"] = loc_df["location"].fillna("").astype(str).str.strip().str.lower()

        for _, r in loc_df.iterrows():
            name = str(r.get("location", "")).strip()
            if not name:
                continue

            try:
                lat = float(r["lat"])
                lon = float(r["lon"])
            except Exception:
                continue

            name_norm = str(r.get("location_norm", "")).strip().lower()

            # live status (recent reports or fallback)
            live = safe_compute_live_status(
                location=name,
                parking_df=parking_df_spot,
                predictor=predictor,
                recency_minutes=RECENCY_WINDOW_MINUTES,
                spot_type=map_spot_type,
            )

            # total reports for bubble size (TOTAL across entire dataset for that spot_type)
            try:
                total_reports = int((parking_df_spot["location_norm"] == name_norm).sum())
            except Exception:
                total_reports = 0

            radius = scale_radius(total_reports)

            # availability
            try:
                availability = float(live.get("availability", 0.5))
            except Exception:
                availability = 0.5

            # color by availability
            if availability >= 0.65:
                color = [40, 167, 69, 200]   # green
            elif availability >= 0.35:
                color = [255, 193, 7, 200]   # yellow
            else:
                color = [220, 53, 69, 200]   # red

            # ensure icon/text exist
            status_text = str(live.get("status_text", "Unknown"))
            icon = str(live.get("icon", "⚪"))
            if status_text == "Unknown" and icon == "⚪":
                status_text, icon = _get_status_and_icon(availability)

            map_rows.append(
                {
                    "location": name,
                    "lat": lat,
                    "lon": lon,
                    "availability": availability,
                    "availability_pct": f"{availability:.0%}",
                    "status_text": status_text,
                    "icon": icon,
                    "last_report_text": str(live.get("last_report_text", "No reports")),
                    "recent_reports": int(live.get("report_count", 0) or 0),
                    "total_reports": total_reports,
                    "radius": radius,
                    "color": color,
                }
            )

    map_df = pd.DataFrame(map_rows)

    # Choose map center
    if map_df is not None and len(map_df) > 0 and map_df["lat"].notna().any() and map_df["lon"].notna().any():
        center_lat = float(map_df["lat"].mean())
        center_lon = float(map_df["lon"].mean())
    else:
        center_lat, center_lon = DEFAULT_CENTER_LAT, DEFAULT_CENTER_LON

    # Build ONE folium map
    m = make_folium_map(center_lat, center_lon, zoom=14)

    # Add bubbles for locations
    add_bubble_markers(m, map_df)

    # Show a visible pin for the last clicked point (if any)
    if "map_click_lat" in st.session_state and "map_click_lon" in st.session_state:
        try:
            folium.Marker(
                [float(st.session_state["map_click_lat"]), float(st.session_state["map_click_lon"])],
                tooltip="Selected point",
                icon=folium.Icon(color="blue", icon="info-sign"),
            ).add_to(m)
        except Exception:
            pass

    # Render and capture clicks
    click_data = st_folium(
        m,
        height=520,
        width=None,
        returned_objects=["last_clicked"],
        key="smartpark_single_map",
    )

    if click_data and click_data.get("last_clicked"):
        st.session_state["map_click_lat"] = float(click_data["last_clicked"]["lat"])
        st.session_state["map_click_lon"] = float(click_data["last_clicked"]["lng"])
        st.success(f"Captured click: ({st.session_state['map_click_lat']:.6f}, {st.session_state['map_click_lon']:.6f})")

    # Legend
    st.markdown("---")
    l1, l2, l3 = st.columns(3)
    with l1:
        st.markdown("🟢 **High Availability** (≥65%)")
    with l2:
        st.markdown("🟡 **Medium Availability** (35–65%)")
    with l3:
        st.markdown("🔴 **Low Availability** (<35%)")

    # Add-location panel
    st.markdown("---")
    st.subheader("➕ Add a new location from the click")

    default_lat = st.session_state.get("map_click_lat", center_lat)
    default_lon = st.session_state.get("map_click_lon", center_lon)

    with st.expander("Add location", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            new_lat = st.number_input(
                "Latitude",
                min_value=-90.0,
                max_value=90.0,
                value=float(default_lat),
                step=0.0001,
                format="%.6f",
                key="map_add_lat",
            )
        with c2:
            new_lon = st.number_input(
                "Longitude",
                min_value=-180.0,
                max_value=180.0,
                value=float(default_lon),
                step=0.0001,
                format="%.6f",
                key="map_add_lon",
            )

        new_name = st.text_input(
            "Location name (optional)",
            value="",
            max_chars=50,
            key="map_add_name",
            placeholder="Example: Rhame Ave - North Side",
        )
        new_notes = st.text_input(
            "Notes (optional)",
            value="",
            max_chars=100,
            key="map_add_notes",
            placeholder="Example: near CVS, 2-hour limit",
        )

        if st.button("✅ Add Location", type="primary", use_container_width=True, key="map_add_submit"):
            lat_val = float(new_lat)
            lon_val = float(new_lon)

            name_clean = (new_name or "").strip()
            if not name_clean:
                name_clean = f"Map Pin ({lat_val:.5f}, {lon_val:.5f})"

            # Duplicate protection (case-insensitive)
            exists = False
            try:
                all_df = location_manager.get_all_locations_df()
                if all_df is not None and len(all_df) > 0 and "location" in all_df.columns:
                    exists = name_clean.lower() in (
                        all_df["location"].fillna("").astype(str).str.strip().str.lower().values
                    )
            except Exception:
                exists = False

            if exists:
                st.warning("⚠️ A location with that name already exists. Pick a different name.")
            else:
                ok = False
                try:
                    ok = location_manager.add_location(
                        location=name_clean,
                        notes=(new_notes or "").strip(),
                        created_by=user_nickname,
                        lat=lat_val,
                        lon=lon_val,
                    )
                except TypeError:
                    # if your LocationManager doesn't support lat/lon yet
                    ok = location_manager.add_location(
                        location=name_clean,
                        notes=(new_notes or "").strip(),
                        created_by=user_nickname,
                    )

                if ok:
                    event_logger.log(
                        "location_add_map",
                        location=name_clean,
                        user=user_nickname,
                        details=f"Coords: ({lat_val:.6f},{lon_val:.6f}) | Notes: {(new_notes or '').strip()}",
                    )
                    st.success(f"✅ Added location: **{name_clean}**")
                    st.cache_resource.clear()
                    time.sleep(0.25)
                    st.rerun()
                else:
                    st.error("❌ Could not add location (it may already exist or there was a write error).")

    # Refresh loop
    if seconds_since_map_refresh >= AUTO_REFRESH_SECONDS:
        st.session_state.map_last_refresh = datetime.now()
        time.sleep(0.08)
        st.rerun()

# ============================================================
# TAB: LIVE BOARD
# ============================================================
with tab_live:
    st.header("🟢 Live Parking Board")
    st.markdown(f"Updates every {AUTO_REFRESH_SECONDS}s")

    live_spot_type = st.radio(
        "♿ View availability for:",
        options=["general", "accessible"],
        format_func=spot_label,
        horizontal=True,
        key="live_spot_type",
    )

    if "last_refresh" not in st.session_state:
        st.session_state.last_refresh = datetime.now()

    seconds_since_refresh = (datetime.now() - st.session_state.last_refresh).total_seconds()
    next_refresh_in = max(0, AUTO_REFRESH_SECONDS - int(seconds_since_refresh))
    st.caption(f"🔄 Next refresh in {next_refresh_in}s")

    active_locations = location_manager.get_active_locations()
    if not active_locations:
        st.info("No active locations yet. Add some in Manage Locations.")
    else:
        df = safe_read_parking_data(PARKING_DATA_FILE)
        df["spot_type_norm"] = df["spot_type"].fillna("general").astype(str).str.strip().str.lower()
        df_live = df[df["spot_type_norm"] == live_spot_type].copy()

        now = datetime.now()
        for loc in sorted(active_locations):
            live = safe_compute_live_status(
                location=loc,
                parking_df=df_live,
                predictor=predictor,
                recency_minutes=RECENCY_WINDOW_MINUTES,
                spot_type=live_spot_type,
            )

            # If fallback happened and didn't apply spot_type somewhere, override safely
            if not live.get("is_live", False):
                try:
                    live["availability"] = predictor.predict(loc, now.strftime("%A"), now.hour, spot_type=live_spot_type)
                    live["status_text"], live["icon"] = _get_status_and_icon(live["availability"])
                except Exception:
                    pass

            st.markdown(
                f"""
                <div class="live-status-card">
                    <h3 style="margin: 0 0 0.5rem 0;">{live.get('icon','⚪')} {loc}</h3>
                    <div style="color:#666; font-size:0.9rem;">{spot_label(live_spot_type)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            a, b, c = st.columns([2, 2, 3])
            with a:
                st.markdown(f"**Status:** {live.get('status_text','Unknown')}")
                st.markdown(f"**Availability:** {float(live.get('availability',0.5)):.0%}")
            with b:
                st.markdown(f"**Last Report:** {live.get('last_report_text','No reports')}")
                st.markdown(f"**Recent Reports:** {int(live.get('report_count',0))}")
            with c:
                trend = live.get("trend_icons", "")
                st.markdown(
                    f"**Recent Trend:** <span class='trend-icons'>{trend}</span>"
                    if trend else "**Recent Trend:** No recent data",
                    unsafe_allow_html=True,
                )
                st.caption("✨ Live data" if live.get("is_live", False) else "📊 Historical prediction")

            st.markdown("")

    if seconds_since_refresh >= AUTO_REFRESH_SECONDS:
        st.session_state.last_refresh = datetime.now()
        time.sleep(0.08)
        st.rerun()

# ============================================================
# TAB: CHECK AVAILABILITY
# ============================================================
with tab_check:
    st.header("🔍 Check Parking Availability")

    col1, col2, col3 = st.columns(3)

    with col1:
        locations = location_manager.get_active_locations()
        if not locations:
            st.warning("No locations available. Add some in Manage Locations.")
            selected_location = None
        else:
            default_idx = 0
            if "tab_selected_location" in st.session_state and st.session_state["tab_selected_location"] in locations:
                default_idx = locations.index(st.session_state["tab_selected_location"])
            selected_location = st.selectbox("📍 Select Location", options=locations, index=default_idx)

        default_spot_idx = 0
        if st.session_state.get("tab_selected_spot_type") == "accessible":
            default_spot_idx = 1

        spot_type = st.radio(
            "♿ Spot Type",
            options=["general", "accessible"],
            format_func=spot_label,
            horizontal=True,
            index=default_spot_idx,
            key="spot_type_predict",
        )

    with col2:
        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        selected_day = st.selectbox("📅 Day of Week", options=days, index=datetime.now().weekday())

    with col3:
        selected_hour = st.slider("🕐 Hour of Day", 0, 23, datetime.now().hour)

    st.markdown("")
    predict_button = st.button(
        "🔮 Predict Availability",
        type="primary",
        use_container_width=True,
        disabled=(selected_location is None),
    )

    if predict_button and selected_location:
        with st.spinner("Analyzing patterns..."):
            time.sleep(0.2)

            probability = predictor.predict(selected_location, selected_day, int(selected_hour), spot_type=spot_type)

            df = safe_read_parking_data(PARKING_DATA_FILE)
            df["spot_type_norm"] = df["spot_type"].fillna("general").astype(str).str.strip().str.lower()
            df_spot = df[df["spot_type_norm"] == spot_type].copy()

            confidence = compute_prediction_confidence(selected_location, selected_day, int(selected_hour), df_spot)
            explanation = generate_prediction_explanation(selected_location, selected_day, int(selected_hour), df_spot, float(probability))

            event_logger.log(
                event_type="predict",
                location=selected_location,
                day_of_week=selected_day,
                hour=int(selected_hour),
                user=user_nickname,
                details=f"Spot: {spot_type} | Probability: {float(probability):.2%} | Confidence: {confidence.get('level','Low')}",
                spot_type=spot_type,
            )

        st.markdown("---")
        st.subheader("🎯 Prediction Results")

        prob = float(probability)
        if prob >= 0.7:
            color, emoji, message, recommendation = "#28a745", "✅", "Excellent chance of finding parking!", "Go ahead — parking should be available."
        elif prob >= 0.5:
            color, emoji, message, recommendation = "#ffc107", "👍", "Good chance of finding parking", "Favorable odds, but have a backup plan."
        elif prob >= 0.3:
            color, emoji, message, recommendation = "#fd7e14", "⚠️", "Moderate availability — arrive early", "Arrive 10–15 minutes early or keep a backup ready."
        else:
            color, emoji, message, recommendation = "#dc3545", "🚫", "Low availability expected", "Consider alternatives (another lot, transit, or different time)."

        st.markdown(f"<h1 style='text-align:center; color:{color}; margin:0;'>{emoji} {prob:.0%}</h1>", unsafe_allow_html=True)
        st.markdown(f"<p style='text-align:center; font-size:20px; font-weight:600; margin-top:0.5rem;'>{message}</p>", unsafe_allow_html=True)
        st.progress(prob)

        stars = confidence.get("stars", "★☆☆☆☆")
        level = confidence.get("level", "Low")
        count = int(confidence.get("report_count", 0))
        expl = confidence.get("explanation", "")

        st.markdown(
            "<div style='text-align:center;'>"
            "<span style='font-size:0.9rem; color:#666;'>Confidence: </span>"
            f"<span class='confidence-stars'>{stars}</span> "
            f"<span style='font-size:0.9rem; color:#666;'>({level})</span>"
            "</div>",
            unsafe_allow_html=True,
        )
        st.caption(f"Based on {count} relevant report(s). {expl}".strip())
        st.info(f"💡 **Recommendation:** {recommendation}")

        with st.expander("📖 How was this prediction made?", expanded=True):
            st.markdown(f"**Spot Type:** {spot_label(spot_type)}")
            st.markdown(explanation.get("text", ""))
            if explanation.get("details"):
                st.caption(explanation["details"])

# ============================================================
# TAB: SUBMIT REPORT
# ============================================================
with tab_submit:
    st.header("📝 Submit Parking Report")

    with st.form("parking_report_form", clear_on_submit=True):
        c1, c2 = st.columns(2)

        with c1:
            locations = location_manager.get_active_locations()
            if not locations:
                st.warning("No locations available. Add some in Manage Locations.")
                report_location = None
            else:
                report_location = st.selectbox("📍 Location", options=locations)

        with c2:
            found_parking_option = st.radio(
                "Did you find parking?",
                options=["✅ Yes, I found parking", "❌ No, no parking available"],
            )

        report_spot_type = st.radio(
            "♿ What kind of spot were you looking for?",
            options=["general", "accessible"],
            format_func=spot_label,
            horizontal=True,
            key="spot_type_report",
        )

        report_notes = st.text_area("Additional Notes (optional)", max_chars=200)

        submit_button = st.form_submit_button(
            "📤 Submit Report",
            type="primary",
            use_container_width=True,
            disabled=(report_location is None),
        )

    if submit_button and report_location:
        found = "Yes" in found_parking_option
        now = datetime.now()

        new_report = {
            "timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
            "location": report_location,
            "day_of_week": now.strftime("%A"),
            "hour": int(now.hour),
            "found_parking": 1 if found else 0,
            "spot_type": report_spot_type,
        }

        try:
            append_parking_report(new_report, PARKING_DATA_FILE)

            details = f"Found: {found}"
            if report_notes and report_notes.strip():
                details += f" | Notes: {report_notes.strip()}"

            event_logger.log(
                event_type="report_submit",
                location=report_location,
                day_of_week=new_report["day_of_week"],
                hour=new_report["hour"],
                found_parking=new_report["found_parking"],
                user=user_nickname,
                details=details,
                spot_type=report_spot_type,
            )

            # If your LocationManager has trust features, try incrementing verified
            if hasattr(location_manager, "increment_verified"):
                try:
                    location_manager.increment_verified(report_location)
                except Exception:
                    pass

            st.cache_resource.clear()
            st.success("✅ Report submitted. Thank you!")
            st.balloons()
            time.sleep(0.4)
            st.rerun()

        except Exception as e:
            st.error(f"❌ Error submitting report: {e}")

# ============================================================
# TAB: MANAGE LOCATIONS
# ============================================================
with tab_manage:
    st.header("📍 Manage Parking Locations")

    st.subheader("➕ Add New Location")
    with st.form("add_location_form", clear_on_submit=True):
        a1, a2 = st.columns(2)
        with a1:
            new_location_name = st.text_input("Location Name *", max_chars=50)
        with a2:
            new_location_notes = st.text_input("Notes (optional)", max_chars=100)

        st.markdown("**📍 Coordinates (optional — for map)**")
        cc1, cc2 = st.columns(2)
        with cc1:
            new_lat = st.number_input(
                "Latitude",
                min_value=-90.0,
                max_value=90.0,
                value=DEFAULT_CENTER_LAT,
                step=0.0001,
                format="%.6f",
            )
        with cc2:
            new_lon = st.number_input(
                "Longitude",
                min_value=-180.0,
                max_value=180.0,
                value=DEFAULT_CENTER_LON,
                step=0.0001,
                format="%.6f",
            )

        add_location_button = st.form_submit_button("➕ Add Location", type="primary", use_container_width=True)

    if add_location_button:
        name = (new_location_name or "").strip()
        notes = (new_location_notes or "").strip()

        if not name:
            st.error("❌ Please enter a location name.")
        else:
            try:
                success = location_manager.add_location(
                    location=name, notes=notes, created_by=user_nickname, lat=new_lat, lon=new_lon
                )
            except TypeError:
                success = location_manager.add_location(location=name, notes=notes, created_by=user_nickname)

            if success:
                event_logger.log(
                    event_type="location_add",
                    location=name,
                    user=user_nickname,
                    details=f"Notes: {notes} | Coords: ({new_lat}, {new_lon})" if notes else f"Coords: ({new_lat}, {new_lon})",
                )
                st.success(f"✅ Location '{name}' added!")
                st.cache_resource.clear()
                time.sleep(0.3)
                st.rerun()
            else:
                st.warning("⚠️ Location already exists or could not be added.")

    st.markdown("---")
    st.subheader("📋 Current Locations")

    try:
        locations_df = location_manager.get_all_locations_df()
    except Exception:
        locations_df = pd.DataFrame()

    if locations_df is None or len(locations_df) == 0:
        st.info("No locations yet.")
    else:
        if "lat" not in locations_df.columns:
            locations_df["lat"] = None
        if "lon" not in locations_df.columns:
            locations_df["lon"] = None
        if "is_active" in locations_df.columns:
            locations_df["is_active"] = locations_df["is_active"].astype(str).str.lower().isin(["true", "1", "yes"])
        else:
            locations_df["is_active"] = True

        active_count = int(locations_df["is_active"].sum())
        coords_count = int(
            (pd.to_numeric(locations_df["lat"], errors="coerce").notna() & pd.to_numeric(locations_df["lon"], errors="coerce").notna()).sum()
        )
        st.caption(f"{len(locations_df)} location(s) • {active_count} active • {coords_count} with coordinates")

        for idx, row in locations_df.iterrows():
            loc_name = str(row.get("location", "")).strip()
            if not loc_name:
                continue

            notes = row.get("notes", "")
            notes = "" if pd.isna(notes) else str(notes).strip()

            has_coords = pd.notna(pd.to_numeric(row.get("lat", None), errors="coerce")) and pd.notna(
                pd.to_numeric(row.get("lon", None), errors="coerce")
            )
            has_coords_marker = "🗺️ " if has_coords else ""
            label = f"{has_coords_marker}{'✅' if bool(row.get('is_active', True)) else '❌'} {loc_name}"
            if notes:
                label += f" - {notes}"

            with st.expander(label, expanded=False):
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    st.markdown(f"**Created:** {row.get('created_at','')}")
                with c2:
                    st.markdown(f"**Created by:** {row.get('created_by','')}")
                with c3:
                    st.markdown(f"**Status:** {'🟢 Active' if bool(row.get('is_active', True)) else '🔴 Inactive'}")
                with c4:
                    trust = row.get("trust_score", None)
                    try:
                        trust = float(trust)
                        badge = "🟢" if trust >= 70 else ("🟡" if trust >= 40 else "🔴")
                        st.markdown(f"**Trust:** {badge} {trust:.0f}")
                    except Exception:
                        st.markdown("**Trust:** —")

                if has_coords:
                    try:
                        st.markdown(f"**📍 Coordinates:** {float(row['lat']):.6f}, {float(row['lon']):.6f}")
                    except Exception:
                        st.caption("Coordinates present but unreadable.")
                else:
                    st.caption("No coordinates set for this location (won't appear on map).")

                if bool(row.get("is_active", True)):
                    b1, b2 = st.columns(2)
                    with b1:
                        if st.button("🚫 Deactivate", key=f"deactivate_{idx}"):
                            try:
                                location_manager.deactivate_location(loc_name)
                            except Exception:
                                pass
                            event_logger.log("location_deactivate", location=loc_name, user=user_nickname, details="Deactivated")
                            st.cache_resource.clear()
                            st.success("Deactivated.")
                            time.sleep(0.2)
                            st.rerun()
                    with b2:
                        if st.button("🚩 Flag", key=f"flag_{idx}"):
                            if hasattr(location_manager, "increment_flagged"):
                                try:
                                    location_manager.increment_flagged(loc_name)
                                except Exception:
                                    pass
                            event_logger.log("location_flag", location=loc_name, user=user_nickname, details="Flagged")
                            st.cache_resource.clear()
                            st.warning("Flagged.")
                            time.sleep(0.2)
                            st.rerun()
                else:
                    st.info("This location is inactive.")

# ============================================================
# TAB: ANALYTICS
# ============================================================
with tab_analytics:
    st.header("📊 Parking Analytics")

    df = safe_read_parking_data(PARKING_DATA_FILE)
    if len(df) == 0:
        st.info("No data yet — submit reports to see analytics.")
    else:
        df["timestamp_dt"] = pd.to_datetime(df["timestamp"], errors="coerce")

        spot_filter = st.selectbox(
            "♿ Filter by spot type",
            options=["All", "general", "accessible"],
            format_func=lambda x: "All" if x == "All" else spot_label(x),
        )

        df_view = df.copy()
        df_view["spot_type_norm"] = df_view["spot_type"].fillna("general").astype(str).str.strip().str.lower()
        if spot_filter in ["general", "accessible"]:
            df_view = df_view[df_view["spot_type_norm"] == spot_filter].copy()

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Total Reports", f"{len(df_view):,}")
        with m2:
            try:
                sr2 = float(pd.to_numeric(df_view["found_parking"], errors="coerce").fillna(0).clip(0, 1).mean())
            except Exception:
                sr2 = 0.0
            st.metric("Success Rate", f"{sr2:.1%}")
        with m3:
            st.metric("Locations", int(df_view["location"].nunique()) if "location" in df_view.columns else 0)
        with m4:
            if df_view["timestamp_dt"].notna().any():
                date_range = (df_view["timestamp_dt"].max().date() - df_view["timestamp_dt"].min().date()).days
                st.metric("Data Range", f"{date_range} days")
            else:
                st.metric("Data Range", "N/A")

        st.markdown("---")

        st.subheader("📍 Success Rate by Location")
        try:
            loc_stats = (
                df_view.groupby("location")
                .agg(found_mean=("found_parking", "mean"), reports=("found_parking", "count"))
                .reset_index()
                .sort_values("reports", ascending=False)
            )
            loc_stats["Success Rate"] = loc_stats["found_mean"].apply(lambda x: f"{float(x):.1%}")
            loc_stats = loc_stats.rename(columns={"location": "Location", "reports": "Reports"})
            st.dataframe(loc_stats[["Location", "Success Rate", "Reports"]], use_container_width=True, hide_index=True)
        except Exception as e:
            st.warning(f"Could not compute location stats: {e}")

        st.markdown("---")
        st.subheader("🕒 Recent Reports")
        try:
            recent_df = df_view.sort_values("timestamp", ascending=False).head(12)
            recent_display = recent_df[["timestamp", "location", "spot_type", "day_of_week", "hour", "found_parking"]].copy()
            recent_display["found_parking"] = pd.to_numeric(recent_display["found_parking"], errors="coerce").fillna(0).astype(int)
            recent_display["found_parking"] = recent_display["found_parking"].apply(lambda x: "✅ Yes" if x == 1 else "❌ No")
            recent_display["spot_type"] = recent_display["spot_type"].apply(lambda x: spot_label(str(x)))
            recent_display.columns = ["Timestamp", "Location", "Spot Type", "Day", "Hour", "Found Parking"]
            st.dataframe(recent_display, use_container_width=True, hide_index=True)
        except Exception as e:
            st.warning(f"Could not show recent reports: {e}")
def apply_fullscreen_css(enabled: bool):
    if not enabled:
        st.markdown(
            """
            <style>
            /* normal mode */
            .main > div { padding-top: 2rem; }
            </style>
            """,
            unsafe_allow_html=True,
        )
        return

    st.markdown(
        """
        <style>
        /* "fullscreen-like" map mode */
        .main > div { padding-top: 0.2rem !important; }
        header, footer { visibility: hidden; height: 0px; }
        /* Hide the sidebar */
        [data-testid="stSidebar"] { display: none !important; width: 0 !important; }
        /* Give main area more space */
        section.main { padding-left: 0.5rem !important; padding-right: 0.5rem !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )

# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #888;'>"
    "SmartPark v2.3 | Built with ❤️ using Streamlit | Data stored locally and anonymously"
    "</p>",
    unsafe_allow_html=True,
)
