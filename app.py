from pathlib import Path
import json
import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components


# ============================================================
# App setup
# ============================================================

st.set_page_config(
    page_title="Grandma's Marathon Course Visualizer",
    layout="wide"
)

YOUTUBE_ID = "lLwf_fIW0L8"
YOUTUBE_URL = f"https://www.youtube.com/watch?v={YOUTUBE_ID}"

ROOT = Path(__file__).parent


# ============================================================
# File path helper
# Checks root folder first, then /data folder
# ============================================================

def resolve_data_file(filename: str) -> Path:
    candidates = [
        ROOT / filename,
        ROOT / "data" / filename,
    ]

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        f"Could not find {filename}. Put it beside app.py or in a data/ folder."
    )


# ============================================================
# Time parsing helpers
# ============================================================

def parse_time_to_seconds(value):
    """
    Accepts:
    - 369
    - "369"
    - "6:09"
    - "00:06:09"
    Returns seconds as float or NaN.
    """
    if pd.isna(value):
        return np.nan

    s = str(value).strip()

    if s == "":
        return np.nan

    # Already numeric
    try:
        return float(s)
    except ValueError:
        pass

    # mm:ss or hh:mm:ss
    if ":" in s:
        parts = s.split(":")
        try:
            parts = [float(p) for p in parts]
        except ValueError:
            return np.nan

        if len(parts) == 2:
            minutes, seconds = parts
            return minutes * 60 + seconds

        if len(parts) == 3:
            hours, minutes, seconds = parts
            return hours * 3600 + minutes * 60 + seconds

    return np.nan


def format_seconds(seconds):
    if pd.isna(seconds):
        return ""
    seconds = int(round(float(seconds)))
    m = seconds // 60
    s = seconds % 60
    return f"{m}:{s:02d}"


# ============================================================
# Data loaders
# ============================================================

@st.cache_data
def load_course(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    required = [
        "mile",
        "lat",
        "lon",
        "elevation_smooth_ft",
        "grade_smooth_percent",
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Course file is missing required columns: {missing}")

    df["mile"] = pd.to_numeric(df["mile"], errors="coerce")
    df["elevation_smooth_ft"] = pd.to_numeric(df["elevation_smooth_ft"], errors="coerce")
    df["grade_smooth_percent"] = pd.to_numeric(df["grade_smooth_percent"], errors="coerce")
    df = df.dropna(subset=["mile", "lat", "lon"])
    return df.sort_values("mile").reset_index(drop=True)


@st.cache_data
def load_markers(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Be forgiving about column names from earlier prep versions.
    if "marker_type" not in df.columns:
        if "type" in df.columns:
            df = df.rename(columns={"type": "marker_type"})
        elif "category" in df.columns:
            df = df.rename(columns={"category": "marker_type"})
        else:
            df["marker_type"] = "marker"

    required = ["mile", "marker_type", "name", "items", "notes"]
    for col in required:
        if col not in df.columns:
            df[col] = ""

    df["mile"] = pd.to_numeric(df["mile"], errors="coerce")
    df = df.dropna(subset=["mile"])
    return df.sort_values(["mile", "marker_type"]).reset_index(drop=True)


@st.cache_data
def load_video_sync(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Allow several possible time column names.
    possible_time_cols = [
        "video_second",
        "video_seconds",
        "seconds",
        "time_seconds",
        "time_mmss",
        "timestamp",
        "time",
    ]

    if "video_second" not in df.columns:
        for col in possible_time_cols:
            if col in df.columns:
                df["video_second"] = df[col]
                break
        else:
            df["video_second"] = np.nan

    if "mile" not in df.columns:
        raise ValueError("Video sync file needs a 'mile' column.")

    # First try video_second. If it is blank/non-numeric, try time_mmss/timestamp/time.
    df["video_second"] = df["video_second"].apply(parse_time_to_seconds)

    for backup_col in ["time_mmss", "timestamp", "time"]:
        if backup_col in df.columns:
            missing_mask = df["video_second"].isna()
            df.loc[missing_mask, "video_second"] = (
                df.loc[missing_mask, backup_col].apply(parse_time_to_seconds)
            )

    df["mile"] = pd.to_numeric(df["mile"], errors="coerce")

    if "landmark_or_note" not in df.columns:
        if "notes" in df.columns:
            df["landmark_or_note"] = df["notes"]
        else:
            df["landmark_or_note"] = ""

    df = df.dropna(subset=["video_second", "mile"])
    df = df.sort_values("video_second").reset_index(drop=True)
    df["time_mmss_display"] = df["video_second"].apply(format_seconds)

    return df


@st.cache_data
def load_summary(path: str | None) -> dict:
    if path is None:
        return {}

    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


# ============================================================
# Analysis helpers
# ============================================================

def interpolate_mile_from_video(video_second: float, sync: pd.DataFrame) -> float:
    sync = sync.sort_values("video_second")

    if len(sync) == 0:
        return 0.0

    if video_second <= sync["video_second"].min():
        return float(sync.iloc[0]["mile"])

    if video_second >= sync["video_second"].max():
        return float(sync.iloc[-1]["mile"])

    before = sync[sync["video_second"] <= video_second].iloc[-1]
    after = sync[sync["video_second"] >= video_second].iloc[0]

    if before["video_second"] == after["video_second"]:
        return float(before["mile"])

    ratio = (
        (video_second - before["video_second"]) /
        (after["video_second"] - before["video_second"])
    )

    return float(before["mile"] + ratio * (after["mile"] - before["mile"]))


def nearest_course_point(course: pd.DataFrame, mile: float) -> pd.Series:
    idx = (course["mile"] - mile).abs().idxmin()
    return course.loc[idx]


def get_course_window(course: pd.DataFrame, mile: float, ahead: float) -> pd.DataFrame:
    return course[(course["mile"] >= mile) & (course["mile"] <= mile + ahead)].copy()


def summarize_window(course: pd.DataFrame, mile: float, ahead: float) -> dict:
    window = get_course_window(course, mile, ahead)
    if window.empty:
        return {"gain": 0, "loss": 0, "max_grade": 0}

    diff = window["elevation_smooth_ft"].diff().fillna(0)
    return {
        "gain": float(diff[diff > 0].sum()),
        "loss": float(abs(diff[diff < 0].sum())),
        "max_grade": float(window["grade_smooth_percent"].max()),
    }


def nearby_markers(markers: pd.DataFrame, mile: float, window: float) -> pd.DataFrame:
    return markers[
        (markers["mile"] >= mile) &
        (markers["mile"] <= mile + window)
    ].copy()


def race_cue(mile: float, grade: float) -> tuple[str, str]:
    if mile < 5:
        return "Early control", "Settle in. The goal is to arrive at mile 5 feeling almost too relaxed."
    if 20.5 <= mile <= 21.7:
        return "Lemon Drop Hill area", "Shorten stride, keep cadence, and hold effort rather than chasing pace."
    if mile >= 24:
        return "Finish setup", "Use the crowd, but keep form together. Start pressing only if you still feel controlled."
    if grade > 1.5:
        return "Climbing cue", "Hold effort and avoid forcing goal pace uphill."
    if grade < -1.5:
        return "Downhill cue", "Stay smooth and avoid overstriding."
    return "Rhythm cue", "Run relaxed and steady. Keep fueling and hydration on schedule."


def render_youtube(video_second: int | None = None):
    """
    Streamlit cannot read live playback time from st.video().
    This iframe version can jump the YouTube video to the selected timestamp
    when the app reruns.
    """
    start_arg = ""
    if video_second is not None:
        start_arg = f"&start={int(video_second)}"

    embed_url = (
        f"https://www.youtube.com/embed/{YOUTUBE_ID}"
        f"?rel=0&modestbranding=1{start_arg}"
    )

    components.html(
        f"""
        <div style="position: relative; padding-bottom: 56.25%; height: 0; overflow: hidden;
                    border-radius: 12px;">
            <iframe
                src="{embed_url}"
                title="Grandma's Marathon Course Tour"
                frameborder="0"
                allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
                allowfullscreen
                style="position: absolute; top:0; left:0; width:100%; height:100%;">
            </iframe>
        </div>
        """,
        height=430,
    )


# ============================================================
# Load files
# ============================================================

st.title("Grandma's Marathon Course Visualizer")
st.caption("MVP version: video timestamp/course mile slider + GPX-derived course data + aid/fuel markers.")

try:
    course_path = resolve_data_file("grandmas_course_points.csv")
    markers_path = resolve_data_file("grandmas_course_markers.csv")
    sync_path = resolve_data_file("video_mile_sync_template.csv")

    try:
        summary_path = resolve_data_file("summary.json")
    except Exception:
        summary_path = None

    course = load_course(str(course_path))
    markers = load_markers(str(markers_path))
    sync = load_video_sync(str(sync_path))
    summary = load_summary(str(summary_path)) if summary_path else {}

except Exception as e:
    st.error(f"Could not load app data: {e}")
    st.stop()


# ============================================================
# Sidebar controls
# ============================================================

with st.sidebar:
    st.header("Controls")

    sync_available = len(sync) >= 2

    if sync_available:
        mode = st.radio(
            "Navigation mode",
            ["Video timestamp slider", "Course mile slider"],
            index=0
        )
    else:
        mode = "Course mile slider"
        st.radio(
            "Navigation mode",
            ["Course mile slider"],
            index=0
        )

    preview_window = st.slider(
        "Upcoming preview window, miles",
        min_value=0.5,
        max_value=3.0,
        value=1.0,
        step=0.5
    )

    max_mile = float(course["mile"].max())

    if mode == "Video timestamp slider" and sync_available:
        min_sec = int(sync["video_second"].min())
        max_sec = int(sync["video_second"].max())

        video_second = st.slider(
            "Video timestamp",
            min_value=min_sec,
            max_value=max_sec,
            value=min_sec,
            step=1,
            format="%d sec"
        )

        current_mile = interpolate_mile_from_video(video_second, sync)

        st.caption(f"Selected video time: **{format_seconds(video_second)}**")
        st.caption(f"Interpolated course mile: **{current_mile:.2f}**")

        jump_video = st.checkbox(
            "Jump embedded video to selected timestamp",
            value=True,
            help="When checked, moving the video timestamp slider reloads the embedded YouTube video at that timestamp."
        )

    else:
        video_second = None
        current_mile = st.slider(
            "Course mile",
            min_value=0.0,
            max_value=max_mile,
            value=0.0,
            step=0.05
        )
        jump_video = False

    st.divider()
    st.subheader("Data check")
    st.write(f"Course points: **{len(course):,}**")
    st.write(f"Course distance: **{course['mile'].max():.2f} mi**")
    if "raw_distance_miles" in summary:
        st.write(f"Raw GPX distance: **{summary['raw_distance_miles']:.2f} mi**")
    st.write(f"Video sync rows filled: **{len(sync)}**")

    with st.expander("Resolved data files"):
        st.write(str(course_path))
        st.write(str(markers_path))
        st.write(str(sync_path))
        if summary_path:
            st.write(str(summary_path))

    if len(sync) > 0:
        with st.expander("Video sync preview"):
            st.dataframe(sync.head(10), use_container_width=True)


# ============================================================
# Current point calculations
# ============================================================

current_point = nearest_course_point(course, current_mile)
window_summary = summarize_window(course, current_mile, preview_window)
nearby = nearby_markers(markers, current_mile, preview_window)
cue_title, cue_text = race_cue(current_mile, current_point["grade_smooth_percent"])


# ============================================================
# Top layout
# ============================================================

left, right = st.columns([1.25, 1])

with left:
    st.subheader("Course Video")

    if jump_video and video_second is not None:
        render_youtube(video_second)
    else:
        render_youtube(None)

    if sync_available:
        st.info(
            "Use the **Video timestamp slider** in the sidebar to move the course data. "
            "If the jump option is checked, the embedded YouTube video reloads at that timestamp. "
            "This is a practical workaround because Streamlit cannot read the live playback time from YouTube."
        )
    else:
        st.warning(
            "No usable video sync rows were found. Check that `video_mile_sync_template.csv` "
            "has columns like `video_second,mile` or `time_mmss,mile`."
        )

with right:
    st.subheader("Current Course Context")

    m1, m2, m3 = st.columns(3)
    m1.metric("Mile", f"{current_mile:.2f}")
    m2.metric("Elevation", f"{current_point['elevation_smooth_ft']:.0f} ft")
    m3.metric("Grade", f"{current_point['grade_smooth_percent']:.1f}%")

    st.markdown(f"### {cue_title}")
    st.write(cue_text)

    st.markdown(f"### Next {preview_window:.1f} mile(s)")
    c1, c2, c3 = st.columns(3)
    c1.metric("Gain", f"{window_summary['gain']:.0f} ft")
    c2.metric("Loss", f"{window_summary['loss']:.0f} ft")
    c3.metric("Max grade", f"{window_summary['max_grade']:.1f}%")

    st.markdown("### Nearby markers")
    if nearby.empty:
        st.caption("No aid/fuel/course marker in this preview window.")
    else:
        for _, row in nearby.iterrows():
            st.markdown(
                f"**Mile {row['mile']:.1f}: {row['name']}**  \n"
                f"{row['items']}  \n"
                f"{row['notes']}"
            )


# ============================================================
# Elevation profile
# ============================================================

st.subheader("Elevation Profile")

elev_fig = go.Figure()

elev_fig.add_trace(
    go.Scatter(
        x=course["mile"],
        y=course["elevation_smooth_ft"],
        mode="lines",
        name="Elevation"
    )
)

if len(markers) > 0:
    marker_points = []
    for _, row in markers.iterrows():
        cp = nearest_course_point(course, row["mile"])
        marker_points.append({
            "mile": row["mile"],
            "elevation": cp["elevation_smooth_ft"],
            "name": row["name"],
            "marker_type": row["marker_type"]
        })

    marker_df = pd.DataFrame(marker_points)

    elev_fig.add_trace(
        go.Scatter(
            x=marker_df["mile"],
            y=marker_df["elevation"],
            mode="markers",
            name="Markers",
            text=marker_df["name"],
            hovertemplate="Mile %{x:.1f}<br>%{text}<extra></extra>"
        )
    )

elev_fig.add_trace(
    go.Scatter(
        x=[current_mile],
        y=[current_point["elevation_smooth_ft"]],
        mode="markers",
        marker=dict(size=16),
        name="Current position"
    )
)

elev_fig.update_layout(
    xaxis_title="Mile",
    yaxis_title="Elevation, ft",
    height=380,
    margin=dict(l=20, r=20, t=20, b=20)
)

st.plotly_chart(elev_fig, use_container_width=True)


# ============================================================
# Grade profile
# ============================================================

st.subheader("Smoothed Grade Profile")

grade_fig = go.Figure()

grade_fig.add_trace(
    go.Scatter(
        x=course["mile"],
        y=course["grade_smooth_percent"],
        mode="lines",
        name="Grade"
    )
)

grade_fig.add_hline(y=0, line_dash="dash")

grade_fig.add_trace(
    go.Scatter(
        x=[current_mile],
        y=[current_point["grade_smooth_percent"]],
        mode="markers",
        marker=dict(size=16),
        name="Current position"
    )
)

grade_fig.update_layout(
    xaxis_title="Mile",
    yaxis_title="Grade, %",
    height=320,
    margin=dict(l=20, r=20, t=20, b=20)
)

st.plotly_chart(grade_fig, use_container_width=True)


# ============================================================
# Map
# ============================================================

st.subheader("Course Map")

map_fig = go.Figure()

map_fig.add_trace(
    go.Scattermapbox(
        lat=course["lat"],
        lon=course["lon"],
        mode="lines",
        name="Course"
    )
)

map_fig.add_trace(
    go.Scattermapbox(
        lat=[current_point["lat"]],
        lon=[current_point["lon"]],
        mode="markers",
        marker=dict(size=14),
        name="Current position"
    )
)

map_fig.update_layout(
    mapbox_style="open-street-map",
    mapbox=dict(
        center=dict(
            lat=float(current_point["lat"]),
            lon=float(current_point["lon"])
        ),
        zoom=12
    ),
    height=500,
    margin=dict(l=0, r=0, t=0, b=0)
)

st.plotly_chart(map_fig, use_container_width=True)


# ============================================================
# Tables
# ============================================================

with st.expander("Course markers table"):
    st.dataframe(markers, use_container_width=True)

with st.expander("Video sync table"):
    st.dataframe(sync, use_container_width=True)

with st.expander("Course points table"):
    st.dataframe(course, use_container_width=True)
