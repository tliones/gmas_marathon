import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# ============================================================
# Grandma's Marathon Course Visualizer - Streamlit MVP
# ============================================================

st.set_page_config(
    page_title="Grandma's Marathon Course Visualizer",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_DIR = Path("data")
COURSE_POINTS_FILE = DATA_DIR / "grandmas_course_points.csv"
COURSE_MARKERS_FILE = DATA_DIR / "grandmas_course_markers.csv"
VIDEO_SYNC_FILE = DATA_DIR / "video_mile_sync_template.csv"
SUMMARY_FILE = DATA_DIR / "summary.json"

YOUTUBE_URL = "https://www.youtube.com/watch?v=lLwf_fIW0L8"


# ------------------------------------------------------------
# Load data
# ------------------------------------------------------------
@st.cache_data
def load_course_points(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {
        "mile",
        "lat",
        "lon",
        "elevation_smooth_ft",
        "grade_smooth_percent",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {path}: {sorted(missing)}")
    return df.sort_values("mile").reset_index(drop=True)


@st.cache_data
def load_markers(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df.sort_values(["mile", "marker_type"]).reset_index(drop=True)


@st.cache_data
def load_video_sync(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["video_second", "mile", "landmark_or_note"])

    df = pd.read_csv(path)

    # Template may have blank video_second values. Keep only usable rows.
    if "video_second" not in df.columns or "mile" not in df.columns:
        return pd.DataFrame(columns=["video_second", "mile", "landmark_or_note"])

    df["video_second"] = pd.to_numeric(df["video_second"], errors="coerce")
    df["mile"] = pd.to_numeric(df["mile"], errors="coerce")
    df = df.dropna(subset=["video_second", "mile"]).sort_values("video_second")

    return df


@st.cache_data
def load_summary(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    return {}


def nearest_course_point(course: pd.DataFrame, mile: float) -> pd.Series:
    idx = (course["mile"] - mile).abs().idxmin()
    return course.loc[idx]


def course_window(course: pd.DataFrame, mile: float, behind=0.25, ahead=1.0) -> pd.DataFrame:
    return course[(course["mile"] >= mile - behind) & (course["mile"] <= mile + ahead)].copy()


def upcoming_markers(markers: pd.DataFrame, mile: float, ahead=1.0) -> pd.DataFrame:
    return markers[(markers["mile"] >= mile) & (markers["mile"] <= mile + ahead)].copy()


def previous_markers(markers: pd.DataFrame, mile: float, behind=0.25) -> pd.DataFrame:
    return markers[(markers["mile"] < mile) & (markers["mile"] >= mile - behind)].copy()


def next_mile_summary(course: pd.DataFrame, mile: float, ahead=1.0) -> dict:
    w = course_window(course, mile, behind=0.0, ahead=ahead)
    if w.empty:
        return {"gain_ft": 0.0, "loss_ft": 0.0, "max_grade": 0.0, "avg_grade": 0.0}

    elev_diff = w["elevation_smooth_ft"].diff().fillna(0)
    gain = elev_diff[elev_diff > 0].sum()
    loss = abs(elev_diff[elev_diff < 0].sum())

    return {
        "gain_ft": float(gain),
        "loss_ft": float(loss),
        "max_grade": float(w["grade_smooth_percent"].max()),
        "avg_grade": float(w["grade_smooth_percent"].mean()),
    }


def interpolate_mile_from_video(video_second: float, sync: pd.DataFrame) -> float:
    if sync.empty:
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
        (video_second - before["video_second"])
        / (after["video_second"] - before["video_second"])
    )
    return float(before["mile"] + ratio * (after["mile"] - before["mile"]))


def race_cue(mile: float, summary: dict, current_grade: float) -> tuple[str, str]:
    if 20.5 <= mile <= 21.5:
        return (
            "Lemon Drop / late-course climbing area",
            "Shorten stride, keep cadence, and hold effort instead of chasing pace.",
        )

    if summary["gain_ft"] >= 45 or current_grade >= 1.5:
        return (
            "Climbing ahead",
            "Use effort-based pacing. Let pace float slower uphill.",
        )

    if summary["loss_ft"] >= 45 or current_grade <= -1.5:
        return (
            "Downhill / net-loss section",
            "Stay smooth and avoid overstriding, especially late in the race.",
        )

    if mile < 5:
        return (
            "Early control",
            "Settle in. The goal is to arrive at mile 5 feeling almost too relaxed.",
        )

    if 13 <= mile < 18:
        return (
            "Middle miles",
            "Stay patient, fuel on schedule, and keep effort boringly steady.",
        )

    if mile >= 23:
        return (
            "Final 5K focus",
            "Use the crowd, but keep form compact before committing to a final push.",
        )

    return (
        "Rhythm section",
        "Keep cadence relaxed and focus on steady fueling and effort.",
    )


# ------------------------------------------------------------
# App
# ------------------------------------------------------------
st.title("Grandma's Marathon Course Visualizer")
st.caption("MVP version: manual mile/video slider + GPX-derived course data + aid/fuel markers.")

try:
    course = load_course_points(COURSE_POINTS_FILE)
    markers = load_markers(COURSE_MARKERS_FILE)
    sync = load_video_sync(VIDEO_SYNC_FILE)
    summary = load_summary(SUMMARY_FILE)
except Exception as exc:
    st.error(f"Could not load app data: {exc}")
    st.stop()

max_mile = float(course["mile"].max())

with st.sidebar:
    st.header("Controls")

    mode_options = ["Course mile slider"]
    if not sync.empty:
        mode_options.append("Video time slider")

    mode = st.radio("Navigation mode", mode_options)

    if mode == "Video time slider":
        max_video_second = int(sync["video_second"].max())
        video_second = st.slider(
            "Video time, seconds",
            min_value=0,
            max_value=max_video_second,
            value=0,
            step=5,
        )
        current_mile = interpolate_mile_from_video(video_second, sync)
    else:
        current_mile = st.slider(
            "Course mile",
            min_value=0.0,
            max_value=round(max_mile, 1),
            value=0.0,
            step=0.1,
        )

    preview_window = st.slider(
        "Upcoming preview window, miles",
        min_value=0.5,
        max_value=3.0,
        value=1.0,
        step=0.5,
    )

    st.divider()
    st.markdown("### Data check")
    st.write(f"Course points: **{len(course):,}**")
    st.write(f"Course distance: **{max_mile:.2f} mi**")
    if summary:
        st.write(f"Raw GPX distance: **{summary.get('raw_gpx_distance_miles', 0):.2f} mi**")
    st.write(f"Video sync rows filled: **{len(sync)}**")

current = nearest_course_point(course, current_mile)
future = next_mile_summary(course, current_mile, ahead=preview_window)
upcoming = upcoming_markers(markers, current_mile, ahead=preview_window)
recent = previous_markers(markers, current_mile, behind=0.25)
cue_title, cue_text = race_cue(current_mile, future, current["grade_smooth_percent"])

video_col, context_col = st.columns([1.25, 1.0])

with video_col:
    st.subheader("Course Video")
    st.video(YOUTUBE_URL)

    st.info(
        "For this MVP, use the course-mile slider while watching the video. "
        "Once you fill `video_mile_sync_template.csv`, the app will enable a video-time slider."
    )

with context_col:
    st.subheader("Current Course Context")

    m1, m2, m3 = st.columns(3)
    m1.metric("Mile", f"{current_mile:.1f}")
    m2.metric("Elevation", f"{current['elevation_smooth_ft']:.0f} ft")
    m3.metric("Grade", f"{current['grade_smooth_percent']:.1f}%")

    st.markdown(f"### {cue_title}")
    st.write(cue_text)

    st.markdown(f"### Next {preview_window:g} mile(s)")
    n1, n2, n3 = st.columns(3)
    n1.metric("Gain", f"{future['gain_ft']:.0f} ft")
    n2.metric("Loss", f"{future['loss_ft']:.0f} ft")
    n3.metric("Max grade", f"{future['max_grade']:.1f}%")

    if len(upcoming) > 0 or len(recent) > 0:
        st.markdown("### Nearby markers")
        nearby = pd.concat([recent, upcoming]).drop_duplicates().sort_values("mile")
        for _, row in nearby.iterrows():
            st.markdown(
                f"**Mile {row['mile']:.1f}: {row['name']}**  \n"
                f"{row['items']}  \n"
                f"{row['notes'] if pd.notna(row['notes']) else ''}"
            )
    else:
        st.caption("No aid/fuel/course marker in the selected preview window.")

st.divider()

# ------------------------------------------------------------
# Elevation profile
# ------------------------------------------------------------
st.subheader("Elevation Profile")

elev_fig = go.Figure()

elev_fig.add_trace(
    go.Scatter(
        x=course["mile"],
        y=course["elevation_smooth_ft"],
        mode="lines",
        name="Elevation",
        hovertemplate="Mile %{x:.2f}<br>Elevation %{y:.0f} ft<extra></extra>",
    )
)

aid = markers[markers["marker_type"].isin(["aid_station", "fuel", "food", "hill"])]
if not aid.empty:
    # Put marker y values on nearest elevation profile point.
    marker_y = [
        nearest_course_point(course, float(m))["elevation_smooth_ft"]
        for m in aid["mile"]
    ]
    elev_fig.add_trace(
        go.Scatter(
            x=aid["mile"],
            y=marker_y,
            mode="markers",
            name="Markers",
            text=aid["name"],
            hovertemplate="%{text}<br>Mile %{x:.1f}<extra></extra>",
        )
    )

elev_fig.add_trace(
    go.Scatter(
        x=[current_mile],
        y=[current["elevation_smooth_ft"]],
        mode="markers",
        marker=dict(size=16),
        name="Current position",
        hovertemplate="Current mile %{x:.2f}<br>%{y:.0f} ft<extra></extra>",
    )
)

elev_fig.update_layout(
    height=360,
    xaxis_title="Mile",
    yaxis_title="Elevation, ft",
    margin=dict(l=20, r=20, t=30, b=20),
)

st.plotly_chart(elev_fig, use_container_width=True)

# ------------------------------------------------------------
# Grade profile
# ------------------------------------------------------------
st.subheader("Smoothed Grade Profile")

grade_fig = go.Figure()

grade_fig.add_trace(
    go.Scatter(
        x=course["mile"],
        y=course["grade_smooth_percent"],
        mode="lines",
        name="Smoothed grade",
        hovertemplate="Mile %{x:.2f}<br>Grade %{y:.1f}%<extra></extra>",
    )
)

grade_fig.add_hline(y=0, line_dash="dash")

grade_fig.add_trace(
    go.Scatter(
        x=[current_mile],
        y=[current["grade_smooth_percent"]],
        mode="markers",
        marker=dict(size=16),
        name="Current grade",
        hovertemplate="Current mile %{x:.2f}<br>%{y:.1f}%<extra></extra>",
    )
)

grade_fig.update_layout(
    height=300,
    xaxis_title="Mile",
    yaxis_title="Grade, %",
    margin=dict(l=20, r=20, t=30, b=20),
)

st.plotly_chart(grade_fig, use_container_width=True)

# ------------------------------------------------------------
# Map
# ------------------------------------------------------------
st.subheader("Course Map")

map_fig = go.Figure()

map_fig.add_trace(
    go.Scattermapbox(
        lat=course["lat"],
        lon=course["lon"],
        mode="lines",
        name="Course",
        hoverinfo="skip",
    )
)

map_fig.add_trace(
    go.Scattermapbox(
        lat=[current["lat"]],
        lon=[current["lon"]],
        mode="markers",
        marker=dict(size=16),
        name="Current position",
        hovertemplate=f"Mile {current_mile:.2f}<extra></extra>",
    )
)

marker_map = markers[markers["marker_type"].isin(["aid_station", "fuel", "food", "hill", "start", "finish"])]
map_fig.add_trace(
    go.Scattermapbox(
        lat=[nearest_course_point(course, float(m))["lat"] for m in marker_map["mile"]],
        lon=[nearest_course_point(course, float(m))["lon"] for m in marker_map["mile"]],
        mode="markers",
        marker=dict(size=9),
        name="Course markers",
        text=marker_map["name"],
        hovertemplate="%{text}<extra></extra>",
    )
)

map_fig.update_layout(
    mapbox_style="open-street-map",
    mapbox=dict(
        center=dict(lat=float(current["lat"]), lon=float(current["lon"])),
        zoom=12,
    ),
    height=500,
    margin=dict(l=0, r=0, t=0, b=0),
)

st.plotly_chart(map_fig, use_container_width=True)

# ------------------------------------------------------------
# Tables
# ------------------------------------------------------------
with st.expander("Aid, fuel, food, and course markers"):
    st.dataframe(markers, use_container_width=True)

with st.expander("Video sync template"):
    if sync.empty:
        st.write(
            "No video timestamps have been filled yet. Edit "
            "`data/video_mile_sync_template.csv` and add video seconds for known miles."
        )
        st.dataframe(pd.read_csv(VIDEO_SYNC_FILE), use_container_width=True)
    else:
        st.dataframe(sync, use_container_width=True)

with st.expander("Raw GPX-derived course points"):
    st.dataframe(course, use_container_width=True)
