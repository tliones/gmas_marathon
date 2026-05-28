from pathlib import Path
import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


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
# File helpers
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
        f"Could not find {filename}. Put it beside app.py or inside a data/ folder."
    )


# ============================================================
# Type-safe helpers
# ============================================================

def parse_time_to_seconds(value):
    if pd.isna(value):
        return np.nan

    s = str(value).strip()

    if s == "":
        return np.nan

    try:
        return float(s)
    except ValueError:
        pass

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


def as_text(value):
    if pd.isna(value):
        return ""
    return str(value).strip()


def youtube_timestamp_url(seconds):
    seconds = int(round(float(seconds)))
    return f"https://www.youtube.com/watch?v={YOUTUBE_ID}&t={seconds}s"


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

    for col in ["mile", "lat", "lon", "elevation_smooth_ft", "grade_smooth_percent"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")

    df = df.dropna(subset=["mile", "lat", "lon"]).copy()
    return df.sort_values("mile").reset_index(drop=True)


@st.cache_data
def load_markers(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Be forgiving about column names.
    if "marker_type" not in df.columns:
        if "type" in df.columns:
            df = df.rename(columns={"type": "marker_type"})
        elif "category" in df.columns:
            df = df.rename(columns={"category": "marker_type"})
        else:
            df["marker_type"] = "marker"

    for col in ["mile", "marker_type", "name", "items", "notes"]:
        if col not in df.columns:
            df[col] = ""

    df["mile"] = pd.to_numeric(df["mile"], errors="coerce").astype("float64")
    df = df.dropna(subset=["mile"]).copy()

    for col in ["marker_type", "name", "items", "notes"]:
        df[col] = df[col].fillna("").astype(str).str.strip()

    # Drop blank markers. These create repeated entries like "**Mile 0.4:**".
    df = df[
        (df["name"] != "") |
        (df["items"] != "") |
        (df["notes"] != "")
    ].copy()

    # Remove exact duplicates.
    df = df.drop_duplicates(
        subset=["mile", "marker_type", "name", "items", "notes"]
    )

    return df.sort_values(["mile", "marker_type", "name"]).reset_index(drop=True)


@st.cache_data
def load_video_sync(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str).fillna("")

    unnamed_cols = [c for c in df.columns if str(c).startswith("Unnamed")]
    if unnamed_cols:
        df = df.drop(columns=unnamed_cols)

    if "mile" not in df.columns:
        raise ValueError("Video sync file needs a 'mile' column.")

    if "video_second" in df.columns:
        seconds = df["video_second"].apply(parse_time_to_seconds)
    else:
        seconds = pd.Series([np.nan] * len(df), dtype="float64")

    for backup_col in ["time_mmss", "timestamp", "time", "video_seconds", "seconds", "time_seconds"]:
        if backup_col in df.columns:
            missing_mask = seconds.isna()
            if missing_mask.any():
                seconds.loc[missing_mask] = df.loc[missing_mask, backup_col].apply(parse_time_to_seconds)

    df["video_second"] = pd.to_numeric(seconds, errors="coerce").astype("float64")
    df["mile"] = pd.to_numeric(df["mile"], errors="coerce").astype("float64")

    if "landmark_or_note" not in df.columns:
        if "notes" in df.columns:
            df["landmark_or_note"] = df["notes"]
        else:
            df["landmark_or_note"] = ""

    df["landmark_or_note"] = df["landmark_or_note"].fillna("").astype(str).str.strip()

    df = df.dropna(subset=["video_second", "mile"]).copy()
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
    if len(sync) == 0:
        return 0.0

    sync = sync.sort_values("video_second").reset_index(drop=True)

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


def summarize_window(course: pd.DataFrame, mile: float, ahead: float) -> dict:
    window = course[(course["mile"] >= mile) & (course["mile"] <= mile + ahead)].copy()

    if window.empty:
        return {"gain": 0.0, "loss": 0.0, "max_grade": 0.0}

    diff = window["elevation_smooth_ft"].diff().fillna(0)
    return {
        "gain": float(diff[diff > 0].sum()),
        "loss": float(abs(diff[diff < 0].sum())),
        "max_grade": float(window["grade_smooth_percent"].max()),
    }


def upcoming_markers(markers: pd.DataFrame, mile: float, window: float) -> pd.DataFrame:
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


# ============================================================
# Load files
# ============================================================

st.title("Grandma's Marathon Course Visualizer")
st.caption("Stable version: Streamlit-native video + timestamp/course-mile slider + GPX course data.")

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
    max_mile = float(course["mile"].max())

    if sync_available:
        mode = st.radio(
            "Navigation mode",
            ["Video timestamp slider", "Course mile slider", "Mile marker select"],
            index=0
        )
    else:
        mode = st.radio(
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

    video_second = None

    if mode == "Video timestamp slider" and sync_available:
        min_sec = int(sync["video_second"].min())
        max_sec = int(sync["video_second"].max())

        video_second = st.slider(
            "Video timestamp",
            min_value=min_sec,
            max_value=max_sec,
            value=min_sec,
            step=1
        )

        current_mile = interpolate_mile_from_video(video_second, sync)

        st.caption(f"Selected video time: **{format_seconds(video_second)}**")
        st.caption(f"Interpolated course mile: **{current_mile:.2f}**")

    elif mode == "Mile marker select" and sync_available:
        labels = [
            f"{row.time_mmss_display} | Mile {row.mile:g} | {row.landmark_or_note}"
            for row in sync.itertuples(index=False)
        ]

        selected_label = st.selectbox("Jump to synced marker", labels, index=0)
        selected_index = labels.index(selected_label)
        video_second = float(sync.iloc[selected_index]["video_second"])
        current_mile = float(sync.iloc[selected_index]["mile"])

        st.caption(f"Selected video time: **{format_seconds(video_second)}**")
        st.caption(f"Selected course mile: **{current_mile:.2f}**")

    else:
        current_mile = st.slider(
            "Course mile",
            min_value=0.0,
            max_value=max_mile,
            value=0.0,
            step=0.05
        )

        if sync_available:
            sorted_sync = sync.sort_values("mile")
            video_second = float(np.interp(
                current_mile,
                sorted_sync["mile"].to_numpy(dtype=float),
                sorted_sync["video_second"].to_numpy(dtype=float)
            ))
            st.caption(f"Estimated video time: **{format_seconds(video_second)}**")

    sync_video = st.checkbox(
        "Start video at selected timestamp",
        value=True,
        help=(
            "Uses Streamlit's native st.video(start_time=...). "
            "The video updates after Streamlit reruns from a slider/selectbox change."
        )
    )

    autoplay_video = st.checkbox(
        "Autoplay video",
        value=False,
        help="Some browsers block autoplay, especially with sound."
    )

    st.divider()
    st.subheader("Data check")
    st.write(f"Course points: **{len(course):,}**")
    st.write(f"Course distance: **{course['mile'].max():.2f} mi**")
    if "raw_distance_miles" in summary:
        st.write(f"Raw GPX distance: **{summary['raw_distance_miles']:.2f} mi**")
    st.write(f"Course markers: **{len(markers)}**")
    st.write(f"Video sync rows filled: **{len(sync)}**")

    with st.expander("Resolved data files"):
        st.write(str(course_path))
        st.write(str(markers_path))
        st.write(str(sync_path))
        if summary_path:
            st.write(str(summary_path))

    if len(sync) > 0:
        with st.expander("Video sync preview"):
            st.dataframe(sync, use_container_width=True)


# ============================================================
# Current point calculations
# ============================================================

current_point = nearest_course_point(course, current_mile)
current_elev = float(current_point["elevation_smooth_ft"])
current_grade = float(current_point["grade_smooth_percent"])
window_summary = summarize_window(course, current_mile, preview_window)
nearby = upcoming_markers(markers, current_mile, preview_window)
cue_title, cue_text = race_cue(current_mile, current_grade)


# ============================================================
# Top layout
# ============================================================

left, right = st.columns([1.25, 1])

with left:
    st.subheader("Course Video")

    if sync_video and video_second is not None:
        st.video(
            YOUTUBE_URL,
            start_time=int(round(float(video_second))),
            autoplay=autoplay_video,
            muted=True
        )
        st.caption(
            f"Video requested start time: {format_seconds(video_second)}. "
            f"[Open this timestamp on YouTube]({youtube_timestamp_url(video_second)})"
        )
    else:
        st.video(YOUTUBE_URL)
        st.caption("Video sync is off, so the video loads from the beginning.")

    st.info(
        "The controls update the course context, charts, and video start time after you release the slider. "
        "For exact video seeking, use the YouTube timestamp link under the video."
    )

with right:
    st.subheader("Current Course Context")

    m1, m2, m3 = st.columns(3)
    m1.metric("Mile", f"{current_mile:.2f}")
    m2.metric("Elevation", f"{current_elev:.0f} ft")
    m3.metric("Grade", f"{current_grade:.1f}%")

    st.markdown(f"### {cue_title}")
    st.write(cue_text)

    st.markdown(f"### Next {preview_window:.1f} mile(s)")
    c1, c2, c3 = st.columns(3)
    c1.metric("Gain", f"{window_summary['gain']:.0f} ft")
    c2.metric("Loss", f"{window_summary['loss']:.0f} ft")
    c3.metric("Max grade", f"{window_summary['max_grade']:.1f}%")

    st.markdown("### Upcoming course markers")
    if nearby.empty:
        st.caption("No aid, fuel, food, or course marker in this preview window.")
    else:
        max_markers_to_show = 8

        for _, row in nearby.head(max_markers_to_show).iterrows():
            note_text = as_text(row["notes"])
            items_text = as_text(row["items"])
            name_text = as_text(row["name"])
            type_text = as_text(row["marker_type"]).replace("_", " ").title()

            marker_text = f"**Mile {float(row['mile']):.1f}: {name_text}**"
            if type_text and type_text.lower() != "marker":
                marker_text += f"  \n_Type: {type_text}_"
            if items_text:
                marker_text += f"  \n{items_text}"
            if note_text:
                marker_text += f"  \n{note_text}"

            st.markdown(marker_text)

        if len(nearby) > max_markers_to_show:
            st.caption(
                f"Showing first {max_markers_to_show} of {len(nearby)} upcoming markers."
            )


# ============================================================
# Elevation profile
# ============================================================

st.subheader("Elevation Profile")

elev_fig = go.Figure()

elev_fig.add_trace(
    go.Scatter(
        x=course["mile"].to_numpy(dtype=float),
        y=course["elevation_smooth_ft"].to_numpy(dtype=float),
        mode="lines",
        name="Elevation"
    )
)

if len(markers) > 0:
    marker_rows = []
    for _, row in markers.iterrows():
        cp = nearest_course_point(course, float(row["mile"]))
        marker_rows.append({
            "mile": float(row["mile"]),
            "elevation": float(cp["elevation_smooth_ft"]),
            "name": as_text(row["name"]),
        })

    marker_df = pd.DataFrame(marker_rows)

    if not marker_df.empty:
        elev_fig.add_trace(
            go.Scatter(
                x=marker_df["mile"].to_numpy(dtype=float),
                y=marker_df["elevation"].to_numpy(dtype=float),
                mode="markers",
                name="Course markers",
                text=marker_df["name"].astype(str).tolist(),
                hovertemplate="Mile %{x:.1f}<br>%{text}<extra></extra>"
            )
        )

elev_fig.add_trace(
    go.Scatter(
        x=[float(current_mile)],
        y=[current_elev],
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
        x=course["mile"].to_numpy(dtype=float),
        y=course["grade_smooth_percent"].to_numpy(dtype=float),
        mode="lines",
        name="Grade"
    )
)

grade_fig.add_hline(y=0, line_dash="dash")

grade_fig.add_trace(
    go.Scatter(
        x=[float(current_mile)],
        y=[current_grade],
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
        lat=course["lat"].to_numpy(dtype=float),
        lon=course["lon"].to_numpy(dtype=float),
        mode="lines",
        name="Course"
    )
)

map_fig.add_trace(
    go.Scattermapbox(
        lat=[float(current_point["lat"])],
        lon=[float(current_point["lon"])],
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
