from pathlib import Path
from string import Template
import json

import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# App setup
# ============================================================

st.set_page_config(
    page_title="Grandma's Marathon Course Visualizer",
    layout="wide",
    initial_sidebar_state="collapsed"
)

YOUTUBE_ID = "lLwf_fIW0L8"
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

    df = df[
        (df["name"] != "") |
        (df["items"] != "") |
        (df["notes"] != "")
    ].copy()

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
    return df.sort_values("video_second").reset_index(drop=True)


# ============================================================
# Grade smoothing
# ============================================================

def add_windowed_grade(course: pd.DataFrame, window_miles: float = 0.15) -> pd.DataFrame:
    """
    Recalculate grade over a distance window so GPX elevation noise does not
    create unrealistic short spikes.
    """
    df = course.sort_values("mile").reset_index(drop=True).copy()
    miles = df["mile"].to_numpy(dtype=float)
    elev = df["elevation_smooth_ft"].to_numpy(dtype=float)

    half_window = window_miles / 2.0
    grades = np.zeros(len(df), dtype=float)

    for i, m in enumerate(miles):
        left_mile = max(miles[0], m - half_window)
        right_mile = min(miles[-1], m + half_window)

        left_elev = np.interp(left_mile, miles, elev)
        right_elev = np.interp(right_mile, miles, elev)

        distance_ft = max((right_mile - left_mile) * 5280.0, 1.0)
        grades[i] = ((right_elev - left_elev) / distance_ft) * 100.0

    df["grade_course_percent"] = (
        pd.Series(grades)
        .rolling(window=21, center=True, min_periods=1)
        .mean()
        .clip(lower=-8.0, upper=8.0)
        .astype("float64")
    )
    return df


# ============================================================
# Load data
# ============================================================

try:
    course = load_course(str(resolve_data_file("grandmas_course_points.csv")))
    course = add_windowed_grade(course, window_miles=0.15)

    markers = load_markers(str(resolve_data_file("grandmas_course_markers.csv")))
    sync = load_video_sync(str(resolve_data_file("video_mile_sync_template.csv")))

except Exception as e:
    st.error(f"Could not load app data: {e}")
    st.stop()


if len(sync) < 2:
    st.warning(
        "The live-sync player needs at least two rows in video_mile_sync_template.csv "
        "with video_second/time_mmss and mile."
    )
    st.stop()


# ============================================================
# Browser payload
# ============================================================

course_for_browser = course.copy()
if len(course_for_browser) > 2400:
    step = int(np.ceil(len(course_for_browser) / 2400))
    course_for_browser = course_for_browser.iloc[::step].copy()
    if course_for_browser.iloc[-1]["mile"] != course.iloc[-1]["mile"]:
        course_for_browser = pd.concat([course_for_browser, course.tail(1)], ignore_index=True)

course_records = course_for_browser[
    ["mile", "lat", "lon", "elevation_smooth_ft", "grade_course_percent"]
].rename(columns={"grade_course_percent": "grade_smooth_percent"}).to_dict(orient="records")

sync_records = sync[
    ["video_second", "mile", "landmark_or_note"]
].to_dict(orient="records")

marker_records = markers[
    ["mile", "marker_type", "name", "items", "notes"]
].to_dict(orient="records")

# Official aid-station fallback if marker CSV is blank or overwritten.
if len(marker_records) == 0:
    aid_miles = [3, 5, 7, 9, 11, 13, 15, 17, 19, 20, 21, 22, 23, 24, 25]
    marker_records = []
    for m in aid_miles:
        items = "Water; Powerade Ion4 Mountain Berry Blast; toilets; first aid"
        notes = ""
        if m in [9, 25]:
            items += "; Hiccup Earth reusable cups"
            notes = "Reusable cup station. Do not take cups with you."
        if m == 17:
            items += "; PURE FUEL by Anderson’s Maple Syrup"
        if m == 19:
            items += "; fresh fruit near mile 19"
        if m == 23:
            items += "; fresh fruit near mile 23.5 nearby"
        marker_records.append({
            "mile": float(m),
            "marker_type": "aid_station",
            "name": f"Aid Station Mile {m}",
            "items": items,
            "notes": notes
        })
    marker_records.append({
        "mile": 23.5,
        "marker_type": "food",
        "name": "Fresh Fruit",
        "items": "Fresh fruit by Super One Foods",
        "notes": "Near mile 23.5"
    })


course_json = json.dumps(course_records)
sync_json = json.dumps(sync_records)
markers_json = json.dumps(marker_records)


# ============================================================
# End-user layout
# ============================================================

st.markdown(
    """
    <style>
      #MainMenu, footer, header {visibility: hidden;}
      .block-container {
        padding-top: 0.4rem;
        padding-bottom: 0.3rem;
        padding-left: 0.6rem;
        padding-right: 0.6rem;
        max-width: 100%;
      }
      [data-testid="stSidebar"] {
        display: none;
      }
    </style>
    """,
    unsafe_allow_html=True
)

html_template = Template(r"""
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8" />
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>

  <style>
    :root {
      --bg: #0b1020;
      --panel: #111827;
      --panel2: #0f172a;
      --border: #253047;
      --text: #f8fafc;
      --muted: #a7b0c0;
      --blue: #3b82f6;
      --orange: #f97316;
    }

    html, body {
      margin: 0;
      padding: 0;
      background: var(--bg);
      color: var(--text);
      font-family: Arial, Helvetica, sans-serif;
      overflow: hidden;
    }

    .app {
      height: 96vh;
      min-height: 820px;
      display: grid;
      grid-template-rows: 42px 1fr;
      gap: 8px;
      padding: 8px;
      box-sizing: border-box;
    }

    .topbar {
      display: flex;
      align-items: center;
      justify-content: space-between;
      background: linear-gradient(90deg, #111827, #172554);
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 8px 14px;
      box-sizing: border-box;
    }

    .title {
      font-size: 20px;
      font-weight: 800;
      letter-spacing: 0.1px;
    }

    .subtitle {
      color: var(--muted);
      font-size: 13px;
    }

    .dashboard {
      display: grid;
      grid-template-columns: 38% 26% 36%;
      grid-template-rows: 34% 31% 35%;
      gap: 8px;
      min-height: 0;
    }

    .panel {
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 10px;
      box-sizing: border-box;
      min-height: 0;
      overflow: hidden;
    }

    .video-panel {
      grid-column: 1 / 2;
      grid-row: 1 / 3;
      display: flex;
      flex-direction: column;
      gap: 8px;
    }

    .context-panel {
      grid-column: 2 / 3;
      grid-row: 1 / 3;
      overflow: auto;
    }

    .map-panel {
      grid-column: 3 / 4;
      grid-row: 1 / 3;
    }

    .elev-panel {
      grid-column: 1 / 3;
      grid-row: 3 / 4;
    }

    .grade-panel {
      grid-column: 3 / 4;
      grid-row: 3 / 4;
    }

    .video-wrap {
      position: relative;
      width: 100%;
      flex: 1;
      min-height: 250px;
      overflow: hidden;
      border-radius: 12px;
      background: #000;
    }

    #player {
      position: absolute;
      top: 0;
      left: 0;
      width: 100% !important;
      height: 100% !important;
    }
    .metrics {
      display: grid;
      grid-template-columns: repeat(2, 1fr);
      gap: 8px;
      margin-bottom: 8px;
    }

    .metric {
      background: var(--panel2);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 8px;
    }

    .metric-label {
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 2px;
    }

    .metric-value {
      color: #fff;
      font-size: 24px;
      line-height: 1.1;
      font-weight: 800;
    }

    .subhead {
      font-size: 16px;
      font-weight: 800;
      margin: 8px 0 6px 0;
    }

    .cue {
      background: #102a43;
      border: 1px solid #1f4e79;
      border-radius: 12px;
      padding: 9px;
      line-height: 1.35;
      font-size: 13px;
      margin-bottom: 8px;
    }

    .marker {
      border-left: 4px solid var(--blue);
      padding: 7px 9px;
      margin: 7px 0;
      background: var(--panel2);
      border-radius: 8px;
      line-height: 1.32;
      font-size: 13px;
    }

    .small {
      font-size: 12px;
      color: var(--muted);
      line-height: 1.35;
    }

    #elevationPlot, #gradePlot, #map {
      height: calc(100% - 22px);
      width: 100%;
    }

    .panel-title {
      font-size: 15px;
      font-weight: 800;
      margin: 0 0 4px 0;
    }

    @media (max-width: 1050px) {
      html, body {
        overflow: auto;
      }

      .app {
        height: auto;
        min-height: 0;
      }

      .dashboard {
        display: grid;
        grid-template-columns: 1fr;
        grid-template-rows: auto;
      }

      .video-panel, .context-panel, .map-panel, .elev-panel, .grade-panel {
        grid-column: auto;
        grid-row: auto;
        min-height: 360px;
      }

      .context-panel {
        min-height: 420px;
      }

      .map-panel, .elev-panel, .grade-panel {
        height: 380px;
      }
    }
  </style>
</head>

<body>
  <div class="app">
    <div class="topbar">
      <div>
        <div class="title">Grandma’s Marathon Course Visualizer</div>
        <div class="subtitle">Video-synced elevation, grade, aid stations, and course position</div>
      </div>
      <div class="subtitle" id="statusText">Loading course...</div>
    </div>

    <div class="dashboard">
      <div class="panel video-panel">
        <div class="video-wrap">
          <div id="player"></div>
        </div>
      </div>

      <div class="panel context-panel">
        <div class="metrics">
          <div class="metric">
            <div class="metric-label">Video Time</div>
            <div class="metric-value" id="timeMetric">0:00</div>
          </div>
          <div class="metric">
            <div class="metric-label">Course Mile</div>
            <div class="metric-value" id="mileMetric">0.00</div>
          </div>
          <div class="metric">
            <div class="metric-label">Elevation</div>
            <div class="metric-value" id="elevMetric">--</div>
          </div>
          <div class="metric">
            <div class="metric-label">Smoothed Grade</div>
            <div class="metric-value" id="gradeMetric">--</div>
          </div>
          <div class="metric">
            <div class="metric-label">Next 1 mi Gain</div>
            <div class="metric-value" id="gainMetric">--</div>
          </div>
          <div class="metric">
            <div class="metric-label">Next 1 mi Loss</div>
            <div class="metric-value" id="lossMetric">--</div>
          </div>
        </div>

        <div class="subhead" id="cueTitle">Course Cue</div>
        <div class="cue" id="cueText">Loading course context...</div>

        <div class="subhead">Upcoming course markers</div>
        <div id="markerList" class="small">Loading markers...</div>
      </div>

      <div class="panel map-panel">
        <div class="panel-title">Course Map</div>
        <div id="map"></div>
      </div>

      <div class="panel elev-panel">
        <div class="panel-title">Elevation Profile</div>
        <div id="elevationPlot"></div>
      </div>

      <div class="panel grade-panel">
        <div class="panel-title">Smoothed Grade Profile</div>
        <div id="gradePlot"></div>
      </div>
    </div>
  </div>

<script>
const course = $course_json.map(d => ({
  ...d,
  mile: Number(d.mile),
  lat: Number(d.lat),
  lon: Number(d.lon),
  elevation_smooth_ft: Number(d.elevation_smooth_ft),
  grade_smooth_percent: Number(d.grade_smooth_percent)
}));

const sync = $sync_json.map(d => ({
  ...d,
  video_second: Number(d.video_second),
  mile: Number(d.mile)
}));

const markers = $markers_json
  .map(d => ({
    ...d,
    mile: Number(d.mile),
    marker_type: String(d.marker_type || "marker"),
    name: String(d.name || ""),
    items: String(d.items || ""),
    notes: String(d.notes || "")
  }))
  .filter(d =>
    Number.isFinite(d.mile) &&
    (d.name.trim() !== "" || d.items.trim() !== "" || d.notes.trim() !== "")
  )
  .sort((a, b) => a.mile - b.mile);

let player;
let map;
let courseMarker;
let courseLine;
let liveUpdateTimer;

const elevationX = course.map(d => d.mile);
const elevationY = course.map(d => d.elevation_smooth_ft);
const gradeY = course.map(d => d.grade_smooth_percent);
const latLngs = course.map(d => [d.lat, d.lon]);

function formatSeconds(seconds) {
  seconds = Math.round(seconds);
  const m = Math.floor(seconds / 60);
  const s = seconds % 60;
  return String(m) + ":" + String(s).padStart(2, "0");
}

function interpolateMileFromVideo(t) {
  if (t <= sync[0].video_second) return sync[0].mile;
  if (t >= sync[sync.length - 1].video_second) return sync[sync.length - 1].mile;

  for (let i = 0; i < sync.length - 1; i++) {
    const a = sync[i];
    const b = sync[i + 1];

    if (t >= a.video_second && t <= b.video_second) {
      const ratio = (t - a.video_second) / (b.video_second - a.video_second);
      return a.mile + ratio * (b.mile - a.mile);
    }
  }

  return sync[0].mile;
}

function nearestCoursePoint(mile) {
  let best = course[0];
  let bestDiff = Math.abs(course[0].mile - mile);

  for (const p of course) {
    const diff = Math.abs(p.mile - mile);
    if (diff < bestDiff) {
      best = p;
      bestDiff = diff;
    }
  }
  return best;
}

function summarizeNextMile(mile) {
  const end = mile + 1.0;
  const window = course.filter(p => p.mile >= mile && p.mile <= end);

  if (window.length < 2) return {gain: 0, loss: 0};

  let gain = 0;
  let loss = 0;

  for (let i = 1; i < window.length; i++) {
    const diff = window[i].elevation_smooth_ft - window[i - 1].elevation_smooth_ft;
    if (diff > 0) gain += diff;
    if (diff < 0) loss += Math.abs(diff);
  }

  return {gain, loss};
}

function getCue(mile, grade) {
  if (mile < 5) {
    return {
      title: "Early control",
      text: "Settle in. The goal is to arrive at mile 5 feeling almost too relaxed."
    };
  }

  if (mile >= 20.5 && mile <= 21.7) {
    return {
      title: "Lemon Drop Hill area",
      text: "Shorten stride, keep cadence, and hold effort rather than chasing pace."
    };
  }

  if (mile >= 24) {
    return {
      title: "Finish setup",
      text: "Use the crowd, but keep form together. Start pressing only if you still feel controlled."
    };
  }

  if (grade > 1.5) {
    return {
      title: "Climbing cue",
      text: "Hold effort and avoid forcing goal pace uphill."
    };
  }

  if (grade < -1.5) {
    return {
      title: "Downhill cue",
      text: "Stay smooth and avoid overstriding."
    };
  }

  return {
    title: "Rhythm cue",
    text: "Run relaxed and steady. Keep fueling and hydration on schedule."
  };
}

function markerHtml(m, mile) {
  const type = (m.marker_type || "marker").replaceAll("_", " ");
  const items = m.items ? "<div>" + m.items + "</div>" : "";
  const notes = m.notes ? "<div>" + m.notes + "</div>" : "";
  const distanceAway = Math.max(0, m.mile - mile);
  const awayText = distanceAway > 0.05
    ? "<div><b>" + distanceAway.toFixed(1) + " mi ahead</b></div>"
    : "<div><b>Now / very close</b></div>";

  return `
    <div class="marker">
      <b>Mile ${Number(m.mile).toFixed(1)}: ${m.name}</b>
      <div><i>${type}</i></div>
      ${awayText}
      ${items}
      ${notes}
    </div>
  `;
}

function updateMarkers(mile) {
  const previewWindow = 1.0;

  const upcoming = markers
    .filter(m => m.mile >= mile - 0.02 && m.mile <= mile + previewWindow)
    .slice(0, 5);

  const nextMarker = markers.find(m => m.mile >= mile - 0.02);
  const el = document.getElementById("markerList");

  if (markers.length === 0) {
    el.innerHTML = "No course markers are loaded.";
    return;
  }

  if (upcoming.length > 0) {
    el.innerHTML = upcoming.map(m => markerHtml(m, mile)).join("");
    return;
  }

  if (nextMarker) {
    el.innerHTML = "<div class='small'>No marker in the next mile. Next marker:</div>" + markerHtml(nextMarker, mile);
    return;
  }

  el.innerHTML = "No more course markers ahead.";
}

function markerLabel(m) {
  const t = String(m.marker_type || "").toLowerCase();
  const name = String(m.name || "").toLowerCase();
  const items = String(m.items || "").toLowerCase();

  if (t.includes("aid") || name.includes("aid station") || items.includes("water")) return "💧";
  if (t.includes("food") || items.includes("fruit")) return "🍌";
  if (items.includes("pure fuel") || items.includes("maple")) return "🍁";
  if (t.includes("hill") || name.includes("hill")) return "⛰️";
  if (t.includes("start")) return "▶";
  if (t.includes("finish")) return "🏁";
  return "●";
}

function markerColor(m) {
  const t = String(m.marker_type || "").toLowerCase();
  const name = String(m.name || "").toLowerCase();
  const items = String(m.items || "").toLowerCase();

  if (t.includes("aid") || name.includes("aid station") || items.includes("water")) return "#38bdf8";
  if (t.includes("food") || items.includes("fruit")) return "#22c55e";
  if (items.includes("pure fuel") || items.includes("maple")) return "#f59e0b";
  if (t.includes("hill") || name.includes("hill")) return "#ef4444";
  if (t.includes("start") || t.includes("finish")) return "#f97316";
  return "#e5e7eb";
}

function elevationAtMile(mile) {
  if (mile <= course[0].mile) return course[0].elevation_smooth_ft;
  if (mile >= course[course.length - 1].mile) return course[course.length - 1].elevation_smooth_ft;

  for (let i = 0; i < course.length - 1; i++) {
    const a = course[i];
    const b = course[i + 1];
    if (mile >= a.mile && mile <= b.mile) {
      const ratio = (mile - a.mile) / (b.mile - a.mile);
      return a.elevation_smooth_ft + ratio * (b.elevation_smooth_ft - a.elevation_smooth_ft);
    }
  }
  return course[0].elevation_smooth_ft;
}

function buildElevationMarkerTrace() {
  const usable = markers.filter(m => Number.isFinite(m.mile));
  return {
    x: usable.map(m => m.mile),
    y: usable.map(m => elevationAtMile(m.mile)),
    mode: "markers+text",
    name: "Course markers",
    text: usable.map(m => markerLabel(m)),
    textposition: "top center",
    textfont: { size: 13 },
    marker: {
      size: 10,
      color: usable.map(m => markerColor(m)),
      line: { color: "#0b1020", width: 1 }
    },
    customdata: usable.map(m => [
      m.name || "",
      m.marker_type || "",
      m.items || "",
      m.notes || ""
    ]),
    hovertemplate:
      "<b>Mile %{x:.1f}: %{customdata[0]}</b><br>" +
      "%{customdata[1]}<br>" +
      "%{customdata[2]}<br>" +
      "%{customdata[3]}" +
      "<extra></extra>"
  };
}

function initializePlots() {
  const baseLayout = {
    paper_bgcolor: "#111827",
    plot_bgcolor: "#111827",
    font: { color: "#f8fafc", size: 10 },
    margin: { l: 44, r: 12, t: 5, b: 32 },
    xaxis: {
      title: "Mile",
      gridcolor: "#334155",
      zerolinecolor: "#334155"
    },
    showlegend: false
  };

  const elevationMarkerTrace = buildElevationMarkerTrace();

  Plotly.newPlot("elevationPlot", [
    {
      x: elevationX,
      y: elevationY,
      mode: "lines",
      name: "Elevation",
      line: { width: 2, color: "#60a5fa" }
    },
    elevationMarkerTrace,
    {
      x: [0],
      y: [elevationY[0]],
      mode: "markers",
      name: "Current position",
      marker: { size: 14, color: "#f97316" }
    }
  ], {
    ...baseLayout,
    yaxis: {
      title: "Elevation, ft",
      gridcolor: "#334155",
      zerolinecolor: "#334155"
    }
  }, {responsive: true});

  Plotly.newPlot("gradePlot", [
    {
      x: elevationX,
      y: gradeY,
      mode: "lines",
      name: "Grade",
      line: { width: 2, color: "#a78bfa" }
    },
    {
      x: [0],
      y: [gradeY[0]],
      mode: "markers",
      name: "Current position",
      marker: { size: 14, color: "#f97316" }
    }
  ], {
    ...baseLayout,
    yaxis: {
      title: "Grade, %",
      gridcolor: "#334155",
      zerolinecolor: "#334155"
    },
    shapes: [{
      type: "line",
      x0: 0,
      x1: 26.2,
      y0: 0,
      y1: 0,
      line: { color: "#94a3b8", width: 1, dash: "dash" }
    }]
  }, {responsive: true});
}

function initializeMap() {
  map = L.map("map", { zoomControl: false }).setView(latLngs[0], 12);

  L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    maxZoom: 19,
    attribution: "&copy; OpenStreetMap"
  }).addTo(map);

  courseLine = L.polyline(latLngs, {color: "#3b82f6", weight: 4}).addTo(map);

  markers.forEach(m => {
    const p = nearestCoursePoint(m.mile);
    const label = markerLabel(m);
    const popup = `<b>Mile ${Number(m.mile).toFixed(1)}: ${m.name}</b><br>${m.items || ""}<br>${m.notes || ""}`;
    L.circleMarker([p.lat, p.lon], {
      radius: 5,
      color: markerColor(m),
      fillColor: markerColor(m),
      fillOpacity: 0.85,
      weight: 1
    }).bindPopup(label + " " + popup).addTo(map);
  });

  courseMarker = L.circleMarker(latLngs[0], {
    radius: 8,
    color: "#f97316",
    fillColor: "#f97316",
    fillOpacity: 1
  }).addTo(map);

  map.fitBounds(courseLine.getBounds(), {padding: [8, 8]});
}

function isPlayerReady() {
  return player && typeof player.getCurrentTime === "function";
}

function normalizeVideoSecond(seconds) {
  const parsed = Number(seconds);
  if (Number.isFinite(parsed)) return parsed;

  const fallback = Number(sync[0] && sync[0].video_second);
  return Number.isFinite(fallback) ? fallback : 0;
}

function renderDisplayAtTime(seconds, options = {}) {
  const panMap = options.panMap !== false;
  const t = normalizeVideoSecond(seconds);
  const mile = interpolateMileFromVideo(t);
  const p = nearestCoursePoint(mile);
  const next = summarizeNextMile(mile);
  const cue = getCue(mile, p.grade_smooth_percent);

  document.getElementById("timeMetric").innerText = formatSeconds(t);
  document.getElementById("mileMetric").innerText = mile.toFixed(2);
  document.getElementById("elevMetric").innerText = Math.round(p.elevation_smooth_ft) + " ft";
  document.getElementById("gradeMetric").innerText = p.grade_smooth_percent.toFixed(1) + "%";
  document.getElementById("gainMetric").innerText = Math.round(next.gain) + " ft";
  document.getElementById("lossMetric").innerText = Math.round(next.loss) + " ft";
  document.getElementById("cueTitle").innerText = cue.title;
  document.getElementById("cueText").innerText = cue.text;
  document.getElementById("statusText").innerText = "Mile " + mile.toFixed(2) + " • " + formatSeconds(t);

  const elevationPlot = document.getElementById("elevationPlot");
  if (window.Plotly && elevationPlot && elevationPlot.data && elevationPlot.data.length >= 3) {
    try {
      Plotly.restyle("elevationPlot", {
        x: [[mile]],
        y: [[p.elevation_smooth_ft]]
      }, [2]);
    } catch (error) {
      console.warn("Elevation plot update skipped", error);
    }
  }

  const gradePlot = document.getElementById("gradePlot");
  if (window.Plotly && gradePlot && gradePlot.data && gradePlot.data.length >= 2) {
    try {
      Plotly.restyle("gradePlot", {
        x: [[mile]],
        y: [[p.grade_smooth_percent]]
      }, [1]);
    } catch (error) {
      console.warn("Grade plot update skipped", error);
    }
  }

  if (courseMarker && map) {
    const latlng = [p.lat, p.lon];
    courseMarker.setLatLng(latlng);
    if (panMap) {
      map.panTo(latlng, {animate: true, duration: 0.25});
    }
  }

  updateMarkers(mile);
}

function updateLiveDisplay() {
  if (!isPlayerReady()) return;

  const current = Number(player.getCurrentTime());
  if (!Number.isFinite(current)) return;

  renderDisplayAtTime(current, {panMap: true});
}

function createYouTubePlayer() {
  if (player) return;

  if (!window.YT || typeof YT.Player !== "function") {
    return;
  }

  player = new YT.Player("player", {
    videoId: "$youtube_id",
    playerVars: {
      rel: 0,
      modestbranding: 1,
      playsinline: 1
    },
    events: {
      onReady: function(event) {
        initializePlots();
        initializeMap();
        updateLiveDisplay();

        if (liveUpdateTimer) {
          clearInterval(liveUpdateTimer);
        }
        liveUpdateTimer = setInterval(updateLiveDisplay, 500);
      },
      onError: function(event) {
        document.getElementById("statusText").innerText = "Video player failed to load.";
      }
    }
  });
}

window.onYouTubeIframeAPIReady = createYouTubePlayer;

function loadYouTubeIframeApi() {
  if (window.YT && typeof YT.Player === "function") {
    createYouTubePlayer();
    return;
  }

  if (document.getElementById("youtube-iframe-api")) return;

  const tag = document.createElement("script");
  tag.id = "youtube-iframe-api";
  tag.src = "https://www.youtube.com/iframe_api";
  tag.async = true;
  tag.onerror = function() {
    document.getElementById("statusText").innerText = "Could not load YouTube API.";
  };
  document.head.appendChild(tag);

  const retryTimer = setInterval(function() {
    if (player) {
      clearInterval(retryTimer);
      return;
    }

    if (window.YT && typeof YT.Player === "function") {
      createYouTubePlayer();
      clearInterval(retryTimer);
    }
  }, 250);

  setTimeout(function() {
    clearInterval(retryTimer);
  }, 10000);
}

loadYouTubeIframeApi();
</script>
</body>
</html>
""")

html = html_template.safe_substitute(
    course_json=course_json,
    sync_json=sync_json,
    markers_json=markers_json,
    youtube_id=YOUTUBE_ID,
)

# Embed the full HTML dashboard in an iframe.
st.iframe(html, height=900)
