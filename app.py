from pathlib import Path
import json
import numpy as np
import pandas as pd
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
    df = df.sort_values("video_second").reset_index(drop=True)

    return df




# ============================================================
# Grade smoothing
# ============================================================

def add_windowed_grade(course: pd.DataFrame, window_miles: float = 0.10) -> pd.DataFrame:
    """
    Recalculate grade using a distance window instead of point-to-point GPS changes.

    Why:
    GPX elevation has small jumps/noise. If grade is calculated over very short
    distances, a tiny elevation artifact can look like a 10-20% grade spike.
    A rolling distance window gives a more race-useful course grade estimate.

    window_miles:
    0.10 means grade is estimated over roughly 0.10 miles around each point.
    Try 0.15 or 0.20 for even smoother output.
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

    # Final light rolling average to remove sawtooth effects.
    df["grade_course_percent"] = (
        pd.Series(grades)
        .rolling(window=21, center=True, min_periods=1)
        .mean()
        .clip(lower=-8.0, upper=8.0)
        .astype("float64")
    )

    return df


# ============================================================
# Prepare browser-side data
# ============================================================

st.title("Grandma's Marathon Course Visualizer")
st.caption("Live-sync version: video-linked course position with smoothed distance-window grade estimates.")

try:
    course_path = resolve_data_file("grandmas_course_points.csv")
    markers_path = resolve_data_file("grandmas_course_markers.csv")
    sync_path = resolve_data_file("video_mile_sync_template.csv")

    course = load_course(str(course_path))
    # Recalculate grade with a distance-window method to avoid GPX noise spikes.
    # Increase window_miles to 0.15 or 0.20 if you want an even smoother profile.
    course = add_windowed_grade(course, window_miles=0.10)

    markers = load_markers(str(markers_path))
    sync = load_video_sync(str(sync_path))

except Exception as e:
    st.error(f"Could not load app data: {e}")
    st.stop()


with st.sidebar:
    st.header("Data check")
    st.write(f"Course points: **{len(course):,}**")
    st.write(f"Course distance: **{course['mile'].max():.2f} mi**")
    st.write(f"Course markers: **{len(markers)}**")
    st.write(f"Video sync rows filled: **{len(sync)}**")
    st.write("Grade smoothing: **0.10-mile window, capped at ±8%**")

    st.caption("This version uses browser-side JavaScript, so the animation does not require Streamlit reruns.")

    with st.expander("Resolved data files"):
        st.write(str(course_path))
        st.write(str(markers_path))
        st.write(str(sync_path))


if len(sync) < 2:
    st.warning(
        "The live-sync player needs at least two rows in video_mile_sync_template.csv "
        "with video_second/time_mmss and mile."
    )
    st.stop()


# Reduce browser payload but keep enough resolution for maps/charts.
# 5456 points is OK, but this keeps the embedded HTML lighter on Streamlit Cloud.
course_for_browser = course.copy()
if len(course_for_browser) > 2500:
    step = int(np.ceil(len(course_for_browser) / 2500))
    course_for_browser = course_for_browser.iloc[::step].copy()
    # Ensure final point is included.
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

# Fallback: if the CSV has no usable markers, create the official aid/fuel markers.
# This keeps the live browser panel useful even if the marker CSV was accidentally overwritten.
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
# Browser-side live sync component
# ============================================================

html = f"""
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8" />
  <script src="https://www.youtube.com/iframe_api"></script>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <link
    rel="stylesheet"
    href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"
  />
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>

  <style>
    body {{
      margin: 0;
      font-family: Arial, Helvetica, sans-serif;
      background: #0e1117;
      color: #fafafa;
    }}

    .container {{
      display: grid;
      grid-template-columns: 1.2fr 0.8fr;
      gap: 22px;
      padding: 8px 8px 18px 8px;
    }}

    .panel {{
      background: #111827;
      border: 1px solid #263244;
      border-radius: 14px;
      padding: 14px;
    }}

    .video-wrap {{
      position: relative;
      padding-bottom: 56.25%;
      height: 0;
      overflow: hidden;
      border-radius: 12px;
      background: #000;
    }}

    #player {{
      position: absolute;
      top: 0;
      left: 0;
      width: 100% !important;
      height: 100% !important;
    }}

    .metrics {{
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 12px;
      margin-bottom: 14px;
    }}

    .metric {{
      background: #0b1220;
      border: 1px solid #253047;
      border-radius: 12px;
      padding: 12px;
    }}

    .metric-label {{
      color: #a7b0c0;
      font-size: 13px;
      margin-bottom: 4px;
    }}

    .metric-value {{
      color: #ffffff;
      font-size: 30px;
      font-weight: 700;
    }}

    .subhead {{
      font-size: 22px;
      font-weight: 700;
      margin: 10px 0 8px 0;
    }}

    .cue {{
      background: #102a43;
      border: 1px solid #1f4e79;
      border-radius: 12px;
      padding: 12px;
      line-height: 1.45;
      margin-bottom: 12px;
    }}

    .marker {{
      border-left: 4px solid #3b82f6;
      padding: 8px 10px;
      margin: 8px 0;
      background: #0b1220;
      border-radius: 8px;
      line-height: 1.4;
    }}

    .controls {{
      margin-top: 12px;
      background: #0b1220;
      border: 1px solid #253047;
      border-radius: 12px;
      padding: 12px;
    }}

    input[type=range] {{
      width: 100%;
    }}

    button {{
      background: #2563eb;
      color: white;
      border: none;
      border-radius: 8px;
      padding: 8px 12px;
      cursor: pointer;
      margin-right: 8px;
    }}

    button:hover {{
      background: #1d4ed8;
    }}

    .charts {{
      display: grid;
      grid-template-columns: 1fr;
      gap: 18px;
      padding: 0 8px 8px 8px;
    }}

    #elevationPlot, #gradePlot {{
      height: 330px;
    }}

    #map {{
      height: 460px;
      border-radius: 12px;
      overflow: hidden;
    }}

    .small {{
      font-size: 13px;
      color: #a7b0c0;
      line-height: 1.4;
    }}

    @media (max-width: 900px) {{
      .container {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>

<body>
  <div class="container">
    <div class="panel">
      <div class="video-wrap">
        <div id="player"></div>
      </div>

      <div class="controls">
        <div class="small">
          Live mode: as the YouTube video plays, the course mile, dot, elevation profile, grade profile, and map update in the browser.
        </div>
        <br />
        <label for="seekSlider"><b>Manual video seek</b>: <span id="seekLabel">0:00</span></label>
        <input id="seekSlider" type="range" min="0" max="369" value="1" step="1" />
        <button onclick="seekToSlider()">Jump video</button>
        <button onclick="player.playVideo()">Play</button>
        <button onclick="player.pauseVideo()">Pause</button>
      </div>
    </div>

    <div class="panel">
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
      </div>

      <div class="metrics">
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
      <div id="markerDebug" class="small"></div>
      <div id="markerList" class="small">Loading markers...</div>
    </div>
  </div>

  <div class="charts">
    <div class="panel">
      <div class="subhead">Elevation Profile</div>
      <div id="elevationPlot"></div>
    </div>

    <div class="panel">
      <div class="subhead">Smoothed Grade Profile</div>
      <div id="gradePlot"></div>
    </div>

    <div class="panel">
      <div class="subhead">Course Map</div>
      <div id="map"></div>
    </div>
  </div>

<script>
const course = {course_json}.map(d => ({
  ...d,
  mile: Number(d.mile),
  lat: Number(d.lat),
  lon: Number(d.lon),
  elevation_smooth_ft: Number(d.elevation_smooth_ft),
  grade_smooth_percent: Number(d.grade_smooth_percent)
}));

const sync = {sync_json}.map(d => ({
  ...d,
  video_second: Number(d.video_second),
  mile: Number(d.mile)
}));

const markers = {markers_json}
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

const elevationX = course.map(d => d.mile);
const elevationY = course.map(d => d.elevation_smooth_ft);
const gradeY = course.map(d => d.grade_smooth_percent);
const latLngs = course.map(d => [d.lat, d.lon]);

document.getElementById("seekSlider").max = Math.round(sync[sync.length - 1].video_second);
document.getElementById("seekSlider").value = Math.round(sync[0].video_second);

function formatSeconds(seconds) {{
  seconds = Math.round(seconds);
  const m = Math.floor(seconds / 60);
  const s = seconds % 60;
  return `${{m}}:${{s.toString().padStart(2, "0")}}`;
}}

function interpolateMileFromVideo(t) {{
  if (t <= sync[0].video_second) return sync[0].mile;
  if (t >= sync[sync.length - 1].video_second) return sync[sync.length - 1].mile;

  for (let i = 0; i < sync.length - 1; i++) {{
    const a = sync[i];
    const b = sync[i + 1];

    if (t >= a.video_second && t <= b.video_second) {{
      const ratio = (t - a.video_second) / (b.video_second - a.video_second);
      return a.mile + ratio * (b.mile - a.mile);
    }}
  }}

  return sync[0].mile;
}}

function nearestCoursePoint(mile) {{
  let best = course[0];
  let bestDiff = Math.abs(course[0].mile - mile);

  for (const p of course) {{
    const diff = Math.abs(p.mile - mile);
    if (diff < bestDiff) {{
      best = p;
      bestDiff = diff;
    }}
  }}
  return best;
}}

function summarizeNextMile(mile) {{
  const end = mile + 1.0;
  const window = course.filter(p => p.mile >= mile && p.mile <= end);

  if (window.length < 2) return {{gain: 0, loss: 0}};

  let gain = 0;
  let loss = 0;

  for (let i = 1; i < window.length; i++) {{
    const diff = window[i].elevation_smooth_ft - window[i - 1].elevation_smooth_ft;
    if (diff > 0) gain += diff;
    if (diff < 0) loss += Math.abs(diff);
  }}

  return {{gain, loss}};
}}

function getCue(mile, grade) {{
  if (mile < 5) {{
    return {{
      title: "Early control",
      text: "Settle in. The goal is to arrive at mile 5 feeling almost too relaxed."
    }};
  }}

  if (mile >= 20.5 && mile <= 21.7) {{
    return {{
      title: "Lemon Drop Hill area",
      text: "Shorten stride, keep cadence, and hold effort rather than chasing pace."
    }};
  }}

  if (mile >= 24) {{
    return {{
      title: "Finish setup",
      text: "Use the crowd, but keep form together. Start pressing only if you still feel controlled."
    }};
  }}

  if (grade > 1.5) {{
    return {{
      title: "Climbing cue",
      text: "Hold effort and avoid forcing goal pace uphill."
    }};
  }}

  if (grade < -1.5) {{
    return {{
      title: "Downhill cue",
      text: "Stay smooth and avoid overstriding."
    }};
  }}

  return {{
    title: "Rhythm cue",
    text: "Run relaxed and steady. Keep fueling and hydration on schedule."
  }};
}}

function updateMarkers(mile) {{
  const upcoming = markers
    .filter(m => m.mile >= mile && m.mile <= mile + 1.0)
    .slice(0, 8);

  const el = document.getElementById("markerList");

  if (upcoming.length === 0) {{
    el.innerHTML = "No aid, fuel, food, or course marker in the next mile.";
    return;
  }}

  el.innerHTML = upcoming.map(m => {{
    const type = (m.marker_type || "marker").replaceAll("_", " ");
    const items = m.items ? `<div>${{m.items}}</div>` : "";
    const notes = m.notes ? `<div>${{m.notes}}</div>` : "";
    return `
      <div class="marker">
        <b>Mile ${{Number(m.mile).toFixed(1)}}: ${{m.name}}</b>
        <div><i>${{type}}</i></div>
        ${{items}}
        ${{notes}}
      </div>
    `;
  }}).join("");
}}

function initializePlots() {{
  const plotLayoutBase = {{
    paper_bgcolor: "#111827",
    plot_bgcolor: "#111827",
    font: {{ color: "#fafafa" }},
    margin: {{ l: 50, r: 20, t: 15, b: 45 }},
    xaxis: {{
      title: "Mile",
      gridcolor: "#334155",
      zerolinecolor: "#334155"
    }},
    showlegend: true,
    legend: {{ orientation: "h" }}
  }};

  Plotly.newPlot("elevationPlot", [
    {{
      x: elevationX,
      y: elevationY,
      mode: "lines",
      name: "Elevation",
      line: {{ width: 2 }}
    }},
    {{
      x: [0],
      y: [elevationY[0]],
      mode: "markers",
      name: "Current position",
      marker: {{ size: 14 }}
    }}
  ], {{
    ...plotLayoutBase,
    yaxis: {{
      title: "Elevation, ft",
      gridcolor: "#334155",
      zerolinecolor: "#334155"
    }}
  }}, {{responsive: true}});

  Plotly.newPlot("gradePlot", [
    {{
      x: elevationX,
      y: gradeY,
      mode: "lines",
      name: "Grade",
      line: {{ width: 2 }}
    }},
    {{
      x: [0],
      y: [gradeY[0]],
      mode: "markers",
      name: "Current position",
      marker: {{ size: 14 }}
    }}
  ], {{
    ...plotLayoutBase,
    yaxis: {{
      title: "Grade, %",
      gridcolor: "#334155",
      zerolinecolor: "#334155"
    }},
    shapes: [{{
      type: "line",
      x0: 0,
      x1: 26.2,
      y0: 0,
      y1: 0,
      line: {{ color: "#94a3b8", width: 1, dash: "dash" }}
    }}]
  }}, {{responsive: true}});
}}

function initializeMap() {{
  map = L.map("map").setView(latLngs[0], 12);

  L.tileLayer("https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png", {{
    maxZoom: 19,
    attribution: "&copy; OpenStreetMap contributors"
  }}).addTo(map);

  courseLine = L.polyline(latLngs, {{color: "#3b82f6", weight: 4}}).addTo(map);
  courseMarker = L.circleMarker(latLngs[0], {{
    radius: 9,
    color: "#f97316",
    fillColor: "#f97316",
    fillOpacity: 1
  }}).addTo(map);

  map.fitBounds(courseLine.getBounds(), {{padding: [10, 10]}});
}}

function updateLiveDisplay() {{
  if (!player || typeof player.getCurrentTime !== "function") return;

  const t = player.getCurrentTime();
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

  document.getElementById("seekSlider").value = Math.round(t);
  document.getElementById("seekLabel").innerText = formatSeconds(t);

  Plotly.restyle("elevationPlot", {{
    x: [[mile]],
    y: [[p.elevation_smooth_ft]]
  }}, [1]);

  Plotly.restyle("gradePlot", {{
    x: [[mile]],
    y: [[p.grade_smooth_percent]]
  }}, [1]);

  if (courseMarker) {{
    const latlng = [p.lat, p.lon];
    courseMarker.setLatLng(latlng);

    // Pan gently every update without changing zoom.
    map.panTo(latlng, {{animate: true, duration: 0.25}});
  }}

  updateMarkers(mile);
}}

function seekToSlider() {{
  const t = Number(document.getElementById("seekSlider").value);
  player.seekTo(t, true);
  updateLiveDisplay();
}}

document.getElementById("seekSlider").addEventListener("input", function() {{
  document.getElementById("seekLabel").innerText = formatSeconds(Number(this.value));
}});

function onYouTubeIframeAPIReady() {{
  player = new YT.Player("player", {{
    videoId: "{YOUTUBE_ID}",
    playerVars: {{
      rel: 0,
      modestbranding: 1,
      playsinline: 1
    }},
    events: {{
      onReady: function(event) {{
        document.getElementById("markerDebug").innerText = `${markers.length} course markers loaded.`;
        initializePlots();
        initializeMap();
        updateLiveDisplay();
        setInterval(updateLiveDisplay, 500);
      }}
    }}
  }});
}}
</script>
</body>
</html>
"""

components.html(html, height=1500, scrolling=True)

st.markdown(
    """
    **Note:** This live-sync section is driven in the browser with JavaScript. That is why the dots and map can move while the video plays.
    The regular Streamlit/Python charts cannot update continuously from YouTube playback without this kind of browser-side component.
    """
)
