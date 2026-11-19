"""
app.py
Streamlit dashboard for the Traffic Sign Recognition and Road Safety Monitor.

Run from the project root:
    streamlit run app.py

Pipeline (all in src\\video_processor.py): YOLOv8 COCO detection + ByteTrack tracking
-> traffic sign detection + GTSRB classification -> temporal rule-based risk analysis
-> alerts -> annotated video.

The risk values are a PROTOTYPE rule-based estimate in pixel units. They are not real
distances, speeds or collision probabilities.
"""
import re
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import streamlit as st

import config

UPLOADS_DIR = config.OUTPUTS_DIR / "uploads"

st.set_page_config(page_title="Road Safety Monitor", page_icon="🚦", layout="wide")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading models (first run only)...")
def get_processor(use_signs: bool):
    """Load the pipeline once and reuse it. Imported here so import errors show in the page."""
    from src.video_processor import VideoProcessor
    return VideoProcessor(use_signs=use_signs)


def safe_name(name: str) -> str:
    """Make an uploaded file name safe for use as a file stem."""
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(name).stem).strip("._")
    return stem or "video"


def make_browser_playable(src: Path):
    """Re-encode the annotated video to H.264 so browsers can play it.

    OpenCV writes the 'mp4v' codec, which many browsers cannot play inside the page.
    Needs ffmpeg (system ffmpeg, or the one bundled with 'pip install imageio-ffmpeg').
    Returns the new path, or None if conversion is not possible.
    """
    exe = shutil.which("ffmpeg")
    if exe is None:
        try:
            import imageio_ffmpeg
            exe = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:  # noqa: BLE001
            return None
    out = src.with_name(src.stem + "_h264.mp4")
    cmd = [exe, "-y", "-i", str(src), "-an", "-vcodec", "libx264", "-pix_fmt", "yuv420p",
           "-preset", "veryfast", "-crf", "23", "-movflags", "+faststart", str(out)]
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=1800)
    except Exception:  # noqa: BLE001
        return None
    if result.returncode != 0 or not out.is_file() or out.stat().st_size == 0:
        return None
    return out


def list_test_videos():
    if not config.TEST_VIDEOS_DIR.exists():
        return []
    return sorted(p for p in config.TEST_VIDEOS_DIR.iterdir() if p.suffix.lower() in config.ALLOWED_VIDEO_EXT)


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("🚦 Traffic Sign Recognition and Road Safety Monitor")
st.caption("Object detection + tracking + traffic sign recognition + temporal risk analysis. "
           "Risk values are a prototype, rule-based estimate in pixel units "
           "(no real distances, speeds or collision probabilities).")

# ---------------------------------------------------------------------------
# Sidebar settings
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Settings")
    sign_weights_ok = config.SIGN_DETECTOR_WEIGHTS.is_file()
    use_signs = st.checkbox("Enable traffic sign detection", value=sign_weights_ok, disabled=not sign_weights_ok)
    if not sign_weights_ok:
        st.caption(f"Sign detector weights not found ({config.SIGN_DETECTOR_WEIGHTS.name}); sign detection is off.")
    sign_every = st.slider("Run sign detector every N frames", 1, 10, 2,
                           help="Higher = faster, but signs are checked less often.")
    limit_frames = st.checkbox("Process only the first N frames (quick test)", value=False)
    max_frames = st.number_input("N frames", min_value=10, max_value=100000, value=150, step=10,
                                 disabled=not limit_frames)
    st.divider()
    with st.expander("About this prototype"):
        st.markdown(
            "- **Detection/tracking:** YOLOv8 (COCO) + ByteTrack for pedestrians, bicycles, cars, "
            "motorcycles, buses and trucks.\n"
            "- **Signs:** YOLO sign detector (GTSDB) + CNN classifier (GTSRB).\n"
            "- **Risk:** rule-based scoring of proximity, traffic density and persistence over "
            "consecutive frames, with smoothing and an alert cooldown.\n"
            "- **Limits:** no camera calibration, so no real-world distance, speed or collision "
            "probability is claimed."
        )

# ---------------------------------------------------------------------------
# Video source
# ---------------------------------------------------------------------------
source_mode = st.radio("Video source", ["Upload a video", "Use a video from datasets/test_videos"], horizontal=True)

video_path = None
if source_mode == "Upload a video":
    allowed = sorted(e.lstrip(".") for e in config.ALLOWED_VIDEO_EXT)
    uploaded = st.file_uploader(f"Upload a road video ({', '.join(allowed)}; max {config.MAX_UPLOAD_MB} MB)", type=allowed)
    if uploaded is not None:
        if uploaded.size > config.MAX_UPLOAD_MB * 1024 * 1024:
            st.error(f"File is {uploaded.size / 1024 / 1024:.0f} MB; the limit is {config.MAX_UPLOAD_MB} MB.")
        else:
            UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
            video_path = UPLOADS_DIR / f"{safe_name(uploaded.name)}{Path(uploaded.name).suffix.lower()}"
            video_path.write_bytes(uploaded.getbuffer())
else:
    vids = list_test_videos()
    if not vids:
        st.info(f"No videos found in {config.TEST_VIDEOS_DIR}. Upload one instead.")
    else:
        choice = st.selectbox("Test video", [v.name for v in vids])
        video_path = config.TEST_VIDEOS_DIR / choice

if video_path is not None:
    with st.expander("Preview original video"):
        st.video(str(video_path))

run = st.button("▶ Analyze video", type="primary", disabled=video_path is None)

# ---------------------------------------------------------------------------
# Processing
# ---------------------------------------------------------------------------
if run and video_path is not None:
    st.session_state.pop("result", None)
    try:
       processor = get_processor(bool(use_signs))
       processor.sign_every = max(1, int(sign_every)) 
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not load the models: {exc}")
        st.stop()
    if use_signs and not processor.signs_enabled:
        st.warning("Sign detection could not start (see the terminal for details). Continuing without it.")

    bar = st.progress(0.0, text="Starting...")

    def on_progress(done: int, total: int) -> None:
        frac = min(done / total, 1.0) if total else 0.0
        bar.progress(frac, text=f"Processed {done} / {total or '?'} frames")

    try:
        summary = processor.process(video_path, max_frames=int(max_frames) if limit_frames else None,
                                    progress_cb=on_progress)
    except Exception as exc:  # noqa: BLE001
        bar.empty()
        st.error(f"Processing failed: {exc}")
        st.stop()
    bar.empty()

    with st.spinner("Preparing a browser-playable copy of the annotated video..."):
        playable = make_browser_playable(Path(summary["annotated_video"]))
    st.session_state["result"] = {"summary": summary, "playable": str(playable) if playable else None}

# ---------------------------------------------------------------------------
# Results (kept in session_state so download buttons do not wipe them)
# ---------------------------------------------------------------------------
res = st.session_state.get("result")
if res:
    s = res["summary"]
    annotated = Path(s["annotated_video"])
    playable = Path(res["playable"]) if res["playable"] else None
    n_frames = max(s["frames_processed"], 1)

    st.subheader("Results")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Frames processed", s["frames_processed"])
    c2.metric("Video length", f"{s['duration_s']} s")
    c3.metric("Processing speed", f"{s['processing_fps']} fps")
    c4.metric("Unique tracks", s["unique_tracks"])
    c5.metric("Peak risk score", s["peak_score"])

    left, right = st.columns([3, 2])
    with left:
        st.markdown("**Annotated video**")
        if playable is not None:
            st.video(str(playable))
        else:
            st.warning("A browser-playable copy could not be made (ffmpeg not found). "
                       "Download the video below and open it in VLC, or run: pip install imageio-ffmpeg")
        file_to_send = playable or annotated
        st.download_button("⬇ Download annotated video", data=file_to_send.read_bytes(),
                           file_name=f"{Path(s['video']).stem}_annotated.mp4", mime="video/mp4")

    with right:
        st.markdown("**Frames per risk level**")
        lf = s["level_frames"]
        st.bar_chart(pd.DataFrame({"frames": [lf.get(k, 0) for k in config.RISK_LEVELS]}, index=list(config.RISK_LEVELS)))
        st.caption(" | ".join(f"{k}: {lf.get(k, 0)} ({100 * lf.get(k, 0) / n_frames:.0f}%)" for k in config.RISK_LEVELS))
        st.markdown("**Tracked objects by class**")
        if s["tracks_by_class"]:
            st.dataframe(pd.DataFrame(list(s["tracks_by_class"].items()), columns=["class", "unique tracks"]))
        else:
            st.caption("No objects were tracked.")

    st.markdown("**Risk score over time**")
    try:
        tl = pd.read_csv(s["timeline_csv"])
        st.line_chart(tl.set_index("time_s")[["score", "raw_score"]])
        st.caption("score = smoothed score used for levels and alerts; raw_score = per-frame score before smoothing.")
    except Exception as exc:  # noqa: BLE001
        st.caption(f"Timeline chart unavailable: {exc}")

    st.markdown(f"**Risk alerts ({len(s['alerts'])})**")
    if s["alerts"]:
        st.dataframe(pd.DataFrame(s["alerts"]))
    else:
        st.success("No alerts were raised for this video.")

    if s["signs_enabled"]:
        st.markdown("**Traffic signs detected**")
        if s["sign_detections"]:
            st.dataframe(pd.DataFrame(list(s["sign_detections"].items()), columns=["sign", "frame-level detections"]))
            st.caption("Counts are confident detections per processed frame, not the number of unique physical signs.")
        else:
            st.caption("No confident sign detections.")
    else:
        st.caption("Sign detection was off for this run.")

    d1, d2 = st.columns(2)
    with d1:
        if Path(s["alerts_csv"]).is_file():
            st.download_button("⬇ Download alerts (CSV)", data=Path(s["alerts_csv"]).read_bytes(),
                               file_name=Path(s["alerts_csv"]).name, mime="text/csv")
    with d2:
        if Path(s["timeline_csv"]).is_file():
            st.download_button("⬇ Download risk timeline (CSV)", data=Path(s["timeline_csv"]).read_bytes(),
                               file_name=Path(s["timeline_csv"]).name, mime="text/csv")

    st.caption("Prototype, rule-based temporal risk estimation in pixel units. "
               "It does not measure real distances or speeds and does not predict collisions.")