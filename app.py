# app.py
# 🛩️ VTOL Precision Landing Simulator — Light Theme (Accessible)
# Streamlit app: eVTOL dataset • Scenario Presets • ArUco/AprilTag panel
# In-place Landing Playback (2D/3D) • Kalman filter • Metrics & Score
# Log Export (CSV/JSON/ZIP) • Auto-Tuner with Apply Best Settings
# Vectorized marker pixel model • RTK/Lidar toggles • Vision controls
# GPS-denied nav toggle • UAV-linked marker profiles (pattern changes by UAV)

import io
import json
import uuid
import math
import time
import zipfile
import datetime as dt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import streamlit as st

# Optional vision libs (graceful fallback if missing)
try:
    import cv2
    _ARUCO_OK = hasattr(cv2, "aruco")
except Exception:
    cv2 = None
    _ARUCO_OK = False

try:
    import pupil_apriltags as apriltag  # detector only (no generator)
    _APRILTAG_OK = True
except Exception:
    apriltag = None
    _APRILTAG_OK = False

APP_VERSION = "1.4.0"  # UAV-linked marker profiles + GPS-denied toggle

# ─────────────────────────────────────────────
# Page / Light Theme Styling
# ─────────────────────────────────────────────
st.set_page_config(page_title="🛩️ VTOL Precision Landing", page_icon="🛩️", layout="wide")

ACCENT = "#0B6E4F"     # teal
TEXT_DARK = "#0B1F2A"  # near-black

st.markdown(f"""
<style>
  .block-container {{ padding-top: 1.2rem; }}
  h1, h2, h3 {{ color: {ACCENT} !important; }}
  .stButton > button {{
    background: {ACCENT}; color: #ffffff; font-weight: 600; border: 0;
  }}
</style>
""", unsafe_allow_html=True)

# Matplotlib defaults for light background
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": TEXT_DARK,
    "axes.labelcolor": TEXT_DARK,
    "xtick.color": TEXT_DARK,
    "ytick.color": TEXT_DARK,
    "grid.color": "#B9C2CC"
})

st.title("🛩️ VTOL Precision Landing Simulator")
st.caption("RTK • Lidar • EKF-style fusion • ArUco/AprilTag assist • Kalman smoothing • 3D cone • Auto-Tuner • Run Log Export")

# ─────────────────────────────────────────────
# eVTOL Dataset (all VTOL-capable) + per-UAV vision marker profile
# ─────────────────────────────────────────────
uav_data = {
    # Hybrids / tailsitters
    "Quantum Systems Vector": {
        "type": "Hybrid Fixed-Wing eVTOL", "rtk": True, "lidar": True,
        "hover_draw_W": 220, "cruise_draw_W": 95,
        "vision_marker": {"family": "aruco4x4", "id": 23}
    },
    "Quantum Systems Trinity F90+": {
        "type": "Hybrid Fixed-Wing eVTOL (mapping)", "rtk": True, "lidar": False,
        "hover_draw_W": 180, "cruise_draw_W": 80,
        "vision_marker": {"family": "apriltag36h11", "id": 5}
    },
    "WingtraOne Gen II": {
        "type": "Tail-sitter eVTOL (mapping)", "rtk": True, "lidar": False,
        "hover_draw_W": 190, "cruise_draw_W": 70,
        "vision_marker": {"family": "aruco4x4", "id": 77}
    },
    "DeltaQuad Evo": {
        "type": "Hybrid Fixed-Wing eVTOL", "rtk": True, "lidar": True,
        "hover_draw_W": 260, "cruise_draw_W": 110,
        "vision_marker": {"family": "apriltag36h11", "id": 8}
    },
    "Censys Sentaero VTOL": {
        "type": "Hybrid Fixed-Wing eVTOL", "rtk": True, "lidar": True,
        "hover_draw_W": 240, "cruise_draw_W": 100,
        "vision_marker": {"family": "aruco4x4", "id": 31}
    },
    "Atmos Marlyn Cobalt": {
        "type": "Hybrid Fixed-Wing eVTOL", "rtk": True, "lidar": True,
        "hover_draw_W": 230, "cruise_draw_W": 90,
        "vision_marker": {"family": "apriltag36h11", "id": 12}
    },
    "ALTI Transition": {
        "type": "Hybrid Fixed-Wing eVTOL", "rtk": True, "lidar": True,
        "hover_draw_W": 300, "cruise_draw_W": 140,
        "vision_marker": {"family": "aruco4x4", "id": 54}
    },
    # Multirotor (still eVTOL)
    "Percepto Air Max": {
        "type": "Multirotor eVTOL (industrial)", "rtk": True, "lidar": True,
        "hover_draw_W": 220, "cruise_draw_W": 0,
        "vision_marker": {"family": "aruco4x4", "id": 99}
    },
    # Custom demo
    "Urban Hawk Tiltrotor (Custom)": {
        "type": "Hybrid Tiltrotor eVTOL", "rtk": True, "lidar": True,
        "hover_draw_W": 300, "cruise_draw_W": 120,
        "vision_marker": {"family": "apriltag36h11", "id": 42}
    },
}

# ─────────────────────────────────────────────
# Scenario Presets (dropdown)
# ─────────────────────────────────────────────
PRESETS = {
    "— None —": {},
    "Rooftop Urban": {
        "wind_gust": True, "occlusion_prob": 0.20, "illum": 0.65, "blur": 0.25,
        "beacon_gain": 0.45, "lock_thresh_px": 30, "lock_dwell_frames": 9
    },
    "Ship Deck": {
        "wind_gust": True, "occlusion_prob": 0.05, "illum": 0.85, "blur": 0.30,
        "beacon_gain": 0.50, "lock_thresh_px": 32, "lock_dwell_frames": 10
    },
    "Forest Clearing": {
        "wind_gust": False, "occlusion_prob": 0.35, "illum": 0.60, "blur": 0.15,
        "beacon_gain": 0.40, "lock_thresh_px": 26, "lock_dwell_frames": 8
    },
    "Desert Pad": {
        "wind_gust": True, "occlusion_prob": 0.05, "illum": 0.95, "blur": 0.20,
        "beacon_gain": 0.38, "lock_thresh_px": 28, "lock_dwell_frames": 7
    },
    "Warehouse Doorway": {
        "wind_gust": False, "occlusion_prob": 0.40, "illum": 0.50, "blur": 0.10,
        "beacon_gain": 0.48, "lock_thresh_px": 24, "lock_dwell_frames": 12
    }
}

def apply_preset(preset_name: str):
    cfg = PRESETS.get(preset_name, {})
    if not cfg:
        return
    for k, v in cfg.items():
        st.session_state[k] = v

# ─────────────────────────────────────────────
# Apply Best Settings from Auto-Tuner (session_state)
# ─────────────────────────────────────────────
APPLY_KEYS = [
    "beacon_gain", "lock_thresh_px", "lock_dwell_frames", "kf_q", "kf_r_base",
    "cam_hfov_deg", "cam_res_x", "marker_size_cm", "enable_vision", "rtk_fix", "use_lidar"
]
if st.session_state.get("pending_apply"):
    payload = st.session_state.get("apply_payload", {})
    for k in APPLY_KEYS:
        if k in payload:
            st.session_state[k] = payload[k]
    st.session_state["pending_apply"] = False  # clear after apply

# ─────────────────────────────────────────────
# Sidebar Controls (keys allow session_state updates)
# ─────────────────────────────────────────────
st.sidebar.header("Mission / Sensor Settings")

# UAV dropdown
uav = st.sidebar.selectbox("UAV Model", list(uav_data.keys()), key="uav_model")
specs = uav_data[uav]
uav_marker = specs.get("vision_marker", {"family": "aruco4x4", "id": 23})

# Scenario preset dropdown
st.sidebar.markdown("### Scenario Preset")
preset_choice = st.sidebar.selectbox("Preset", list(PRESETS.keys()), index=0, key="preset_choice")
if st.sidebar.button("Apply Preset ▶️"):
    apply_preset(preset_choice)
    st.success(f"Preset applied: {preset_choice}")

rtk_fix = st.sidebar.checkbox("RTK Fix Lock", value=True, key="rtk_fix")
use_lidar = st.sidebar.checkbox("Use Lidar Altitude Lock", value=specs["lidar"], key="use_lidar")

# Vision backend + link-to-UAV toggle
st.sidebar.markdown("### Vision Backend")
link_to_uav = st.sidebar.checkbox("Link marker to UAV profile", value=st.session_state.get("link_marker_to_uav", True), key="link_marker_to_uav")

# Set/refresh marker family + backend when linked and UAV changed
def _apply_uav_marker_profile():
    fam = uav_marker.get("family", "aruco4x4").lower()
    mid = int(uav_marker.get("id", 23))
    # set backend to match family
    if "apriltag" in fam:
        st.session_state["vision_backend"] = "AprilTag (pupil_apriltags)"
    else:
        st.session_state["vision_backend"] = "ArUco (OpenCV)"
    st.session_state["marker_id"] = mid
    st.session_state["last_uav_for_marker"] = uav

if link_to_uav and st.session_state.get("last_uav_for_marker") != uav:
    _apply_uav_marker_profile()

vision_backend = st.sidebar.selectbox("Backend", ["ArUco (OpenCV)", "AprilTag (pupil_apriltags)"],
                                      index=0 if st.session_state.get("vision_backend", "ArUco (OpenCV)").startswith("ArUco") else 1,
                                      key="vision_backend")
enable_vision = st.sidebar.checkbox("Enable Vision Assist", value=st.session_state.get("enable_vision", True), key="enable_vision")

# Marker / camera model (shared)
marker_id = st.sidebar.number_input("Marker ID (for ArUco/AprilTag)", min_value=0, max_value=999,
                                    value=int(st.session_state.get("marker_id", uav_marker.get("id", 23))), step=1, key="marker_id")
marker_size_cm = st.sidebar.slider("Marker Size (cm)", 10, 80, int(st.session_state.get("marker_size_cm", 40)), key="marker_size_cm")
cam_res_x = st.sidebar.selectbox("Camera Width (px)", [640, 960, 1280, 1920], index=2, key="cam_res_x")
cam_res_y = st.sidebar.selectbox("Camera Height (px)", [480, 720, 1080], index=2, key="cam_res_y")
cam_hfov_deg = st.sidebar.slider("Camera HFOV (deg)", 40.0, 110.0, float(st.session_state.get("cam_hfov_deg", 78.0)), 0.5, key="cam_hfov_deg")

lock_thresh_px = st.sidebar.slider("Vision Lock Threshold (min pixels)", 10, 120, int(st.session_state.get("lock_thresh_px", 28)), key="lock_thresh_px")
lock_dwell_frames = st.sidebar.slider("Lock Dwell (frames)", 1, 30, int(st.session_state.get("lock_dwell_frames", 8)), key="lock_dwell_frames")

illum = st.sidebar.slider("Illumination (0–1)", 0.1, 1.0, float(st.session_state.get("illum", 0.85)), 0.05, key="illum")
blur = st.sidebar.slider("Motion Blur (0–1)", 0.0, 1.0, float(st.session_state.get("blur", 0.2)), 0.05, key="blur")
occlusion_prob = st.sidebar.slider("Occlusion Probability", 0.0, 0.6, float(st.session_state.get("occlusion_prob", 0.1)), 0.05, key="occlusion_prob")

# Beacon correction (exposed so tuner & presets can apply)
beacon_gain = st.sidebar.slider("Beacon Correction Gain (locked)", 0.0, 0.8, float(st.session_state.get("beacon_gain", 0.35)), 0.01, key="beacon_gain")

# Kalman filter tuning
st.sidebar.markdown("### Kalman Filter (XY)")
kf_q = st.sidebar.slider("Process Noise q", 1e-5, 5e-2, float(st.session_state.get("kf_q", 5e-3)), format="%.5f", key="kf_q")
kf_r_base = st.sidebar.slider("Meas Noise (GNSS σ, m)", 0.02 if rtk_fix else 0.2, 2.0, float(st.session_state.get("kf_r_base", 0.03 if rtk_fix else 1.0)), 0.01, key="kf_r_base")

# Playback / environment
seed = st.sidebar.number_input("Random Seed", value=int(st.session_state.get("seed", 0)), step=1, key="seed")
steps = st.sidebar.slider("Playback Steps", 30, 500, int(st.session_state.get("steps", 160)), key="steps")
play_speed = st.sidebar.slider("Playback Speed (sec/frame)", 0.01, 0.20, float(st.session_state.get("play_speed", 0.05)), key="play_speed")
wind_gust = st.sidebar.checkbox("Inject Wind Gust (XY bias)", value=bool(st.session_state.get("wind_gust", False)), key="wind_gust")
gps_glitch = st.sidebar.checkbox("Inject GPS Glitch (spike)", value=bool(st.session_state.get("gps_glitch", False)), key="gps_glitch")

# NEW: GPS-denied toggle (no GNSS; rely on INS drift + vision/lidar)
gps_denied = st.sidebar.checkbox("GPS-denied / GNSS outage", value=bool(st.session_state.get("gps_denied", False)), key="gps_denied")

# UAV summary
st.markdown(
    f"**Selected:** {uav}  \n"
    f"**Type:** {specs['type']}  \n"
    f"**RTK-capable:** {'✅' if specs['rtk'] else '❌'}  |  "
    f"**Lidar onboard:** {'✅' if specs['lidar'] else '❌'}  |  "
    f"**Vision:** {vision_backend} {'✅' if enable_vision else '❌'}"
)

# ─────────────────────────────────────────────
# Helpers (marker image + camera model)
# ─────────────────────────────────────────────
def generate_aruco_png_bytes(marker_id: int, size_px: int = 800, border_bits: int = 1) -> bytes:
    """Return PNG bytes of an ArUco marker (DICT_4X4_1000). Falls back to a simple checker if OpenCV/aruco is unavailable."""
    from PIL import Image as PImage, ImageOps as PImageOps, ImageDraw
    if _ARUCO_OK:
        dict_ = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)
        # Support both APIs (OpenCV versions differ)
        if hasattr(cv2.aruco, "generateImageMarker"):
            img = cv2.aruco.generateImageMarker(dict_, marker_id, size_px)
        elif hasattr(cv2.aruco, "drawMarker"):
            img = cv2.aruco.drawMarker(dict_, marker_id, size_px)
        else:
            img = np.zeros((size_px, size_px), dtype=np.uint8)
        if border_bits > 0:
            img = cv2.copyMakeBorder(img, border_bits*10, border_bits*10, border_bits*10, border_bits*10,
                                     cv2.BORDER_CONSTANT, value=255)
        pil = PImage.fromarray(img)
    else:
        # Simple placeholder if aruco is not available
        pil = PImage.new("L", (size_px, size_px), 255)
        draw = ImageDraw.Draw(pil)
        s = size_px // 8
        for i in range(8):
            for j in range(8):
                if (i + j + marker_id) % 2 == 0:
                    draw.rectangle([i*s, j*s, (i+1)*s, (j+1)*s], fill=0)
        pil = PImageOps.expand(pil, border=20, fill=255)
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return buf.getvalue()

def generate_apriltag_png_bytes(tag_id: int, size_px: int = 800, border_bits: int = 1) -> bytes:
    """
    Return PNG bytes of an AprilTag (tag36h11) using OpenCV's aruco dictionaries.
    Falls back to a simple checker if aruco is unavailable.
    """
    from PIL import Image as PImage, ImageOps as PImageOps, ImageDraw
    pil = None
    if _ARUCO_OK and hasattr(cv2.aruco, "getPredefinedDictionary"):
        if hasattr(cv2.aruco, "DICT_APRILTAG_36h11"):
            dict_ = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
            if hasattr(cv2.aruco, "generateImageMarker"):
                img = cv2.aruco.generateImageMarker(dict_, tag_id, size_px)
            elif hasattr(cv2.aruco, "drawMarker"):
                img = cv2.aruco.drawMarker(dict_, tag_id, size_px)
            else:
                img = None
            if img is not None:
                if border_bits > 0:
                    img = cv2.copyMakeBorder(img, border_bits*10, border_bits*10, border_bits*10, border_bits*10,
                                             cv2.BORDER_CONSTANT, value=255)
                pil = PImage.fromarray(img)

    if pil is None:
        # Simple placeholder if aruco/AprilTag generation not available
        pil = PImage.new("L", (size_px, size_px), 255)
        draw = ImageDraw.Draw(pil)
        s = size_px // 10
        for i in range(10):
            for j in range(10):
                if (i * 7 + j * 3 + tag_id) % 2 == 0:
                    draw.rectangle([i*s, j*s, (i+1)*s, (j+1)*s], fill=0)
        pil = PImageOps.expand(pil, border=20, fill=255)

    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return buf.getvalue()

def focal_length_px(hfov_deg: float, width_px: int) -> float:
    hfov = np.radians(hfov_deg)
    return width_px / (2.0 * np.tan(hfov / 2.0))

def marker_pixels_from_alt(alt_m, marker_size_m, f_px):
    """Vectorized marker pixel size. Works with scalars or NumPy arrays."""
    alt = np.asarray(alt_m, dtype=float)
    px = (float(f_px) * float(marker_size_m)) / np.maximum(alt, 1e-6)
    return float(px) if px.ndim == 0 else px

def sigmoid(x): return 1.0 / (1.0 + np.exp(-x))

# ─────────────────────────────────────────────
# ArUco / AprilTag panel
# ─────────────────────────────────────────────
st.subheader("🎯 Vision Target & Camera Model")
colA, colB = st.columns([1, 1])

with colA:
    # Decide which family to render based on backend selection
    if st.session_state.get("vision_backend", "ArUco (OpenCV)").startswith("ArUco"):
        marker_png = generate_aruco_png_bytes(st.session_state.get("marker_id", 23), size_px=800)
        st.image(marker_png, caption=f"ArUco ID {marker_id} — print at {marker_size_cm} cm")
        st.download_button("Download ArUco PNG", data=marker_png, file_name=f"aruco_{marker_id}.png", mime="image/png")
        st.markdown(f"_OpenCV/aruco available:_ {'✅' if _ARUCO_OK else '❌ (using placeholder)'}")
    else:
        tag_png = generate_apriltag_png_bytes(st.session_state.get("marker_id", 23), size_px=800)
        st.image(tag_png, caption=f"AprilTag-36h11 ID {marker_id} — print at {marker_size_cm} cm")
        st.download_button("Download AprilTag PNG", data=tag_png, file_name=f"apriltag36h11_{marker_id}.png", mime="image/png")
        ok = _ARUCO_OK and hasattr(cv2.aruco, "DICT_APRILTAG_36h11")
        st.markdown(f"_AprilTag generation via OpenCV/aruco:_ {'✅' if ok else '❌ (using placeholder)'}")
        st.markdown(f"_pupil_apriltags detector available:_ {'✅' if _APRILTAG_OK else '❌'}")
        st.info("This sim estimates detection; the PNGs are suitable for printing and bench tests.")

with colB:
    st.markdown("**Pixel Size vs Altitude (pinhole model, nadir)**")
    fpx = focal_length_px(cam_hfov_deg, cam_res_x)
    alts = np.linspace(1.0, 20.0, 100)
    px = marker_pixels_from_alt(alts, marker_size_cm/100.0, fpx)
    fig_pix, ax_pix = plt.subplots()
    ax_pix.plot(alts, px)
    ax_pix.axhline(lock_thresh_px, linestyle="--")
    ax_pix.set_xlabel("Altitude AGL (m)")
    ax_pix.set_ylabel("Estimated Marker Size (px)")
    ax_pix.set_title("Marker Pixels vs Altitude")
    ax_pix.grid(True)
    st.pyplot(fig_pix)

# ─────────────────────────────────────────────
# Core Modules: XY / Z / EKF summary
# ─────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.subheader("📍 Position Accuracy (RTK GNSS)")
    sigma_xy = 0.03 if rtk_fix else 1.5
    n_pts = 350
    np.random.seed(seed)
    xy_noise = np.random.normal(0, sigma_xy, size=(n_pts, 2))
    if gps_glitch:
        xy_noise[np.random.randint(0, n_pts)] += np.array([3.0, -2.0])
    if wind_gust:
        xy_noise += np.array([0.15, -0.05])
    fig_xy, ax_xy = plt.subplots()
    ax_xy.scatter(xy_noise[:, 0], xy_noise[:, 1], alpha=0.35, s=10)
    ax_xy.set_title("Position Scatter (centered at pad)")
    ax_xy.set_xlabel("X (m)"); ax_xy.set_ylabel("Y (m)")
    ax_xy.set_xlim(-2, 2); ax_xy.set_ylim(-2, 2)
    ax_xy.grid(True)
    st.pyplot(fig_xy)

with col2:
    st.subheader("📏 Altitude Accuracy (Lidar vs Barometer)")
    n = 350
    np.random.seed(seed + 1)
    baro = np.random.normal(0, 0.25, n).cumsum() / 40.0
    if use_lidar:
        lidar = np.random.normal(0, 0.02, n)
    fig_alt, ax_alt = plt.subplots()
    ax_alt.plot(baro, label="Barometer (drift)")
    if use_lidar:
        ax_alt.plot(lidar, label="Lidar (cm-level)")
    ax_alt.set_title("Altitude Sensor Error (relative)")
    ax_alt.set_xlabel("Samples"); ax_alt.set_ylabel("Error (m)")
    ax_alt.grid(True); ax_alt.legend()
    st.pyplot(fig_alt)

st.subheader("🧠 Sensor Fusion Summary (EKF-style)")
st.markdown(
    "- **GNSS/RTK**: global XY; RTK shrinks CEP to cm-level.\n"
    "- **Lidar**: constrains Z in final meters (flare), removing baro drift.\n"
    "- **ArUco/AprilTag**: pad-relative pose; lock when pixel span ≥ threshold for ≥ dwell frames.\n"
    "- **IMU/Compass**: stabilize attitude/heading during VTOL hover.\n"
    "- **Kalman Filter**: smooths XY track using a CV (constant velocity) model."
)

# ─────────────────────────────────────────────
# 3D Landing Cone (Preview)
# ─────────────────────────────────────────────
with st.expander("Show 3D Landing Cone (preview)"):
    fig3d = plt.figure()
    ax3d = fig3d.add_subplot(111, projection='3d')
    z_top = 10.0
    zs = np.linspace(z_top, 0.0, 24)
    theta = np.linspace(0, 2*np.pi, 48)
    Z, TH = np.meshgrid(zs, theta)
    R = (Z / z_top) * 1.0
    X = R * np.cos(TH); Y = R * np.sin(TH)
    ax3d.plot_wireframe(X, Y, Z, linewidth=0.5, alpha=0.6)
    ax3d.set_xlabel("X (m)"); ax3d.set_ylabel("Y (m)"); ax3d.set_zlabel("Z (m)")
    ax3d.set_title("3D Landing Cone")
    ax3d.set_xlim(-1.0, 1.0); ax3d.set_ylim(-1.0, 1.0); ax3d.set_zlim(0, z_top)
    st.pyplot(fig3d)

# ─────────────────────────────────────────────
# Kalman Filter (CV model, 2D)
# ─────────────────────────────────────────────
def kf_init():
    x = np.zeros((4, 1))                 # [x, y, vx, vy]
    P = np.eye(4) * 10.0
    return x, P

def kf_step(x, P, z, q, r, dt=1.0):
    A = np.array([[1, 0, dt, 0],
                  [0, 1, 0, dt],
                  [0, 0, 1, 0 ],
                  [0, 0, 0, 1 ]], dtype=float)
    H = np.array([[1, 0, 0, 0],
                  [0, 1, 0, 0]], dtype=float)
    Q = q * np.array([[dt**4/4, 0, dt**3/2, 0],
                      [0, dt**4/4, 0, dt**3/2],
                      [dt**3/2, 0, dt**2, 0],
                      [0, dt**3/2, 0, dt**2]], dtype=float)
    R = np.eye(2) * (r**2)
    # Predict
    x = A @ x
    P = A @ P @ A.T + Q
    # Update
    y = z - (H @ x)
    S = H @ P @ H.T + R
    K = P @ H.T @ np.linalg.inv(S)
    x = x + K @ y
    P = (np.eye(4) - K @ H) @ P
    return x, P

# ─────────────────────────────────────────────
# Landing Success Metrics & Scoring
# ─────────────────────────────────────────────
def compute_metrics(path_kf, z_series, locked_series, dt=1.0):
    arr_kf = np.array(path_kf)
    z = np.maximum(np.array(z_series), 0.0)
    radial = np.linalg.norm(arr_kf, axis=1)
    r_allowed = (z / 10.0) * 1.0
    cone_viol = (radial > r_allowed).mean() if len(radial) else 1.0
    xy_err = float(np.linalg.norm(arr_kf[-1])) if len(arr_kf) else 99.0
    k = min(5, len(z)-1)
    vs = max(0.0, (z[-k-1] - z[-1]) / (k * dt)) if k >= 1 else 5.0
    n = len(locked_series)
    tail = max(1, int(0.3 * n))
    lock_stability = float(np.mean(locked_series[-tail:])) if n else 0.0
    return {"xy_error_m": xy_err, "touchdown_vspeed_mps": vs, "cone_violation_rate": float(cone_viol), "lock_stability": lock_stability}

def landing_score(m):
    xy_term = math.exp(-m["xy_error_m"] / 0.20)                       # ~20 cm scale
    vs_term = math.exp(-max(0.0, m["touchdown_vspeed_mps"] - 0.5) / 0.5)
    cone_term = math.exp(-5.0 * m["cone_violation_rate"])
    lock_term = m["lock_stability"]
    return float(100.0 * (0.40 * xy_term + 0.20 * vs_term + 0.20 * cone_term + 0.20 * lock_term))

# ─────────────────────────────────────────────
# Landing Playback (Vision-assisted + Kalman + Scoring + Export)
# ─────────────────────────────────────────────
st.subheader("🎬 Landing Playback (Vision-assisted + Kalman + Score)")

def vision_detect_prob(px, thresh_px, illum, blur, backend):
    k = 0.25
    base = sigmoid((px - thresh_px) * k)
    backend_boost = 1.0 if backend.startswith("ArUco") else 1.1  # AprilTag a tad more robust (sim)
    blur_penalty = (1.0 - 0.6 * blur)
    light_boost = 0.6 + 0.4 * illum
    return np.clip(base * blur_penalty * light_boost * backend_boost, 0.0, 1.0)

def focal_px():
    return focal_length_px(cam_hfov_deg, cam_res_x)

start = st.button("Run Playback")

if start:
    np.random.seed(seed)
    run_uuid = str(uuid.uuid4())
    run_time_utc = dt.datetime.utcnow().isoformat() + "Z"

    # XY per-step model
    if gps_denied:
        # INS-style drift (bias + noise), no absolute GNSS correction
        rng = np.random.default_rng(seed)
        drift = np.zeros(2)
        steps_xy = np.zeros((steps, 2))
        noise_std = 0.006         # small random motion (m per step)
        bias_walk_std = 0.002     # slowly wandering bias (m per step)
        for i in range(steps):
            drift += rng.normal(0.0, bias_walk_std, size=2)
            steps_xy[i] = drift + rng.normal(0.0, noise_std, size=2)
    else:
        # Original GNSS random walk
        per_step_sigma = 0.03 if rtk_fix else 1.0
        steps_xy = np.random.normal(0, per_step_sigma, size=(steps, 2))

    # Apply environment effects to both modes
    if wind_gust:
        steps_xy += np.array([0.01, -0.003])
    if gps_glitch and steps > 10 and not gps_denied:  # legacy glitch only when GNSS is present
        j = np.random.randint(5, steps - 5)
        steps_xy[j] += np.array([2.5, -1.5])

    # Z descent with lidar noise model
    z_descent = np.linspace(10.0, 0.0, steps)
    lidar_sigma = 0.05 if use_lidar else 0.5
    z_descent = z_descent + np.random.normal(0, lidar_sigma, steps)

    fpx = focal_px()
    hfov_rad = np.radians(cam_hfov_deg)

    # Placeholders to update in-place (no stacked frames)
    placeholder2d = st.empty()
    status_box = st.empty()
    placeholder3d = st.empty()

    # Kalman init
    x, P = kf_init()
    pos_raw = np.array([0.0, 0.0])
    path_kf, path_raw = [], []
    dwell = 0
    locked = False
    det_timeline, px_timeline, locked_timeline = [], [], []
    z_timeline = []

    for i in range(steps):
        # Integrate raw XY position
        pos_raw = pos_raw + steps_xy[i]
        z_now = max(z_descent[i], 0.0)
        z_timeline.append(z_now)

        # Camera geometry / pixel model
        radial = np.linalg.norm(pos_raw)
        in_fov = radial <= max(z_now, 1e-6) * np.tan(hfov_rad / 2.0)
        px_est = marker_pixels_from_alt(max(z_now, 1e-6), marker_size_cm / 100.0, fpx)
        px_timeline.append(px_est)

        # Simulated detection
        detected = False
        if enable_vision and in_fov:
            p_det = vision_detect_prob(px_est, lock_thresh_px, illum, blur, vision_backend)
            if np.random.rand() < (p_det * (1.0 - occlusion_prob)):
                detected = True
        else:
            p_det = 0.0

        # Dwell/lock logic
        if detected:
            dwell += 1
            if dwell >= lock_dwell_frames:
                locked = True
        else:
            dwell = 0
            if locked and i > steps // 3 and np.random.rand() < 0.05:
                locked = False

        det_timeline.append(1 if detected else 0)
        locked_timeline.append(1 if locked else 0)

        # Apply extra beacon-like correction when locked (pull toward pad)
        if locked and beacon_gain > 0:
            pos_raw = pos_raw + (-float(beacon_gain) * pos_raw)

        # Kalman measurement noise (R) selection
        if gps_denied:
            # Without vision lock, treat as poor INS (very large σ); with lock, use pixel-based tightening
            if locked:
                sigma_meas = max(0.05, min(0.25, 0.8 / max(px_est, 1.0)))
            else:
                sigma_meas = 3.0  # meters (low-confidence dead-reckon)
        else:
            # Original behavior
            sigma_meas = (max(0.02, min(0.20, 0.8 / max(px_est, 1.0)))) if locked else kf_r_base

        z_meas = pos_raw.reshape(2, 1)
        x, P = kf_step(x, P, z_meas, q=kf_q, r=sigma_meas, dt=1.0)
        pos_kf = x[:2].ravel()

        path_raw.append(pos_raw.copy())
        path_kf.append(pos_kf.copy())

        # 2D plot (in-place)
        fig2d, ax2d = plt.subplots()
        arr_kf = np.array(path_kf); arr_raw = np.array(path_raw)
        if len(arr_raw) > 1:
            ax2d.plot(arr_raw[:, 0], arr_raw[:, 1], alpha=0.25, label="Raw (GNSS/INS path)")
        if len(arr_kf) > 1:
            ax2d.plot(arr_kf[:, 0], arr_kf[:, 1], label="Kalman-smoothed")
        ax2d.scatter(pos_kf[0], pos_kf[1], s=40, label="Current (KF)")
        # Allowed cone radius at this altitude
        r_allowed = (z_now / 10.0) * 1.0
        ring = plt.Circle((0, 0), max(r_allowed, 0.05), fill=False, linestyle="--")
        ax2d.add_artist(ring)
        ax2d.set_title(f"Descent: {z_now:.2f} m AGL  |  In FOV: {in_fov}  |  Locked: {locked}")
        ax2d.set_xlabel("X (m)"); ax2d.set_ylabel("Y (m)")
        lim = 2.0
        ax2d.set_xlim(-lim, lim); ax2d.set_ylim(-lim, lim)
        ax2d.grid(True); ax2d.legend(loc="upper right")
        placeholder2d.pyplot(fig2d)

        # 3D quick trace (lightweight, also in-place)
        fig3d_step = plt.figure()
        ax3d_step = fig3d_step.add_subplot(111, projection='3d')
        zs = np.linspace(10.0, 0.0, 12)
        thetas = np.linspace(0, 2*np.pi, 24)
        Zm, THm = np.meshgrid(zs, thetas)
        Rm = (Zm / 10.0) * 1.0
        Xm = Rm * np.cos(THm); Ym = Rm * np.sin(THm)
        ax3d_step.plot_wireframe(Xm, Ym, Zm, linewidth=0.4, alpha=0.35)
        ax3d_step.plot(arr_kf[:, 0], arr_kf[:, 1], np.maximum(z_descent[:len(arr_kf)], 0), linewidth=1.5)
        ax3d_step.scatter(pos_kf[0], pos_kf[1], z_now, s=20)
        ax3d_step.set_xlim(-1.2, 1.2); ax3d_step.set_ylim(-1.2, 1.2); ax3d_step.set_zlim(0, 10.0)
        ax3d_step.set_xlabel("X (m)"); ax3d_step.set_ylabel("Y (m)"); ax3d_step.set_zlabel("Z (m)")
        ax3d_step.set_title("3D Cone Trace (KF path)")
        placeholder3d.pyplot(fig3d_step)

        status_box.markdown(
            f"**Vision:** {'🟢 LOCKED' if locked else ('🟡 DETECTED' if detected else '🔴 SEEKING')}  "
            f"| px≈{px_est:.1f}  | p≈{p_det:.2f}  | dwell={dwell}/{lock_dwell_frames}  "
            f"| σ_meas≈{sigma_meas:.2f} m | beacon_gain={beacon_gain:.2f} "
            f"| {'🛰️🚫 GPS-denied' if gps_denied else '🛰️ GNSS OK'}"
        )

        time.sleep(play_speed)

    # Metrics & overall score
    metrics = compute_metrics(path_kf, z_descent, locked_timeline, dt=1.0)
    score = landing_score(metrics)
    st.success(f"✅ Playback complete — touchdown achieved.  **Landing Success Score: {score:.1f}/100**")

    mcol1, mcol2, mcol3, mcol4 = st.columns(4)
    mcol1.metric("XY Touchdown Error", f"{metrics['xy_error_m']:.3f} m")
    mcol2.metric("Touchdown V-Speed", f"{metrics['touchdown_vspeed_mps']:.2f} m/s", "target ≤ 0.5")
    mcol3.metric("Cone Violation Rate", f"{metrics['cone_violation_rate']*100:.1f}%")
    mcol4.metric("Lock Stability (final 30%)", f"{metrics['lock_stability']*100:.1f}%")

    # Diagnostics
    d1, d2 = st.columns(2)
    with d1:
        st.markdown("**Detection Timeline (1=detected)**")
        fig_t, ax_t = plt.subplots()
        ax_t.plot(det_timeline)
        ax_t.set_xlabel("Frame"); ax_t.set_ylabel("Detected (0/1)")
        ax_t.grid(True)
        st.pyplot(fig_t)
    with d2:
        st.markdown("**Marker Pixels per Frame**")
        fig_p, ax_p = plt.subplots()
        ax_p.plot(px_timeline)
        ax_p.axhline(lock_thresh_px, linestyle="--")
        ax_p.set_xlabel("Frame"); ax_p.set_ylabel("Marker Size (px)")
        ax_p.grid(True)
        st.pyplot(fig_p)

    # Frame-by-frame CSV
    run_df = pd.DataFrame({
        "t": np.arange(steps),
        "x_raw": np.array(path_raw)[:, 0],
        "y_raw": np.array(path_raw)[:, 1],
        "x_kf": np.array(path_kf)[:, 0],
        "y_kf": np.array(path_kf)[:, 1],
        "z_agl": np.maximum(np.array(z_timeline), 0.0),
        "detected": det_timeline,
        "locked": locked_timeline,
        "px_est": px_timeline
    })

    # ── Log Export: JSON, CSV, ZIP
    settings_payload = {
        "app_version": APP_VERSION,
        "run_uuid": run_uuid,
        "run_time_utc": run_time_utc,
        "uav_model": uav,
        "uav_specs": specs,
        "preset": st.session_state.get("preset_choice"),
        "rtk_fix": bool(rtk_fix),
        "use_lidar": bool(use_lidar),
        "vision_backend": vision_backend,
        "enable_vision": bool(enable_vision),
        "marker_id": int(marker_id),
        "marker_size_cm": int(marker_size_cm),
        "camera": {"width_px": int(cam_res_x), "height_px": int(cam_res_y), "hfov_deg": float(cam_hfov_deg)},
        "lock_thresh_px": int(lock_thresh_px),
        "lock_dwell_frames": int(lock_dwell_frames),
        "illum": float(illum),
        "blur": float(blur),
        "occlusion_prob": float(occlusion_prob),
        "beacon_gain": float(beacon_gain),
        "kf_q": float(kf_q),
        "kf_r_base": float(kf_r_base),
        "seed": int(seed),
        "steps": int(steps),
        "play_speed": float(play_speed),
        "wind_gust": bool(wind_gust),
        "gps_glitch": bool(gps_glitch),
        "gps_denied": bool(gps_denied),
        "link_marker_to_uav": bool(link_to_uav),
    }
    metrics_payload = {"score": score, **metrics}

    log_json = {
        "meta": {"app_version": APP_VERSION, "run_uuid": run_uuid, "run_time_utc": run_time_utc},
        "uav": {"model": uav, "specs": specs},
        "settings": settings_payload,
        "metrics": metrics_payload,
        "trace_columns": list(run_df.columns),
        "trace_preview_head": run_df.head(5).to_dict(orient="list")
    }
    json_bytes = json.dumps(log_json, indent=2).encode("utf-8")
    csv_bytes = run_df.to_csv(index=False).encode("utf-8")

    st.download_button("Download Playback CSV", csv_bytes, file_name=f"vtol_playback_{run_uuid[:8]}.csv", mime="text/csv")
    st.download_button("Download Run Log (JSON)", json_bytes, file_name=f"vtol_runlog_{run_uuid[:8]}.json", mime="application/json")

    # ZIP with both + full settings
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(f"run_{run_uuid[:8]}/trace.csv", csv_bytes)
        zf.writestr(f"run_{run_uuid[:8]}/runlog.json", json_bytes)
        zf.writestr(f"run_{run_uuid[:8]}/settings_only.json", json.dumps(settings_payload, indent=2).encode("utf-8"))
    st.download_button("Download All (ZIP)", zip_buf.getvalue(), file_name=f"vtol_run_{run_uuid[:8]}.zip", mime="application/zip")

# ─────────────────────────────────────────────
# Auto-Tuner (experimental) — maximize score
# ─────────────────────────────────────────────
st.subheader("🧪 Auto-Tuner (experimental)")

with st.expander("Open Auto-Tuner"):
    n_trials = st.number_input("Trials", min_value=5, max_value=200, value=30, step=5, key="trials")
    seeds_per_trial = st.slider("Seeds per Trial (averaged)", 1, 10, 3, key="seeds_per_trial")
    steps_tune = st.slider("Sim Steps (tuner)", 40, 240, min(st.session_state.get("steps", 160), 120), 10, key="steps_tune")
    run_tuner = st.button("Run Auto-Tune")

    def kf_fast_once(params, seed_val):
        np.random.seed(seed_val)

        # XY model honors gps_denied (add-only; default behavior unchanged)
        if params.get("gps_denied", False):
            rng = np.random.default_rng(seed_val)
            drift = np.zeros(2)
            steps_xy = np.zeros((steps_tune, 2))
            noise_std = 0.006
            bias_walk_std = 0.002
            for i in range(steps_tune):
                drift += rng.normal(0.0, bias_walk_std, size=2)
                steps_xy[i] = drift + rng.normal(0.0, noise_std, size=2)
        else:
            per_step_sigma = 0.03 if params["rtk_fix"] else 1.0
            steps_xy = np.random.normal(0, per_step_sigma, size=(steps_tune, 2))

        if params["wind_gust"]:
            steps_xy += np.array([0.01, -0.003])
        if params["gps_glitch"] and steps_tune > 10 and not params.get("gps_denied", False):
            j = np.random.randint(5, steps_tune - 5)
            steps_xy[j] += np.array([2.5, -1.5])

        z_descent = np.linspace(10.0, 0.0, steps_tune)
        lidar_sigma = 0.05 if params["use_lidar"] else 0.5
        z_descent = z_descent + np.random.normal(0, lidar_sigma, steps_tune)

        fpx_local = focal_length_px(params["cam_hfov_deg"], params["cam_res_x"])
        hfov_rad = np.radians(params["cam_hfov_deg"])

        x, P = kf_init()
        pos_raw = np.array([0.0, 0.0])
        path_kf, locked_series = [], []
        dwell, locked = 0, False

        for i in range(steps_tune):
            pos_raw = pos_raw + steps_xy[i]
            z_now = max(z_descent[i], 0.0)

            radial = np.linalg.norm(pos_raw)
            in_fov = radial <= max(z_now, 1.0e-6) * np.tan(hfov_rad / 2.0)
            px_est = marker_pixels_from_alt(max(z_now, 1.0e-6), params["marker_size_cm"]/100.0, fpx_local)

            # Detection probability (same shape as UI model)
            k = 0.25
            base = sigmoid((px_est - params["lock_thresh_px"]) * k)
            blur_penalty = (1.0 - 0.6 * params["blur"])
            light_boost = 0.6 + 0.4 * params["illum"]
            p_det = np.clip(base * blur_penalty * light_boost, 0.0, 1.0)

            detected = params["enable_vision"] and in_fov and (np.random.rand() < (p_det * (1.0 - params["occlusion_prob"])))
            if detected:
                dwell += 1
                if dwell >= params["lock_dwell_frames"]:
                    locked = True
            else:
                dwell = 0
                if locked and i > steps_tune // 3 and np.random.rand() < 0.05:
                    locked = False

            # Apply correction when locked (same as main loop)
            if locked and params["beacon_gain"] > 0:
                pos_raw = pos_raw + (-params["beacon_gain"] * pos_raw)

            # R selection mirrors main loop (adds gps_denied branch)
            if params.get("gps_denied", False):
                if locked:
                    sigma_meas = max(0.05, min(0.25, 0.8 / max(px_est, 1.0)))
                else:
                    sigma_meas = 3.0
            else:
                sigma_meas = max(0.02, min(0.20, 0.8 / max(px_est, 1.0))) if locked else params["kf_r_base"]

            z_meas = pos_raw.reshape(2, 1)
            x, P = kf_step(x, P, z_meas, q=params["kf_q"], r=sigma_meas, dt=1.0)
            pos_kf = x[:2].ravel()
            path_kf.append(pos_kf.copy()); locked_series.append(1 if locked else 0)

        m = compute_metrics(path_kf, z_descent, locked_series, dt=1.0)
        return landing_score(m), m

    def simulate_mean_score(params, seeds_list):
        scores = []
        for s in seeds_list:
            sc, _ = kf_fast_once(params, s)
            scores.append(sc)
        return float(np.mean(scores))

    if run_tuner:
        results = []
        base = dict(
            rtk_fix=rtk_fix, use_lidar=use_lidar, enable_vision=enable_vision,
            cam_hfov_deg=cam_hfov_deg, cam_res_x=cam_res_x,
            marker_size_cm=marker_size_cm, blur=blur, illum=illum,
            occlusion_prob=occlusion_prob, wind_gust=wind_gust, gps_glitch=gps_glitch,
            gps_denied=gps_denied, kf_q=kf_q, kf_r_base=kf_r_base,
            lock_thresh_px=lock_thresh_px, lock_dwell_frames=lock_dwell_frames,
            beacon_gain=beacon_gain
        )

        rng = np.random.default_rng(42)
        seeds_list = list(rng.integers(0, 10_000, size=int(seeds_per_trial)))

        for t in range(int(n_trials)):
            trial = base.copy()
            trial["beacon_gain"] = float(rng.uniform(0.15, 0.60))
            trial["lock_thresh_px"] = int(rng.integers(18, 48))
            trial["lock_dwell_frames"] = int(rng.integers(4, 14))
            trial["kf_q"] = float(10 ** rng.uniform(-4.5, -1.9))      # ~3e-5 .. 1e-2
            trial["kf_r_base"] = float(rng.uniform(0.02, 0.60))       # meters

            mean_score = simulate_mean_score(trial, seeds_list)
            results.append({
                "trial": t+1,
                "score_mean": mean_score,
                "beacon_gain": trial["beacon_gain"],
                "lock_thresh_px": trial["lock_thresh_px"],
                "lock_dwell_frames": trial["lock_dwell_frames"],
                "kf_q": trial["kf_q"],
                "kf_r_base": trial["kf_r_base"],
            })

        df = pd.DataFrame(results).sort_values("score_mean", ascending=False).reset_index(drop=True)
        st.markdown("**Top Results**")
        st.dataframe(df.head(10))

        # Export tuner results
        st.download_button("Download Tuner Results (CSV)", df.to_csv(index=False).encode("utf-8"),
                           file_name="tuner_results.csv", mime="text/csv")

        # Best row dict
        best = df.iloc[0].to_dict()
        c1, c2 = st.columns(2)
        with c1:
            st.success(
                "🏆 **Recommended Settings**\n\n"
                f"- Beacon Gain ≈ **{best['beacon_gain']:.2f}**\n"
                f"- Vision Lock Threshold ≈ **{int(best['lock_thresh_px'])} px**\n"
                f"- Lock Dwell ≈ **{int(best['lock_dwell_frames'])} frames**\n"
                f"- Kalman q ≈ **{best['kf_q']:.4g}**\n"
                f"- Kalman R (GNSS σ) ≈ **{best['kf_r_base']:.2f} m**\n"
                f"- Mean Score ≈ **{best['score_mean']:.1f}/100**"
            )
        with c2:
            # One-click Apply Best Settings
            if st.button("Apply Best Settings ▶️"):
                st.session_state["apply_payload"] = {
                    "beacon_gain": float(best["beacon_gain"]),
                    "lock_thresh_px": int(best["lock_thresh_px"]),
                    "lock_dwell_frames": int(best["lock_dwell_frames"]),
                    "kf_q": float(best["kf_q"]),
                    "kf_r_base": float(best["kf_r_base"]),
                    # Keep current camera & toggles, but include for completeness
                    "cam_hfov_deg": float(cam_hfov_deg),
                    "cam_res_x": int(cam_res_x),
                    "marker_size_cm": int(marker_size_cm),
                    "enable_vision": bool(enable_vision),
                    "rtk_fix": bool(rtk_fix),
                    "use_lidar": bool(use_lidar),
                }
                st.session_state["pending_apply"] = True
                st.rerun()

# Footer
with st.expander("UAV Spec Snapshot"):
    st.dataframe(pd.DataFrame(uav_data).T)
st.caption("Tip: Use Scenario Preset to configure conditions quickly, then Auto-Tune and Apply Best Settings. Aim for XY≤0.2 m, V-speed≤0.5 m/s, high lock stability.")
