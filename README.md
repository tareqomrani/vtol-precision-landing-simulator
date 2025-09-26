🛩️ VTOL Precision Landing Simulator

A Streamlit app that simulates precision landings for eVTOL / Hybrid VTOL aircraft in confined spaces using:
	•	RTK GNSS (centimeter XY)
	•	Lidar (centimeter Z near ground)
	•	Vision targets (ArUco / AprilTag) for pad-relative lock
	•	Kalman filtering (constant-velocity XY smoothing)
	•	3D Landing Cone, scoring, and an Auto-Tuner with one-click Apply Best Settings

⚠️ Simulation for education & prototyping only. Not flight software.

⸻

✨ Features
	•	eVTOL Dataset (preloaded): Vector, WingtraOne, DeltaQuad Pro, Urban Hawk (custom)
	•	RTK vs GPS Drift: toggle fix; inject wind/gps glitches
	•	Lidar vs Baro altitude comparison (drift removal below ~5–10 m AGL)
	•	Vision Assist: ArUco/AprilTag size model, lock threshold & dwell, illumination/blur/occlusion controls
	•	3D Landing Cone: visualize allowable corridor from 10 m AGL to touchdown
	•	In-place Playback Animation: clean, single-plot animation (no stacked graphs)
	•	Kalman Filter (CV): smoother XY path and better touchdown error
	•	Landing Success Score: XY error, Z/vertical speed, cone compliance, vision lock stability
	•	Auto-Tuner: random search for high scores; Apply Best Settings updates all sliders instantly
🔧 Modules & Models

RTK vs GPS (XY)
	•	RTK fix → σ≈3 cm scatter; No RTK → σ≈1–1.5 m
	•	Optional wind bias and GPS glitch spike

Lidar vs Barometer (Z)
	•	Barometer: slow drift over time
	•	Lidar: centimeter-level Z near ground; used primarily below ~5–10 m AGL

Vision Assist (ArUco / AprilTag)
	•	Pinhole model estimates marker pixel size:
px ≈ f_px * (marker_size_m / altitude_m)
	•	Lock when pixel size ≥ threshold for N dwell frames
	•	Sliders for illumination, motion blur, occlusion to stress test detection
	•	Print your ArUco marker from the app (PNG download); pick a size that reaches pixel-threshold at your intended lock altitude

Kalman Filter (XY)
	•	Constant-velocity (CV) model; measurement noise decreases when vision lock is active
	•	Smooths raw GNSS path and improves touchdown XY error

3D Landing Cone
	•	Tapered corridor: ~1 m radius at 10 m AGL → 0 at pad
	•	Live overlay in 2D (allowed radius ring) + 3D wireframe trace

⸻

🧮 Scoring

Landing Success Score (0–100) blends:
	•	XY touchdown error (target ≤ 0.20 m)
	•	Vertical speed at touchdown (target ≤ 0.5 m/s)
	•	Cone violation rate (fraction of frames outside allowed radius)
	•	Lock stability in final 30% of the approach

Weighting is documented in landing_score(m). Tune as needed.

⸻

🤖 Auto-Tuner
	•	Random search over:
	•	Beacon/vision correction gain
	•	Lock threshold (px) & dwell (frames)
	•	Kalman q (process noise) & R base (GNSS σ)
	•	Averages multiple random seeds → mean score
	•	Shows top results and a one-click “Apply Best Settings” button to push the winner back into the sidebar controls (via st.session_state)
	•	Rerun playback to validate

⸻

🧠 Tips
	•	Too many graphs? Stream uses a single placeholder for animation — no stacking.
	•	No vision lock? Increase marker size or reduce HFOV; boost illumination; reduce blur/occlusion; lower pixel threshold or raise dwell.
	•	Wobbly path? Increase lock_dwell_frames; raise Kalman q slightly if lagging, or lower q if noisy.
	•	Confined pads: use higher beacon gain once locked and verify cone compliance.

⸻

❓ FAQ

Q: Can I use my own camera params?
A: Yes — set your camera width/height & HFOV in the sidebar. The pixel model uses those directly.

Q: Does AprilTag require a camera?
A: This sim emulates detection behavior. For real detection, integrate your video feed and pose estimator (outside scope here).

Q: Can I log and export runs?
A: Yes — Download Playback CSV after each run.

⸻

🛠️ Troubleshooting
	•	OpenCV errors on macOS (M1/M2): try opencv-contrib-python (non-headless) or install via conda-forge.
	•	AprilTag install issues (Windows): comment it out in requirements.txt and use ArUco.
	•	Streamlit Cloud memory/timeouts: reduce Playback Steps and Auto-Tuner Trials.

⸻

🗺️ Roadmap
	•	Real-time camera input with live ArUco/AprilTag detection
	•	Beacon array / multi-target blending
	•	3D vehicle dynamics and wind fields
	•	SITL bridge (ArduPilot) for HIL demos
	•	Scenario presets (rooftop / ship deck / clearing)

⸻

📄 License

MIT — feel free to fork, tweak, and share. Attribution appreciated.

⸻

🙏 Acknowledgements
	•	Streamlit, NumPy, Pandas, Matplotlib
	•	OpenCV (aruco), pupil_apriltags (AprilTag)
	•	Community insights on RTK GNSS, lidar altimetry, and EKF3-style fusion
