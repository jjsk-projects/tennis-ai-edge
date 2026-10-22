# Technical steps — Serve Lab pipeline

Per-stage implementation detail, tuning knobs, edge-deployment notes and troubleshooting. Setup is in the README.

## 1. Data flow

```
video (any fps, any resolution)
  -> Stage 1  extract_frames: runs/<id>/frames/000000.jpg ... (1280 px wide, JPEG q92) + meta.json
  -> Stage 2  player box (YOLOX, at most 10x per second) -> BlazePose on a zoomed crop (at most 60x per second) -> (33,3) [x_px, y_px, conf]
              then fill_gaps (linear) + smooth_poses (confidence-weighted, centred, ~33 ms each side)
  -> Stage 3  per frame: motion candidates - forehand: the WHOLE frame; serve: a region around the server
              per stroke: racquet boxes (YOLOX on a zoomed crop), contact / hitting point, bounce, baseline, scale
  -> Stage 4  measurements (metres, degrees, seconds) -> rules -> good / improve / unknown + tips in priority order
  -> Stage 5  coach text (template | OpenAI-compatible | Anthropic)
  -> visuals  keyframes.jpg, skeleton.png, sequence.gif, overview.jpg (full-frame ball trajectory)  +  result.json
```

Coordinates are full-frame pixels at the processed resolution. Heights come from `Scale`: the player's real height divided by their standing height in pixels gives metres per pixel at the player's depth, so heights are valid for things in the player's plane.

## 2. Stage 1 — Frames (`src/pipeline.py: extract_frames`)
- The video is split into one JPEG per frame, scaled to `pipeline.video.max_width` (default 1280). Every frame is kept (120 per second for a 120 fps clip), and `pipeline.video.max_fps` (default 120) sub-samples anything faster.
- `meta.json` caches fps, frame count and size. A second run on the same output folder reuses the frames.
- Every later stage reads from disk. A 15 s 1080p/120 fps clip is about 390 MB of JPEGs; set `keep_frames: false` to delete them after the run.

## 3. Stage 2 — Player and skeleton
**Player box** (`detection/person_detector.py`)
- Model: YOLOX-tiny ONNX at 416 px input, class `person`, every 3rd frame but at most 10 times a second (`pipeline.detector.every_frames`, `max_hz`; every 12th frame at 120 fps). Between detections the box follows the skeleton.
- Player selection: the box closest to the previous one. The first pick, and any re-pick after the skeleton is lost, is the most prominent person; when the skeleton is lost the detector runs again on the next frame.
- ONNX Runtime uses a GPU / NPU provider when the machine has one (CoreML on a Mac, CUDA, DirectML), else the CPU. About 50 ms per call on a 2-core CPU.

**Skeleton** (`pose_model/pose_model_backend.py`)
- MediaPipe BlazePose (33 landmarks, including heels and toes) in VIDEO mode, at most 60 times a second (`pipeline.pose.max_hz`; every 2nd frame at 120 fps, gaps filled by linear interpolation). It runs on a square crop of 1.9× the player height, upscaled to 384 px, and landmarks are mapped back to full-frame pixels.
- On IMG_3025 this crop raised detection to 100% of frames (about 60% with full-frame inference in the original code).
- `to_coco17()` gives the standard COCO-17 layout: wrists 9/10, ankles 15/16. The original code put wrists at 15/16, which are the ankles in COCO.

**Smoothing** (`pose_model/smoothing.py`, reused idea from the original `PoseTracker`)
- Confidence-weighted average over a centred window: ±1 frame at 30 fps, ±4 at 120 fps.
- Because the window is centred, there is no lag at contact (a running average would lag).

## 4. Stage 3 — Racquet, ball, court
**Racquet** (`person_detector.find_racket`)
- YOLOX (`tennis racket`, COCO class 38) on a crop of 1.8× the player height around the player. The racquet is too small for full-frame detection.
- The racquet head is the box corner farthest from the hitting wrist, pulled back 22% toward the wrist.
- Detected in 93–137 frames per forehand on the 120 fps clip. On the 30 fps serve clips it is found less often (small player, motion blur).

**Ball candidates** (`ball_tracker.motion_candidates`)
- Search area: **the whole frame for forehands** (incoming ball, bounce, outgoing ball); a region around the server for serves (the toss RANSAC needs a clean cloud).
- Camera shake is removed with whole-frame phase correlation, computed once per frame pair on a copy at most 640 px wide (a 320 px copy lost sub-pixel accuracy and flooded the serve clip with false candidates). Moving pixels are kept where `min(|cur-prev|, |cur-next|) > 14`.
- Blobs are kept if they are 2 px to (0.07 × player height)² in area.
- `drop_static` removes flicker that reappears at the same spot (leaves, cables).

**Serve contact** (`fit_toss`, `find_contact`)
- The toss is fitted by RANSAC over the (t, x, y) cloud: `y = a t² + b t + c` with `2a ≈ g_px = 9.81/fps² × stature_px/height_m`, little sideways drift, and the apex above the head beside the player.
- The hit ball is a fast straight track that is at least 2× the toss speed, leaves at least 35° off the toss direction, and lands on the parabola when stepped back one frame.
- **Contact = midpoint of the last toss position and the first hit-ball position; contact frame = midpoint of the two frames.**
- Sanity gate: contact between 1.2× and 2.1× body height (arm + racquet + jump), and toss peak at most 1.6 m above contact. Otherwise the serve falls back to timing from the skeleton, and its ball-based checks show "unknown".

**Forehand contact** (`find_racquet_contact`, preferred on HD footage)
- Ball blobs must be tennis-ball yellow (HSV hue 20–50, S ≥ 60, V ≥ 110). Blobs on the torso, head or shoes are discarded, because yellow shoes look like balls.
- **Hitting point = midpoint of the ball in the two frames closest to the racquet head** (within 0.35 × body height).
- The incoming (up to 1.4 s) and outgoing (up to 0.5 s) balls are chained with a constant-velocity prediction that bridges up to 0.1 s of missing frames (ball behind the player). At the bounce, where the direction flips, the chain accepts the candidate at the right distance nearest the prediction. This is the original Kalman idea in its simplest form.
- A swing counts as a stroke only if (a) the ball is followed for at least 5 frames into or out of the racquet, and (b) `confirms_hit`: the racquet head moves at least 0.2 body heights per 0.1 s at contact and the ball leaves in the swing direction (cosine ≥ 0.3). Missed strokes are preferred to coaching a non-stroke.
- Racquet detection runs only ±0.3 s around a swing that has a ball near it, on every 2nd frame at 120 fps (heads interpolated); ball colour tests are memoised across overlapping swings.
- Fallbacks:
  - Low-res or colourless footage: `find_groundstroke_contact`, which matches motion-only incoming and outgoing tracks at the racquet (gap of 1–4 frames, at least 60° change of direction).
  - No ball near the racquet: the racquet head at its fastest moment.

**Bounce** (`bounce_from_chain`)
- The lowest point of the incoming chain, near court level, at least 0.12 s before contact, with the ball falling into it and rising out. If the track starts at court level and only rises (the ball was not seen before the bounce), its first point is used.
- Reported as `bounce_to_contact_s`. Time is used instead of metres because the camera angle squeezes distances along the court (a metre toward the camera covers far fewer pixels than a metre of height).
- Drawn on `overview.jpg` with the full incoming path, hitting point and outgoing arrow.

**Baseline** (`court_segmentation/court_detector.py`)
- White mask (V ≥ 150, S ≤ 80) in a band around the feet, then a probabilistic Hough transform. The near-horizontal line closest to the feet is kept, taking the median over frames.
- Manual override: two clicks in the UI (`baseline=x1,y1,x2,y2`).
- `camera_side`: `far` means filmed from across the net (the court is lower in the image); `near` means filmed from behind the player.

**Kalman filter** (`ball_tracking/kalman_filter.py`)
- 6-state constant-acceleration filter with adaptive measurement noise.
- Bug fixed: `predict(dt)` now really rebuilds `F`. It is kept for live, frame-by-frame tracking on the edge device.

## 5. Stage 4 — Judge (coaching logic)
- Each rule is `(key, category, title, measurement, test, target, tip)`. List order is priority. A tip can depend on other measurements; for example, the "too high" advice depends on court position.
- Failed checks become `suggestions` with `priority` and `category`. A check is `unknown` when its measurement is missing.
- There is no machine learning in the judgement. The Random Forest experiment was removed, because it only echoed the rules.

**Forehand** (`serve_analysis/forehand.py`, `FH_RULES`, 18 core checks + 8 joint checks, see Levels)

| # | Check | Measurement | Good if |
|---|---|---|---|
| 1 | Hitting point not above the head | contact height − (nose height + 0.10 m) | ≤ 0 m (tip depends on behind / inside baseline) |
| 2 | Ball bounced close enough | time from bounce to contact | ≤ 0.8 s |
| 3 | Body sideways in preparation | min shoulder width / square-on width | ≤ 0.80 |
| 4 | Contact in front of the body | shoulder width at contact / square-on | ≥ 0.60 |
| 5 | Left foot in front at contact | right ankle ahead of left, toward the net | ≤ 0.05 m |
| 6 | Contact not too low | contact height / body height | ≥ 0.40 |
| 7 | Room to the side | contact − hip centre, toward the hitting side | 0.25–0.9 m |
| 8 | Arm not cramped | hitting elbow angle (player ≥ 150 px only) | ≥ 100° |
| 9 | Head still | nose movement ±0.06 s around contact | ≤ 0.15 m |
| 10 | Early preparation | time from full turn to contact | ≥ 0.30 s |
| 11 | Left arm tracks the ball | left wrist height / shoulder height at backswing | ≥ 0.75 |
| 12 | Knee bend | 180° − 10th-percentile knee angle while loading | ≥ 20° |
| 13 | Stable base | ankle distance at contact | ≥ 0.45 m |
| 14 | Racquet drops below the ball | contact height − lowest racquet head (0.4 s before) | ≥ 0.15 m |
| 15 | Low-to-high swing | wrist rise from its lowest point to the finish | ≥ 0.40 m |
| 16 | Legs drive up | hip rise into contact | ≥ 0.03 m |
| 17 | Finish over the left shoulder | wrist crosses the body at shoulder height | yes |
| 18 | Balanced finish | head offset from the centre of the feet at the finish | ≤ 0.25 m |

**Serve** (`serve_analysis/analysis.py`, `RULES`, 15 core checks + 5 joint checks, see Levels):
1. No foot fault: foot over the baseline before contact ≤ 0.10 m.
2. Contact ≥ 1.5 × height.
3. Toss peak 0.15–1.0 m above contact.
4. Toss side −0.1 to 0.5 m.
5. Knee bend at trophy ≥ 18°.
6. Leg drive (hip rise) ≥ 0.10 m.
7. Elbow at or above shoulder at trophy (≥ −0.05 m).
8. Shoulder tilt ≥ 0.05 m.
9. Toss arm ≥ 150°.
10. Toss hand above head at trophy.
11. Racquet head ≥ 0.10 m below the hand during the drop (racquet detected only).
12. Hitting arm ≥ 150° at contact (player ≥ 150 px only).
13. Head drop ≤ 0.10 m.
14. Feet ≥ 0.05 m off the ground at contact.
15. Landing ≥ 0.10 m inside the baseline.

Phases:
- **Serve:** ready (toss start − 8 frames), trophy (deepest knee bend in the 1.3 s before contact), contact, finish (+0.5 s).
- **Forehand:** ready (−0.8 s), backswing (most turned), contact, follow-through (+0.35 s).

**Levels** (`serve_analysis/levels.py`)
- `FH_LADDER` / `SERVE_LADDER`: per check `(direction, poor, beginner, intermediate, club, excellent, unit[, ideal])`; direction `min` (higher is better), `max` (lower is better) or `band` (distance from `ideal`).
- `FOREHAND` / `SERVE`: which checks each level judges, in priority order, with level-specific tips (Beginner wording is plain).
- `judge(mode, m, level)` → `RuleResult`s with `status` (meets the level's rung?), `reached` (highest rung met), `credit` and `ladder` text.
- `credit`: piecewise linear, 0 at poor → 0.4 (`PASS_CREDIT`) at the level's rung → 1.0 at excellent.
- `score` = 100 × (Σ w·credit / Σ w) ^ 1.5 (`STRICTNESS`), w from 2 (first check) to 1 (last), unmeasurable checks left out.
- `_report` judges all three levels and stores them under `levels` (rules, suggestions, score, coach_text), A language-model coach is called only for the chosen level; the other two use the template.
- CLI `--level beginner|intermediate|club`; API `start?level=`.

**Advanced joint analysis** (`serve_analysis/joints.py`, Club level; some in Intermediate)
- Forehand (`forehand_extras`):
  - `coil_sep`: hip width ratio minus shoulder width ratio at the backswing (each against its widest frame).
  - `chain_lead_s`: time from the hips' fastest opening (d/dt hip width) to the hitting wrist's peak speed.
  - `spine_change_deg`: change in trunk lean (hip mid to shoulder mid) from backswing to contact.
  - `head_level_m`: 5th-95th percentile range of nose height from backswing to contact + 0.1 s.
  - `split_step_m`: largest simultaneous rise of both ankles 1.3-0.35 s before contact.
  - `weight_transfer_m`: hip-centre shift from backswing to contact, in the direction the ball leaves.
  - `left_arm_tuck_m`: left wrist to chest distance at contact.
  - `knee_flex_contact_deg`: 180 minus the mean knee angle at contact.
- Serve (`serve_extras`):
  - `trophy_lean_deg`: trunk lean at the trophy.
  - `chain_lead_s`: knee-extension speed peak to the hitting hand's upward speed peak.
  - `body_line_deg`: mid-ankle to hitting-wrist line against vertical at contact.
  - `contact_tilt_m`: hitting shoulder above the tossing shoulder at contact.
  - `toss_arm_down_m`: tossing wrist below its shoulder at contact.
- Gates: `MIN_PX = 80` (stature), `MIN_CONF = 0.5` per joint, and `FINE_PX = 150` for shoulder-height differences (also applied to the trophy `shoulder_tilt_m`). Below a gate the value is `None`, so the check is "not measurable" and is left out of the score.
- Ladders for these checks are in `levels.py` with the others.

**Forehand vs backhand** (`forehand.facing_sign`, `forehand.stroke_side`)
- `facing_sign`: median sign of (right shoulder x − left shoulder x) over the clip, so the few frames where the pose model swaps sides (often during a backhand) do not matter.
- `stroke_side`: (contact x − hip centre x) × facing ≥ 0 → forehand; without a contact point, the hitting wrist 0.25 s before contact decides.
- Forehand mode keeps `forehand` strokes only; `result.other_side_skipped` counts the rest. A future backhand mode only has to set `wanted = "backhand"`.

## 6. Stage 5 — Coach (`local_llm/coach.py`)
- **`template`:** works offline. Reports **every** check of the chosen level, with the score: all good ones, every fix in priority order, and what could not be measured. Beginner: "Well done / Try this next time" in plain words; Club: each fix with its measured value and target.
- The LLM prompt asks for the same coverage, reworded as feel cues.
- **`openai`:** any OpenAI-compatible server (oMLX on a Mac, llama.cpp or vLLM on the edge box). Set `COACH_BASE_URL`, `COACH_MODEL` and `COACH_API_KEY`.
- **`anthropic`:** the Claude Messages API, with `ANTHROPIC_API_KEY` and `COACH_MODEL`.
- Plain `urllib`, no SDK. Any error falls back to the template.

## 7. Configuration (`config.yaml`)
```yaml
pipeline:
  video:    {max_width: 1280, max_fps: 120, keep_frames: true}
  detector: {model_path: models/yolox_tiny.onnx, every_frames: 3, max_hz: 10}
  pose:     {model_path: models/pose_landmarker.task, max_hz: 60}   # or pose_landmarker_heavy.task
coach:      {backend: template, base_url: http://127.0.0.1:8000/v1, model: "", api_key: "", timeout_s: 60}
```
UI and CLI parameters:
- `--mode serve|forehand`
- `--height` (player height in metres)
- `--camera-side far|near`
- Manual baseline: UI only

## 8. Edge deployment (ASUS UGen300 + Android)
- **Split:**
  - UGen300 / Hailo NPU: YOLOX player and racquet detection, pose model, optional small LLM.
  - Android phone: capture, frames, ball tracking, baseline, coaching logic, results screen.
- **Steps:**
  1. Export YOLOX-tiny to HEF with the Hailo Dataflow Compiler (it is in the Hailo Model Zoo).
  2. Choose a pose model with a commercial licence that compiles for Hailo, and benchmark it.
  3. Stream frames over USB-C.
- **Time today** (2-core cloud CPU, forehand clip, 1,818 frames at 120 fps, 1280×720): about 200 s end to end - frames 34 s, player detection 10 s, skeleton 25 s, ball candidates 46 s, strokes and racquet 47 s, pictures and GIFs 26 s (it was 270-390 s before the speed-ups; the detect + skeleton + ball pass went from 198 s to 80 s). Coaching logic and the template take well under 1 ms per stroke.
- **Budget knob from the original design:** run pose every Nth frame and let the smoothing bridge the gaps.
- **Live mode:** process frames as they arrive. The original `HitDetector` pattern (fire once per swing, re-arm after the hand slows) is the template for streaming stroke detection.

## 9. Testing
```bash
python -m pytest -q        # 38 tests (levels ladder, strict score, joint analysis, coach wording, backhand filter): Kalman, pose mapping, toss RANSAC + contact midpoint, rules, rulebook coverage
```

## 10. Troubleshooting
- **"Pose model not found" / "yolox_tiny.onnx missing":** run `python tools/download_models.py`.
- **"mediapipe is not installed" / "onnxruntime":** run `pip install -r requirements.txt`.
- **No serve found:** check that the whole server and the toss are in view. If it's a groundstroke clip, use Forehand mode.
- **No forehand found:**
  - On low-res footage the ball loses its colour; it then falls back to motion-only tracking.
  - Film in HD, 60–120 fps, with the player at least a third of the frame height.
- **Many checks "unknown":**
  - Arm checks need a player at least 150 px tall.
  - The bounce must be visible: not hidden by the player, and not right at the feet (yellow shoes are masked out).
  - Foot-fault and landing checks need the baseline (click it in the UI).
- **Slow on 120 fps video:** lower `pipeline.pose.max_hz` (e.g. 30) or `pipeline.video.max_fps` (60); on a Mac, `pip install onnxruntime` with CoreML lets the detector use the GPU / Neural Engine; on the product, detection and pose move to the Hailo NPU.
