# Centre Court: Baseline — tennis-ai-edge

On-court tennis coaching from a phone video, built for edge computing on an **Android phone + ASUS UGen300 (Hailo NPU)**.

A serve or forehand video goes in. The app splits it into frames and finds the player, skeleton, racquet, ball, bounce, baseline and hitting point. It then judges the technique with plain coaching logic and writes coach-style feedback with annotated pictures.

**MVP scope:** two modes (serve and forehand), a right-handed player, and one player in view. Today it runs as a web demo on a laptop. The edge build splits the same pipeline between the phone and the UGen300 (see [Edge deployment](#edge-deployment-asus-ugen300--hailo)).

```mermaid
flowchart TD
  A[Phone video - serve or forehand, up to 120 fps] --> S1
  subgraph S1[Stage 1 - Frames]
    F[Split into one JPEG per frame, 1280 px wide]
  end
  S1 --> S2
  subgraph S2[Stage 2 - Player and skeleton]
    B[Player box - YOLOX-tiny, 10x per second] --> C[33-point skeleton - BlazePose on a zoomed crop, 60x per second]
    C --> C2[Confidence-weighted smoothing]
  end
  S2 --> S3
  subgraph S3[Stage 3 - Racquet, ball, court]
    R[Racquet - YOLOX on a zoomed crop]
    D[Ball - motion blobs over the whole screen; colour on HD; gravity parabola for the serve toss]
    H[Hitting point = midpoint of the ball in the two frames around the hit]
    G[Bounce, full ball trajectory, baseline, scale px to metres]
    R --> H
    D --> H
  end
  S3 --> S4
  subgraph S4[Stage 4 - Judge]
    J[Coaching logic: 26 forehand / 20 serve checks (incl. joint analysis); good, improve, unknown; tips in priority order]
  end
  S4 --> S5
  subgraph S5[Stage 5 - Coach]
    K[Template, local LLM, or Claude API]
  end
  S5 --> O[Results: coach note covering every check, ball trajectory, slow-motion GIF, key moments, skeletons, height chart, checklist]
```

## Quick start (macOS / Linux, Python 3.9+)

```bash
cd tennis-ai-edge
source .venv/bin/activate                 # or: python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt           # numpy, opencv, mediapipe, onnxruntime, Pillow
python tools/download_models.py           # only if models/ is missing YOLOX-tiny or BlazePose
python tools/download_models.py --heavy   # optional: BlazePose-Heavy for better arm detail

python app/server.py                      # then open http://127.0.0.1:8080
```

In the browser:
1. Choose **Serve** or **Forehand**.
2. Pick an example clip or drop in your own.
3. Optionally click two points on the baseline.
4. Press **Analyse**.

The command line runs the same pipeline:

```bash
python tools/run_video.py clips/IMG_3025.mov --mode serve --height 1.80
python tools/run_video.py clips/forehand_120fps.mp4 --mode forehand --level intermediate --height 1.80
python -m pytest -q
```

Example clips:
- `IMG_3025.mov` and `IMG_3322.mov`: serves at 30 fps, with a small player.
- `forehand_120fps.mp4`: forehands at 1080p and 120 fps. Copy it into `clips/` yourself; it is too large to sync automatically.

## The five stages

| Stage | What it does | Main code |
|---|---|---|
| 1 Frames | Splits the video into one image per frame (all 120 per second at 120 fps), 1280 px wide, stored in `runs/<id>/frames/`. Every later stage reads from disk, so long videos never fill memory. | `src/pipeline.py`: `extract_frames` |
| 2 Skeleton | YOLOX-tiny finds the player. BlazePose runs on a zoomed crop, because players are often under 150 px tall, and its points are mapped back to full-frame pixels. Joints are then smoothed, an idea reused from the original `PoseTracker`. | `stages/detection`, `stages/pose_model` |
| 3 Racquet, ball, court | The racquet is detected with YOLOX on a zoomed crop. **Serve:** the toss is a RANSAC gravity parabola; contact = midpoint of the last toss position and the first position of the hit ball. **Forehand:** yellow moving blobs anywhere on screen, off the body, chained frame to frame; hitting point = midpoint of the ball in the two frames closest to the racquet head; the incoming chain gives the bounce, and the whole path is drawn on the full frame. A stroke counts only when the ball leaves a fast-moving racquet in the swing direction. Backhands (ball hit on the player's left side) are recognised and skipped. The baseline is found near the feet. | `stages/ball_tracking`, `stages/court_segmentation` |
| 4 Judge | Plain coaching logic, no machine learning. Every check is a rule on a measured number. Failed checks become tips in priority order. A check shows "unknown" when the clip can't support it. | `stages/serve_analysis/forehand.py`, `analysis.py` |
| 5 Coach | A short note in plain words, from the built-in template (works offline), any OpenAI-compatible local LLM, or Claude. Falls back to the template on any error. | `stages/local_llm/coach.py` |

## Player levels: Beginner, Intermediate, Club

Pick the level before analysing, or switch it on the results page; all three are judged on every stroke, so switching is instant.

**Every check has a ladder from Beginner to Club.** For example, knee bend: Beginner ≥ 10°, Intermediate ≥ 15°, Club ≥ 20°. The checklist shows the measured value, the level it reaches (three steps: Beginner → Intermediate → Club) and the ladder.

| Level | Forehand checks judged | Serve checks judged | Coach style |
|---|---|---|---|
| Beginner | 5: contact in front, ball not above your head, turn sideways, finish over the shoulder, stay balanced | 5: no foot fault, toss in front on the racquet side, toss high enough, hit high, watch the ball | Plain words, one simple picture per fix |
| Intermediate | 13: adds bounce distance, squat when getting ready, left foot in front, spacing, early preparation, split step, low-to-high, weight transfer, head level | 11: adds knee bend, elbow up, toss arm up, leg drive, body stacked at contact, landing | Technique named, the level each check is at now |
| Club level | All 26, including the joint analysis below | All 20, including the joint analysis below | Every fix with its measured value and the full ladder |

### Advanced joint analysis (Club level)

These checks follow how the joints move over time (`stages/serve_analysis/joints.py`), not just where they end up:

- **Forehand:**
  - Shoulders coil past the hips at the backswing (hip-shoulder separation).
  - The hips unwind 0.04-0.20 s before the hand peaks (kinetic chain).
  - The spine angle stays steady and the head stays level from backswing to contact.
  - A split step before the ball arrives.
  - Weight moves toward the target.
  - The left arm folds in at contact.
  - The knees are still bent at contact.
- **Serve:**
  - The body leans up and in at the trophy.
  - The legs fire 0.05-0.25 s before the arm (kinetic chain).
  - The body is stacked under the ball at contact (feet-to-hand line within 15° of vertical).
  - The hitting shoulder is up at contact.
  - The toss arm pulls down before contact.

Each is measured only when the skeleton is reliable (player at least 80 px tall, joints seen with confidence ≥ 0.5). Shoulder-height differences of a few centimetres need a player at least 150 px tall. Otherwise the check shows "not measurable" instead of a guess.

**Score (0-100, strict).** Each judged check earns credit: 0 at "poor", 0.4 for just meeting the chosen level's target, and 1.0 only at "excellent" (a strong competitive player, well past Club). Credits are averaged with priority weights (most important check 2, least 1) and the average is raised to the power 1.5. Just meeting every target gives about 25; 70+ needs most checks well past the target; 90+ is close to excellent everywhere. Unmeasurable checks are left out. The same stroke scores lower at a higher level, because its target is higher.

## Forehand mode: forehands only

The filming guide asks for the phone beside the net post on the player's forehand side (the right side for a right-hander), facing the player. In forehand mode every stroke is classified by which side of the hips the ball is hit on (judged against the way the player faces over the whole clip). Backhands are recognised and skipped. The same code will keep only backhands in a future backhand mode.

## Judge: the rulebook (Club level)

### Forehand: the 18 core checks, in priority order (plus 8 joint checks above)

The first three are the coach's key rules.

1. **Hitting point not above your head.**
   - Behind the baseline: move forward and take the ball earlier.
   - Inside the court: get out of no-man's land, step back and let the ball drop.
2. **The ball bounced close enough** (reaches you within 0.8 s of the bounce; time is used because the camera angle squeezes distances along the court). Otherwise, move forward.
3. **Body sideways in preparation.**
4. Contact in front of the body.
5. Left foot in front at contact (the right foot must not be ahead).
6. Contact not too low.
7. Room to the side at contact (0.25–0.9 m).
8. Arm not cramped.
9. Head still.
10. Early preparation.
11. Left arm tracks the ball.
12. Knee bend (squat).
13. Stable base.
14. Racquet drops below the ball.
15. Low-to-high swing (topspin).
16. Legs drive up.
17. Finish over the left shoulder.
18. Balanced finish.

Racquet head speed is measured and shown, but not judged. With the camera facing the player, the swing points at the lens, so the speed seen in the image is far below the real one.

### Serve: the 15 core checks (plus 5 joint checks above)

- No foot fault.
- Contact at full reach (at least 1.5 × your height).
- Toss peaks 0.15–1.0 m above contact.
- Toss on the hitting side.
- Knee bend.
- Leg drive.
- Elbow up at the trophy position.
- Shoulder tilt.
- Toss arm straight.
- Toss arm stays up.
- Racquet drop.
- Hitting arm straight.
- Head up.
- Drive up off the ground.
- Land inside the court.

## What you get per stroke

| Output | What it shows |
|---|---|
| Coach note | Every check: everything that is good, every fix in priority order (with its phase), and what could not be measured. |
| Top fixes | Every failed check as a tip, most important first. |
| Ball trajectory | The full frame with the incoming ball, bounce, hitting point (with its height) and the ball leaving. |
| Animated sequence (GIF) | Skeleton, racquet, incoming ball, bounce, hitting point and the ball leaving; slowed down, pausing on key moments. |
| Key moments | Ready, backswing (trophy for serves), contact and finish screenshots, with a metre ruler and the baseline. |
| Skeleton | The same four moments as clean skeletons plus the racquet, with joint heights in metres. |
| Height chart | Height (m) of the ball, the hitting wrist and the other wrist around contact. |
| Checklist | All checks grouped by phase: good, improve or unknown. |

## Accuracy notes

- **Best footage:** player at least a third of the frame height, a steady camera, 60–120 fps, HD. On HD footage the ball keeps its colour and the racquet is detected in most frames.
- **Arm angles** are only judged when the player is at least 150 px tall; they blur on smaller players.
- **Forehand strokes** only count when the ball is followed into or out of the racquet for at least 5 frames. This removes shadow swings, but can miss a stroke where the ball is barely visible.
- **The bounce** is searched on the whole screen. It is missed when the ball is hidden at the bounce or bounces right at the player's feet (shoes are masked out because yellow shoes look like balls); that check then shows "unknown".
- **Strokes:** unclear strokes are skipped on purpose - better to miss one than to coach a non-stroke.
- **The baseline** is detected automatically near the feet, with about 10 cm tolerance. Click it in the demo for precision.
- **Ball and racquet speeds** are measured in the image plane, so they are lower bounds.

## Edge deployment (ASUS UGen300 / Hailo)

| Runs on | Stages |
|---|---|
| Android phone | Camera capture (120 fps), frame splitting, ball tracking, baseline, coaching logic, coach template, results screen |
| UGen300 (Hailo NPU) | YOLOX player and racquet detection, pose model; optionally a small local LLM coach |

The plan:
1. Compile YOLOX-tiny for Hailo; it is in the Hailo Model Zoo.
2. Convert and benchmark a pose model.
3. Stream frames over USB-C, and keep the phone for light CPU work.
4. If the NPU can't keep up with 120 fps, run pose on every Nth frame and let the smoothing bridge the gaps. The original pipeline used every 3rd frame.

Measured time today, CPU only: the 15 s forehand clip (1,818 frames at 120 fps) takes about 3.3 min on a 2-core cloud CPU (it was 4.5-6.5 min before the speed-ups), the serve clip IMG_3025 about 45 s. The neural models (detection, racquet, skeleton) are the part the Hailo NPU takes over. On a Mac, ONNX Runtime uses CoreML (GPU / Neural Engine) for the detector automatically when available.

## Project layout

```
app/server.py, app/static/index.html     web demo (standard-library HTTP server + one HTML page)
src/pipeline.py                          orchestrates stages 1-5 and the visuals per stroke
src/stages/detection/                    YOLOX player and racquet detector (ONNX Runtime)
src/stages/pose_model/                   BlazePose on a zoomed crop, COCO-17 conversion, smoothing
src/stages/ball_tracking/                motion candidates, serve toss RANSAC, racquet contact, bounce, Kalman filter
src/stages/court_segmentation/           baseline detection
src/stages/serve_analysis/               serve and forehand measurements and coaching rules
src/stages/local_llm/coach.py            template / OpenAI-compatible / Anthropic coach
src/stages/visualize/render.py           keyframes, skeletons, racquet, GIF, overview
tools/                                   run_video, download_models, omllab (local oMLX helper)
tests/                                   unit tests (toss and contact, rules, Kalman, pose mapping)
pitch/                                   ASUS hackathon deck (ServeLab_pitch.pptx + build_deck.js) and demo images
```

## Coach backends

```bash
COACH_BACKEND=openai COACH_BASE_URL=http://127.0.0.1:8000/v1 COACH_MODEL=<model> COACH_API_KEY=<key> python app/server.py
COACH_BACKEND=anthropic ANTHROPIC_API_KEY=... COACH_MODEL=claude-sonnet-5-5 python app/server.py
```

## Licences

Every component uses a permissive licence that allows closed-source commercial use; see [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md). Ultralytics YOLO was deliberately not used, because it is AGPL-3.0.
