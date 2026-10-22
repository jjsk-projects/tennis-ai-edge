# Progress log — Serve Lab (tennis-ai-edge)

**Last updated:** 2026-10-12 (morning)

**Status:**
- The MVP works end to end in both modes (serve, forehand), in the web UI and the CLI.
- Forehand: ball searched on the whole screen; bounce and full ball trajectory drawn; strokes confirmed by the racquet swing.
- Coach reports every check (good, all fixes in priority order, not measurable).
- About 1.5x faster (forehand clip 3.3 min on a 2-core cloud CPU, was 4.5-6.5 min).
- 33 unit tests pass.
- Hackathon deck cut to 10 slides for Jefferson and Ben: `pitch/ServeLab_pitch.pptx`.

## Stage status

| Stage | Status | Notes |
|---|---|---|
| 1 Frames | ✅ Done | Every frame written to disk (120 per second for 120 fps), 1280 px wide |
| 2 Skeleton | ✅ Done | YOLOX player box (10 Hz), BlazePose on a zoomed crop (60 Hz, gaps interpolated), smoothing; re-detects when the player is lost |
| 3 Racquet / ball / court | ✅ Working | Forehand: full-screen ball search, chains bridge 0.1 s gaps and the bounce; hitting point = midpoint of the two frames closest to the racquet head; stroke confirmed when the ball leaves in the swing direction. Serve: gravity-parabola toss + contact midpoint |
| 4 Judge | ✅ Done | 18 forehand / 15 serve checks, general club thresholds; bounce judged as time from bounce to contact (≤ 0.8 s) |
| 5 Coach | ✅ Done | Template lists every check; LLM prompt asks for the same coverage |
| Edge port | ⬜ Not started | Needs the UGen300, the Hailo SDK and an Android app |

## Results on the example clips

**IMG_3025** (serve, 30 fps, player about 86 px tall) - about 45 s
- 2 serves, both with ball-based contact; contact frame 50.5 matches a manual check.
- Contact 3.25 m, toss peak 4.23 m, ball ≥ 100 km/h. 10 checks good, 3 fixes (shoulder tilt, toss arm up, landing inside).

**IMG_3322** (serve, 30 fps, player about 65 px tall, windy trees)
- 4 serves found (about 100 s). Serve 1 has a ball-based contact (3.68 m, 2.0 × height); the other 3 fall back to skeleton timing and their ball checks show "not measurable".
- The stricter contact gate (≤ 2.1 × height) now rejects an implausible 4.09 m contact instead of coaching on it.

**forehand_120fps.mp4** (forehand, 1080p, 120 fps, 1,818 frames) - about 200 s
- 4 forehands (contact frames 137.5, 692.5, 1064.5, 1510.5), each with the racquet and a hitting point on the strings. The stroke at 322.5 is a backhand (spotted by Jefferson) and is now recognised and skipped.
- Bounce found for 2 of 4 (0.86 s and 0.55 s before contact); in the other two the ball bounces at the player's feet or is hidden.
- Skipped on purpose: frame 481 (racquet head not tracked well enough to confirm the hit) and shadow swings (215, 1370).

## Log

### 2026-10-18
- **Web demo restyled as "Centre Court" (second pass, after feedback):** no background images; the banner is an illustrated tennis net (dark mesh, white tape along the top, centre strap); every section is drawn as a court (outer lines, tramlines both sides, centre marks); all text in wood tones (walnut, oak, ash; maple on the net) for wooden racquets; crossed wooden-racquet logo. Type: Cormorant Garamond + Libre Franklin (Georgia/Arial offline).

### 2026-10-12 (night)
- **Backhands filtered out of forehand mode:** each stroke is classified by the side of the hips it is hit on; the forehand clip gives 4 forehands, 1 backhand skipped. Filming guide in the UI: phone beside the net post on the forehand side.
- **Player levels:** Beginner (5 checks, plain words), Intermediate (10, adds squat, footwork, preparation) and Club (all 18 forehand / 15 serve checks, values shown). Each level has its own targets and a weighted 0-100 score. Dropdown before analysis and on the results page.
- **Ladder from Beginner to Club:** every check now has Beginner / Intermediate / Club targets on one scale (plus "poor" and "excellent" anchors). The checklist shows the level each value reaches. ("Plays like" was tried and removed at Jefferson's request.)
- **Advanced joint analysis:** 8 new forehand checks (coil, kinetic chain, steady spine, level head, split step, weight transfer, left-arm tuck, knee bend at contact) and 5 serve checks (trophy lean, kinetic chain, body stacked, hitting shoulder up, toss arm down). Club level now judges 26 forehand / 20 serve checks; Intermediate 13 / 11. Fine shoulder checks need a player at least 150 px tall.
- Scores with joint analysis (beginner / intermediate / club): forehands 91/68/56, 95/73/50, 90/65/61, 78/62/52; serve IMG_3025 68/67/62 and 67/63/54. Jefferson's own read: the forehands are intermediate level, the serve club level. On this low-res serve clip (player 85 px) the club fixes are toss height, toss side, trophy lean and landing, and 4 checks are not measurable.
- **Stricter score:** 0.4 credit for just meeting a target, full marks only at "excellent" (well past Club), average raised to the power 1.5.
- Stricter score, before the joint analysis (beginner / intermediate / club): forehands 91/75/68, 95/87/59, 90/72/64, 78/70/51 (was 100/100/92, 100/100/81, 100/88/94, 87/81/70); serve 68/67/61 and 67/62/58 (was 100/93/79 and 100/93/72).
- 38 unit tests pass.

### 2026-10-12 (morning)
- **Ball on the whole screen (forehand):** motion candidates over the full frame; chains up to 1.4 s in / 0.5 s out, bridging 0.1 s gaps and the direction flip at the bounce; bounce fallback when the track starts at court level.
- **Trajectory picture:** `overview.jpg` now shows the full path (ball in, bounce, hitting point with height, ball out, legend); shown in the UI under "Ball trajectory"; GIF starts when the ball comes in and keeps a nearby bounce in view.
- **Stroke check:** `confirms_hit` - the racquet head must be swinging and the ball must leave in the swing direction. The contact pair is the one the ball can be followed into and out of.
- **Bounce rule:** judged as time from bounce to contact (≤ 0.8 s), because the camera angle squeezes distances along the court.
- **Coach:** template and LLM prompt now cover every check.
- **Speed:** detector at 10 Hz, pose at 60 Hz with interpolation, shake estimate once per frame pair on a ≤ 640 px copy, racquet detection only ±0.3 s around swings with a ball (every 2nd frame at 120 fps), memoised colour tests, no images held in memory. ONNX Runtime picks CoreML / CUDA / DirectML when present.
- **Tracking fix:** when the skeleton is lost the detector runs again and re-acquires the player (this had lost frames 204-412 and stroke 322).
- **Serve gate:** contact must be ≤ 2.1 × height (was 2.4).
- **Backhands skipped:** a hit on the player's left side (judged against which way the player faces over the whole clip) is a backhand and is not coached as a forehand.
- **Deck:** 10 slides with presenter split in the notes (Jefferson: 1-5, 7, 10; Ben: 6, 8, 9).

### 2026-10-12 (overnight)
- **Stage 2:** confidence-weighted, centred skeleton smoothing (`pose_model/smoothing.py`), reusing the original PoseTracker idea.
- **Stage 3:**
  - YOLOX racquet detection on a zoomed crop (30 of 35 frames in a test).
  - Racquet drawn on all visuals, with a "hitting point" label.
  - Forehand contact rebuilt: yellow ball blobs off the body, two frames closest to the racquet head, midpoint. The chains bridge up to 6 missing frames (the Kalman idea).
  - A swing counts only if the ball is followed into or out of the racquet. This removed shadow swings, e.g. frame 215.
  - Bounce detection from the incoming chain.
  - Baseline detection now also runs in forehand mode, giving the court position (behind / inside).
- **Stage 4:**
  - Random Forest removed.
  - Forehand rulebook expanded to 18 checks. The user's rules come first:
    1. Contact above the head (advice depends on behind / inside the baseline).
    2. Bounce too far in front means move forward.
    3. Body sideways.
  - Serve rulebook expanded to 15 checks (elbow up, shoulder tilt, toss arm up, racquet drop, head up, drive up, landing).
  - Arm angles are judged only for players ≥ 150 px tall.
- **UI:** checks grouped by phase, "Top fixes in priority order", new measurement labels.
- **Deck:** reframed for the ASUS hackathon (edge: Android + UGen300), organised by the 5 stages, Random Forest removed.
- **Docs:** README, technical-steps and progress fully rewritten.

### 2026-10-12 (evening)
- Forehand mode added. Frames-first processing. Downscaling for 1080p/120 fps.
- First hackathon deck. Pitch images saved to `pitch/`.

### 2026-10-11
- Rebuilt as a serve-feedback MVP: YOLOX player detector (not Ultralytics, which is AGPL), BlazePose on a crop, serve toss RANSAC, contact midpoint, baseline, web demo.
- Fixed bugs in the original code:
  - Kalman `predict(dt)` discarded the new `F`.
  - The COCO mapping put the wrists in the ankle slots.
  - The k-NN classifier was broken (double standardisation, mismatched synthetic features).
  - The "ball" was actually the wrist.

### 2026-10-08 / 09 (original code)
- Kalman filter, pose tracker, hit detector, BlazePose backend, k-NN classifier.
- First run on `IMG_3316.mov`.

## Known issues
1. **Bounce at the player's feet** is masked out (yellow shoes look like balls); hidden bounces are not found.
2. **Forehand recall:** strokes where the racquet head or ball is unclear are skipped (deliberate).
3. **Small players (< 100 px):** arm detail blurs; serve clip B (IMG_3322) is hard.
4. **Thresholds** are general club-coaching ranges and need validating with a coach.
5. **Speeds** are measured in the image plane only (lower bounds).
6. **Not yet tested on the edge device;** all timings so far are CPU-only.

## Answers received (2026-10-12 morning)
1. Bounce: search the full screen and plot the trajectory - done.
2. Thresholds: general thresholds are fine - kept as club-coaching ranges.
3. Output and coach must show all the feedback - done.
4. Skipping missed and wrong strokes is OK.
5. Camera: across the net, facing the player (default `far`).
6. Deck: Jefferson and Ben (serve) present; 10 slides - done.
7. Polish serve and forehand before new modes.
