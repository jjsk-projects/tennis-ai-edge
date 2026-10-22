// Serve Lab - ASUS Hackathon deck - McKinsey-style, colours from the logo (charcoal + leaf green)
const pptxgen = require("pptxgenjs");
const React = require("react");
const ReactDOMServer = require("react-dom/server");
const sharp = require("sharp");
const fa = require("react-icons/fa");
const fs = require("fs");
const { applyTheme } = require("/root/.claude/skills/synced/47ae7250-da63-4c8a-9fbf-832e2f029f9f_1dc3fcad-99e8-40be-9857-1391f9704814/pptx/scripts/apply_theme.js");

const THEME = {
  name: "Serve Lab",
  headFontFace: "Cambria",
  bodyFontFace: "Arial",
  colors: {
    dk1: "1F211C", lt1: "FFFFFF", dk2: "373A31", lt2: "F1F2EE",
    accent1: "79BA00", accent2: "373A31", accent3: "8A8F80", accent4: "C6E39A",
    accent5: "3E6B00", accent6: "B9BCB2", hlink: "3E6B00", folHlink: "5E6355",
  },
};
const HEX = THEME.colors;

async function icon(Comp, color, size = 256) {
  const svg = ReactDOMServer.renderToStaticMarkup(React.createElement(Comp, { color: "#" + color, size: String(size) }));
  const buf = await sharp(Buffer.from(svg)).png().toBuffer();
  return "image/png;base64," + buf.toString("base64");
}
const img = (p) => "image/" + (p.endsWith(".png") ? "png" : "jpeg") + ";base64," + fs.readFileSync(p).toString("base64");

(async () => {
  const pres = new pptxgen();
  pres.layout = "LAYOUT_WIDE"; // 13.333 x 7.5
  pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
  pres.title = "Serve Lab - ASUS Hackathon";
  pres.company = "Serve Lab";
  const C = pres.SchemeColor;
  const LOGO = img("img/logo.png");

  // ---------------- layouts ----------------
  pres.defineSlideMaster({
    title: "TITLE_DARK",
    background: { color: HEX.dk2 },
    objects: [
      { placeholder: { options: { name: "title", type: "title", x: 0.8, y: 2.35, w: 9.6, h: 1.9,
          fontFace: THEME.headFontFace, fontSize: 40, bold: true, color: C.background1, valign: "bottom", align: "left", margin: 0 }, text: "" } },
      { placeholder: { options: { name: "body", type: "body", x: 0.8, y: 4.45, w: 9.6, h: 1.0,
          fontSize: 18, color: C.accent4, valign: "top", align: "left", margin: 0 }, text: "" } },
    ],
  });
  pres.defineSlideMaster({
    title: "CONTENT",
    background: { color: HEX.lt1 },
    margin: [0.5, 0.6, 0.75, 0.6],
    objects: [
      { placeholder: { options: { name: "tracker", type: "body", x: 0.6, y: 0.32, w: 9, h: 0.3,
          fontSize: 11, bold: true, color: C.accent1, charSpacing: 1, margin: 0, valign: "top" }, text: "" } },
      { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 0.66, w: 11.4, h: 1.05,
          fontFace: THEME.headFontFace, fontSize: 26, bold: false, color: C.text2, valign: "top", align: "left", margin: 0 }, text: "" } },
      { placeholder: { options: { name: "source", type: "body", x: 0.6, y: 6.92, w: 10.6, h: 0.32,
          fontSize: 10, color: C.accent3, valign: "top", margin: 0 }, text: "" } },
      { image: { x: 12.38, y: 0.28, w: 0.42, h: 0.42, data: LOGO } },
    ],
    slideNumber: { x: 12.3, y: 6.92, w: 0.5, h: 0.3, fontSize: 10, color: HEX.accent3, align: "right" },
  });

  const content = (section, tracker, title, source) => {
    const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: section });
    s.addText(tracker.toUpperCase(), { placeholder: "tracker" });
    s.addText(title, { placeholder: "title" });
    if (source) s.addText(source, { placeholder: "source" });
    return s;
  };
  const txt = (s, text, o) => s.addText(text, Object.assign({ isTextBox: true, margin: 0, fontSize: 14, color: C.text1, valign: "top" }, o));

  // ===== 10-slide version for Jefferson + Ben. Numbers come from data.json (pipeline results) =====
  const D = JSON.parse(fs.readFileSync("data.json", "utf8"));
  const who = (name, note) => `Presenter: ${name}. ${note}`;
  const bullets = (sl, items, o) => sl.addText(items.map((t, j) => ({ text: t, options: { bullet: true, breakLine: j < items.length - 1 } })),
    Object.assign({ fontSize: 14, color: C.text1, paraSpaceAfter: 6, valign: "top", isTextBox: true, margin: 0 }, o));

  // ---------------- 1. title ----------------
  pres.addSection({ title: "Opening" });
  let s = pres.addSlide({ masterName: "TITLE_DARK", sectionTitle: "Opening" });
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.8, y: 0.8, w: 1.1, h: 1.1, rectRadius: 0.15, fill: { color: C.background1 }, line: { type: "none" }, objectName: "logo tile" });
  s.addImage({ data: LOGO, x: 0.9, y: 0.9, w: 0.9, h: 0.9, objectName: "logo" });
  s.addText("Serve Lab: instant tennis coaching, computed on the court", { placeholder: "title" });
  s.addText("ASUS Hackathon 2026  |  Android phone + ASUS UGen300 (Hailo NPU)  |  Jefferson & Ben", { placeholder: "body" });
  s.addNotes(who("Jefferson", "A phone films a serve or forehand; the AI runs at the edge - phone plus ASUS UGen300 - and coaches the player right there on court, with no cloud. Jefferson covers the product and the forehand; Ben covers the serve, the demo and edge readiness."));

  // ---------------- 2. problem + why edge ----------------
  pres.addSection({ title: "Problem" });
  s = content("Problem", "Problem and why edge",
    "Players cannot see their own technique, and the fix has to be instant, private and offline - a natural edge problem",
    "Our hypothesis, to be validated with players and coaches");
  const probs = [
    [fa.FaEyeSlash, "Invisible", "Foot order, toss height and the hitting point happen in a fraction of a second."],
    [fa.FaClock, "Delayed", "Video review happens after the session, when the feel of the stroke is gone."],
    [fa.FaUserTie, "Coach-dependent", "Frame-by-frame analysis needs a coach's time for every stroke."],
  ];
  txt(s, "The problem", { x: 0.6, y: 1.9, w: 6.2, h: 0.4, fontSize: 17, bold: true, color: C.accent5 });
  for (let i = 0; i < 3; i++) {
    const y = 2.45 + i * 1.35;
    s.addShape(pres.shapes.OVAL, { x: 0.6, y, w: 0.75, h: 0.75, fill: { color: C.text2 }, line: { type: "none" } });
    s.addImage({ data: await icon(probs[i][0], HEX.accent1), x: 0.78, y: y + 0.18, w: 0.39, h: 0.39 });
    txt(s, probs[i][1], { x: 1.6, y: y - 0.02, w: 5.2, h: 0.4, fontSize: 17, bold: true, color: C.text2 });
    txt(s, probs[i][2], { x: 1.6, y: y + 0.38, w: 5.2, h: 0.75, fontSize: 14 });
  }
  s.addShape(pres.shapes.RECTANGLE, { x: 7.3, y: 1.9, w: 5.4, h: 4.6, fill: { color: C.text2 }, line: { type: "none" }, objectName: "why edge" });
  txt(s, "Why edge computing", { x: 7.6, y: 2.1, w: 4.8, h: 0.45, fontSize: 18, bold: true, color: C.accent1, fontFace: THEME.headFontFace });
  const why = [["Offline", "Courts often have weak or no Wi-Fi; every step runs on the phone and the UGen300."],
    ["Private", "Video of players (often juniors) never leaves the court."],
    ["Instant", "Feedback between points, while the player can still feel the stroke."]];
  why.forEach(([k, v], i) => {
    const y = 2.75 + i * 1.2;
    txt(s, k, { x: 7.6, y, w: 4.8, h: 0.38, fontSize: 16, bold: true, color: C.background1 });
    txt(s, v, { x: 7.6, y: y + 0.38, w: 4.8, h: 0.75, fontSize: 14, color: C.accent4 });
  });
  s.addNotes(who("Jefferson", "Frame the problem as a hypothesis; the edge is not a gimmick here - network, privacy and speed all point to on-device processing."));

  // ---------------- 3. solution ----------------
  pres.addSection({ title: "Solution" });
  s = content("Solution", "Solution and edge architecture",
    "Five stages turn a phone clip into coaching; the phone does capture and logic, the UGen300's NPU runs the neural models",
    "Hackathon demo runs the same pipeline as a web app on a laptop; YOLOX is in the Hailo Model Zoo, the pose model still has to be converted and benchmarked");
  const steps = [
    [fa.FaFilm, "1 Frames", "Video split into one image per frame (120 per second)", "Phone"],
    [fa.FaRunning, "2 Skeleton", "Player box + 33-point skeleton on a zoomed crop", "UGen300"],
    [fa.FaBaseballBall, "3 Ball", "Racquet, full-screen ball path, bounce, hitting point", "UGen300 + phone"],
    [fa.FaClipboardCheck, "4 Judge", `${D.n_fh_rules} forehand / ${D.n_serve_rules} serve coaching checks`, "Phone"],
    [fa.FaCommentDots, "5 Coach", "Every result as plain-language feedback", "Phone or UGen300"],
  ];
  for (let i = 0; i < 5; i++) {
    const x = 0.6 + i * 2.44;
    s.addShape(i === 0 ? pres.shapes.PENTAGON : pres.shapes.CHEVRON, { x, y: 2.0, w: 2.4, h: 1.0, fill: { color: i === 4 ? C.accent1 : C.text2 }, line: { type: "none" }, objectName: `stage ${i + 1}` });
    s.addImage({ data: await icon(steps[i][0], i === 4 ? HEX.dk2 : HEX.accent1), x: x + (i ? 0.5 : 0.25), y: 2.3, w: 0.4, h: 0.4 });
    txt(s, steps[i][1], { x: x + (i ? 0.98 : 0.73), y: 2.2, w: 1.3, h: 0.6, fontSize: 15, bold: true, color: i === 4 ? C.text2 : C.background1, valign: "middle" });
    txt(s, steps[i][2], { x: x + 0.1, y: 3.2, w: 2.25, h: 0.95, fontSize: 13 });
    txt(s, "Runs on: " + steps[i][3], { x: x + 0.1, y: 4.15, w: 2.25, h: 0.35, fontSize: 12, bold: true, color: C.accent5 });
  }
  const arch = [["Android phone", "120 fps camera, frame splitting, ball maths (OpenCV), coaching logic, results screen"],
    ["ASUS UGen300 (Hailo NPU)", "Player and racquet detection (YOLOX), skeleton model; optional small local LLM coach"]];
  arch.forEach(([k, v], i) => {
    const x = 0.6 + i * 6.15;
    s.addShape(pres.shapes.RECTANGLE, { x, y: 4.8, w: 5.95, h: 1.7, fill: { color: i ? C.text2 : C.background2 }, line: { type: "none" }, objectName: k });
    txt(s, k, { x: x + 0.3, y: 4.95, w: 5.4, h: 0.4, fontSize: 16, bold: true, color: i ? C.accent1 : C.text2 });
    txt(s, v, { x: x + 0.3, y: 5.4, w: 5.4, h: 1.0, fontSize: 14, color: i ? C.background1 : C.text1 });
  });
  s.addNotes(who("Jefferson", "Frames go from the phone to the UGen300 over USB-C; boxes and skeletons come back; the light logic runs on the phone."));

  // ---------------- 4. stages 1-3 ----------------
  s = content("Solution", "Stages 1-3: see the stroke",
    "Every frame is kept, the skeleton is fitted on a zoom, and the ball is tracked across the whole screen to find the bounce and the hitting point",
    `Forehand ${D.fh_pick} of the 120 fps clip. Skeleton: MediaPipe BlazePose; player and racquet: YOLOX-tiny (both Apache-2.0)`);
  s.addImage({ data: img("img/fh_traj.jpg"), x: 0.6, y: 1.9, w: 7.2, h: 4.05, objectName: "trajectory" });
  txt(s, "Full-frame ball path: incoming ball, bounce (cross), hitting point (star) and the ball leaving (arrow)", { x: 0.6, y: 6.0, w: 7.2, h: 0.5, fontSize: 11, color: C.accent3 });
  const st = [["1 Frames", "120 images per second, written once to disk, so a 1080p clip never fills memory."],
    ["2 Skeleton", "33 joints on a zoomed crop, smoothed over ~33 ms and converted to metres from the player's height."],
    ["3 Ball and racquet", "Yellow moving blobs anywhere on screen are chained frame to frame; the hitting point is the midpoint of the two frames closest to the racquet head."]];
  st.forEach(([k, v], i) => {
    const y = 1.95 + i * 1.45;
    txt(s, k, { x: 8.2, y, w: 4.5, h: 0.4, fontSize: 16, bold: true, color: C.accent5 });
    txt(s, v, { x: 8.2, y: y + 0.4, w: 4.5, h: 1.0, fontSize: 13 });
  });
  s.addNotes(who("Jefferson", "The ball is now searched on the full screen, which is what makes the bounce visible. A stroke only counts when the ball is followed out of a fast-moving racquet."));

  // ---------------- 5. stages 4-5 ----------------
  s = content("Solution", "Stages 4-5: judge and coach",
    `Plain coaching logic - ${D.n_fh_rules} forehand and ${D.n_serve_rules} serve checks - and the coach reports every one of them`,
    "Targets are general club-coaching ranges for a right-handed player; a check shows 'not measurable' when the clip cannot support it");
  const top = [
    ["1", "Hitting point above your head", "Behind the baseline: move forward. Inside the court: leave no-man's land and let the ball drop."],
    ["2", "Ball bounced too far in front", "Ball took longer than 0.8 s from bounce to racquet: move forward to meet it."],
    ["3", "Not sideways in preparation", "Turn shoulders and hips sideways before the swing."],
  ];
  txt(s, "Top-priority forehand rules", { x: 0.6, y: 1.9, w: 6.0, h: 0.4, fontSize: 16, bold: true, color: C.accent5 });
  top.forEach(([n, k, v], i) => {
    const y = 2.4 + i * 1.1;
    s.addShape(pres.shapes.OVAL, { x: 0.6, y, w: 0.48, h: 0.48, fill: { color: C.accent1 }, line: { type: "none" } });
    txt(s, n, { x: 0.6, y, w: 0.48, h: 0.48, fontSize: 14, bold: true, color: C.text2, align: "center", valign: "middle" });
    txt(s, k, { x: 1.25, y: y - 0.02, w: 5.4, h: 0.36, fontSize: 14, bold: true, color: C.text2 });
    txt(s, v, { x: 1.25, y: y + 0.34, w: 5.4, h: 0.7, fontSize: 12 });
  });
  txt(s, [{ text: "Also checked: ", options: { bold: true, color: C.accent5 } },
    { text: "contact in front, left foot in front, contact height, spacing, arm, head still, early preparation, left arm, knees, base, racquet drop, low-to-high, leg drive, finish, balance." }],
    { x: 0.6, y: 5.75, w: 6.0, h: 0.85, fontSize: 12 });
  s.addShape(pres.shapes.RECTANGLE, { x: 7.0, y: 1.9, w: 5.7, h: 4.7, fill: { color: C.accent4 }, transparency: 55, line: { type: "none" }, objectName: "coach note" });
  s.addImage({ data: await icon(fa.FaQuoteLeft, HEX.accent5), x: 7.25, y: 2.1, w: 0.4, h: 0.4 });
  txt(s, `Coach, forehand ${D.coach_pick}`, { x: 7.8, y: 2.1, w: 4.7, h: 0.4, fontSize: 14, bold: true, color: C.accent5 });
  txt(s, D.coach_text, { x: 7.25, y: 2.65, w: 5.25, h: 3.85, fontSize: 11, color: C.text1, fit: "shrink" });
  s.addNotes(who("Jefferson", "No machine learning in the judgement: each check is an explainable rule on a measured number. The coach lists everything: what is good, every fix in priority order, and what could not be measured. The template runs offline; a local LLM or Claude can reword it."));

  // ---------------- 6. serve result (Ben) ----------------
  pres.addSection({ title: "Results" });
  s = content("Results", "Demo result: serve",
    "Serve: the toss, the exact contact moment and the heights that matter, from a low-resolution phone clip",
    "Clip IMG_3025: phone, 30 fps, player about 86 px tall. Heights scaled from the player's entered height (1.80 m); ball speed is a lower bound");
  s.addImage({ data: img("img/serve_keys.jpg"), x: 0.6, y: 1.85, w: 9.0, h: 3.0, objectName: "serve key moments" });
  D.serve_kpis.forEach(([n, l], i) => {
    const y = 1.85 + i * 0.78;
    txt(s, n, { x: 9.95, y, w: 2.8, h: 0.42, fontSize: 22, bold: true, color: C.accent5, fontFace: THEME.headFontFace });
    txt(s, l, { x: 9.95, y: y + 0.4, w: 2.8, h: 0.32, fontSize: 12 });
  });
  txt(s, [{ text: "What the coach says: ", options: { bold: true, color: C.accent5 } }, { text: D.serve_readout }],
    { x: 0.6, y: 5.1, w: 12.1, h: 1.4, fontSize: 14 });
  s.addNotes(who("Ben", "Yellow = toss fitted to a gravity curve; star = contact (midpoint between the last toss frame and the first frame of the hit ball); orange = ball leaving. Contact frame 50.5 matches the manual frame-by-frame check."));

  // ---------------- 7. forehand result ----------------
  s = content("Results", "Demo result: forehand",
    `Forehand: ${D.n_fh} strokes found in one 15 s clip, each with racquet, hitting point and a full list of checks`,
    "Clip forehand_120fps.mp4: 1080p at 120 fps. A stroke counts only when the ball leaves a fast-moving racquet in the swing direction; unclear strokes are skipped");
  s.addImage({ data: img("img/fh_keys.jpg"), x: 0.6, y: 1.9, w: 7.6, h: 2.54, objectName: "forehand key moments" });
  txt(s, `Forehand ${D.fh_pick}: white = racquet, star = hitting point, yellow = incoming ball, orange = ball leaving`, { x: 0.6, y: 4.52, w: 7.6, h: 0.3, fontSize: 11, color: C.accent3 });
  txt(s, [{ text: "Read-out: ", options: { bold: true, color: C.accent5 } }, { text: D.fh_readout }], { x: 0.6, y: 5.0, w: 7.6, h: 1.4, fontSize: 14 });
  const fh = [["#", "Hitting point", "Bounce to hit", "Fixes"]].concat(D.fh_rows);
  s.addTable(fh.map((r, i) => r.map((c, j) => ({ text: String(c), options: {
      bold: i === 0, fontSize: 12, color: i === 0 ? C.background1 : C.text1,
      fill: { color: i === 0 ? C.text2 : (i % 2 ? C.background1 : C.background2) }, valign: "middle", margin: [0.04, 0.08, 0.04, 0.08] } }))),
    { x: 8.6, y: 1.9, w: 4.1, colW: [0.4, 1.2, 1.2, 1.3], rowH: 0.45, border: { type: "none" }, objectName: "forehand table" });
  s.addNotes(who("Jefferson", "Each row is one forehand; 'Fixes' counts the failed checks, all of which the coach reports in priority order."));

  // ---------------- 8. demo experience (Ben) ----------------
  s = content("Results", "Demo experience",
    "The player gets the answer first - the coach's note - and the visual evidence underneath",
    "Screenshot of the Serve Lab web demo (serve mode, clip IMG_3025)");
  s.addImage({ data: img("img/ui.png"), x: 0.6, y: 1.85, w: 6.3, h: 4.75, objectName: "web demo screenshot", sizing: { type: "contain", w: 6.3, h: 4.75 } });
  const outs = [["Coach's note", "everything that is good, every fix in priority order, what could not be measured"],
    ["Ball trajectory", "full-screen path: incoming ball, bounce, hitting point, ball leaving"],
    ["Slow motion", "annotated sequence with skeleton and racquet, key moments, joint heights"],
    ["Checklist", "every check: good, improve or not measurable, with the measured value"]];
  outs.forEach(([k, v], i) => {
    const y = 1.95 + i * 1.15;
    s.addShape(pres.shapes.OVAL, { x: 7.3, y, w: 0.55, h: 0.55, fill: { color: C.accent1 }, line: { type: "none" } });
    txt(s, String(i + 1), { x: 7.3, y, w: 0.55, h: 0.55, fontSize: 16, bold: true, color: C.text2, align: "center", valign: "middle" });
    txt(s, k, { x: 8.05, y: y - 0.02, w: 4.6, h: 0.4, fontSize: 17, bold: true, color: C.text2 });
    txt(s, v, { x: 8.05, y: y + 0.38, w: 4.6, h: 0.65, fontSize: 13 });
  });
  s.addNotes(who("Ben", `Live demo: python app/server.py, then analyse IMG_3025.mov (serve, about ${D.t_serve} s) or forehand_120fps.mp4 (about ${D.t_fh_min} min on a laptop CPU).`));

  // ---------------- 9. edge readiness (Ben) ----------------
  pres.addSection({ title: "Readiness" });
  s = content("Readiness", "Edge readiness",
    "The heavy part is exactly what the NPU accelerates, and the whole stack is licensed for commercial use",
    `CPU time on a 2-core cloud machine, forehand clip (${D.frames} frames, 1280x720 at 120 fps). Not yet benchmarked on the UGen300`);
  s.addChart(pres.charts.BAR, [{ name: "seconds", labels: D.cost_labels, values: D.cost_values }], {
    x: 0.6, y: 1.85, w: 6.4, h: 4.6, barDir: "bar", showTitle: true, title: `Where the ${D.t_fh_total} s go today (CPU only)`,
    titleFontSize: 13, titleColor: HEX.dk2, titleFontFace: "+mn-lt",
    chartColors: [HEX.accent1], showValue: true, dataLabelPosition: "outEnd", dataLabelFontSize: 12,
    dataLabelColor: HEX.dk1, dataLabelFontFace: "+mn-lt", dataLabelFormatCode: '0" s"',
    catAxisLabelFontSize: 12, catAxisLabelColor: HEX.dk1, catAxisLabelFontFace: "+mn-lt", catAxisOrientation: "maxMin",
    valAxisHidden: true, valAxisMinVal: 0, valGridLine: { style: "none" }, catGridLine: { style: "none" },
    showLegend: false, barGapWidthPct: 60, objectName: "compute chart" });
  const rd = [["Speed-ups already in", "Detector at 10 Hz, skeleton at 60 Hz with gaps filled, shake estimate on a small copy: the forehand clip went from " + D.t_before + " s to " + D.t_fh_total + " s."],
    ["To the NPU", "Detection, racquet and skeleton are standard networks - the work the Hailo NPU is built for."],
    ["Commercially clean", "OpenCV, MediaPipe, YOLOX, ONNX Runtime: Apache / BSD / MIT. Ultralytics YOLO (AGPL) deliberately avoided."]];
  rd.forEach(([k, v], i) => {
    const y = 1.95 + i * 1.5;
    txt(s, k, { x: 7.5, y, w: 5.2, h: 0.4, fontSize: 16, bold: true, color: i === 2 ? C.accent5 : C.text2 });
    txt(s, v, { x: 7.5, y: y + 0.4, w: 5.2, h: 1.0, fontSize: 13 });
  });
  s.addNotes(who("Ben", "Today's numbers are CPU-only. First step at the hackathon: compile YOLOX-tiny for Hailo and measure per-frame latency on the UGen300. On a Mac, ONNX Runtime uses CoreML automatically when available."));

  // ---------------- 10. roadmap + thank you ----------------
  pres.addSection({ title: "Close" });
  s = pres.addSlide({ masterName: "TITLE_DARK", sectionTitle: "Close" });
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.8, y: 0.8, w: 1.1, h: 1.1, rectRadius: 0.15, fill: { color: C.background1 }, line: { type: "none" }, objectName: "logo tile" });
  s.addImage({ data: LOGO, x: 0.9, y: 0.9, w: 0.9, h: 0.9, objectName: "logo" });
  s.addText("Next: from web demo to an Android app with the UGen300 doing the heavy lifting", { placeholder: "title", fontSize: 32 });
  const ph = [["Now", "Serve and forehand working end to end; web demo"], ["Next", "YOLOX + pose on the Hailo NPU; Android app with filming guide"], ["Then", "Coaches validate thresholds; backhand; club trials"]];
  ph.forEach(([k, v], i) => {
    const x = 0.8 + i * 3.95;
    txt(s, k, { x, y: 4.6, w: 3.7, h: 0.4, fontSize: 18, bold: true, color: C.accent1, fontFace: THEME.headFontFace });
    txt(s, v, { x, y: 5.05, w: 3.7, h: 0.9, fontSize: 14, color: C.background1 });
  });
  txt(s, "Thank you - Jefferson & Ben", { x: 0.8, y: 6.3, w: 11, h: 0.5, fontSize: 18, bold: true, color: C.accent4 });
  s.addNotes(who("Jefferson", "Close and hand over to the live demo if time allows."));

  await pres.writeFile({ fileName: "ServeLab_pitch.pptx" });
  await applyTheme("ServeLab_pitch.pptx", THEME);
  console.log("written");
})();
