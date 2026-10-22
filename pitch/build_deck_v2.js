// Centre Court: Baseline - ASUS UGen AI League pitch, styled like the web demo:
// a net for the banner slides, court lines framing every content slide, wood-toned words.
const pptxgen = require("pptxgenjs");
const fs = require("fs");
const { applyTheme } = require("/root/.claude/skills/synced/47ae7250-da63-4c8a-9fbf-832e2f029f9f_1dc3fcad-99e8-40be-9857-1391f9704814/pptx/scripts/apply_theme.js");

const THEME = {
  name: "Centre Court",
  headFontFace: "Cormorant Garamond",
  bodyFontFace: "Libre Franklin",
  colors: {
    dk1: "4A3121",   // walnut - body text
    lt1: "FCFBF6",   // ivory card
    dk2: "1F2A24",   // net
    lt2: "F5F4EC",   // paper
    accent1: "3F6B3A", // court lines
    accent2: "7A5233", // oak - headings, numbers
    accent3: "8C6B4F", // ash - secondary text
    accent4: "E2C79A", // maple - words on the net
    accent5: "A06A1C", // amber - "improve"
    accent6: "DCD8C8", // hairline
    hlink: "3F6B3A", folHlink: "7A5233",
  },
};
const H = THEME.colors;
const img = (p) => "image/" + (p.endsWith(".png") ? "png" : "jpeg") + ";base64," + fs.readFileSync(p).toString("base64");
const N = JSON.parse(fs.readFileSync("notes.json", "utf8"));
const note = (n) => `Presenter: ${N[n][0]}. ${N[n][1]}`;

(async () => {
  const pres = new pptxgen();
  pres.layout = "LAYOUT_WIDE"; // 13.333 x 7.5
  pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
  pres.title = "Centre Court: Baseline - ASUS UGen AI League";
  pres.company = "Centre Court";
  const C = pres.SchemeColor;
  const NET = img("img/net_bg.png"), LOGO_M = img("img/logo_maple.png"), LOGO_O = img("img/logo_oak.png");

  // ---------- layouts ----------
  pres.defineSlideMaster({
    title: "NET",
    background: { data: NET },
    objects: [
      { image: { x: 0.8, y: 0.85, w: 1.1, h: 1.1, data: LOGO_M } },
      { placeholder: { options: { name: "title", type: "title", x: 0.8, y: 2.55, w: 9.4, h: 1.5,
          fontFace: THEME.headFontFace, fontSize: 46, bold: true, color: C.accent4, valign: "bottom", align: "left", margin: 0 }, text: "" } },
      { placeholder: { options: { name: "body", type: "body", x: 0.8, y: 4.15, w: 9.4, h: 2.6,
          fontSize: 16, color: C.accent4, valign: "top", align: "left", margin: 0 }, text: "" } },
    ],
  });
  // court lines: outer rectangle, tramlines inside both sides, centre marks top and bottom
  const L = 0.35, T = 0.35, R = 12.983, B = 7.15, TRAM = 0.22;
  const court = [
    { rect: { x: L, y: T, w: R - L, h: B - T, fill: { type: "none" }, line: { color: H.accent1, width: 1.5 } } },
    { line: { x: L + TRAM, y: T, w: 0, h: B - T, line: { color: H.accent1, width: 0.75 } } },
    { line: { x: R - TRAM, y: T, w: 0, h: B - T, line: { color: H.accent1, width: 0.75 } } },
    { line: { x: (L + R) / 2, y: T, w: 0, h: 0.16, line: { color: H.accent1, width: 1.5 } } },
    { line: { x: (L + R) / 2, y: B - 0.16, w: 0, h: 0.16, line: { color: H.accent1, width: 1.5 } } },
  ];
  pres.defineSlideMaster({
    title: "COURT",
    background: { color: H.lt2 },
    margin: [0.6, 0.85, 0.8, 0.85],
    objects: [
      ...court,
      { image: { x: 11.95, y: 0.6, w: 0.55, h: 0.55, data: LOGO_O } },
      { placeholder: { options: { name: "tracker", type: "body", x: 0.85, y: 0.62, w: 9, h: 0.3,
          fontSize: 12, italic: true, color: C.accent3, margin: 0, valign: "top" }, text: "" } },
      { placeholder: { options: { name: "title", type: "title", x: 0.85, y: 0.92, w: 10.9, h: 1.0,
          fontFace: THEME.headFontFace, fontSize: 28, bold: true, color: C.accent2, valign: "top", align: "left", margin: 0 }, text: "" } },
      { placeholder: { options: { name: "source", type: "body", x: 0.85, y: 6.68, w: 10.8, h: 0.3,
          fontSize: 10, color: C.accent3, valign: "top", margin: 0 }, text: "" } },
    ],
    slideNumber: { x: 11.95, y: 6.68, w: 0.55, h: 0.3, fontSize: 10, color: H.accent3, align: "right" },
  });

  const content = (section, tracker, title, source) => {
    const s = pres.addSlide({ masterName: "COURT", sectionTitle: section });
    s.addText(tracker, { placeholder: "tracker" });
    s.addText(title, { placeholder: "title" });
    if (source) s.addText(source, { placeholder: "source" });
    return s;
  };
  const txt = (s, text, o) => s.addText(text, Object.assign({ isTextBox: true, margin: 0, fontSize: 14, color: C.text1, valign: "top" }, o));
  const head = (s, text, o) => txt(s, text, Object.assign({ fontFace: THEME.headFontFace, fontSize: 20, bold: true, color: C.accent2 }, o));
  const hair = (s, x, y, w) => s.addShape(pres.shapes.LINE, { x, y, w, h: 0, line: { color: H.accent6, width: 1 } });
  // a small court-line frame (sections inside a slide)
  const courtBox = (s, x, y, w, h, name) => {
    s.addShape(pres.shapes.RECTANGLE, { x, y, w, h, fill: { color: H.lt1 }, line: { color: H.accent1, width: 1 }, objectName: name });
    s.addShape(pres.shapes.LINE, { x: x + 0.12, y, w: 0, h, line: { color: H.accent1, width: 0.5 } });
    s.addShape(pres.shapes.LINE, { x: x + w - 0.12, y, w: 0, h, line: { color: H.accent1, width: 0.5 } });
  };

  // ---------- 1. title ----------
  pres.addSection({ title: "Opening" });
  let s = pres.addSlide({ masterName: "NET", sectionTitle: "Opening" });
  s.addText("Centre Court: Baseline", { placeholder: "title" });
  s.addText([
    { text: "Instant tennis coaching, computed on the court", options: { fontFace: THEME.headFontFace, italic: true, fontSize: 24, breakLine: true } },
    { text: " ", options: { fontSize: 10, breakLine: true } },
    { text: "2026 ASUS UGen AI League Innovation Hackathon", options: { breakLine: true } },
    { text: "Battlefield Lightning: ASUS UGen300 AI Accelerator", options: { breakLine: true } },
    { text: " ", options: { fontSize: 10, breakLine: true } },
    { text: "Jefferson Tse-Wei Lee, Senior Data Scientist, Aurecon", options: { breakLine: true } },
    { text: "Ben Wylie, Senior Asset Consultant, Aurecon" },
  ], { placeholder: "body" });
  s.addNotes(note(1));

  // ---------- 2. team ----------
  pres.addSection({ title: "Team" });
  s = content("Team", "The team", "Two tennis players from Aurecon, both based in Brisbane");
  s.addImage({ data: img("img/team.jpg"), x: 9.0, y: 1.85, w: 3.45, h: 4.6, sizing: { type: "cover", w: 3.45, h: 4.6 }, objectName: "team photo" });
  const people = [
    ["Jefferson Tse-Wei Lee", "Born and raised in Taiwan; started his career as an intern at ASUS Taiwan. Senior Data Scientist in Asset Performance at Aurecon."],
    ["Ben Wylie", "Twenty-plus years of tennis; plays club level in Brisbane. Senior Asset Consultant in Infrastructure Advisory at Aurecon."],
  ];
  people.forEach(([k, v], i) => {
    const y = 1.95 + i * 1.35;
    head(s, k, { x: 0.85, y, w: 7.6, h: 0.42 });
    txt(s, v, { x: 0.85, y: y + 0.45, w: 7.6, h: 0.8, fontSize: 14 });
  });
  hair(s, 0.85, 4.75, 7.6);
  txt(s, [
    { text: "Aurecon ", options: { bold: true, color: C.accent2 } },
    { text: "is a 7,000-person engineering consultancy from Australia, with projects across Asia-Pacific including Hong Kong International Airport and the tallest building in mainland China.", options: { breakLine: true } },
    { text: " ", options: { fontSize: 6, breakLine: true } },
    { text: "Disclosure: ", options: { bold: true, color: C.accent2 } },
    { text: "this is an independent project, with no commercial relationship with, or endorsement from, Aurecon." },
  ], { x: 0.85, y: 4.95, w: 7.6, h: 1.5, fontSize: 13 });
  s.addNotes(note(2));

  // ---------- 3. problem + why edge ----------
  pres.addSection({ title: "Problem" });
  s = content("Problem", "Problem and why edge",
    "Players cannot see their own technique, and the fix has to be instant, private and offline",
    "Our hypothesis, to be validated with players and coaches");
  head(s, "The problem", { x: 0.85, y: 2.0, w: 6.2, h: 0.45 });
  [["Invisible", "Foot order, toss height and the hitting point happen in a fraction of a second."],
   ["Delayed", "Video review happens after the session, when the feel of the stroke is gone."],
   ["Coach-dependent", "Frame-by-frame analysis needs a coach's time for every stroke."]].forEach(([k, v], i) => {
    const y = 2.65 + i * 1.25;
    txt(s, k, { x: 0.85, y, w: 6.2, h: 0.4, fontSize: 16, bold: true, color: C.accent2 });
    txt(s, v, { x: 0.85, y: y + 0.4, w: 6.2, h: 0.7, fontSize: 14 });
  });
  s.addShape(pres.shapes.RECTANGLE, { x: 7.55, y: 1.95, w: 4.95, h: 4.45, fill: { color: H.dk2 }, line: { type: "none" }, objectName: "net panel" });
  s.addShape(pres.shapes.RECTANGLE, { x: 7.55, y: 1.95, w: 4.95, h: 0.12, fill: { color: "F4F1E6" }, line: { type: "none" }, objectName: "net tape" });
  head(s, "Why edge computing", { x: 7.9, y: 2.3, w: 4.3, h: 0.45, color: C.accent4 });
  [["Offline", "Courts often have weak or no Wi-Fi; every step runs on the phone and the UGen300."],
   ["Private", "Video of players, often juniors, never leaves the court."],
   ["Instant", "Feedback between points, while the player can still feel the stroke."]].forEach(([k, v], i) => {
    const y = 2.95 + i * 1.1;
    txt(s, k, { x: 7.9, y, w: 4.3, h: 0.36, fontSize: 15, bold: true, color: C.accent4 });
    txt(s, v, { x: 7.9, y: y + 0.36, w: 4.3, h: 0.7, fontSize: 13, color: C.background1 });
  });
  s.addNotes(note(3));

  // ---------- 4. five stages ----------
  pres.addSection({ title: "Solution" });
  s = content("Solution", "Solution and edge architecture",
    "Five stages turn a phone clip into coaching; the UGen300's NPU runs the neural models",
    "The demo runs the same pipeline as a web app on a laptop. YOLOX is in the Hailo Model Zoo; the pose model still has to be converted and benchmarked");
  const stages = [
    ["Frames", "Video split into one image per frame, 120 per second", "Phone"],
    ["Skeleton", "Player box and a 33-point skeleton on a zoomed crop", "UGen300"],
    ["Ball", "Racquet, ball path, bounce and hitting point", "UGen300 + phone"],
    ["Judge", "26 forehand and 20 serve checks, at three player levels", "Phone"],
    ["Coach", "Every result in plain words, worded for the player's level", "Phone or UGen300"],
  ];
  stages.forEach(([k, v, on], i) => {
    const x = 0.85 + i * 2.33, y = 1.95, w = 2.15, h = 2.55;
    courtBox(s, x, y, w, h, `stage ${i + 1}`);
    txt(s, String(i + 1), { x: x + 0.25, y: y + 0.18, w: 0.6, h: 0.5, fontFace: THEME.headFontFace, fontSize: 26, bold: true, color: C.accent1 });
    txt(s, k, { x: x + 0.25, y: y + 0.7, w: w - 0.5, h: 0.4, fontFace: THEME.headFontFace, fontSize: 20, bold: true, color: C.accent2 });
    txt(s, v, { x: x + 0.25, y: y + 1.12, w: w - 0.5, h: 0.95, fontSize: 12 });
    txt(s, "Runs on: " + on, { x: x + 0.25, y: y + 2.1, w: w - 0.5, h: 0.3, fontSize: 11, italic: true, color: C.accent3 });
  });
  [["Android phone", "120 fps camera, frame splitting, ball maths (OpenCV), coaching logic, results screen"],
   ["ASUS UGen300 (Hailo NPU)", "Player and racquet detection (YOLOX), the skeleton model; optional small local language-model coach"]].forEach(([k, v], i) => {
    const x = 0.85 + i * 5.9;
    hair(s, x, 4.85, 5.6);
    head(s, k, { x, y: 4.98, w: 5.6, h: 0.4, fontSize: 18 });
    txt(s, v, { x, y: 5.42, w: 5.6, h: 0.9, fontSize: 13 });
  });
  s.addNotes(note(4));

  // ---------- 5. stages 1-3 ----------
  s = content("Solution", "Stages 1 to 3: see the stroke",
    "Every frame is kept, the skeleton is fitted on a zoom, and the ball is tracked across the whole screen",
    "Forehand 1 of the 120 fps clip. Skeleton: MediaPipe BlazePose; player and racquet: YOLOX-tiny (both Apache-2.0)");
  s.addImage({ data: img("img/fh_traj.jpg"), x: 0.85, y: 1.95, w: 7.3, h: 4.11, objectName: "ball trajectory" });
  s.addShape(pres.shapes.RECTANGLE, { x: 0.85, y: 1.95, w: 7.3, h: 4.11, fill: { type: "none" }, line: { color: H.accent1, width: 1 } });
  txt(s, "Ball in, bounce (cross), hitting point (star) and ball out (arrow), found anywhere on screen", { x: 0.85, y: 6.12, w: 7.3, h: 0.3, fontSize: 11, italic: true, color: C.accent3 });
  [["Frames", "120 images per second, written once to disk, so a 1080p clip never fills memory."],
   ["Skeleton", "33 joints on a zoomed crop, smoothed over about 33 ms and converted to metres from the player's height."],
   ["Ball and racquet", "Moving yellow blobs are chained frame to frame; the hitting point is the midpoint of the two frames closest to the racquet head. A hit on the left side is a backhand and is skipped."]].forEach(([k, v], i) => {
    const y = 1.95 + i * 1.42;
    head(s, k, { x: 8.55, y, w: 3.9, h: 0.4, fontSize: 18 });
    txt(s, v, { x: 8.55, y: y + 0.42, w: 3.9, h: 0.95, fontSize: 12.5 });
  });
  s.addNotes(note(5));

  // ---------- 6. judge + coach by level ----------
  s = content("Solution", "Stages 4 and 5: judge and coach",
    "One ladder from Beginner to Club: the checks, the targets and the coach's words follow the player's level",
    "Forehand 1 of the 120 fps clip. Score: 40% for just meeting a target, full marks only near excellent; priority-weighted, then raised to the power 1.5");
  const lad = [
    ["Check", "Value", "Beginner", "Intermed.", "Club", "Reaches"],
    ["Contact in front of the body", "0.97", "≥ 0.45", "≥ 0.55", "≥ 0.60", "Club"],
    ["Ball reaches you after the bounce", "0.86 s", "≤ 1.2 s", "≤ 1.0 s", "≤ 0.8 s", "Intermediate"],
    ["Head stays level through the swing", "0.13 m", "≤ 0.18 m", "≤ 0.12 m", "≤ 0.07 m", "Beginner"],
    ["Still in your legs at contact", "4°", "≥ 3°", "≥ 6°", "≥ 10°", "Beginner"],
    ["Shoulders coil past the hips", "0.31", "≥ 0", "≥ 0.05", "≥ 0.12", "Club"],
  ];
  s.addTable(lad.map((r, i) => r.map((c, j) => ({ text: c, options: {
      bold: i === 0 || j === 5, fontSize: i === 0 ? 11 : 12,
      color: i === 0 ? C.accent3 : (j === 5 ? (c === "Club" ? C.accent1 : c === "Beginner" ? C.accent5 : C.accent2) : C.text1),
      fill: { color: H.lt2 }, valign: "middle", margin: [0.05, 0.08, 0.05, 0.08],
      border: [{ type: "none" }, { type: "none" }, { pt: 0.75, color: H.accent6 }, { type: "none" }] } }))),
    { x: 0.85, y: 2.0, w: 7.2, colW: [2.25, 0.8, 0.95, 0.95, 0.95, 1.3], rowH: 0.5, objectName: "ladder table" });
  txt(s, [{ text: "Scores for this forehand: ", options: { bold: true, color: C.accent2 } },
          { text: "91 Beginner, 68 Intermediate, 56 Club." }], { x: 0.85, y: 5.25, w: 7.2, h: 0.4, fontSize: 14 });
  txt(s, "Club level adds joint analysis: hip-shoulder coil, kinetic chain, steady spine and head, split step, weight transfer, left-arm tuck, knee bend at contact.",
    { x: 0.85, y: 5.7, w: 7.2, h: 0.7, fontSize: 12, color: C.accent3 });
  courtBox(s, 8.45, 2.0, 4.05, 4.45, "coach note");
  head(s, "What the coach says", { x: 8.75, y: 2.15, w: 3.5, h: 0.4, fontSize: 18 });
  [["Beginner", "Well done: contact in front, ball not above your head, turned sideways, finish over the shoulder, balanced."],
   ["Intermediate", "Keep your head at the same height from the backswing through contact."],
   ["Club", "Move forward to meet the ball (0.86 s; club ≤ 0.8 s). Head level (0.13 m; club ≤ 0.07 m). Stay down through the ball (4°; club ≥ 10°). Hips before arm."]].forEach(([k, v], i) => {
    const y = 2.68 + i * 1.18;
    txt(s, k, { x: 8.75, y, w: 3.5, h: 0.3, fontSize: 12, bold: true, color: C.accent1 });
    txt(s, v, { x: 8.75, y: y + 0.3, w: 3.5, h: 0.86, fontSize: 11.5 });
  });
  s.addNotes(note(6));

  // ---------- 7. serve ----------
  pres.addSection({ title: "Results" });
  s = content("Results", "Demo result: serve",
    "Serve: the toss, the exact contact moment and the heights that matter, from a low-resolution phone clip",
    "Clip IMG_3025: phone, 30 fps, player about 86 px tall. Heights scaled from the entered height (1.80 m); ball speed is a lower bound");
  s.addImage({ data: img("img/serve_keys.jpg"), x: 0.85, y: 1.95, w: 8.4, h: 2.8, objectName: "serve key moments" });
  s.addShape(pres.shapes.RECTANGLE, { x: 0.85, y: 1.95, w: 8.4, h: 2.8, fill: { type: "none" }, line: { color: H.accent1, width: 1 } });
  [["3.2 m", "contact height, with a jump"], ["4.2 m", "toss peak"], ["≥ 99 km/h", "ball speed off the racquet"], ["Frame 50.5", "contact, same as the manual check"]].forEach(([n, l], i) => {
    const y = 1.95 + i * 0.72;
    txt(s, n, { x: 9.65, y, w: 2.8, h: 0.42, fontFace: THEME.headFontFace, fontSize: 24, bold: true, color: C.accent2 });
    txt(s, l, { x: 9.65, y: y + 0.4, w: 2.8, h: 0.3, fontSize: 11, color: C.accent3 });
  });
  hair(s, 0.85, 5.05, 11.6);
  [["68", "Beginner"], ["67", "Intermediate"], ["62", "Club level"]].forEach(([n, l], i) => {
    const x = 0.85 + i * 2.1;
    txt(s, n, { x, y: 5.2, w: 1.9, h: 0.6, fontFace: THEME.headFontFace, fontSize: 34, bold: true, color: C.accent2 });
    txt(s, l + " score", { x, y: 5.82, w: 1.9, h: 0.3, fontSize: 11, color: C.accent3 });
  });
  txt(s, [{ text: "Club-level fixes: ", options: { bold: true, color: C.accent2 } },
          { text: "toss a little lower and more to the hitting side, lean up and into the court at the trophy, land inside the court." }],
    { x: 7.25, y: 5.25, w: 5.2, h: 0.9, fontSize: 13 });
  s.addNotes(note(7));

  // ---------- 8. forehand ----------
  s = content("Results", "Demo result: forehand",
    "Forehand: four forehands found in one 15-second clip, the backhand skipped, each scored at every level",
    "Clip forehand_120fps.mp4: 1080p at 120 fps. A stroke counts only when the ball leaves a fast-moving racquet in the swing direction; backhands are recognised and skipped");
  s.addImage({ data: img("img/fh_keys.jpg"), x: 0.85, y: 1.95, w: 6.9, h: 2.3, objectName: "forehand key moments" });
  s.addShape(pres.shapes.RECTANGLE, { x: 0.85, y: 1.95, w: 6.9, h: 2.3, fill: { type: "none" }, line: { color: H.accent1, width: 1 } });
  txt(s, "Forehand 1: white = racquet, star = hitting point, yellow = ball in, orange = ball out", { x: 0.85, y: 4.32, w: 6.9, h: 0.3, fontSize: 11, italic: true, color: C.accent3 });
  txt(s, [{ text: "What separates the levels: ", options: { bold: true, color: C.accent2 } },
          { text: "all four forehands pass the Beginner basics. At Club level the head moves 12-16 cm in the swing and the knees are almost straight at contact - typical of an intermediate player." }],
    { x: 0.85, y: 4.8, w: 6.9, h: 1.4, fontSize: 13 });
  s.addChart(pres.charts.BAR, [
    { name: "Beginner", labels: ["Forehand 1", "Forehand 2", "Forehand 3", "Forehand 4"], values: [91, 95, 90, 78] },
    { name: "Intermediate", labels: ["Forehand 1", "Forehand 2", "Forehand 3", "Forehand 4"], values: [68, 73, 65, 62] },
    { name: "Club level", labels: ["Forehand 1", "Forehand 2", "Forehand 3", "Forehand 4"], values: [56, 50, 61, 52] },
  ], {
    x: 8.05, y: 1.9, w: 4.45, h: 4.5, barDir: "col", barGrouping: "clustered",
    showTitle: true, title: "Score out of 100, by level", titleFontSize: 13, titleColor: H.accent2, titleFontFace: "+mj-lt",
    chartColors: [H.accent4, H.accent2, H.accent1], showValue: true, dataLabelPosition: "outEnd", dataLabelFontSize: 9,
    dataLabelColor: H.dk1, dataLabelFontFace: "+mn-lt",
    catAxisLabelFontSize: 10, catAxisLabelColor: H.dk1, catAxisLabelFontFace: "+mn-lt", catAxisLineShow: false,
    valAxisHidden: true, valAxisMinVal: 0, valAxisMaxVal: 110, valGridLine: { style: "none" }, catGridLine: { style: "none" },
    showLegend: true, legendPos: "b", legendFontSize: 10, legendColor: H.dk1, legendFontFace: "+mn-lt",
    barGapWidthPct: 60, objectName: "forehand scores chart" });
  s.addNotes(note(8));

  // ---------- 9. demo experience ----------
  s = content("Results", "Demo experience",
    "Pick your level, film, and get the coach's note first, with the evidence underneath",
    "Screenshots of the Centre Court web demo: start page, and a serve analysed at Intermediate level");
  s.addImage({ data: img("img/ui_landing.png"), x: 0.85, y: 1.95, w: 6.1, h: 3.91, objectName: "start page" });
  s.addImage({ data: img("img/ui_results.png"), x: 7.25, y: 1.95, w: 5.25, h: 4.34, sizing: { type: "crop", x: 0, y: 0, w: 5.25, h: 4.34 }, objectName: "results page" });
  [s].forEach(() => {
    s.addShape(pres.shapes.RECTANGLE, { x: 0.85, y: 1.95, w: 6.1, h: 3.91, fill: { type: "none" }, line: { color: H.accent1, width: 1 } });
    s.addShape(pres.shapes.RECTANGLE, { x: 7.25, y: 1.95, w: 5.25, h: 4.34, fill: { type: "none" }, line: { color: H.accent1, width: 1 } });
  });
  txt(s, "Coach's note, score and key numbers, ball trajectory, slow motion, and every check with the level it reaches.",
    { x: 0.85, y: 6.0, w: 6.1, h: 0.5, fontSize: 12, color: C.accent3 });
  s.addNotes(note(9));

  // ---------- 10. edge readiness ----------
  pres.addSection({ title: "Readiness" });
  s = content("Readiness", "Edge readiness",
    "The heavy part is exactly what the NPU accelerates, and the whole stack is licensed for commercial use",
    "CPU time on a 2-core cloud machine, forehand clip (1,818 frames, 1280x720 at 120 fps). Not yet benchmarked on the UGen300");
  s.addChart(pres.charts.BAR, [{ name: "seconds", labels: ["Split into frames", "Player detection", "Skeleton", "Ball candidates", "Strokes + racquet", "Pictures + GIFs"], values: [34, 10, 25, 46, 47, 26] }], {
    x: 0.85, y: 1.9, w: 6.4, h: 4.55, barDir: "bar", showTitle: true, title: "Where the 199 s go today (CPU only)",
    titleFontSize: 13, titleColor: H.accent2, titleFontFace: "+mj-lt",
    chartColors: [H.accent1], showValue: true, dataLabelPosition: "outEnd", dataLabelFontSize: 11,
    dataLabelColor: H.dk1, dataLabelFontFace: "+mn-lt", dataLabelFormatCode: '0" s"',
    catAxisLabelFontSize: 11, catAxisLabelColor: H.dk1, catAxisLabelFontFace: "+mn-lt", catAxisOrientation: "maxMin", catAxisLineShow: false,
    valAxisHidden: true, valAxisMinVal: 0, valGridLine: { style: "none" }, catGridLine: { style: "none" },
    showLegend: false, barGapWidthPct: 60, objectName: "compute chart" });
  [["Speed-ups already in", "Detector at 10 Hz, skeleton at 60 Hz with gaps filled, shake estimate on a smaller copy: the forehand clip went from 270-390 s to 199 s."],
   ["To the NPU", "Detection, racquet and skeleton are standard neural networks - the work the Hailo NPU is built for."],
   ["Commercially clean", "OpenCV, MediaPipe, YOLOX, ONNX Runtime: Apache, BSD and MIT licences. Ultralytics YOLO (AGPL) deliberately avoided."]].forEach(([k, v], i) => {
    const y = 1.95 + i * 1.5;
    head(s, k, { x: 7.75, y, w: 4.7, h: 0.4, fontSize: 18 });
    txt(s, v, { x: 7.75, y: y + 0.42, w: 4.7, h: 1.0, fontSize: 13 });
  });
  s.addNotes(note(10));

  // ---------- 11. next + thank you ----------
  pres.addSection({ title: "Close" });
  s = pres.addSlide({ masterName: "NET", sectionTitle: "Close" });
  s.addText("Next: an Android app, with the UGen300 doing the heavy lifting", { placeholder: "title", fontSize: 28 });
  s.addText([
    { text: "Now: serve and forehand working end to end, three player levels, web demo", options: { breakLine: true } },
    { text: "Next: YOLOX and pose on the Hailo NPU; Android app with a filming guide", options: { breakLine: true } },
    { text: "Then: coaches validate the ladders; backhand mode; club trials", options: { breakLine: true } },
    { text: " ", options: { fontSize: 10, breakLine: true } },
    { text: "Thank you - Jefferson and Ben", options: { fontFace: THEME.headFontFace, italic: true, fontSize: 24 } },
  ], { placeholder: "body", paraSpaceAfter: 6 });
  s.addNotes(note(11));

  await pres.writeFile({ fileName: "Centre_Court_pitch_v2.pptx" });
  await applyTheme("Centre_Court_pitch_v2.pptx", THEME);
  console.log("written");
})();
