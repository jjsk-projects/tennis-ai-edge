# Centre Court: Baseline - 5-minute demo speech

Ben opens (team, Aurecon, the pain point on court, about 1.5 minutes); Jefferson covers the technical side and the demo (about 2 minutes); Ben closes. Timings assume about 150 words a minute.

## Slide 1 - Title

**Ben** (15s)

Hi everyone, I'm Ben Wylie, and this is Jefferson Lee. We're from Brisbane, Australia and this is Centre Court: Baseline - instant tennis coaching, computed right on the court, on an Android phone and the ASUS UGen300.

## Slide 2 - The team

**Ben** (30s)

We both work at Aurecon, a 7,000-person Australian engineering consultancy with projects across Asia-Pacific, from Hong Kong International Airport to the tallest building in mainland China. I'm a senior asset consultant in infrastructure advisory, and Jefferson is a senior data scientist in asset performance. Jefferson grew up in Taiwan and started his career as an intern at ASUS, so this is a bit of a homecoming for him. Our day jobs use data to understand how physical assets perform; Centre Court does the same for a tennis stroke. 

## Slide 3 - The problem and why edge

**Ben** (40s)

I've played tennis for over twenty years, and still learning. Is my toss too low? Am I hitting the ball above my head? It all happens in a fraction of a second. Today you either film yourself and watch it at home, when the feel of the stroke is gone, or pay a coach to watch every ball, which most club players can't do every week. We wish that feedback can be instant and on court That's an edge-computing problem, and that's why we built it for the UGen300. Jefferson will show you how it works.

## Slide 4 - Five stages

**Jefferson** (30s)

Thanks, Ben. The phone films at 120 frames a second, and then there are five stages. One: split the video into frames. Two: find the player and fit a 33-point skeleton. Three: find the racquet, the ball, the bounce and the exact hitting point. Four: judge the stroke against coaching rules. Five: turn it into plain-language coaching. The phone does the camera and the light maths; the UGen300's Hailo NPU runs the neural networks.

## Slide 5 - Stages 1-3: see the stroke

**Jefferson** (30s)

Here's what stages one to three see. We keep every frame, because at 120 frames a second we catch the few milliseconds around contact. The skeleton is fitted on a zoomed crop and converted into metres from the player's height. And the ball is tracked across the whole screen: you can see it come in, bounce, and the star is the hitting point - the midpoint of the two frames where the ball is closest to the racquet head. The arrow is the ball leaving.

## Slide 6 - Stages 4-5: judge and coach

**Jefferson** (30s)

Stage four is deliberately not a black box. It's coaching logic: 18 forehand checks and 15 serve checks, each a rule on a measured number, in the order a coach would fix them. Top priority: if you hit the ball above your head, move forward - or, if you're already inside the court, get out of no-man's land and let it drop. Then the coach reports everything: what's good, every fix in priority order, and what it couldn't measure.

## Slide 7 - Serve result

**Jefferson** (30s)

Here's a serve from an ordinary phone clip: 30 frames a second, with the player only 86 pixels tall. It fits the toss to a gravity curve and finds the contact at frame 50.5 - exactly what we found checking frame by frame. Contact at 3.25 metres, and three fixes: tilt the shoulders, keep the toss arm up, and land inside the court.

## Slide 8 - Forehand result

**Jefferson** (30s)

On a 15-second forehand clip at 120 frames a second, it found four forehands. It also recognised the one backhand in the clip and skipped it - we'd rather miss a stroke than coach the wrong one. Forehand one: hit at 1.1 metres, in front of the body, left foot forward. One fix: the ball took almost 0.9 seconds after the bounce, so move forward to meet it.

## Slide 9 - Demo

**Jefferson** (15s)

This is what the player sees. [DEMO: switch to the web app with IMG_3025 already analysed.] The coach's note comes first, then the evidence: the ball trajectory, slow motion with skeleton and racquet, and the checklist with every measured value.

## Slide 10 - Edge readiness

**Jefferson** (15s)

Today this runs on a CPU: about three minutes for that 15-second clip at 120 frames a second. Most of that time is detection, skeleton and racquet - standard neural networks, exactly what the Hailo NPU accelerates. And it's commercially clean: every component is Apache, BSD or MIT licensed. We deliberately avoided AGPL models.

## Slide 11 - Next steps and thanks

**Ben** (5:08-5:25)

Next, we put the models on the UGen300, build the Android app with a filming guide, and get coaches to validate the thresholds on real courts. Every club player deserves a coach on the court. Thank you.

## Delivery notes

- **Demo:** run IMG_3025 in the web app before you start (about 45 s on a laptop), so slide 9 is a quick switch to the finished result rather than a wait.
- **Stage I video limit:** the hackathon's Stage I demo video is capped at 3 minutes (due 14 Oct). This 5-minute script suits the live pitch; for the video, cut slides 2, 6 and 10 to one line each.
- **If you run long:** drop the serve detail on slide 7 to one sentence - the numbers are on the slide.
