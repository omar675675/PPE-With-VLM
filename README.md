# PPE Violation Detection with VLM Verification

Finds people on video who are not wearing their safety gear (helmet, vest), double-checks every finding with a vision-language model, and shows the results on a live web dashboard.

<!-- Screenshots: save them to docs/ and uncomment
**Before verifying:** what the PPE model flagged

![Dashboard before VLM verification](docs/before-verify.png)

**After verifying:** the VLM's verdicts, with the cases it disagreed with marked for review

![Dashboard after VLM verification](docs/after-verify.png)
-->

It works in two stages:

1. **Detection (CNN).** YOLO models find and track every person, then check each person's crop for missing PPE.
2. **Verification (VLM).** A vision-language model (`qwen3-vl:8b`, run locally through Ollama) looks at each flagged person and says whether the violation is real. When it disagrees with the CNN, a person makes the final call in the dashboard.

## How it works

```
video / stream
   │
   ▼
person detector (YOLOv8) + ByteTrack  ──►  one ID per person
   │
   ▼
crop each person  ──►  PPE model (YOLO)  ──►  helmet / no_helmet / vest / no_vest
   │
   ▼
per-person voting over many frames (min_hits, min_ratio)
   │
   ▼
flagged person + best crop + screenshot  ──►  violations.json  ──►  dashboard (live)
   │
   ▼
Verify button  ──►  VLM looks at the crop: is the item really missing?
   │
   ▼
agrees     → ✓ Verified (green)
disagrees  → ! Needs review (red) → a person picks: Accept VLM  or  Keep violation
```

### Stage 1: detection

- **Tracking.** Every person gets an ID that follows them through the video, so each person is reported once, not once per frame.
- **Region checks.** A helmet only counts if it sits in the top part of the person's crop, and a vest only counts in the middle part. This stops a neighbour's helmet or vest from counting for the wrong person.
- **Voting over frames.** A person is only flagged if the violation shows up in at least `min_hits` frames and in at least `min_ratio` of the frames they were checked in. One bad frame is not enough.
- **Best screenshot.** For each person, the frame where the model was most confident is saved, both as a crop and as the full frame with a red box.

### Stage 2: VLM verification

After processing finishes, press **Verify** in the dashboard.

- **What gets sent.** The person's crop, not the full frame. In testing, crops were faster (about 3.5 s vs 6 s per person) and more accurate. With the full frame, the VLM also read the "no vest" label drawn on the image and simply repeated it.
- **What gets asked.** A neutral question, "is this person wearing a hard hat / a high-visibility vest?", rather than "confirm this violation". This keeps the VLM from just agreeing with the CNN.
- **What gets checked depends on the page.** On the *No vest* page only the vest is checked, on *No helmet* only the helmet, and on *All types* every violation the person has. Tick the people you want checked, or leave all unticked to check everything on the page.
- **Results:**
  - **✓ Verified** (green frame): the VLM agrees the item is missing.
  - **! Needs review** (red frame): the VLM sees the item. A person decides:
    - **Accept VLM**: the VLM is right. The tag changes to what it saw (e.g. *✓ Helmet*) and the violation stops counting. A person with nothing left is marked *✓ Cleared*.
    - **Keep violation**: the VLM is wrong, and the violation stays.
- **GPU memory.** Built for an 8 GB GPU. The YOLO models free their memory once processing ends, and the VLM is unloaded from Ollama when verification ends, so the two never fight over memory.

## Testing the VLM: the `flag_everyone` switch

The PPE model turned out to be too good to test against. On our test videos it made almost no mistakes, even with every threshold turned down to nearly zero, so the VLM had nothing to catch and the "Needs review" path never showed up.

So we added a test-only switch to `config.yaml`:

```yaml
ppe:
  flag_everyone: true   # TEST ONLY - default is false
```

With it on, every checked person is flagged for every violation type, whatever the PPE model says. That gives the VLM a mix of real and fake violations with an obvious right answer. On two clips with three workers (one in full PPE, two with a helmet but no vest), the VLM got all 6 checks right:

| Person | Flagged for | Truth | VLM |
|---|---|---|---|
| Worker A (helmet, no vest) | No helmet | wrong | ! Needs review ✓ |
| Worker A | No vest | right | ✓ Verified ✓ |
| Worker B (helmet + vest) | No helmet | wrong | ! Needs review ✓ |
| Worker B | No vest | wrong | ! Needs review ✓ |
| Worker C (helmet, no vest) | No helmet | wrong | ! Needs review ✓ |
| Worker C | No vest | right | ✓ Verified ✓ |

Turn it off again for real use. In this mode the PPE confidence on the cards shows `0.00`, because the model never saw a violation.

## Dashboard

- **Filters** by video and by violation type. The type filter also decides what Verify checks.
- **Tiles:** people flagged, count per violation type, and a *VLM review* tile (how many checked, confirmed, needing review, corrected).
- **Table** with one row per video. Click a row to filter to it.
- **Cards** with one per flagged person: crop, violation tags, PPE confidence and frame counts, and the VLM's answer with its reason.
- **Lightbox:** click a card to see the full frame with the red box. Arrow keys move between people.

It updates live while processing and verification run, and follows the system's light or dark theme.

## Project layout

| File | What it does |
|---|---|
| `process.py` | Reads the streams, runs detection and tracking, writes `output/violations.json` and images |
| `verify.py` | Sends crops to the VLM, stores its answers and human decisions in `output/review.json` |
| `dashboard.py` | FastAPI web page: shows results, runs Verify, takes Accept / Keep decisions |
| `main.py` | Runs processing in the background and the dashboard in front |
| `config.yaml` | All settings: models, streams, thresholds, tracker, VLM, dashboard port |

## Setup

Python 3.12 and an NVIDIA GPU are recommended (CPU works, but slowly).

```bash
python -m venv .venv
.venv/bin/pip install ultralytics opencv-python fastapi uvicorn pyyaml
```

Model weights and videos are not in this repo. Put them here:

```
models/yolov8n-person.pt   # person detector (trained at 1280)
models/ppe-m.pt            # PPE detector with classes: helmet, no_helmet, vest, no_vest
videos/*.mp4               # your footage
```

For the VLM step, install [Ollama](https://ollama.com) and pull the model:

```bash
ollama pull qwen3-vl:8b
```

Then list your videos under `streams` in `config.yaml`:

```yaml
streams:
  - name: site1
    source: videos/site1.mp4
```

## Run

All in one (processing plus live dashboard):

```bash
.venv/bin/python main.py                  # uses config.yaml
.venv/bin/python main.py my-config.yaml   # or another config
```

Or separately:

```bash
.venv/bin/python process.py      # writes output/
.venv/bin/python dashboard.py    # serves the results
```

Open `http://localhost:9000` (the port is set in `config.yaml`), wait for processing to finish, then press **Verify**.

Each processing run clears the `output/` folder first, including VLM results and human decisions. To look at old results again without re-processing, run `dashboard.py` on its own.

## Output

- `output/violations.json`: processing status, one summary per video, and one entry per flagged person (violation types, confidence, hit counts, timestamp, image paths).
- `output/review.json`: per person and per violation type, the VLM's answer and reason and the review state: `pending`, `verified`, `rejected` (needs review), `kept`, `accepted` or `error`.
- `output/<stream>/<track_id>_crop.jpg` and `<track_id>_frame.jpg`: the person crop and the full frame with a red box.

## Tuning

The main settings in `config.yaml`:

| Setting | Effect |
|---|---|
| `person.check_conf` | How sure the person detector must be before running the PPE check |
| `person.min_height` | Skip people too small to judge |
| `person.crop_pad` | Extra space around each person crop (also what the VLM sees) |
| `ppe.conf` | Minimum confidence for any PPE box to come out of the model |
| `ppe.violation_conf` | How sure the PPE model must be about a `no_...` box |
| `ppe.checks.*.region` | Where in the crop each item is expected |
| `ppe.flag_everyone` | Test only: flag everyone, to test the VLM (see above) |
| `violation.min_hits` / `min_ratio` | How many frames must agree before a person is flagged |
| `tracker.track_buffer` | How long a lost person is remembered before getting a new ID |
| `vlm.model` / `vlm.url` | Which Ollama model and server to use |
| `vlm.items` | How each item (helmet, vest) is described to the VLM |

## Known limits

- **Crowded scenes.** A person's crop can include part of a neighbour. The VLM can then judge the wrong person, for example by taking a neighbour's red scarf for a vest. That's what the human review step is for.
- **Streams run one after another.** A live `rtsp://` source never ends, so streams listed after it never start. There is also no reconnect if a stream drops.
- **Silent crashes.** If processing crashes inside `main.py`, the dashboard keeps showing "Processing…". Check the terminal.

## Status

- [x] Person detection and tracking
- [x] PPE check per person, with region filtering and voting over frames
- [x] Live dashboard
- [x] VLM verification of flagged crops, chosen per page or per selection
- [x] Human review when the VLM disagrees (Accept VLM / Keep violation)
- [x] `flag_everyone` test mode
