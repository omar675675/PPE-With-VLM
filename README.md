# PPE Violation Detection with VLM Verification

Finds people on video who are not wearing their safety gear (helmet, vest) and shows them on a live web dashboard.

It works in two stages:

1. **Detection (CNN).** YOLO models find and track every person, then check each person's crop for missing PPE.
2. **Verification (VLM), in progress.** A vision-language model looks at each flagged crop and confirms or rejects the violation. This cuts down on false alarms from the CNN.

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
flagged person + best crop + screenshot
   │
   ▼
VLM check on the crop: real violation or not?      (planned)
   │
   ▼
violations.json  ──►  dashboard (live)
```

A few details on the detection stage:

- **Region checks.** A helmet only counts if it sits in the top part of the person's crop, and a vest only counts in the middle part. This stops a neighbour's helmet or vest from being counted for the wrong person.
- **Voting over frames.** A person is only flagged if the violation shows up in at least `min_hits` frames and in at least `min_ratio` of the frames they were checked in. One bad frame is not enough.
- **One report per person.** Each tracked person is reported once, with all their violations and the clearest screenshot (the frame where the model was most confident).

## Project layout

| File | What it does |
|---|---|
| `process.py` | Reads the streams, runs detection and tracking, writes `output/violations.json` and images |
| `dashboard.py` | FastAPI web page that shows the results and updates live |
| `main.py` | Runs both at once: processing in the background, dashboard in front |
| `config.yaml` | All settings: models, streams, thresholds, tracker, dashboard port |

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

Then list your videos under `streams` in `config.yaml`:

```yaml
streams:
  - name: site1
    source: videos/site1.mp4
```

## Run

All in one (processing plus live dashboard):

```bash
.venv/bin/python main.py
```

Or separately:

```bash
.venv/bin/python process.py      # writes output/
.venv/bin/python dashboard.py    # serves the results
```

Open `http://localhost:9000` (the port is set in `config.yaml`).

Each run clears the `output/` folder first.

## Output

`output/violations.json` holds:

- `status`: `starting`, `running` (with current stream and frame), or `done`
- `videos`: one summary per stream (frames, length, people tracked, people flagged)
- `violations`: one entry per flagged person, with violation types, confidence, hit counts, timestamp, and paths to the crop and full-frame screenshot

Images are saved as `output/<stream>/<track_id>_crop.jpg` and `<track_id>_frame.jpg`.

## Tuning

The main knobs in `config.yaml`:

| Setting | Effect |
|---|---|
| `person.check_conf` | How sure the person detector must be before running the PPE check |
| `person.min_height` | Skip people too small to judge |
| `ppe.violation_conf` | How sure the PPE model must be about a `no_...` box |
| `ppe.checks.*.region` | Where in the crop each item is expected |
| `violation.min_hits` / `min_ratio` | How many frames must agree before a person is flagged |
| `tracker.track_buffer` | How long a lost person is remembered before getting a new ID |

## Status

- [x] Person detection and tracking
- [x] PPE check per person, with region filtering and voting over frames
- [x] Live dashboard
- [ ] VLM verification of flagged crops
