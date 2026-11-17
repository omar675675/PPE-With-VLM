"""
PPE violation finder.

For every stream in config.yaml:
  1. Detect and track people (person model + ByteTrack)
  2. Crop each person and run the PPE model on the crop
  3. Check the violations listed under ppe.checks (no_helmet, no_vest)
  4. Report each tracked person once, with all their violations and one screenshot

Results are written to <output_dir>/violations.json + images while it runs
(about once a second), so the dashboard can show them live.

Run:
    .venv/bin/python process.py [config.yaml]
"""

import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import cv2
import yaml
from ultralytics import YOLO

ROOT = Path(__file__).parent
FLUSH_EVERY_S = 1.0  # how often results are written while a stream is running


def load_config(path):
    path = Path(path)
    cfg = yaml.safe_load(path.read_text())
    base = path.parent

    def resolve(p):
        p = str(p)
        return p if "://" in p or os.path.isabs(p) else str(base / p)

    cfg["models"] = {k: resolve(v) for k, v in cfg["models"].items()}
    for s in cfg["streams"]:
        s["source"] = resolve(s["source"])
    cfg["output_dir"] = Path(resolve(cfg["output_dir"]))
    return cfg


class ResultsWriter:
    """Keeps everything that goes into violations.json and writes it atomically."""

    def __init__(self, cfg):
        self.out_dir = cfg["output_dir"]
        self.types = list(cfg["ppe"]["checks"])
        self.done_records = []   # records from streams that are finished
        self.summaries = []      # one per stream, updated in place while running
        self.status = {"state": "starting"}

    def save(self, current_records=()):
        data = {"status": self.status, "types": self.types, "videos": self.summaries,
                "violations": self.done_records + list(current_records)}
        tmp = self.out_dir / "violations.json.tmp"
        tmp.write_text(json.dumps(data, indent=2))
        os.replace(tmp, self.out_dir / "violations.json")


def judge(ppe_result, names, crop_w, crop_h, cfg):
    """Return ({violation: conf} found in this crop, set of checks that had any evidence)."""
    checks = cfg["ppe"]["checks"]
    best = {}  # (check, class name) -> best conf inside that check's region
    for (x1, y1, x2, y2), cls, conf in zip(
        ppe_result.boxes.xyxy.tolist(), ppe_result.boxes.cls.tolist(), ppe_result.boxes.conf.tolist()
    ):
        name = names[int(cls)]
        cx, cy = (x1 + x2) / 2 / crop_w, (y1 + y2) / 2 / crop_h
        for vio, c in checks.items():
            if name in (c["good"], c["bad"]):
                r = c["region"]
                if r["x"][0] <= cx <= r["x"][1] and r["y"][0] <= cy <= r["y"][1]:
                    best[(vio, name)] = max(best.get((vio, name), 0.0), conf)

    found, checked = {}, set()
    for vio, c in checks.items():
        g, b = best.get((vio, c["good"]), 0.0), best.get((vio, c["bad"]), 0.0)
        if g or b:
            checked.add(vio)
        if b >= cfg["ppe"]["violation_conf"] and b > g:
            found[vio] = b
    return found, checked


def confirmed_violations(t, fps, cfg):
    """Violations of one track that pass min_hits / min_ratio so far."""
    vc = cfg["violation"]
    confirmed = {}
    for vio, hits in t["hits"].items():
        n_checks = t["checks"].get(vio, 1)
        if hits >= vc["min_hits"] and hits / n_checks >= vc["min_ratio"]:
            b = t["best"][vio]
            confirmed[vio] = {"conf": round(b["conf"], 3), "hits": hits, "checks": n_checks,
                              "frame": b["frame"], "time_s": round(b["frame"] / fps, 2)}
    return confirmed


def write_images(out, tid, t, confirmed, main):
    """Save the crop and the annotated full frame for one person."""
    b = t["best"][main]
    full = cv2.imdecode(b["full_jpg"], cv2.IMREAD_COLOR)
    x1, y1, x2, y2 = b["box"]
    label = f"#{tid} " + ", ".join(v.replace("_", " ") for v in confirmed)
    cv2.rectangle(full, (x1, y1), (x2, y2), (59, 59, 208), 3)
    cv2.putText(full, label, (x1, max(25, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (59, 59, 208), 2, cv2.LINE_AA)
    (out / f"{tid}_crop.jpg").write_bytes(b["crop_jpg"].tobytes())
    cv2.imwrite(str(out / f"{tid}_frame.jpg"), full, [cv2.IMWRITE_JPEG_QUALITY, 85])


def process_stream(stream, cfg, ppe_model, tracker_file, writer):
    pc = cfg["person"]
    name = stream["name"]
    out = cfg["output_dir"] / name
    out.mkdir(parents=True, exist_ok=True)

    summary = {"video": name, "frames": 0, "duration_s": 0, "people": 0, "flagged": 0}
    writer.summaries.append(summary)

    person_model = YOLO(cfg["models"]["person"])  # fresh model = fresh tracker per stream
    cap = cv2.VideoCapture(stream["source"])
    if not cap.isOpened():
        print(f"  could not open {stream['source']}")
        summary["error"] = "could not open stream"
        writer.save()
        return
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # per track id:
    #   hits[vio], checks[vio] -> counts
    #   best[vio] = {conf, frame, box, crop_jpg, full_jpg}
    tracks = {}
    records = {}   # tid -> record currently in the results
    written = {}   # tid -> what the images on disk show, so they're only rewritten when it changes
    frame_idx = -1

    def flush():
        for tid, t in tracks.items():
            confirmed = confirmed_violations(t, fps, cfg)
            if not confirmed:
                if tid in records:  # dropped below the thresholds
                    del records[tid]
                    written.pop(tid, None)
                    for f in (f"{tid}_crop.jpg", f"{tid}_frame.jpg"):
                        (out / f).unlink(missing_ok=True)
                continue
            main = max(confirmed, key=lambda v: confirmed[v]["conf"])  # screenshot = most confident
            b = t["best"][main]
            key = (b["frame"], tuple(confirmed))
            if written.get(tid) != key:
                write_images(out, tid, t, confirmed, main)
                written[tid] = key
            records[tid] = {
                "video": name, "track_id": tid, "types": list(confirmed), "details": confirmed,
                "frame": b["frame"], "time_s": round(b["frame"] / fps, 2), "conf": round(b["conf"], 3),
                "crop": f"{name}/{tid}_crop.jpg", "screenshot": f"{name}/{tid}_frame.jpg",
                "v": b["frame"],  # changes when the screenshot changes (cache-busting)
            }
        summary.update(frames=frame_idx + 1, duration_s=round((frame_idx + 1) / fps, 1),
                       people=len(tracks), flagged=len(records))
        writer.status = {"state": "running", "stream": name, "frame": frame_idx + 1, "total": total}
        writer.save(sorted(records.values(), key=lambda r: r["time_s"]))

    last_flush = time.monotonic()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        H, W = frame.shape[:2]

        res = person_model.track(frame, imgsz=pc["imgsz"], conf=pc["det_conf"], persist=True,
                                 tracker=tracker_file, verbose=False,
                                 device=cfg["device"], half=cfg["half"])[0]
        if res.boxes.id is not None:
            crops, metas = [], []
            for (x1, y1, x2, y2), tid, conf in zip(res.boxes.xyxy.tolist(), res.boxes.id.int().tolist(),
                                                   res.boxes.conf.tolist()):
                if conf < pc["check_conf"] or y2 - y1 < pc["min_height"]:
                    continue
                pw, ph = (x2 - x1) * pc["crop_pad"], (y2 - y1) * pc["crop_pad"]
                box = (max(0, int(x1 - pw)), max(0, int(y1 - ph)), min(W, int(x2 + pw)), min(H, int(y2 + ph)))
                crops.append(frame[box[1]:box[3], box[0]:box[2]])
                metas.append((tid, box))

            if crops:
                ppe_results = ppe_model(crops, imgsz=cfg["ppe"]["imgsz"], conf=cfg["ppe"]["conf"],
                                        verbose=False, device=cfg["device"], half=cfg["half"])
                for (tid, box), crop, pr in zip(metas, crops, ppe_results):
                    t = tracks.setdefault(tid, {"hits": {}, "checks": {}, "best": {}})
                    found, checked = judge(pr, ppe_model.names, crop.shape[1], crop.shape[0], cfg)
                    for vio in checked:
                        t["checks"][vio] = t["checks"].get(vio, 0) + 1
                    for vio, conf in found.items():
                        t["hits"][vio] = t["hits"].get(vio, 0) + 1
                        if conf > t["best"].get(vio, {}).get("conf", 0):  # keep the clearest example
                            t["best"][vio] = {
                                "conf": conf, "frame": frame_idx, "box": box,
                                "crop_jpg": cv2.imencode(".jpg", crop)[1],
                                "full_jpg": cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])[1],
                            }

        if time.monotonic() - last_flush >= FLUSH_EVERY_S:
            flush()
            last_flush = time.monotonic()
        if frame_idx % 100 == 0:
            print(f"  {name}: frame {frame_idx}/{total}", flush=True)
    cap.release()

    flush()
    writer.done_records += sorted(records.values(), key=lambda r: r["time_s"])
    print(f"  -> {summary['people']} people checked, {summary['flagged']} flagged")


def main(config_path=None):
    cfg = load_config(config_path or (sys.argv[1] if len(sys.argv) > 1 else ROOT / "config.yaml"))
    out_dir = cfg["output_dir"]
    # empty the output folder (but keep the folder itself, the dashboard serves from it)
    out_dir.mkdir(parents=True, exist_ok=True)
    for p in out_dir.iterdir():
        shutil.rmtree(p) if p.is_dir() else p.unlink()

    writer = ResultsWriter(cfg)
    writer.save()

    # ultralytics wants the tracker settings as a yaml file
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        yaml.safe_dump(cfg["tracker"], f)
        tracker_file = f.name

    ppe_model = YOLO(cfg["models"]["ppe"])
    try:
        for stream in cfg["streams"]:
            print(f"Processing {stream['name']} ({stream['source']})")
            process_stream(stream, cfg, ppe_model, tracker_file, writer)
    finally:
        os.unlink(tracker_file)

    writer.status = {"state": "done"}
    writer.save()
    print(f"Done. {len(writer.done_records)} people flagged -> {out_dir / 'violations.json'}")


if __name__ == "__main__":
    main()
