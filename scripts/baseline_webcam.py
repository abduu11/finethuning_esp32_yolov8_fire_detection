import argparse
import csv
import time
from datetime import datetime
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO


def parse_args():
    p = argparse.ArgumentParser(description="Sprint 1: baseline fire/smoke webcam inference")
    p.add_argument("--model", default="models/fire_smoke_yolov8n.pt")
    p.add_argument("--source", default="0", help="webcam index (int) or video/image path")
    p.add_argument("--conf", type=float, default=0.35)
    p.add_argument("--imgsz", type=int, default=416, help="inference size (320 fast, 640 accurate)")
    p.add_argument("--threads", type=int, default=4, help="CPU thread count (physical cores)")
    p.add_argument("--device", default="auto", help="device: auto / 0 (cuda:0) / cpu")
    p.add_argument("--res", default="640x480", help="webcam capture resolution WxH")
    p.add_argument("--log", action="store_true", help="log detections to CSV")
    p.add_argument("--out", default="data/captures", help="dir for saved frames + CSV")
    return p.parse_args()


def main():
    args = parse_args()
    if args.device == "auto":
        args.device = "0" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(args.threads)
    cv2.setNumThreads(args.threads)
    model = YOLO(args.model)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"cannot open source: {args.source}")
    if args.source.isdigit():
        w, h = (int(v) for v in args.res.lower().split("x"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)

    csv_path = out_dir / f"detections_{datetime.now():%Y%m%d_%H%M%S}.csv"
    csv_file = None
    csv_writer = None
    if args.log:
        csv_file = open(csv_path, "w", newline="")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["timestamp", "frame", "class", "conf", "x1", "y1", "x2", "y2"])

    print(f"[baseline] model={args.model} conf={args.conf} imgsz={args.imgsz} device={args.device}")
    print("[baseline] keys: q=quit  s=save frame  c=toggle CSV log")
    print("[baseline] hint: lower --imgsz (320) = more FPS; raise (640) = more accurate")

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        t0 = time.perf_counter()
        results = model.predict(frame, conf=args.conf, imgsz=args.imgsz, device=args.device, verbose=False)
        dt = time.perf_counter() - t0
        fps = 1.0 / dt if dt > 0 else 0.0

        n_det = 0
        for r in results:
            boxes = r.boxes
            if boxes is None:
                continue
            n_det = len(boxes)
            for b in boxes:
                cls = int(b.cls[0])
                conf = float(b.conf[0])
                name = model.names.get(cls, str(cls))
                x1, y1, x2, y2 = [int(v) for v in b.xyxy[0]]
                color = (0, 165, 255) if name == "fire" else (200, 200, 200)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{name} {conf:.2f}", (x1, max(0, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                if csv_writer is not None:
                    csv_writer.writerow([datetime.now().isoformat(), frame_idx,
                                         name, round(conf, 4), x1, y1, x2, y2])

        cv2.putText(frame, f"FPS {fps:.1f} | conf {args.conf} | n={n_det}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("baseline fire/smoke", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            p = out_dir / f"frame_{frame_idx:05d}.jpg"
            cv2.imwrite(str(p), frame)
            print(f"[save] {p}")
        elif key == ord("c"):
            if csv_writer is None:
                csv_file = open(csv_path, "a", newline="")
                csv_writer = csv.writer(csv_file)
                csv_writer.writerow(["timestamp", "frame", "class", "conf", "x1", "y1", "x2", "y2"])
            else:
                csv_file.close()
                csv_writer = None
            print(f"[log] {'ON' if csv_writer is not None else 'OFF'} -> {csv_path}")

        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()
    if csv_file is not None:
        csv_file.close()
    print("[baseline] done")


if __name__ == "__main__":
    main()
