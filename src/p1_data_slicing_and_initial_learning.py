"""COCO train2017을 n개 부분집합으로 나누고, 부분집합 1로 초기 레이블러(YOLOv8s, scratch)를 학습한다.

하는 일
  1) train2017을 n개(기본 20) 부분집합 JSON으로 균등 분할 (<out-dir>/p1_output/subsets/)
  2) subset_01로 YOLOv8s를 처음부터 학습 (<out-dir>/p1_output/weights/labeler_best.pt)
  3) 학습한 레이블러를 subset 2~n과 val2017에서 평가 (eval_results.csv, eval_by_conf.csv)

저장된 가중치가 있으면 재사용할지 묻는다(--skip-train: 묻지 않고 재사용, --retrain: 묻지 않고 새로 학습).
사용 예: python p1_data_slicing_and_initial_learning.py --coco-root coco --out-dir outputs --n 20 --seed 42
"""

import argparse
import csv
import json
import os
import random
import shutil
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

import p0_common_api as common

IOU_THRS = np.linspace(0.5, 0.95, 10)


def get_train_config():
    """초기 레이블러 학습 설정. Ultralytics의 자동 옵티마이저(auto)는 학습 반복 수에 따라
    SGD/AdamW와 학습률을 바꾸므로, 실험 간 조건을 같게 하려고 SGD로 고정한다.
    """
    return {
        "model": "yolov8s.yaml",  # 가중치 없이 구조만 생성 (scratch)
        "pretrained": False,
        "batch": 16,
        "imgsz": 640,
        "optimizer": "SGD",
        "lr0": 0.01,
        "lrf": 0.01,
        "momentum": 0.937,
        "weight_decay": 0.0005,
        "epochs": 200,  # 상한. patience에 따라 일찍 끝날 수 있음
        "patience": 25,  # 검증 성능이 25 에포크 동안 오르지 않으면 중단
        "mosaic": 1.0,
        "copy_paste": 0.2,
        "mixup": 0.15,
        "seed": 0,
        "workers": 8,
        "device": None,
        "plots": True,
    }


def get_eval_config():
    """평가 설정. conf_grid는 eval_by_conf.csv에 저장할 신뢰도 임계값 목록이다."""
    return {
        "pred_conf": 0.001,
        "pred_iou": 0.6,
        "max_det": 100,
        "batch": 32,
        "conf_thr": 0.25,
        "conf_grid": [0.1, 0.25, 0.3, 0.5, 0.7, 0.9],
    }


def split_ids(image_ids, n, seed):
    """이미지 id를 섞어 n개로 균등 분할한다 (크기 차이는 최대 1)."""
    ids = sorted(image_ids)
    random.Random(seed).shuffle(ids)
    base, extra = divmod(len(ids), n)
    subsets, start = [], 0
    for k in range(n):
        size = base + (1 if k < extra else 0)
        subsets.append(ids[start:start + size])
        start += size
    return subsets


def split_train(ann_path, out_dir, n=20, seed=42, drop_empty=False):
    """instances_train2017.json을 n개 부분집합 JSON(subset_01~n.json)과 split_index.json으로 저장한다."""
    with open(ann_path, "r", encoding="utf-8") as f:
        coco = json.load(f)
    images_by_id = {img["id"]: img for img in coco["images"]}
    anns_by_image = defaultdict(list)
    for a in coco["annotations"]:
        anns_by_image[a["image_id"]].append(a)

    image_ids = list(images_by_id)
    if drop_empty:
        image_ids = [i for i in image_ids if i in anns_by_image]
    print(f"전체 이미지: {len(images_by_id)}장 / 분할 대상: {len(image_ids)}장 / "
          f"어노테이션: {len(coco['annotations'])}개")

    subsets = split_ids(image_ids, n, seed)
    all_ids = [i for s in subsets for i in s]
    assert len(all_ids) == len(set(all_ids)) == len(image_ids), "subset 간 중복/누락 발생"

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    index = {"seed": seed, "n": n, "drop_empty": drop_empty, "subsets": {}}
    subset_paths = []
    for k, ids in enumerate(subsets, start=1):
        sub = {
            "info": coco.get("info", {}),
            "licenses": coco.get("licenses", []),
            "images": [images_by_id[i] for i in sorted(ids)],
            "annotations": [a for i in sorted(ids) for a in anns_by_image.get(i, [])],
            "categories": coco["categories"],
        }
        name = f"subset_{k:02d}.json"
        with open(out_dir / name, "w", encoding="utf-8") as f:
            json.dump(sub, f)
        index["subsets"][name] = ids
        subset_paths.append(out_dir / name)
        print(f"{name}: 이미지 {len(sub['images'])}장, 어노테이션 {len(sub['annotations'])}개")

    with open(out_dir / "split_index.json", "w", encoding="utf-8") as f:
        json.dump(index, f)
    print(f"subset 저장 완료: {out_dir.resolve()}")
    return subset_paths


def coco_json_to_yolo(ann_path, img_src_dir, out_dir, split, cat_id_to_idx, use_symlink=True):
    """COCO JSON 하나를 YOLO 형식(images/<split>, labels/<split>)으로 변환한다."""
    with open(ann_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    img_out = Path(out_dir) / "images" / split
    lbl_out = Path(out_dir) / "labels" / split
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    anns_by_image = defaultdict(list)
    for a in coco["annotations"]:
        anns_by_image[a["image_id"]].append(a)

    n_img = n_box = 0
    for img in coco["images"]:
        src = Path(img_src_dir) / img["file_name"]
        dst = img_out / img["file_name"]
        if not dst.exists():
            if use_symlink:
                try:
                    os.symlink(src.resolve(), dst)
                except OSError as e:
                    print(f"심볼릭 링크 생성 실패({e}) -> 이후 이미지는 복사 방식으로 진행합니다.")
                    use_symlink = False
            if not use_symlink:
                tmp = dst.with_name(dst.name + ".tmp")
                shutil.copy2(src, tmp)
                tmp.replace(dst)

        w, h = img["width"], img["height"]
        lines = []
        for a in anns_by_image.get(img["id"], []):
            if a.get("iscrowd", 0):
                continue
            x, y, bw, bh = a["bbox"]
            if bw <= 0 or bh <= 0:
                continue
            cx = min(max((x + bw / 2) / w, 0.0), 1.0)
            cy = min(max((y + bh / 2) / h, 0.0), 1.0)
            nw = min(bw / w, 1.0)
            nh = min(bh / h, 1.0)
            lines.append(f"{cat_id_to_idx[a['category_id']]} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

        (lbl_out / (Path(img["file_name"]).stem + ".txt")).write_text("\n".join(lines))
        n_img += 1
        n_box += len(lines)

    print(f"[{split}] 이미지 {n_img}장, 박스 {n_box}개 변환 -> {img_out.parent.parent}")
    return use_symlink


def build_yolo_dataset(subset_json, train_img_dir, val_json, val_img_dir, work_dir, use_symlink=True):
    """subset_01(train)과 val2017(val)을 YOLO 형식으로 만들고 data.yaml 경로를 반환한다."""
    work_dir = Path(work_dir).resolve()
    cat_id_to_idx, names = common.load_category_map(subset_json)

    use_symlink = coco_json_to_yolo(subset_json, train_img_dir, work_dir, "train", cat_id_to_idx, use_symlink)
    coco_json_to_yolo(val_json, val_img_dir, work_dir, "val", cat_id_to_idx, use_symlink)

    yaml_path = work_dir / "data.yaml"
    lines = [f"path: {work_dir}", "train: images/train", "val: images/val", "names:"]
    lines += [f"  {i}: {n}" for i, n in enumerate(names)]
    yaml_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"data.yaml 저장: {yaml_path} (클래스 {len(names)}개)")
    return yaml_path


def train_labeler(data_yaml, project, name, cfg):
    """YOLOv8s 레이블러를 학습하고 Ultralytics 결과 폴더를 반환한다."""
    from ultralytics import YOLO

    cfg = dict(cfg)
    model = YOLO(cfg.pop("model"))
    model.train(data=str(data_yaml), project=str(Path(project).resolve()), name=name,
                exist_ok=True, **cfg)
    return Path(model.trainer.save_dir)


def plot_results(csv_path, out_png=None, show=True):
    """학습 기록(results.csv)으로 loss, Precision/Recall, mAP, 학습률 그래프를 저장한다."""
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    x = df["epoch"]

    panels = [
        ("Box loss", ["train/box_loss", "val/box_loss"]),
        ("Cls loss", ["train/cls_loss", "val/cls_loss"]),
        ("DFL loss", ["train/dfl_loss", "val/dfl_loss"]),
        ("Precision / Recall", ["metrics/precision(B)", "metrics/recall(B)"]),
        ("mAP", ["metrics/mAP50(B)", "metrics/mAP50-95(B)"]),
        ("Learning rate", ["lr/pg0", "lr/pg1", "lr/pg2"]),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    for ax, (title, cols) in zip(axes.ravel(), panels):
        for c in cols:
            if c in df.columns:
                ax.plot(x, df[c], label=c)
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Labeler (YOLOv8s) training on subset_01")
    fig.tight_layout()

    out_png = Path(out_png) if out_png else csv_path.parent / "training_curves.png"
    fig.savefig(out_png, dpi=150)
    print(f"학습 그래프 저장: {out_png}")
    if show:
        plt.show()
    plt.close(fig)
    return out_png


def weights_paths(out_dir):
    """저장된 레이블러 가중치 관련 파일 경로."""
    d = Path(out_dir) / "weights"
    return {"dir": d, "pt": d / "labeler_best.pt", "meta": d / "labeler_meta.json",
            "csv": d / "labeler_results.csv", "png": d / "training_curves.png"}


def read_results_csv(csv_path):
    """학습 기록에서 (학습한 에포크 수, 최고 val mAP@0.5:0.95)를 읽는다."""
    try:
        with open(csv_path, newline="", encoding="utf-8") as f:
            rows = [{k.strip(): v for k, v in r.items()} for r in csv.DictReader(f)]
        vals = [float(r["metrics/mAP50-95(B)"]) for r in rows]
        return len(rows), (max(vals) if vals else None)
    except Exception:
        return None, None


def save_trained_weights(out_dir, run_dir, tcfg, args):
    """best.pt, 학습 기록, 학습 일시·조건(labeler_meta.json)을 weights/에 저장한다."""
    wp = weights_paths(out_dir)
    wp["dir"].mkdir(parents=True, exist_ok=True)
    run_dir = Path(run_dir)

    best = run_dir / "weights" / "best.pt"
    if not best.exists():
        best = run_dir / "weights" / "last.pt"
    shutil.copy2(best, wp["pt"])
    if (run_dir / "results.csv").exists():
        shutil.copy2(run_dir / "results.csv", wp["csv"])

    epochs_run, best_map = read_results_csv(wp["csv"])
    meta = {
        "trained_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "epochs_planned": tcfg["epochs"],
        "epochs_run": epochs_run,
        "best_val_mAP50-95": best_map,
        "split": {"n": args.n, "seed": args.seed, "drop_empty": args.drop_empty},
        "train_config": tcfg,
        "source_run_dir": str(run_dir),
    }
    wp["meta"].write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"가중치 저장 완료: {wp['pt']}")
    print(f"학습 날짜 기록: {datetime.fromisoformat(meta['trained_at']):%Y-%m-%d %H:%M:%S} -> {wp['meta']}")
    return wp


def _saved_info(pt, csv_path, png, meta):
    """저장된 가중치 정보를 화면 출력·재사용 판단용 dict로 만든다."""
    trained_at, source = None, "학습 종료 시각 기록"
    if meta.get("trained_at"):
        try:
            trained_at = datetime.fromisoformat(meta["trained_at"])
        except ValueError:
            trained_at = None
    if trained_at is None:
        trained_at = datetime.fromtimestamp(pt.stat().st_mtime).astimezone()
        source = "파일 수정 시각, 학습 기록 없음"
    return {"pt": Path(pt), "csv": Path(csv_path) if Path(csv_path).exists() else None,
            "png": Path(png), "meta": meta, "trained_at": trained_at, "source": source}


def find_saved_labeler(out_dir):
    """저장된 레이블러 가중치 정보를 찾는다. 없으면 None."""
    wp = weights_paths(out_dir)
    if wp["pt"].exists():
        meta = {}
        if wp["meta"].exists():
            try:
                meta = json.loads(wp["meta"].read_text(encoding="utf-8"))
            except Exception:
                meta = {}
        return _saved_info(wp["pt"], wp["csv"], wp["png"], meta)

    legacy = Path(out_dir) / "runs" / "labeler"
    if (legacy / "weights" / "best.pt").exists():
        return _saved_info(legacy / "weights" / "best.pt", legacy / "results.csv",
                           legacy / "training_curves.png", {})
    return None


def describe_saved(info):
    """저장된 가중치의 학습 일시, 에포크, 최고 mAP, 분할 조건을 출력한다."""
    meta = info["meta"]
    epochs_run, best_map = read_results_csv(info["csv"]) if info["csv"] else (None, None)
    epochs_run = meta.get("epochs_run") or epochs_run
    best_map = meta.get("best_val_mAP50-95") or best_map

    print("\n저장된 Labeler 가중치가 있습니다.")
    print(f"  - 학습 날짜 : {info['trained_at']:%Y-%m-%d %H:%M:%S}  ({info['source']})")
    print(f"  - 파일      : {info['pt']}  ({info['pt'].stat().st_size / 1e6:.0f}MB)")
    if epochs_run:
        planned = meta.get("epochs_planned")
        print(f"  - epoch     : {epochs_run}" + (f" / 계획 {planned}" if planned else ""))
    if best_map is not None:
        print(f"  - val mAP50-95(최고): {best_map:.4f}")
    if meta.get("split"):
        s = meta["split"]
        print(f"  - 분할 조건 : n={s.get('n')}, seed={s.get('seed')}, drop_empty={s.get('drop_empty')}")


def ask_reuse():
    """기존 가중치를 쓸지 새로 학습할지 묻는다."""
    prompt = ("\n기존 가중치를 사용하시겠습니까?\n"
              "  [1] 기존 가중치 사용 (학습 생략, 기본값)\n"
              "  [2] 새로 학습 (기존 가중치는 weights/archive/ 에 보관)\n"
              "선택 (1/2, 그냥 Enter = 1): ")
    while True:
        try:
            ans = input(prompt).strip().lower()
        except EOFError:
            print("입력을 받을 수 없어 기존 가중치를 사용합니다.")
            return True
        if ans in ("", "1", "y"):
            return True
        if ans in ("2", "n"):
            return False
        print("1 또는 2를 입력하세요.")


def archive_saved_labeler(out_dir, info):
    """새로 학습하기 전에 기존 가중치를 weights/archive/에 학습 일시 이름으로 보관한다."""
    arch = weights_paths(out_dir)["dir"] / "archive"
    arch.mkdir(parents=True, exist_ok=True)
    stamp = f"{info['trained_at']:%Y%m%d_%H%M%S}"
    dst = arch / f"labeler_{stamp}.pt"
    if not dst.exists():
        shutil.copy2(info["pt"], dst)
        if info["meta"]:
            (arch / f"labeler_{stamp}_meta.json").write_text(
                json.dumps(info["meta"], ensure_ascii=False, indent=2), encoding="utf-8")
        if info["csv"]:
            shutil.copy2(info["csv"], arch / f"labeler_{stamp}_results.csv")
    print(f"기존 가중치 보관: {dst}")


def compute_map(conf, cls, tp, n_gt):
    """mAP@0.5와 mAP@0.5:0.95를 계산한다. GT가 없는 클래스는 평균에서 제외한다."""
    nc = len(n_gt)
    ap = np.full((nc, len(IOU_THRS)), np.nan)
    order = np.argsort(-conf, kind="stable")
    cls, tp = cls[order], tp[order]
    for c in range(nc):
        if n_gt[c] == 0:
            continue
        m = cls == c
        if not m.any():
            ap[c] = 0.0
            continue
        t = tp[m].astype(np.float64)
        tpc = np.cumsum(t, 0)
        fpc = np.cumsum(1 - t, 0)
        rec = tpc / n_gt[c]
        prec = tpc / (tpc + fpc)
        for k in range(len(IOU_THRS)):
            ap[c, k] = common._ap_101(rec[:, k], prec[:, k])
    if np.isnan(ap).all():
        return 0.0, 0.0
    return float(np.nanmean(ap[:, 0])), float(np.nanmean(ap))


def count_by_conf(acc, thresholds):
    """신뢰도 임계값별 TP, FP 개수를 센다 (IoU 0.5, 클래스 일치)."""
    if acc["conf"]:
        conf = np.concatenate(acc["conf"])
        tp0 = np.concatenate(acc["tp"])[:, 0]
    else:
        conf, tp0 = np.zeros(0), np.zeros(0, dtype=bool)
    counts = {}
    for t in thresholds:
        sel = conf >= t
        tp = int(tp0[sel].sum())
        counts[t] = (tp, int(sel.sum()) - tp)
    return {"images": acc["n_img"], "GT": int(acc["n_gt"].sum()), "counts": counts}


def merge_conf_counts(infos):
    """여러 부분집합의 count_by_conf() 결과를 합친다."""
    merged = {"images": 0, "GT": 0, "counts": {}}
    for info in infos:
        merged["images"] += info["images"]
        merged["GT"] += info["GT"]
        for t, (tp, fp) in info["counts"].items():
            a, b = merged["counts"].get(t, (0, 0))
            merged["counts"][t] = (a + tp, b + fp)
    return merged


def evaluate_subset(model, subset_json, img_dir, cat_id_to_idx, ecfg, imgsz, device, desc, limit=None):
    """부분집합 하나에 레이블러를 실행해 평가 누적 결과를 반환한다."""
    try:
        from tqdm import tqdm
    except ImportError:
        def tqdm(it, **kwargs):
            return it

    items = common.load_subset_gt(subset_json, cat_id_to_idx)
    if limit:
        items = items[:limit]
    nc = len(cat_id_to_idx)
    acc = common.new_acc(nc)
    if not items:
        return acc

    paths = [str(Path(img_dir) / it[0]) for it in items]
    def predict_in_chunks():
        step = ecfg["batch"]
        for s in range(0, len(paths), step):
            yield from model.predict(source=paths[s:s + step], imgsz=imgsz, conf=ecfg["pred_conf"],
                                     iou=ecfg["pred_iou"], max_det=ecfg["max_det"],
                                     device=device, verbose=False)

    for (_, gb, gc, gcrowd), r in tqdm(zip(items, predict_in_chunks()), total=len(items), desc=desc, leave=False):
        boxes = r.boxes
        if boxes is None or len(boxes) == 0:
            db, dc, dcls = np.zeros((0, 4)), np.zeros(0), np.zeros(0, np.int64)
        else:
            db = boxes.xyxy.cpu().numpy().astype(np.float64)
            dc = boxes.conf.cpu().numpy().astype(np.float64)
            dcls = boxes.cls.cpu().numpy().astype(np.int64)
        c, k, t = common.match_image(db, dc, dcls, gb, gc, gcrowd)
        acc["conf"].append(c)
        acc["cls"].append(k)
        acc["tp"].append(t)
        acc["n_gt"] += np.bincount(gc[~gcrowd], minlength=nc)
        acc["n_img"] += 1
    return acc


def print_table(rows, csv_path):
    """conf_thr 기준 평가표(TP, FP, Precision, Recall, mAP)를 출력하고 csv로 저장한다."""
    import pandas as pd

    df = pd.DataFrame(rows).set_index("set")
    fmt = {c: "{:.4f}".format for c in ["Precision", "Recall", "mAP@0.5", "mAP@0.5:0.95"]}
    print("\n" + df.to_string(formatters=fmt))
    df.to_csv(csv_path, encoding="utf-8-sig")
    print(f"\n평가 결과 저장: {csv_path}")
    return df


def conf_rows(name, info):
    """count_by_conf() 결과를 임계값별 표의 행으로 바꾼다."""
    gt, n_img = info["GT"], info["images"]
    rows = []
    for t, (tp, fp) in info["counts"].items():
        rows.append({
            "set": name, "conf_thr": t, "images": n_img, "GT": gt, "TP": tp, "FP": fp, "FN": gt - tp,
            "Precision": tp / (tp + fp) if (tp + fp) else float("nan"),
            "Recall": tp / gt if gt else 0.0,
            "unlabeled_ratio": (gt - tp) / gt if gt else 0.0,
            "labels_per_img": (tp + fp) / n_img if n_img else 0.0,
        })
    return rows


def print_conf_table(rows, csv_path, show_sets):
    """임계값별 Precision/Recall 표를 csv로 저장하고 show_sets의 행만 출력한다."""
    import pandas as pd

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    fmt = {c: "{:.4f}".format for c in ["Precision", "Recall", "unlabeled_ratio"]}
    fmt["labels_per_img"] = "{:.3f}".format
    shown = df[df["set"].isin(show_sets)].set_index(["set", "conf_thr"])
    print("\n[의사 라벨링 임계값 참고표] confidence 임계값별 라벨 정확도(Precision)와 라벨링되는 객체 비율(Recall)")
    print(shown.to_string(formatters=fmt))
    print(f"\n임계값별 표 저장(전체 subset 포함): {csv_path}")
    return df


def build_arg_parser():
    """명령행 옵션."""
    p = argparse.ArgumentParser()
    p.add_argument("--coco-root", default="coco", help="train2017/, val2017/, annotations/ 가 있는 폴더")
    p.add_argument("--out-dir", default="outputs", help="subset, YOLO 데이터, 학습 결과, 평가 결과 출력 폴더")
    p.add_argument("--n", type=int, default=20, help="subset 개수")
    p.add_argument("--seed", type=int, default=42, help="분할 seed")
    p.add_argument("--drop-empty", action="store_true", help="어노테이션 없는 이미지를 분할에서 제외")
    p.add_argument("--copy-images", action="store_true", help="심볼릭 링크 대신 이미지 복사 (Windows)")
    p.add_argument("--no-show", action="store_true", help="그래프 창을 띄우지 않고 저장만")
    p.add_argument("--eval-max-images", type=int, default=None,
                   help="subset당 평가 이미지 수 제한 (빠른 동작 확인용, 기본: 전체)")

    p.add_argument("--skip-split", action="store_true", help="이미 만든 subset JSON을 재사용")

    p.add_argument("--skip-train", action="store_true", help="묻지 않고 저장된 가중치 사용 (학습 생략)")
    p.add_argument("--retrain", action="store_true", help="묻지 않고 새로 학습 (기존 가중치는 archive에 보관)")
    p.add_argument("--weights", default=None, help="직접 지정한 가중치(.pt)를 사용 (학습 생략)")
    return p


def resolve_labeler_weights(out, args):
    """저장된 가중치를 쓸지 새로 학습할지 정한다."""
    saved = find_saved_labeler(out)

    if args.weights:
        w = Path(args.weights)
        if not w.exists():
            raise SystemExit(f"가중치를 찾을 수 없습니다: {w}")
        meta_file, meta = w.with_name("labeler_meta.json"), {}
        if meta_file.exists():
            try:
                meta = json.loads(meta_file.read_text(encoding="utf-8"))
            except Exception:
                meta = {}
        saved_used = _saved_info(w, w.parent.parent / "results.csv", w.parent.parent / "training_curves.png", meta)
        return True, saved, saved_used

    if args.skip_train:
        if saved is None:
            raise SystemExit(f"저장된 가중치가 없습니다: {weights_paths(out)['pt']} (--weights로 지정하거나 학습하세요)")
        return True, saved, saved

    if args.retrain or saved is None:
        if saved is None:
            print("\n저장된 가중치가 없어 새로 학습합니다.")
        return False, saved, None

    describe_saved(saved)
    use_saved = ask_reuse()
    return use_saved, saved, (saved if use_saved else None)


def check_coco_paths(root, out, need_train_ann):
    """COCO 필수 경로를 확인한다 (분할할 때만 train 어노테이션이 필요)."""
    train_ann = root / "annotations" / "instances_train2017.json"
    val_ann = root / "annotations" / "instances_val2017.json"
    train_img, val_img = root / "train2017", root / "val2017"

    required = [("train 이미지 폴더", train_img),
                ("val 어노테이션", val_ann), ("val 이미지 폴더", val_img)]
    if need_train_ann:
        required.append(("train 어노테이션", train_ann))

    missing = [(label, path) for label, path in required if not path.exists()]
    if missing:
        print(f"현재 작업 폴더: {Path.cwd()}")
        print(f"--coco-root : {root.resolve()}")
        for label, path in missing:
            print(f"  [없음] {label}: {path.resolve()}")
        raise SystemExit("COCO 경로를 찾을 수 없습니다. --coco-root에 coco 폴더의 절대 경로를 지정하세요.")
    return train_ann, val_ann, train_img, val_img


def main():
    """분할 → 초기 레이블러 학습(또는 재사용) → 부분집합·val2017 평가 순서로 실행한다."""
    args = build_arg_parser().parse_args()

    root = Path(args.coco_root)
    out = Path(args.out_dir).resolve()
    p1_out = out / "p1_output"
    subset_dir = p1_out / "subsets"
    p1_out.mkdir(parents=True, exist_ok=True)
    train_ann, val_ann, train_img, val_img = check_coco_paths(root, p1_out, need_train_ann=not args.skip_split)

    use_saved, saved, saved_used = resolve_labeler_weights(p1_out, args)

    print(f"\n===== [1/4] COCO train2017 -> {args.n}개 subset 분할 =====")
    subset_files = [subset_dir / f"subset_{k:02d}.json" for k in range(1, args.n + 1)]
    if args.skip_split and all(f.exists() for f in subset_files):
        print("기존 subset JSON 재사용")
    else:
        subset_files = split_train(train_ann, subset_dir, args.n, args.seed, args.drop_empty)

    print("\n===== [2/4] subset 1로 Labeler(YOLOv8s) 학습 =====")
    tcfg = get_train_config()
    if use_saved:
        weights, results_csv, curve_png = saved_used["pt"], saved_used["csv"], saved_used["png"]
        print(f"학습 생략, 저장된 가중치 사용: {weights}")
        print(f"학습 날짜: {saved_used['trained_at']:%Y-%m-%d %H:%M:%S} ({saved_used['source']})")

        s = saved_used["meta"].get("split")
        if s and (s.get("n"), s.get("seed"), s.get("drop_empty")) != (args.n, args.seed, args.drop_empty):
            print(f"[경고] 이 가중치는 분할 조건 n={s.get('n')}, seed={s.get('seed')}, drop_empty={s.get('drop_empty')}"
                  f"로 학습되었는데 지금은 n={args.n}, seed={args.seed}, drop_empty={args.drop_empty}입니다. "
                  "subset 구성이 달라져 학습에 쓴 이미지가 평가 subset에 섞일 수 있습니다.")
    else:
        if saved is not None:
            archive_saved_labeler(p1_out, saved)
        data_yaml = build_yolo_dataset(subset_files[0], train_img, val_ann, val_img,
                                       p1_out / "yolo_data", use_symlink=not args.copy_images)
        run_dir = train_labeler(data_yaml, p1_out / "runs", "labeler", tcfg)
        wp = save_trained_weights(p1_out, run_dir, tcfg, args)
        weights, results_csv, curve_png = wp["pt"], (wp["csv"] if wp["csv"].exists() else None), wp["png"]

    if results_csv:
        plot_results(results_csv, out_png=curve_png, show=not args.no_show)

    print("\n===== [3/4] Labeler로 subset 2~n 평가 / [4/4] COCO validate(val2017) 평가 =====")
    from ultralytics import YOLO

    model = YOLO(str(weights))
    ecfg = get_eval_config()
    cat_id_to_idx, _ = common.load_category_map(subset_files[0])

    def run_eval(name, ann_json, img_dir):
        acc = evaluate_subset(model, ann_json, img_dir, cat_id_to_idx, ecfg, tcfg["imgsz"],
                              tcfg["device"], desc=name, limit=args.eval_max_images)
        row = {"set": name}
        row.update(common.summarize(acc, ecfg["conf_thr"]))
        print(f"{name}: TP={row['TP']} FP={row['FP']} P={row['Precision']:.4f} "
              f"R={row['Recall']:.4f} mAP50={row['mAP@0.5']:.4f} mAP50-95={row['mAP@0.5:0.95']:.4f}")
        return row, count_by_conf(acc, ecfg["conf_grid"])

    rows, conf_infos, conf_table = [], [], []
    for k in range(2, args.n + 1):
        name = f"subset_{k:02d}"
        row, info = run_eval(name, subset_files[k - 1], train_img)
        rows.append(row)
        conf_infos.append(info)
        conf_table += conf_rows(name, info)

    val_row, val_info = run_eval("val2017 (COCO validate)", val_ann, val_img)
    rows.append(val_row)

    pooled_name = f"subset 2~{args.n} 합계 (unlabeled)"
    conf_table = (conf_rows(pooled_name, merge_conf_counts(conf_infos))
                  + conf_table
                  + conf_rows("val2017 (COCO validate)", val_info))

    print(f"\n[평가 기준] TP/FP/Precision/Recall: conf>={ecfg['conf_thr']}, IoU>=0.5 / "
          f"mAP: conf>={ecfg['pred_conf']}, COCO 101점 보간")
    print_table(rows, p1_out / "eval_results.csv")
    print_conf_table(conf_table, p1_out / "eval_by_conf.csv", show_sets=[pooled_name, "val2017 (COCO validate)"])


if __name__ == "__main__":
    main()
