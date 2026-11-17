"""실험 코드(p2_1_fixedthr_ext.py, p2_2_pseudo_label_ext.py)가 함께 쓰는 공통 함수와 설정값.

직접 실행하지 않고 `import p0_common_api as common`으로 불러 쓴다. 주요 설정은 세 함수에 모여 있다.
  - get_eval_config(): 예측·평가 설정
  - get_selection_config(): 의사 레이블 선택(클래스별 임계값 산정, 범위, 갱신 주기) 설정
  - get_retrain_config(): 라운드별 재학습 설정과 라운드 조기 종료 기준

평가 지표
  - mAP@0.5:0.95: conf >= 0.001 예측, COCO 101점 보간, IoU 0.5~0.95(0.05 간격) 평균
  - TP/FP/Precision/Recall: conf >= 0.25, IoU >= 0.5, 같은 클래스 GT와 신뢰도 순 1:1 greedy 매칭
  - iscrowd GT와 겹치는 미매칭 예측은 무시하고, iscrowd GT는 GT 수에서 제외한다.
"""

import datetime
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np


IOU_THRS = np.linspace(0.5, 0.95, 10)


def get_eval_config(max_det=100):
    """예측·평가 설정. max_det(이미지당 최대 예측 수)은 호출하는 쪽에서 넘긴다."""
    return {
        "imgsz": 640,
        "device": None,
        "pred_conf": 0.001,  # mAP 계산과 의사 레이블 후보에 쓰는 최소 신뢰도
        "pred_iou": 0.6,  # NMS IoU
        "max_det": max_det,
        "batch": 32,
        "conf_thr": 0.25,  # Precision/Recall/TP/FP를 셀 때의 신뢰도 기준
    }


def get_selection_config():
    """의사 레이블 선택 설정 (클래스별 임계값 산정, 범위, 갱신 주기)."""
    return {
        "target_precision": 0.6,  # 클래스별 임계값: 보정용 데이터셋 정밀도가 이 값 이상인 가장 낮은 신뢰도
        "min_gt_per_class": 20,  # 보정용 데이터셋 GT가 이보다 적으면 fallback_conf 사용
        "fallback_conf": 0.4,  # 임계값을 정할 수 없는 클래스에 쓰는 값

        "min_conf_thr": 0.30,  # 범위 하한 (범위 없음은 0.00)
        "max_conf_thr": 0.50,  # 범위 상한 (범위 없음은 1.00)

        "recalib_every_n_rounds": 3,  # None이면 고정(실험 B), k이면 k라운드마다 갱신(실험 C)

        "use_max_box_filter": False,  # 논문 실험에서는 끔
        "max_boxes_per_image": 20,

        "use_class_quota": False,  # 논문 실험에서는 끔
        "max_per_class_per_round": None,
    }


def get_retrain_config():
    """라운드별 재학습 설정. 직전 라운드 가중치에서 이어서 학습하므로 초기 학습보다 에포크와 학습률을 낮게 둔다."""
    return {
        "batch": 16,
        "imgsz": 640,
        "optimizer": "SGD",
        "momentum": 0.937,
        "weight_decay": 0.0005,
        "mosaic": 1.0,
        "copy_paste": 0.2,
        "mixup": 0.15,
        "workers": 8,
        "device": None,
        "plots": True,

        "epochs": 12,  # 라운드당 에포크
        "patience": 6,  # 에포크 조기 종료 (검증 성능이 6 에포크 동안 오르지 않으면 중단)
        "lr0": 0.002,  # 직전 라운드 가중치에서 이어서 학습하므로 초기 학습(0.01)보다 낮게
        "lrf": 0.01,

        "close_mosaic": 3,  # 마지막 3 에포크는 mosaic 끔
        "warmup_epochs": 1,

        "max_pseudo_rounds_kept": 3,  # subset_01 정답 + 최근 3라운드 의사 레이블만 학습에 사용

        "round_early_stop_patience": 3,  # val mAP@0.5:0.95가 직전 라운드보다 3번 연속 떨어지면 라운드 반복 중단
    }


def add_common_args(parser):
    """두 실험 코드가 공통으로 쓰는 명령행 옵션을 추가한다."""
    parser.add_argument("--coco-root", default="coco", help="train2017/, val2017/, annotations/ 가 있는 폴더")
    parser.add_argument("--out-dir", default="outputs",
                         help="subset JSON·가중치를 읽고 결과를 저장할 폴더 "
                              "(p1_data_slicing_and_initial_learning.py와 동일해야 함)")
    parser.add_argument("--n", type=int, default=20,
                         help="읽어올 subset 개수 (p1_data_slicing_and_initial_learning.py 실행 때의 --n과 같아야 함)")
    parser.add_argument("--weights", default=None,
                         help="기본 위치(outputs/weights/labeler_best.pt) 대신 직접 지정한 가중치(.pt) 사용")
    parser.add_argument("--eval-max-images", type=int, default=None,
                         help="subset/val 당 처리 이미지 수 제한 (빠른 동작 확인용, 기본: 전체)")
    parser.add_argument("--no-show", action="store_true",
                         help="라운드별 val2017 성능 그래프를 창으로 띄우지 않고 파일로만 저장")
    return parser


def load_category_map(subset_json):
    """COCO category_id(1~90, 비연속)를 0~79 인덱스로 바꾸는 매핑과 클래스 이름 목록을 반환한다."""
    with open(subset_json, "r", encoding="utf-8") as f:
        categories = sorted(json.load(f)["categories"], key=lambda c: c["id"])
    return {c["id"]: i for i, c in enumerate(categories)}, [c["name"] for c in categories]


def load_image_sizes(subset_json):
    """부분집합 JSON에서 {file_name: (width, height)}를 만든다."""
    with open(subset_json, "r", encoding="utf-8") as f:
        coco = json.load(f)
    return {img["file_name"]: (img["width"], img["height"]) for img in coco["images"]}


def require_subset_files(subset_dir, n):
    """subset_01~n.json이 모두 있는지 확인하고 경로 목록을 반환한다."""
    files = [Path(subset_dir) / f"subset_{k:02d}.json" for k in range(1, n + 1)]
    missing = [f for f in files if not f.exists()]
    if missing:
        print(f"subset JSON을 찾을 수 없습니다: {subset_dir}")
        for f in missing[:5]:
            print(f"  [없음] {f}")
        if len(missing) > 5:
            print(f"  ... 외 {len(missing) - 5}개")
        raise SystemExit(
            "먼저 p1_data_slicing_and_initial_learning.py로 COCO train2017을 분할하세요.\n"
            "  python p1_data_slicing_and_initial_learning.py --coco-root <coco 경로> --out-dir <out-dir 경로>"
        )
    return files


def load_labeler_weights(out_dir, weights_arg):
    """초기 레이블러 가중치 경로를 찾는다 (--weights가 있으면 그 경로)."""
    if weights_arg:
        path = Path(weights_arg)
        if not path.exists():
            raise SystemExit(f"가중치를 찾을 수 없습니다: {path}")
        meta_path = path.with_name(path.stem.replace("best", "meta") + ".json")
        if not meta_path.exists():
            meta_path = path.with_name("labeler_meta.json")
    else:
        path = Path(out_dir) / "weights" / "labeler_best.pt"
        meta_path = path.with_name("labeler_meta.json")
        if not path.exists():
            legacy = Path(out_dir) / "runs" / "labeler" / "weights" / "best.pt"
            if not legacy.exists():
                raise SystemExit(
                    f"학습된 가중치를 찾을 수 없습니다: {path}\n"
                    "먼저 p1_data_slicing_and_initial_learning.py로 Labeler를 학습하거나, "
                    "--weights로 가중치 경로를 직접 지정하세요."
                )
            path, meta_path = legacy, None

    meta = {}
    if meta_path and meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            meta = {}

    print(f"\nLabeler 가중치 읽음: {path}  ({path.stat().st_size / 1e6:.0f}MB)")
    if meta.get("trained_at"):
        print(f"  - 학습 날짜 : {meta['trained_at']}")
    if meta.get("epochs_run"):
        planned = meta.get("epochs_planned")
        print(f"  - epoch     : {meta['epochs_run']}" + (f" / 계획 {planned}" if planned else ""))
    if meta.get("best_val_mAP50-95") is not None:
        print(f"  - val mAP50-95(최고): {meta['best_val_mAP50-95']:.4f}")
    if meta.get("split"):
        s = meta["split"]
        print(f"  - 분할 조건 : n={s.get('n')}, seed={s.get('seed')}, drop_empty={s.get('drop_empty')}")
    if not meta:
        print("  - 학습 날짜/조건 기록 없음 (labeler_meta.json을 찾지 못함)")
    return path


def check_coco_paths(root):
    """COCO 필수 경로(val 어노테이션, train2017, val2017)를 확인한다."""
    val_ann = root / "annotations" / "instances_val2017.json"
    train_img, val_img = root / "train2017", root / "val2017"

    required = [("train 이미지 폴더", train_img), ("val 어노테이션", val_ann), ("val 이미지 폴더", val_img)]
    missing = [(label, path) for label, path in required if not path.exists()]
    if missing:
        print(f"현재 작업 폴더: {Path.cwd()}")
        print(f"--coco-root : {root.resolve()}")
        for label, path in missing:
            print(f"  [없음] {label}: {path.resolve()}")
        raise SystemExit("COCO 경로를 찾을 수 없습니다. --coco-root에 coco 폴더의 절대 경로를 지정하세요.")
    return val_ann, train_img, val_img


def box_iou_parts(a, b):
    """xyxy 박스 a(n,4), b(m,4)의 IoU(n,m), 교집합 넓이(n,m), a의 넓이(n)를 반환한다."""
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / np.maximum(union, 1e-9), inter, area_a


def match_image(db, dc, dcls, gb, gcls, gcrowd):
    """이미지 1장의 예측을 GT와 매칭한다 (같은 클래스 GT와 신뢰도 순 1:1 greedy, iscrowd GT와 겹치는 미매칭 예측은 제외)."""
    order = np.argsort(-dc, kind="stable")
    db, dc, dcls = db[order], dc[order], dcls[order]
    n = len(db)
    tp = np.zeros((n, len(IOU_THRS)), dtype=bool)
    if n == 0:
        return dc, dcls, tp

    valid = ~gcrowd
    g_b, g_c = gb[valid], gcls[valid]
    if len(g_b):
        iou, _, _ = box_iou_parts(db, g_b)
        iou = np.where(dcls[:, None] == g_c[None, :], iou, 0.0)
        for ti, thr in enumerate(IOU_THRS):
            matched = np.zeros(len(g_b), dtype=bool)
            for i in np.nonzero(iou.max(1) >= thr)[0]:
                row = np.where(matched, -1.0, iou[i])
                j = int(row.argmax())
                if row[j] >= thr:
                    tp[i, ti] = True
                    matched[j] = True

    keep = np.ones(n, dtype=bool)
    if gcrowd.any():
        cr_b, cr_c = gb[gcrowd], gcls[gcrowd]
        _, inter, area_d = box_iou_parts(db, cr_b)
        ratio = np.where(dcls[:, None] == cr_c[None, :], inter / np.maximum(area_d[:, None], 1e-9), 0.0)
        keep = ~((~tp[:, 0]) & (ratio.max(1) >= 0.5))
    return dc[keep], dcls[keep], tp[keep]


def _ap_101(rec, prec):
    """COCO 방식 101점 보간 AP."""
    prec = np.maximum.accumulate(prec[::-1])[::-1]
    idx = np.searchsorted(rec, np.linspace(0, 1, 101), side="left")
    q = np.zeros(101)
    ok = idx < len(rec)
    q[ok] = prec[idx[ok]]
    return q.mean()


def compute_ap_per_class(conf, cls, tp, n_gt):
    """클래스별·IoU 기준별 AP를 (nc, 10) 배열로 반환한다. GT가 없는 클래스는 nan."""
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
            ap[c, k] = _ap_101(rec[:, k], prec[:, k])
    return ap


def compute_map(conf, cls, tp, n_gt):
    """mAP@0.5와 mAP@0.5:0.95를 계산한다. GT가 없는 클래스는 평균에서 제외한다."""
    ap = compute_ap_per_class(conf, cls, tp, n_gt)
    if np.isnan(ap).all():
        return 0.0, 0.0
    return float(np.nanmean(ap[:, 0])), float(np.nanmean(ap))


def new_acc(nc):
    """평가 결과를 모을 빈 누적 객체를 만든다."""
    return {"conf": [], "cls": [], "tp": [], "n_gt": np.zeros(nc, dtype=np.int64), "n_img": 0}


def summarize(acc, conf_thr):
    """누적 결과에서 TP, FP, FN, Precision, Recall(conf_thr 기준)과 mAP를 계산한다."""
    nc = len(acc["n_gt"])
    if acc["conf"]:
        conf = np.concatenate(acc["conf"])
        cls = np.concatenate(acc["cls"])
        tp = np.concatenate(acc["tp"])
    else:
        conf, cls, tp = np.zeros(0), np.zeros(0, int), np.zeros((0, len(IOU_THRS)), bool)

    n_gt = int(acc["n_gt"].sum())
    sel = conf >= conf_thr
    TP = int(tp[sel, 0].sum())
    FP = int(sel.sum()) - TP
    FN = n_gt - TP
    precision = TP / (TP + FP) if (TP + FP) else 0.0
    recall = TP / n_gt if n_gt else 0.0
    map50, map5095 = compute_map(conf, cls, tp, acc["n_gt"]) if len(conf) else (0.0, 0.0)
    return {
        "images": acc["n_img"], "GT": n_gt, "TP": TP, "FP": FP, "FN": FN,
        "Precision": precision, "Recall": recall,
        "mAP@0.5": map50, "mAP@0.5:0.95": map5095,
    }


def summarize_per_class(acc, class_names, conf_thr):
    """summarize()의 클래스별 버전. 클래스마다 GT, TP, FP, FN, Precision, Recall, AP를 반환한다."""
    nc = len(acc["n_gt"])
    if acc["conf"]:
        conf = np.concatenate(acc["conf"])
        cls = np.concatenate(acc["cls"])
        tp = np.concatenate(acc["tp"])
    else:
        conf, cls, tp = np.zeros(0), np.zeros(0, int), np.zeros((0, len(IOU_THRS)), bool)

    ap = compute_ap_per_class(conf, cls, tp, acc["n_gt"]) if len(conf) else np.full((nc, len(IOU_THRS)), np.nan)

    rows = []
    for c, name in enumerate(class_names):
        n_gt_c = int(acc["n_gt"][c])
        m = cls == c
        sel = m & (conf >= conf_thr)
        TP = int(tp[sel, 0].sum()) if sel.any() else 0
        FP = int(sel.sum()) - TP
        FN = n_gt_c - TP
        ap50 = float(ap[c, 0]) if not np.isnan(ap[c, 0]) else 0.0
        ap5095 = float(np.nanmean(ap[c])) if not np.all(np.isnan(ap[c])) else 0.0
        rows.append({
            "class": name, "GT": n_gt_c, "TP": TP, "FP": FP, "FN": FN,
            "Precision": (TP / (TP + FP)) if (TP + FP) else 0.0,
            "Recall": (TP / n_gt_c) if n_gt_c else 0.0,
            "AP@0.5": ap50, "AP@0.5:0.95": ap5095,
        })
    return rows


def derive_class_thresholds(acc, scfg):
    """보정용 데이터셋 예측으로 클래스별 임계값을 정한다.

    클래스마다 예측을 신뢰도 내림차순으로 정렬해, 누적 정밀도가 target_precision 이상인 가장 낮은
    신뢰도를 임계값으로 쓴다. GT가 min_gt_per_class 미만이거나 목표에 못 미치면 fallback_conf를 쓰고,
    마지막에 모든 클래스를 [min_conf_thr, max_conf_thr]로 자른다. (임계값 배열, 클래스별 진단값)을 반환한다.
    """
    nc = len(acc["n_gt"])
    thresholds = np.full(nc, scfg["fallback_conf"], dtype=np.float64)
    used_fallback = np.ones(nc, dtype=bool)

    if acc["conf"]:
        conf = np.concatenate(acc["conf"])
        cls = np.concatenate(acc["cls"])
        tp0 = np.concatenate(acc["tp"])[:, 0]

        for c in range(nc):
            if acc["n_gt"][c] < scfg["min_gt_per_class"]:
                continue
            m = cls == c
            if not m.any():
                continue

            cc, tt = conf[m], tp0[m]
            order = np.argsort(-cc, kind="stable")
            cc, tt = cc[order], tt[order]
            tp_cum = np.cumsum(tt)
            fp_cum = np.cumsum(~tt)
            precision = tp_cum / (tp_cum + fp_cum)

            ok = np.nonzero(precision >= scfg["target_precision"])[0]
            if len(ok):
                thresholds[c] = float(cc[ok[-1]])
                used_fallback[c] = False
    else:
        conf = cls = tp0 = np.zeros(0)

    # 범위 제한: fallback을 쓴 클래스를 포함해 모든 클래스에 적용
    lo, hi = scfg.get("min_conf_thr"), scfg.get("max_conf_thr")
    if lo is not None or hi is not None:
        clipped = np.clip(thresholds, lo if lo is not None else 0.0, hi if hi is not None else 1.0)
        n_clipped = int(np.sum(~np.isclose(clipped, thresholds)))
        if n_clipped:
            print(f"  [임계값 clip] {n_clipped}개 클래스의 역산 임계값이 범위 "
                  f"[{lo}, {hi}] 밖이라 조정됨")
        thresholds = clipped

    diagnostics = _threshold_diagnostics(conf, cls, tp0, acc["n_gt"], thresholds, used_fallback)
    return thresholds, diagnostics


def _threshold_diagnostics(conf, cls, tp0, n_gt, thresholds, used_fallback):
    """확정된 임계값에서 클래스별로 실제 달성되는 Precision, Recall, 채택 예측 수를 계산한다."""
    diagnostics = []
    for c in range(len(n_gt)):
        m = cls == c
        sel = m & (conf >= thresholds[c])
        n_sel = int(sel.sum())
        tp_sel = int(tp0[sel].sum()) if n_sel else 0
        diagnostics.append({
            "used_fallback": bool(used_fallback[c]),
            "n_selected_val": n_sel,
            "achieved_precision": (tp_sel / n_sel) if n_sel else float("nan"),
            "achieved_recall": (tp_sel / n_gt[c]) if n_gt[c] else 0.0,
        })
    return diagnostics


def _progress(iterable, **kwargs):
    """tqdm이 있으면 진행률 표시줄로 순회한다."""
    try:
        from tqdm import tqdm
        return tqdm(iterable, **kwargs)
    except ImportError:
        return iterable


def _predict_stream(model, img_dir, items, ecfg):
    """이미지 목록에 레이블러 예측을 스트리밍으로 실행한다."""
    paths = [str(Path(img_dir) / it[0]) for it in items]
    step = ecfg["batch"]
    for s in range(0, len(paths), step):
        yield from model.predict(source=paths[s:s + step], imgsz=ecfg["imgsz"], conf=ecfg["pred_conf"],
                                 iou=ecfg["pred_iou"], max_det=ecfg["max_det"],
                                 device=ecfg["device"], verbose=False)


def _boxes_from_result(r):
    """Ultralytics 예측 결과 1장을 (xyxy, confidence, 클래스 인덱스) 배열로 바꾼다."""
    boxes = r.boxes
    if boxes is None or len(boxes) == 0:
        return np.zeros((0, 4)), np.zeros(0), np.zeros(0, np.int64)
    return (boxes.xyxy.cpu().numpy().astype(np.float64),
            boxes.conf.cpu().numpy().astype(np.float64),
            boxes.cls.cpu().numpy().astype(np.int64))


def load_subset_gt(subset_json, cat_id_to_idx):
    """부분집합 JSON에서 [(file_name, gt_xyxy, gt_cls, gt_iscrowd), ...]를 만든다."""
    with open(subset_json, "r", encoding="utf-8") as f:
        coco = json.load(f)
    by_img = defaultdict(list)
    for a in coco["annotations"]:
        by_img[a["image_id"]].append(a)

    items = []
    for img in coco["images"]:
        anns = [a for a in by_img.get(img["id"], []) if a["bbox"][2] > 0 and a["bbox"][3] > 0]
        if anns:
            b = np.array([a["bbox"] for a in anns], dtype=np.float64).reshape(-1, 4)
            b[:, 2:] += b[:, :2]
            c = np.array([cat_id_to_idx[a["category_id"]] for a in anns], dtype=np.int64)
            crowd = np.array([bool(a.get("iscrowd", 0)) for a in anns])
        else:
            b, c, crowd = np.zeros((0, 4)), np.zeros(0, np.int64), np.zeros(0, bool)
        items.append((img["file_name"], b, c, crowd))
    return items


def evaluate_subset(model, subset_json, img_dir, cat_id_to_idx, ecfg, desc, limit=None):
    """부분집합 하나에 레이블러를 실행해 평가 누적 결과를 반환한다."""
    items = load_subset_gt(subset_json, cat_id_to_idx)
    if limit:
        items = items[:limit]
    nc = len(cat_id_to_idx)
    acc = new_acc(nc)
    if not items:
        return acc

    results = _predict_stream(model, img_dir, items, ecfg)
    for (_, gb, gc, gcrowd), r in _progress(zip(items, results), total=len(items), desc=desc, leave=False):
        db, dc, dcls = _boxes_from_result(r)
        c, k, t = match_image(db, dc, dcls, gb, gc, gcrowd)
        acc["conf"].append(c)
        acc["cls"].append(k)
        acc["tp"].append(t)
        acc["n_gt"] += np.bincount(gc[~gcrowd], minlength=nc)
        acc["n_img"] += 1
    return acc


def stage_image(src, dst, use_symlink):
    """이미지 1장을 학습 폴더에 둔다 (심볼릭 링크, 실패하면 복사). 다음 호출에 쓸 use_symlink를 반환한다."""
    if dst.exists():
        return use_symlink
    if use_symlink:
        try:
            os.symlink(Path(src).resolve(), dst)
            return True
        except OSError:
            print(f"  [알림] 심볼릭 링크 실패(권한 부족) -> 파일 복사로 전환: {dst.name}")
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    shutil.copy2(src, tmp)
    tmp.replace(dst)
    return False


def coco_gt_to_yolo(ann_path, img_src_dir, work_dir, split, cat_id_to_idx, use_symlink=True):
    """COCO 형식 정답을 YOLO 형식(images/<split>, labels/<split>)으로 변환한다."""
    with open(ann_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    img_out = Path(work_dir) / "images" / split
    lbl_out = Path(work_dir) / "labels" / split
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    anns_by_image = defaultdict(list)
    for a in coco["annotations"]:
        anns_by_image[a["image_id"]].append(a)

    n_img = n_box = 0
    for img in coco["images"]:
        src = Path(img_src_dir) / img["file_name"]
        dst = img_out / img["file_name"]
        use_symlink = stage_image(src, dst, use_symlink)

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
            nw, nh = min(bw / w, 1.0), min(bh / h, 1.0)
            lines.append(f"{cat_id_to_idx[a['category_id']]} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
        (lbl_out / (Path(img["file_name"]).stem + ".txt")).write_text("\n".join(lines))
        n_img += 1
        n_box += len(lines)

    print(f"[{split}] 이미지 {n_img}장, 박스 {n_box}개 변환 -> {img_out.parent.parent}")
    return use_symlink


def append_pseudo_round(work_dir, img_src_dir, selected, use_symlink, round_idx):
    """채택된 의사 레이블을 누적 학습셋에 추가하고, 라운드별 추가 목록(manifests/round_XX.json)을 남긴다."""
    img_out = Path(work_dir) / "images" / "train"
    lbl_out = Path(work_dir) / "labels" / "train"
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)
    for fname, lines in selected:
        src = Path(img_src_dir) / fname
        dst = img_out / fname
        use_symlink = stage_image(src, dst, use_symlink)
        (lbl_out / (Path(fname).stem + ".txt")).write_text("\n".join(lines))

    manifest_dir = Path(work_dir) / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    fnames = [fname for fname, _ in selected]
    (manifest_dir / f"round_{round_idx:02d}.json").write_text(json.dumps(fnames))
    return use_symlink


def prune_pseudo_round(work_dir, round_idx):
    """해당 라운드가 추가한 의사 레이블을 누적 학습셋에서 제거한다 (subset_01 정답은 유지)."""
    manifest_path = Path(work_dir) / "manifests" / f"round_{round_idx:02d}.json"
    if not manifest_path.exists():
        return 0
    fnames = json.loads(manifest_path.read_text())
    img_dir = Path(work_dir) / "images" / "train"
    lbl_dir = Path(work_dir) / "labels" / "train"
    for fname in fnames:
        for p in (img_dir / fname, lbl_dir / (Path(fname).stem + ".txt")):
            p.unlink(missing_ok=True)
    manifest_path.unlink()
    return len(fnames)


def write_data_yaml(work_dir, names):
    """누적 학습셋의 data.yaml을 쓴다."""
    work_dir = Path(work_dir).resolve()
    yaml_path = work_dir / "data.yaml"
    lines = [f"path: {work_dir}", "train: images/train", "val: images/val", "names:"]
    lines += [f"  {i}: {n}" for i, n in enumerate(names)]
    yaml_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return yaml_path


def setup_cumulative_dataset(subset1_json, train_img, val_ann, val_img, cat_id_to_idx, class_names, work_dir):
    """누적 학습셋의 시작점을 만든다: subset_01 정답 → train, val2017 → val."""
    print(f"\n누적 학습 데이터셋 준비 중: {work_dir}")
    use_symlink = coco_gt_to_yolo(subset1_json, train_img, work_dir, "train", cat_id_to_idx, True)
    use_symlink = coco_gt_to_yolo(val_ann, val_img, work_dir, "val", cat_id_to_idx, use_symlink)
    data_yaml = write_data_yaml(work_dir, class_names)
    return data_yaml, use_symlink


def retrain_labeler(data_yaml, prev_weights, round_idx, project_dir, rcfg):
    """직전 라운드 가중치에서 이어서 재학습하고 (best.pt 경로, 결과 폴더)를 반환한다.
    best.pt는 val2017 성능이 가장 좋은 에포크의 가중치다.
    """
    from ultralytics import YOLO

    NON_TRAIN_KEYS = ("max_pseudo_rounds_kept", "round_early_stop_patience")
    train_kwargs = {k: v for k, v in rcfg.items() if k not in NON_TRAIN_KEYS}
    model = YOLO(str(prev_weights))
    name = f"pseudo_round_{round_idx:02d}"
    model.train(data=str(data_yaml), project=str(project_dir), name=name, exist_ok=True, **train_kwargs)
    save_dir = Path(model.trainer.save_dir)
    return save_dir / "weights" / "best.pt", save_dir


def save_final_pseudo_weights(out_dir, weights_path, round_rows, thresholds_csv, source,
                               best_round=None, best_val_map=None):
    """최고 mAP 라운드의 가중치와 라운드별 요약 메타 json을 weights/에 저장한다."""
    dst_dir = Path(out_dir) / "weights"
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / "labeler_pseudo_final.pt"
    shutil.copy2(weights_path, dst)
    meta = {
        "saved_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "source": source,
        "class_thresholds_csv": str(thresholds_csv),
        "best_round": best_round,
        "best_val_mAP@0.5:0.95": best_val_map,
        "rounds": round_rows,
    }
    (dst_dir / "labeler_pseudo_final_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if best_round is not None:
        print(f"\n최종 의사 레이블링 Labeler 저장: {dst} "
              f"(라운드 {best_round} 선택, val mAP@0.5:0.95={best_val_map:.4f})")
    else:
        print(f"\n최종 의사 레이블링 Labeler 저장: {dst}")
    return dst


def update_best_checkpoint(state, round_idx, row, current_weights, patience):
    """최고 mAP 라운드를 갱신하고, mAP가 직전 라운드보다 patience번 연속 떨어졌는지 판단한다.
    (state, 중단 여부)를 반환한다.
    """
    cur_map = row.get("val_mAP@0.5:0.95")
    if not row.get("retrained") or cur_map is None or (isinstance(cur_map, float) and np.isnan(cur_map)):
        return state, False

    if cur_map > state["best_map"]:
        state["best_map"] = cur_map
        state["best_weights"] = current_weights
        state["best_round"] = round_idx

    if state["prev_map"] is not None:
        state["decline_streak"] = state["decline_streak"] + 1 if cur_map < state["prev_map"] else 0
    state["prev_map"] = cur_map

    return state, state["decline_streak"] >= patience


def new_best_checkpoint_state():
    """update_best_checkpoint()에 넘길 초기 상태."""
    return {"best_map": -1.0, "best_weights": None, "best_round": None,
            "prev_map": None, "decline_streak": 0}


def save_round_results_incremental(round_rows, out_dir):
    """라운드가 끝날 때마다 pseudo_round_results.csv를 저장한다 (중단돼도 결과가 남도록)."""
    import pandas as pd

    round_csv = Path(out_dir) / "pseudo_round_results.csv"
    pd.DataFrame(round_rows).to_csv(round_csv, index=False, encoding="utf-8-sig")
    return round_csv


def save_class_metrics_incremental(class_metric_rows, out_dir):
    """라운드가 끝날 때마다 pseudo_round_class_metrics.csv를 저장한다."""
    import pandas as pd

    class_csv = Path(out_dir) / "pseudo_round_class_metrics.csv"
    if not class_metric_rows:
        return None
    pd.DataFrame(class_metric_rows).to_csv(class_csv, index=False, encoding="utf-8-sig")
    return class_csv


def plot_round_results(round_df, out_png, show=False):
    """라운드별 val2017 mAP, Precision, Recall 그래프를 저장한다."""
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = round_df[round_df["retrained"]]
    if df.empty:
        print("[알림] 재학습이 한 번도 일어나지 않아 라운드별 그래프를 생략합니다.")
        return None

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(df["round"], df["val_mAP@0.5"], marker="o", label="val mAP@0.5")
    axes[0].plot(df["round"], df["val_mAP@0.5:0.95"], marker="o", label="val mAP@0.5:0.95")
    axes[0].set_title("val2017 mAP by round")
    axes[0].set_xlabel("round")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(df["round"], df["val_Precision"], marker="o", label="val Precision")
    axes[1].plot(df["round"], df["val_Recall"], marker="o", label="val Recall")
    axes[1].set_title("val2017 Precision/Recall by round")
    axes[1].set_xlabel("round")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    fig.suptitle("Pseudo-labeling self-training: round별 val2017 성능 추이")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    print(f"라운드별 val2017 성능 그래프 저장: {out_png}")
    if show:
        plt.show()
    plt.close(fig)
    return out_png


def report_round_results(round_rows, out_dir, show_plot):
    """라운드별 결과를 csv와 그래프로 저장하고 표로 출력한다."""
    import pandas as pd

    round_df = pd.DataFrame(round_rows)
    round_csv = out_dir / "pseudo_round_results.csv"
    round_df.to_csv(round_csv, index=False, encoding="utf-8-sig")
    fmt = {c: "{:.4f}".format for c in
           ["pseudo_Precision", "pseudo_Recall", "val_Precision", "val_Recall", "val_mAP@0.5", "val_mAP@0.5:0.95"]}
    print(f"\n===== 라운드별 결과 (val2017 기준, {round_csv}) =====")
    print(round_df.to_string(index=False, formatters=fmt))
    plot_round_results(round_df, out_dir / "pseudo_round_curve.png", show=show_plot)
    return round_df
