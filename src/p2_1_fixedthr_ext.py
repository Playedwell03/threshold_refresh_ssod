"""실험 A: 모든 클래스에 같은 신뢰도 임계값을 고정해 의사 레이블링을 반복한다.

subset_01로 학습한 초기 레이블러에서 시작해, 라운드 r마다 subset_(r+2)를 예측하고 임계값(--fixed-conf)
이상인 박스를 의사 레이블로 채택한 뒤 재학습한다. subset_02는 쓰지 않는다(라운드 수를 실험 B·C와 맞춤).

입력: <out-dir>/p1_output/ (p1_data_slicing_and_initial_learning.py의 결과)
출력: <out-dir>/<run-name>/ (라운드별 결과 csv, 클래스별 결과 csv, 라운드별 가중치)
사용 예: python p2_1_fixedthr_ext.py --coco-root coco --out-dir outputs --fixed-conf 0.4
"""

import argparse
import shutil
from pathlib import Path

import numpy as np

import p0_common_api as common


def process_round_subset(model, subset_json, img_dir, cat_id_to_idx, ecfg, scfg, thresholds, desc, limit=None):
    """subset 하나를 예측해 (전체 예측 평가, 채택 라벨 평가, 채택된 의사 레이블 목록)을 반환한다.
    채택 라벨은 YOLO 형식 문자열(class xc yc w h, 정규화 좌표)이다.
    """
    items = common.load_subset_gt(subset_json, cat_id_to_idx)
    img_sizes = common.load_image_sizes(subset_json)
    if limit:
        items = items[:limit]
    nc = len(cat_id_to_idx)
    raw_acc, sel_acc = common.new_acc(nc), common.new_acc(nc)
    selected = []
    if not items:
        return raw_acc, sel_acc, selected

    results = common._predict_stream(model, img_dir, items, ecfg)
    for (fname, gb, gc, gcrowd), r in common._progress(zip(items, results), total=len(items), desc=desc, leave=False):
        db, dc, dcls = common._boxes_from_result(r)

        c1, k1, t1 = common.match_image(db, dc, dcls, gb, gc, gcrowd)
        raw_acc["conf"].append(c1)
        raw_acc["cls"].append(k1)
        raw_acc["tp"].append(t1)
        raw_acc["n_gt"] += np.bincount(gc[~gcrowd], minlength=nc)
        raw_acc["n_img"] += 1

        # 임계값 이상인 예측만 의사 레이블로 채택
        keep = dc >= thresholds[dcls] if len(dcls) else np.zeros(0, dtype=bool)
        sb, sc, scls = db[keep], dc[keep], dcls[keep]

        if scfg["use_max_box_filter"] and len(sc) > scfg["max_boxes_per_image"]:
            top = np.argsort(-sc)[: scfg["max_boxes_per_image"]]
            sb, sc, scls = sb[top], sc[top], scls[top]


        if len(sc) == 0:
            continue

        c2, k2, t2 = common.match_image(sb, sc, scls, gb, gc, gcrowd)
        sel_acc["conf"].append(c2)
        sel_acc["cls"].append(k2)
        sel_acc["tp"].append(t2)
        sel_acc["n_gt"] += np.bincount(gc[~gcrowd], minlength=nc)
        sel_acc["n_img"] += 1

        w, h = img_sizes[fname]
        lines = []
        for (x1, y1, x2, y2), cidx in zip(sb, scls):
            cx = min(max(((x1 + x2) / 2) / w, 0.0), 1.0)
            cy = min(max(((y1 + y2) / 2) / h, 0.0), 1.0)
            bw = min((x2 - x1) / w, 1.0)
            bh = min((y2 - y1) / h, 1.0)
            lines.append(f"{int(cidx)} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        selected.append((fname, lines))

    return raw_acc, sel_acc, selected


def setup_fixed_thresholds(scfg, class_names, out_dir):
    """모든 클래스에 같은 임계값(scfg['fallback_conf'], main에서 --fixed-conf로 설정)을 적용하고 csv로 저장한다."""
    thresholds = np.full(len(class_names), scfg["fallback_conf"], dtype=np.float64)

    import pandas as pd

    thr_rows = [{"class_idx": i, "class": name, "conf_thr": float(thresholds[i])}
                for i, name in enumerate(class_names)]
    thr_csv = out_dir / "pseudo_class_thresholds.csv"
    pd.DataFrame(thr_rows).to_csv(thr_csv, index=False, encoding="utf-8-sig")

    print(f"\n클래스별 임계값: 보정 없이 전 클래스 동일 conf_thr={scfg['fallback_conf']} 적용 "
          f"(subset_02/val2017 모두 미사용) -> {thr_csv}")
    return thresholds, thr_csv


def run_round(round_idx, subset_json, subset_name, model, current_weights, use_symlink,
              train_img, val_ann, val_img, cat_id_to_idx, class_names, ecfg, scfg, rcfg,
              thresholds, work_dir, runs_dir, limit=None):
    """라운드 하나를 처리한다: 의사 레이블 채택 → 누적 학습셋 추가(최근 3라운드만 유지) → 재학습 → val2017 평가.
    (결과 행, 클래스별 결과 행, model, 가중치 경로, use_symlink)를 반환한다.
    """
    from ultralytics import YOLO

    raw_acc, sel_acc, selected = process_round_subset(
        model, subset_json, train_img, cat_id_to_idx, ecfg, scfg, thresholds,
        desc=f"{subset_name} 예측", limit=limit,
    )
    raw_row, sel_row = common.summarize(raw_acc, ecfg["conf_thr"]), common.summarize(sel_acc, ecfg["conf_thr"])
    print(f"  - 선택 전 Labeler 예측 : P={raw_row['Precision']:.4f} R={raw_row['Recall']:.4f} "
          f"mAP50={raw_row['mAP@0.5']:.4f}")
    print(f"  - 채택된 pseudo-label  : 이미지 {sel_acc['n_img']}/{raw_acc['n_img']}장 채택, "
          f"P={sel_row['Precision']:.4f} R={sel_row['Recall']:.4f} "
          f"(P=채택 라벨의 정확도, R=subset 전체 GT 대비 라벨링된 비율)")

    row = {
        "round": round_idx, "subset": subset_name,
        "n_images_total": raw_acc["n_img"], "n_images_selected": sel_acc["n_img"],
        "pseudo_Precision": sel_row["Precision"], "pseudo_Recall": sel_row["Recall"],
    }

    if not selected:
        print(f"  [알림] {subset_name}에서 채택된 이미지가 없어 이번 라운드는 재학습을 건너뜁니다.")
        row.update({"retrained": False, "val_Precision": np.nan, "val_Recall": np.nan,
                    "val_mAP@0.5": np.nan, "val_mAP@0.5:0.95": np.nan})
        return row, [], model, current_weights, use_symlink

    use_symlink = common.append_pseudo_round(work_dir, train_img, selected, use_symlink, round_idx)

    # 슬라이딩 윈도우: 오래된 라운드의 의사 레이블은 학습셋에서 제거
    keep = rcfg.get("max_pseudo_rounds_kept")
    if keep:
        evict_round = round_idx - keep
        if evict_round >= 1:
            n_removed = common.prune_pseudo_round(work_dir, evict_round)
            if n_removed:
                print(f"  - 누적 학습셋 정리: round {evict_round}의 pseudo-label {n_removed}개 제거 "
                      f"(최근 {keep}라운드 + subset_01 GT만 유지)")

    data_yaml = Path(work_dir).resolve() / "data.yaml"
    current_weights, _ = common.retrain_labeler(data_yaml, current_weights, round_idx, runs_dir, rcfg)
    model = YOLO(str(current_weights))
    print(f"  - 재학습 완료 -> {current_weights}")

    val_acc = common.evaluate_subset(model, val_ann, val_img, cat_id_to_idx, ecfg,
                                  desc=f"val2017(round {round_idx} 이후)", limit=limit)
    val_row = common.summarize(val_acc, ecfg["conf_thr"])
    print(f"  - val2017(갱신 후)     : P={val_row['Precision']:.4f} R={val_row['Recall']:.4f} "
          f"mAP50={val_row['mAP@0.5']:.4f} mAP50-95={val_row['mAP@0.5:0.95']:.4f}")

    row.update({
        "retrained": True,
        "val_Precision": val_row["Precision"], "val_Recall": val_row["Recall"],
        "val_mAP@0.5": val_row["mAP@0.5"], "val_mAP@0.5:0.95": val_row["mAP@0.5:0.95"],
    })

    class_rows = [{"round": round_idx, "subset": subset_name, **r}
                  for r in common.summarize_per_class(val_acc, class_names, ecfg["conf_thr"])]
    return row, class_rows, model, current_weights, use_symlink


def build_arg_parser():
    """명령행 옵션. 공통 옵션은 p0_common_api.add_common_args()에 있다."""
    p = argparse.ArgumentParser()
    common.add_common_args(p)
    p.add_argument("--fixed-conf", type=float, default=0.5,
                   help="전 클래스에 동일하게 적용할 고정 confidence 임계값(기본 0.5)")
    p.add_argument("--class-conf", action="append", default=[],
                   help="특정 클래스만 다른 고정 임계값 사용. 예: --class-conf person=0.1 (여러 번 지정 가능)")
    p.add_argument("--run-name", default=None,
                   help="결과 폴더 이름(기본: p2_1_thrXXX_output)")
    p.add_argument("--resume-dir", default=None,
                   help="이전 실행 상태 폴더(resume_state). 지정하면 --start-round 라운드부터 이어서 실행")
    p.add_argument("--start-round", type=int, default=None,
                   help="--resume-dir와 함께 사용. 이 라운드부터 새로 실행(이전 라운드는 resume_state에서 복원)")
    p.add_argument("--no-early-stop", action="store_true",
                   help="라운드 조기 종료를 끈다(기본: 켬)")
    p.add_argument("--dry-run", action="store_true",
                   help="준비/복원까지만 하고 라운드 학습 전에 종료(동작 점검용)")
    return p


def main():
    """준비(임계값 산정, 누적 학습셋 구성) 후 라운드를 반복하고, 최고 mAP 라운드의 가중치를 최종본으로 저장한다."""
    args = build_arg_parser().parse_args()

    root = Path(args.coco_root)
    out = Path(args.out_dir).resolve()
    run_out = out / (args.run_name or f"p2_1_thr{round(args.fixed_conf * 100):03d}_output")
    p1_out = out / "p1_output"
    subset_dir = p1_out / "subsets"
    pseudo_work_dir = run_out / "pseudo" / "yolo_data"
    pseudo_runs_dir = run_out / "pseudo" / "runs"
    if args.n < 3:
        raise SystemExit(f"--n={args.n}: subset_01(학습) + subset_02(건너뜀, p2_2_pseudo_label_ext.py와 라운드 수를 "
                          f"맞추기 위함) 최소 2개 필요 + pseudo-labeling할 subset이 최소 1개 있어야 "
                          f"하므로 --n은 3 이상이어야 합니다.")
    n_rounds = args.n - 2

    if (run_out / "pseudo" / "yolo_data").exists():
        raise SystemExit(f"결과 폴더에 이전 실행의 학습셋이 남아 있습니다: {run_out}\n"
                         f"폴더를 지우거나 이름을 바꾼 뒤 다시 실행하세요.")
    run_out.mkdir(parents=True, exist_ok=True)
    print(f"입력(읽기 전용): {out}")
    print(f"출력(이 실행 전용): {run_out}")

    val_ann, train_img, val_img = common.check_coco_paths(root)

    print(f"\n===== [0/{n_rounds}] 준비: subset JSON({args.n}개) + Labeler 가중치 읽기 =====")
    subset_files = common.require_subset_files(subset_dir, args.n)
    weights = common.load_labeler_weights(p1_out, args.weights)

    from ultralytics import YOLO

    model = YOLO(str(weights))
    ecfg = common.get_eval_config(max_det=100)
    scfg, rcfg = common.get_selection_config(), common.get_retrain_config()
    scfg["fallback_conf"] = args.fixed_conf
    cat_id_to_idx, class_names = common.load_category_map(subset_files[0])

    thresholds, thr_csv = setup_fixed_thresholds(scfg, class_names, run_out)
    if args.class_conf:
        import pandas as pd
        for item in args.class_conf:
            name, val = item.split("=")
            if name not in class_names:
                raise SystemExit(f"--class-conf: 클래스 이름을 찾을 수 없습니다: {name}")
            thresholds[class_names.index(name)] = float(val)
            print(f"  - 클래스별 임계값 덮어쓰기: {name} = {float(val)}")
        pd.DataFrame([{"class_idx": i, "class": n, "conf_thr": float(thresholds[i])}
                      for i, n in enumerate(class_names)]).to_csv(thr_csv, index=False, encoding="utf-8-sig")
    if args.no_early_stop:
        rcfg["round_early_stop_patience"] = 10 ** 6
    print(f"  - 라운드 조기 종료: {'끔' if args.no_early_stop else '켬 (patience=%d)' % rcfg['round_early_stop_patience']}")
    data_yaml, use_symlink = common.setup_cumulative_dataset(
        subset_files[0], train_img, val_ann, val_img, cat_id_to_idx, class_names, pseudo_work_dir)

    current_weights = weights
    round_rows = []
    class_metric_rows = []
    best_state = common.new_best_checkpoint_state()
    start_round = 1
    # --resume-dir: 이전 실행의 결과·가중치·최근 의사 레이블을 복원해 --start-round부터 이어서 실행
    if args.resume_dir:
        import pandas as pd
        rs = Path(args.resume_dir).resolve()
        start_round = args.start_round
        if not start_round or start_round < 2:
            raise SystemExit("--resume-dir를 쓰려면 --start-round(2 이상)를 지정하세요.")
        prev = pd.read_csv(rs / "pseudo_round_results.csv", encoding="utf-8-sig")
        prev = prev[prev["round"] < start_round]
        round_rows = prev.to_dict("records")
        cm = rs / "pseudo_round_class_metrics.csv"
        if cm.exists():
            cdf = pd.read_csv(cm, encoding="utf-8-sig")
            class_metric_rows = cdf[cdf["round"] < start_round].to_dict("records")
        import json as _json
        work = Path(pseudo_work_dir)
        (work / "manifests").mkdir(parents=True, exist_ok=True)
        (work / "labels" / "train").mkdir(parents=True, exist_ok=True)
        (work / "images" / "train").mkdir(parents=True, exist_ok=True)
        n_restored = 0
        for mf in sorted((rs / "yolo_data" / "manifests").glob("round_*.json")):
            fnames = _json.loads(mf.read_text())
            for fname in fnames:
                stem = Path(fname).stem
                shutil.copy2(rs / "yolo_data" / "labels" / "train" / f"{stem}.txt",
                             work / "labels" / "train" / f"{stem}.txt")
                use_symlink = common.stage_image(train_img / fname, work / "images" / "train" / fname, use_symlink)
            shutil.copy2(mf, work / "manifests" / mf.name)
            n_restored += len(fnames)
            print(f"  - 복원: {mf.stem} 의사 라벨 {len(fnames)}개")
        for row in round_rows:
            w = rs / "weights" / f"round_{int(row['round']):02d}_best.pt"
            best_state, _ = common.update_best_checkpoint(
                best_state, int(row["round"]), row, w, rcfg["round_early_stop_patience"])
        current_weights = rs / "weights" / f"round_{start_round - 1:02d}_best.pt"
        if not current_weights.exists():
            raise SystemExit(f"이어 시작할 가중치가 없습니다: {current_weights}")
        model = YOLO(str(current_weights))
        print(f"  - 이어서 실행: R{start_round}부터, 시작 가중치 {current_weights.name}, "
              f"복원한 라운드 {len(round_rows)}개, 최고 R{best_state['best_round']} "
              f"mAP={best_state['best_map']:.4f}, 연속 하락 {best_state['decline_streak']}")
    if args.dry_run:
        print("\n[dry-run] 준비/복원까지 완료. 학습 없이 종료합니다.")
        return

    for r in range(start_round + 2, args.n + 1):
        round_idx = r - 2
        subset_name = f"subset_{r:02d}"
        print(f"\n===== [{round_idx}/{n_rounds}] round {round_idx}: {subset_name} 의사 라벨링 + 재학습 =====")

        row, class_rows, model, current_weights, use_symlink = run_round(
            round_idx, subset_files[r - 1], subset_name, model, current_weights, use_symlink,
            train_img, val_ann, val_img, cat_id_to_idx, class_names, ecfg, scfg, rcfg,
            thresholds, pseudo_work_dir, pseudo_runs_dir, limit=args.eval_max_images,
        )
        round_rows.append(row)
        class_metric_rows.extend(class_rows)
        common.save_round_results_incremental(round_rows, run_out)
        common.save_class_metrics_incremental(class_metric_rows, run_out)

        best_state, should_stop = common.update_best_checkpoint(
            best_state, round_idx, row, current_weights, rcfg["round_early_stop_patience"])
        if should_stop:
            print(f"\n[조기 종료] val mAP@0.5:0.95가 {rcfg['round_early_stop_patience']}라운드 연속 하락해 "
                  f"라운드 {round_idx}에서 중단합니다 (최고 성능: 라운드 {best_state['best_round']}, "
                  f"mAP@0.5:0.95={best_state['best_map']:.4f}). 남은 라운드는 건너뜁니다.")
            break

    common.report_round_results(round_rows, run_out, show_plot=not args.no_show)
    if best_state["best_weights"] is not None:
        common.save_final_pseudo_weights(
            run_out, best_state["best_weights"], round_rows, thr_csv, source="p2_1_fixedthr_ext.py",
            best_round=best_state["best_round"], best_val_map=best_state["best_map"])
    else:
        print("\n[알림] 모든 라운드에서 채택된 pseudo-label이 없어 재학습이 한 번도 일어나지 않았습니다.")


if __name__ == "__main__":
    main()
