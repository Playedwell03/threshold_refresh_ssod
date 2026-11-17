"""results/의 실험 결과 CSV로 논문의 표 3, 5, 6, 7과 그림 6, 7, 8을 다시 만든다.

사용법: python scripts/make_tables_figures.py [--results results] [--out paper_outputs]
- 표는 paper_outputs/tables/*.csv, 그림은 paper_outputs/figures/*.png로 저장한다.
- ρ_P, ρ_R은 마지막 라운드 값 / 1라운드 값, 최고 mAP는 val mAP@0.5:0.95가 가장 높은 라운드다.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

P, R, M = "val_Precision", "val_Recall", "val_mAP@0.5:0.95"
YLIM = {P: (0.25, 0.80), R: (0.25, 0.80), M: (0.18, 0.22)}  # 그림 6·7·8 축 통일
GLOBAL = {f"{t:.1f}": f"A_global_{t:.1f}" for t in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7)}
RANGES = {"범위 없음": "norange", "0.3~0.5": "0.3-0.5", "0.2~0.85": "0.2-0.85"}


def load(res: Path, name: str) -> pd.DataFrame:
    return pd.read_csv(res / name / "pseudo_round_results.csv", encoding="utf-8-sig")


def summary(d: pd.DataFrame) -> dict:
    b = d[M].idxmax()
    return {"최종 정밀도": round(d[P].iloc[-1], 3), "최종 재현율": round(d[R].iloc[-1], 3),
            "ρ_P": round(d[P].iloc[-1] / d[P].iloc[0], 2), "ρ_R": round(d[R].iloc[-1] / d[R].iloc[0], 2),
            "최고 mAP (R)": f"{d[M][b]:.4f}({d['round'][b]})",
            "마지막 mAP (R)": f"{d[M].iloc[-1]:.4f}({d['round'].iloc[-1]})"}


def plot(runs: dict, out: Path, prefix: str) -> None:
    for col, yl, fn in [(P, "Precision", "precision"), (R, "Recall", "recall"), (M, "mAP@0.5:0.95", "map")]:
        f, a = plt.subplots(figsize=(4.2, 3.2), dpi=300)
        for label, (d, color) in runs.items():
            a.plot(d["round"], d[col], "-", marker="o", ms=3, lw=1.4, color=color, label=label)
        a.set_ylim(YLIM[col]); a.set_xlim(0.5, 18.5); a.set_xticks(range(2, 19, 2))
        a.set_xlabel("Round"); a.set_ylabel(yl); a.grid(alpha=.35); a.spines[["top", "right"]].set_visible(False)
        f.tight_layout(); f.savefig(out / f"{prefix}_{fn}.png"); plt.close(f)
    h, l = a.get_legend_handles_labels()
    f = plt.figure(figsize=(4.5, 0.6), dpi=300)
    f.legend(h, l, loc="center", ncol=1, fontsize=7, frameon=False, handlelength=3.0)
    f.savefig(out / f"{prefix}_legend.png", bbox_inches="tight"); plt.close(f)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="paper_outputs")
    a = ap.parse_args()
    res, out = Path(a.results), Path(a.out)
    (out / "tables").mkdir(parents=True, exist_ok=True); (out / "figures").mkdir(parents=True, exist_ok=True)

    # 표 3 / 그림 6: 전역 고정 임계값
    t3 = pd.DataFrame([{"임계값": t, **summary(load(res, n))} for t, n in GLOBAL.items()])
    t3.to_csv(out / "tables/table3_global.csv", index=False, encoding="utf-8-sig")
    colors = ["#8c564b", "#ff7f0e", "#9467bd", "#1f77b4", "#7f7f7f", "#e377c2", "#2ca02c"]
    plot({f"Global {t}": (load(res, n), c) for (t, n), c in zip(GLOBAL.items(), colors)}, out / "figures", "fig6_global")

    # 표 5·6 / 그림 7·8: 클래스별 고정(B), 갱신(C)
    for exp, tag, title in [("B", "B_fixed", "fixed"), ("C", "C_refresh", "refresh every 3 rounds")]:
        t = pd.DataFrame([{"구분": k, **summary(load(res, f"{tag}_{v}"))} for k, v in RANGES.items()])
        t.to_csv(out / f"tables/table{5 if exp == 'B' else 6}_{tag}.csv", index=False, encoding="utf-8-sig")
        cols = ["#d62728", "#ff7f0e", "#9467bd"] if exp == "B" else ["#1f77b4", "#ff7f0e", "#9467bd"]
        plot({f"({'abc'[i]}) Class-wise {title}, {'no range' if v == 'norange' else 'range ' + k}":
              (load(res, f"{tag}_{v}"), cols[i]) for i, (k, v) in enumerate(RANGES.items())},
             out / "figures", "fig7_fixed" if exp == "B" else "fig8_refresh")

    # 표 7: 15라운드 클래스별 TP, FP, AP (A 전역 0.4, B 고정 0.3~0.5, C 갱신 0.3~0.5)
    t7 = []
    for k, n in [("A", "A_global_0.4"), ("B", "B_fixed_0.3-0.5"), ("C", "C_refresh_0.3-0.5")]:
        d = pd.read_csv(res / n / "pseudo_round_class_metrics.csv", encoding="utf-8-sig")
        d = d[d["round"] == 15].set_index("class")
        part = d.loc[["person", "bowl", "tv", "giraffe", "toaster"], ["GT", "TP", "FP", "AP@0.5:0.95"]]
        part.loc["전체"] = [d.GT.sum(), d.TP.sum(), d.FP.sum(), d["AP@0.5:0.95"].mean()]
        t7.append(part.add_prefix(f"{k} "))
    pd.concat(t7, axis=1).round(4).to_csv(out / "tables/table7_class_r15.csv", encoding="utf-8-sig")
    print(f"저장: {out}/tables, {out}/figures")


if __name__ == "__main__":
    main()
