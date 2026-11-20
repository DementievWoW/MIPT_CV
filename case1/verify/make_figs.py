# -*- coding: utf-8 -*-
"""Картинки для отчёта ДЗ1 по артефактам dz1_artifacts (числа берутся из файлов, не вписаны руками).
Запуск: python -I make_figs.py <dz1_artifacts> <dataset_dz1_raw> <папка для png>"""
import sys
from pathlib import Path
import pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ART, DATA, OUT = map(Path, sys.argv[1:4]); OUT.mkdir(exist_ok=True)
BLUE, ORANGE, GRAY, INK, INK2, SURF = "#2a78d6", "#eb6834", "#b4b2ab", "#0b0b0b", "#52514e", "#fcfcfb"
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": INK2,
                     "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False, "font.size": 10})
ALLOWED = ["letter", "form", "email", "invoice", "memo", "budget", "questionnaire", "resume"]
clean = pd.read_csv(ART / "clean_labels.csv"); rej = pd.read_csv(ART / "rejected_log.csv")
labels = pd.read_csv(DATA / "labels.csv")
n_all, n_clean = len(labels), len(clean)
by = rej["reason"].value_counts()
steps = [("не читается", int(by.get("не читается", 0))),
         ("посторонний тип", int(by.get("посторонний тип документа", 0))),
         ("дубликат", int(by.get("дубликат", 0)))]
steps.append(("без метки\n(keep_aside)", n_all - n_clean - sum(v for _, v in steps)))

# 1. водопад
fig, ax = plt.subplots(figsize=(8.5, 3.8))
level = n_all
ax.bar(0, n_all, color=BLUE, width=0.6); ax.text(0, n_all + 40, f"{n_all}", ha="center")
for i, (name, v) in enumerate(steps, 1):
    ax.bar(i, v, bottom=level - v, color=GRAY, width=0.6, edgecolor=SURF, linewidth=2)
    ax.text(i, level + 40, f"−{v}", ha="center"); level -= v
ax.bar(len(steps) + 1, n_clean, color=BLUE, width=0.6)
ax.text(len(steps) + 1, n_clean + 40, f"{n_clean} ({n_clean / n_all:.1%})", ha="center")
assert level == n_clean, "водопад не сходится"
ax.set_xticks(range(len(steps) + 2), ["архив"] + [s for s, _ in steps] + ["clean"])
ax.set_ylabel("страниц"); ax.set_ylim(0, n_all * 1.12)
ax.yaxis.grid(True, color="#e6e5e0"); ax.set_axisbelow(True)
ax.set_title("Чистка: от 3891 страниц архива к clean (журнал по первой причине)", loc="left")
fig.tight_layout(); fig.savefig(OUT / "fig1_waterfall.png", dpi=150); plt.close(fig)

# 2. классы до и после
bad = set(pd.read_csv(ART / "bad_files_log.csv")["file"])  # как в Шаге 1: только читаемые
before = labels[labels["label"].isin(ALLOWED) & ~labels["file"].isin(bad)]["label"].value_counts().reindex(ALLOWED)
after = clean["label"].value_counts().reindex(ALLOWED)
fig, ax = plt.subplots(figsize=(8.5, 3.8)); w = 0.38; x = range(len(ALLOWED))
ax.bar([i - w / 2 for i in x], before, w, color=GRAY, label=f"labels.csv, читаемые (дисбаланс {before.max() / before.min():.1f})")
ax.bar([i + w / 2 for i in x], after, w, color=BLUE, label=f"clean (дисбаланс {after.max() / after.min():.1f})")
for i, (b, a) in enumerate(zip(before, after)):
    ax.text(i + w / 2, a + 12, str(a), ha="center", fontsize=8, color=INK2)
ax.set_xticks(list(x), ALLOWED, rotation=20); ax.set_ylabel("страниц")
ax.yaxis.grid(True, color="#e6e5e0"); ax.set_axisbelow(True); ax.legend(frameon=False)
ax.set_title("Целевые классы до и после чистки", loc="left")
fig.tight_layout(); fig.savefig(OUT / "fig2_classes.png", dpi=150); plt.close(fig)

# 3. доли классов в train и val
lab = dict(zip(clean["file"], clean["label"]))
tr = pd.Series([lab[f] for f in (ART / "train.txt").read_text().split()]).value_counts(normalize=True).reindex(ALLOWED)
va = pd.Series([lab[f] for f in (ART / "val.txt").read_text().split()]).value_counts(normalize=True).reindex(ALLOWED)
fig, ax = plt.subplots(figsize=(8.5, 3.6))
ax.bar([i - w / 2 for i in x], tr * 100, w, color=BLUE, label=f"train ({len((ART / 'train.txt').read_text().split())})")
ax.bar([i + w / 2 for i in x], va * 100, w, color=ORANGE, label=f"val ({len((ART / 'val.txt').read_text().split())})")
ax.set_xticks(list(x), ALLOWED, rotation=20); ax.set_ylabel("доля класса, %")
ax.yaxis.grid(True, color="#e6e5e0"); ax.set_axisbelow(True); ax.legend(frameon=False)
ax.set_title(f"Стратификация: макс. расхождение долей {float((tr - va).abs().max()) * 100:.2f} п.п.", loc="left")
fig.tight_layout(); fig.savefig(OUT / "fig3_split.png", dpi=150); plt.close(fig)
print("ok", [s for s in steps])
