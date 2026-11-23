# -*- coding: utf-8 -*-
"""Независимая проверка ДЗ2: dz2_artifacts.zip сверяется с ДЗ1, с сырыми сканами и с ноутбуком.

Ничего не обучает и ничего не меняет. Модель из classifier.pt перезагружается и заново
прогоняется по holdout с предобработкой, написанной здесь с нуля по контракту из model_config.json.

Запуск (пути по умолчанию — как в Colab):
    DZ2_ART=<распакованный dz2_artifacts> DZ1_ART=<распакованный dz1_artifacts> \
    DZ_DATA=<распакованный dataset_dz1_raw> DZ2_NB=<ноутбук .ipynb> python -m pytest -q test_dz2.py
"""
import hashlib, json, os, re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

ART = Path(os.environ.get("DZ2_ART", "/content/dz2_artifacts"))
DZ1 = Path(os.environ.get("DZ1_ART", "/content/dz1"))
DATA = Path(os.environ.get("DZ_DATA", "/content/archive"))
NB = Path(os.environ.get("DZ2_NB", Path(__file__).resolve().parents[1] / "CV_ДЗ2_Классификатор_документов.ipynb"))


def _lines(p):
    return [s for s in Path(p).read_text(encoding="utf-8").splitlines() if s.strip()]


@pytest.fixture(scope="session")
def art():
    return {
        "metrics": json.loads((ART / "metrics.json").read_text(encoding="utf-8")),
        "config": json.loads((ART / "model_config.json").read_text(encoding="utf-8")),
        "manifest": json.loads((ART / "artifact_manifest.json").read_text(encoding="utf-8")),
        "clean": pd.read_csv(ART / "clean_labels.csv"),
        "train": _lines(ART / "train.txt"), "val": _lines(ART / "val.txt"),
        "fit": _lines(ART / "fit_train.txt"), "tuning": _lines(ART / "tuning.txt"),
        "holdout": _lines(ART / "holdout.txt"),
        "pred": pd.read_csv(ART / "holdout_predictions.csv"),
        "report": pd.read_csv(ART / "classification_report_holdout.csv", index_col=0),
        "experiments": pd.read_csv(ART / "experiments.csv"),
    }


@pytest.fixture(scope="session")
def label_of(art):
    return dict(zip(art["clean"]["file"], art["clean"]["label"]))


# ---------------------------------------------------------------- состав архива
def test_manifest_files_present(art):
    missing = [f for f in art["manifest"]["files"] if not (ART / f).exists()]
    assert not missing, f"в архиве нет файлов из манифеста: {missing}"
    need = set(art["manifest"]["required_for_dz4"])
    assert need <= set(art["manifest"]["files"]), f"для ДЗ4 не хватает: {need - set(art['manifest']['files'])}"


# ---------------------------------------------------------------- связь с ДЗ1
def test_inputs_match_dz1(art):
    """clean, train и val ДЗ2 — ровно те, что вышли из ДЗ1."""
    dz1_clean = pd.read_csv(DZ1 / "clean_labels.csv")
    assert set(dz1_clean["file"]) == set(art["clean"]["file"]), "clean_labels.csv отличается от ДЗ1"
    assert dict(zip(dz1_clean["file"], dz1_clean["label"])) == dict(zip(art["clean"]["file"], art["clean"]["label"]))
    assert sorted(_lines(DZ1 / "train.txt")) == sorted(art["train"]), "train.txt отличается от ДЗ1"
    assert sorted(_lines(DZ1 / "val.txt")) == sorted(art["val"]), "val.txt отличается от ДЗ1"


def test_holdout_is_dz1_val(art):
    assert sorted(art["holdout"]) == sorted(art["val"]), "holdout не равен val из ДЗ1"


# ---------------------------------------------------------------- разбиение fit / tuning / holdout
def test_split_partitions_train(art):
    fit, tun, hold = set(art["fit"]), set(art["tuning"]), set(art["holdout"])
    assert not fit & tun, "fit и tuning пересекаются"
    assert not (fit | tun) & hold, "holdout пересекается с обучением"
    assert fit | tun == set(art["train"]), "fit + tuning не равно train из ДЗ1"
    sp = art["metrics"]["split"]
    assert (len(fit), len(tun), len(hold)) == (sp["fit_size"], sp["tuning_size"], sp["holdout_size"])


def test_split_has_no_content_leak(art):
    """Одинаковые по содержимому сканы (md5) не должны попасть и в обучение, и в holdout."""
    md5 = lambda f: hashlib.md5((DATA / f).read_bytes()).hexdigest()
    train_h = {md5(f) for f in art["fit"] + art["tuning"]}
    leaked = [f for f in art["holdout"] if md5(f) in train_h]
    assert not leaked, f"копии обучающих сканов в holdout: {leaked[:5]}"


def test_fit_tuning_stratified(art, label_of):
    a = pd.Series([label_of[f] for f in art["fit"]]).value_counts(normalize=True)
    b = pd.Series([label_of[f] for f in art["tuning"]]).value_counts(normalize=True)
    gap = float((a - b).abs().max())
    assert gap < 0.01, f"доли классов в fit и tuning расходятся на {gap:.3f}"


def test_class_counts_and_imbalance(art, label_of):
    counts = pd.Series([label_of[f] for f in art["fit"]]).value_counts()
    assert counts.to_dict() == art["metrics"]["class_counts_fit"]
    assert round(counts.max() / counts.min(), 2) == pytest.approx(art["metrics"]["imbalance_ratio"])


def test_class_weights_inverse_frequency(art):
    """Веса потерь обратно пропорциональны частоте класса в fit (среднее нормировано к 1)."""
    m = art["metrics"]
    cnt = pd.Series(m["class_counts_fit"])
    w = pd.Series(m["class_weights"])[cnt.index]
    ref = 1 / cnt
    ref = ref / ref.mean()
    assert np.allclose(w.values, ref.values, atol=1e-5), "веса классов не соответствуют 1/частота"


# ---------------------------------------------------------------- конфиг и контракт
def test_config_matches_metrics(art):
    c, m = art["config"], art["metrics"]
    assert c["classes"] == sorted(art["clean"]["label"].unique()) == m["classes"]
    assert c["class_to_idx"] == {k: i for i, k in enumerate(c["classes"])}
    assert c["num_classes"] == len(c["classes"]) == 8
    assert c["preprocessing"] == m["preprocessing"], "контракт предобработки в config и metrics расходится"
    assert c["selected_run"] == m["best_model"] == m["gradcam_model"]


def test_best_model_chosen_on_tuning(art):
    """Модель выбрана по tuning-выборке, а не по holdout; числа совпадают с experiments.csv."""
    m, ex = art["metrics"], art["experiments"]
    cand = m["candidates_tuning"]
    assert m["best_model"] == max(cand, key=cand.get)
    assert m["selection_macro_f1_tuning"] == cand[m["best_model"]]
    for _, r in ex.iterrows():
        assert cand[r["model"]] == pytest.approx(r["best_val_f1"], abs=1e-4), r["model"]


def test_checkpoints_identical(art):
    import torch
    sd = torch.load(ART / "best_model.pt", map_location="cpu")
    ck = torch.load(ART / "classifier.pt", map_location="cpu", weights_only=False)
    assert ck["config"] == art["config"], "конфиг внутри classifier.pt отличается от model_config.json"
    assert sd.keys() == ck["state_dict"].keys()
    assert all(torch.equal(sd[k], ck["state_dict"][k]) for k in sd), "best_model.pt и classifier.pt разные веса"


# ---------------------------------------------------------------- перепрогон модели по holdout
def _preprocess(path, pp):
    """Предобработка, написанная заново по контракту: grayscale -> square resize linear -> x3 -> ImageNet norm."""
    g = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    g = cv2.resize(g, (pp["image_size"], pp["image_size"]), interpolation=cv2.INTER_LINEAR)
    x = np.repeat(g[..., None], 3, axis=2).astype(np.float32) / 255.0
    x = (x - np.array(pp["mean"], np.float32)) / np.array(pp["std"], np.float32)
    return x.transpose(2, 0, 1)


@pytest.fixture(scope="session")
def rerun(art):
    import torch
    from torchvision import models
    ck = torch.load(ART / "classifier.pt", map_location="cpu", weights_only=False)
    cfg = ck["config"]
    net = models.resnet18(weights=None)
    net.fc = torch.nn.Linear(net.fc.in_features, cfg["num_classes"])
    net.load_state_dict(ck["state_dict"])
    net.eval()
    files = list(art["pred"]["file"])
    probs = []
    with torch.no_grad():
        for i in range(0, len(files), 64):
            xb = torch.from_numpy(np.stack([_preprocess(DATA / f, cfg["preprocessing"]) for f in files[i:i + 64]]))
            probs.append(torch.softmax(net(xb), 1).numpy())
    probs = np.concatenate(probs)
    return pd.DataFrame({"file": files, "pred": [cfg["classes"][k] for k in probs.argmax(1)],
                         "conf": probs.max(1)})


def test_predictions_cover_holdout(art, label_of):
    p = art["pred"]
    assert sorted(p["file"]) == sorted(art["holdout"]) and p["file"].is_unique
    assert (p["true"] == p["file"].map(label_of)).all(), "в holdout_predictions.csv неверные истинные метки"


def test_rerun_reproduces_predictions(art, rerun):
    p = art["pred"]
    mism = int((rerun["pred"].values != p["predicted"].values).sum())
    assert mism <= 2, f"перезагруженная модель расходится с holdout_predictions.csv на {mism} документах"
    assert np.allclose(rerun["conf"].values, p["confidence"].values, atol=1e-3), "уверенности не совпадают"


def test_holdout_metrics_recomputed(art, rerun):
    from sklearn.metrics import accuracy_score, f1_score
    y = art["pred"]["true"]
    for pred, src in [(art["pred"]["predicted"], "csv"), (rerun["pred"], "перепрогон")]:
        f1 = f1_score(y, pred, average="macro")
        acc = accuracy_score(y, pred)
        assert f1 == pytest.approx(art["metrics"]["holdout_macro_f1"], abs=0.003), f"{src}: macro-F1 {f1:.4f}"
        assert acc == pytest.approx(art["metrics"]["holdout_accuracy"], abs=0.003), f"{src}: accuracy {acc:.4f}"
    per_class = f1_score(y, art["pred"]["predicted"], average=None, labels=art["config"]["classes"])
    for k, v in zip(art["config"]["classes"], per_class):
        assert v == pytest.approx(art["metrics"]["f1_per_class"][k], abs=1e-3), k
        assert v == pytest.approx(art["report"].loc[k, "f1-score"], abs=1e-6), k


# ---------------------------------------------------------------- ноутбук
@pytest.fixture(scope="session")
def nb():
    d = json.loads(NB.read_text(encoding="utf-8"))
    return ["".join(c["source"]) for c in d["cells"]], d


def test_notebook_has_no_errors(nb):
    _, d = nb
    errs = [i for i, c in enumerate(d["cells"]) for o in c.get("outputs", []) if o.get("output_type") == "error"]
    assert not errs, f"ячейки с ошибкой: {errs}"


def test_augmentations_are_physical(nb):
    """Аугментации студента на сером листе: углы после поворота светлые, шум одноцветный и слабый."""
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    src, _ = nb
    cell = next(s for s in src if "def build_train_aug" in s)
    code = re.search(r"def build_train_aug\(\):.*?\]\)\s*\n", cell, re.S).group(0)
    mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    ns = {"A": A, "cv2": cv2, "ToTensorV2": ToTensorV2, "IMG_SIZE": 224, "MEAN": mean, "STD": std}
    exec(code, ns)
    tf = A.Compose(ns["build_train_aug"]().transforms, seed=0)   # свой сид: иначе каждый запуск разный
    page = np.full((300, 300, 3), 240, np.uint8)
    gaps, corners, noise = [], [], []
    for _ in range(80):
        x = tf(image=page)["image"].numpy().transpose(1, 2, 0) * np.array(std) + np.array(mean)  # обратно в [0, 1]
        gaps.append(float(np.abs(x[..., 0] - x[..., 1]).mean()))
        corners.append(float(x[:6, :6].mean() - x[100:124, 100:124].mean()))
        noise.append(float(x[80:144, 80:144, 0].std()))
    assert max(gaps) < 1 / 255, f"каналы расходятся (цветной шум): {max(gaps) * 255:.1f} градаций"
    assert min(corners) > -0.05, f"после поворота углы темнее листа на {-min(corners) * 255:.0f} градаций"
    assert max(noise) < 20 / 255, f"шум слишком сильный: std до {max(noise) * 255:.0f} градаций"


def test_results_text_fields_filled(art):
    m = art["metrics"]
    empty = [k for k in ("chosen_for_production", "main_error_diagnosis", "experiments_tried")
             if str(m.get(k, "")).startswith("заполните")]
    assert not empty, f"в metrics.json не заполнены поля RESULTS: {empty}"


def test_notebook_reflections_filled(nb):
    src, _ = nb
    unfilled = [i for i, s in enumerate(src) if re.search(r"✍️ Ваш ответ:\s*\.\.\.", s)]
    assert not unfilled, f"не заполнены ответы в ячейках: {unfilled}"
