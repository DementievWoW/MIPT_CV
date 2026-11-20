# -*- coding: utf-8 -*-
"""Независимая проверка ДЗ1: артефакты dz1_artifacts.zip и ноутбук сверяются с пересчётом с нуля.

Ничего в ноутбуке и артефактах не меняет. Код студента (student_loader.py) только вызывается.

Запуск:
    DZ1_DATA=<распакованный dataset_dz1_raw> DZ1_ART=<распакованный dz1_artifacts> \
    DZ1_NB=<ноутбук .ipynb> python -m pytest -q test_dz1.py
"""
import hashlib, importlib.util, json, os, re, sys, tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

DATA = Path(os.environ["DZ1_DATA"])
ART = Path(os.environ["DZ1_ART"])
NB = Path(os.environ["DZ1_NB"])
ALLOWED = ["letter", "form", "email", "invoice", "memo", "budget", "questionnaire", "resume"]


# ---------------------------------------------------------------- независимый пересчёт
@pytest.fixture(scope="session")
def ref():
    """Пересчёт Шагов 1-3 с нуля по сырому архиву, без кода студента."""
    labels = pd.read_csv(DATA / "labels.csv")
    rows = []
    for f in labels["file"]:
        p = DATA / f
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        # «читается» = декодируется целиком: усечённый JPEG (нет маркера конца FFD9)
        # cv2 читает частично, низ страницы заливается серым — это тоже брак
        ok = img is not None and img.size > 0 and p.read_bytes().rstrip(b"\x00").endswith(b"\xff\xd9")
        rows.append({"file": f, "readable": ok,
                     "hash": hashlib.md5(p.read_bytes()).hexdigest() if ok else None,
                     "brightness": float(img.mean()) if ok else np.nan})
    pages = pd.DataFrame(rows).merge(labels, on="file")
    valid = pages[pages["hash"].notna()]
    by_hash = valid[valid["label"].isin(ALLOWED)].groupby("hash")["label"].agg(lambda s: frozenset(s))
    inherited = {}
    for _, r in pages[pages["label"].isna() & pages["hash"].notna()].iterrows():
        labs = by_hash.get(r["hash"], frozenset())
        if len(labs) == 1:
            inherited[r["file"]] = next(iter(labs))
    pages["label2"] = pages["label"]
    for f, lab in inherited.items():
        pages.loc[pages["file"] == f, "label2"] = lab
    broken = set(pages.loc[~pages["readable"], "file"])
    foreign = set(pages.loc[pages["label"].notna() & ~pages["label"].isin(ALLOWED), "file"])
    dupes = set(valid.loc[valid.duplicated("hash", keep="first"), "file"])
    journal = {}
    for f in broken: journal.setdefault(f, "не читается")
    for f in foreign: journal.setdefault(f, "посторонний тип документа")
    for f in dupes: journal.setdefault(f, "дубликат")
    clean = pages[~pages["file"].isin(journal) & pages["label2"].isin(ALLOWED)]
    return {"pages": pages, "clean": clean, "journal": journal, "inherited": inherited,
            "hash_of": dict(zip(pages["file"], pages["hash"]))}


@pytest.fixture(scope="session")
def art():
    return {
        "clean": pd.read_csv(ART / "clean_labels.csv"),
        "rejected": pd.read_csv(ART / "rejected_log.csv"),
        "bad": pd.read_csv(ART / "bad_files_log.csv"),
        "train": (ART / "train.txt").read_text().split(),
        "val": (ART / "val.txt").read_text().split(),
        "results": json.loads((ART / "results.json").read_text(encoding="utf-8")),
    }


@pytest.fixture(scope="session")
def nb_cells():
    nb = json.loads(NB.read_text(encoding="utf-8"))
    return ["".join(c["source"]) for c in nb["cells"]], nb


# ---------------------------------------------------------------- Шаг 2: загрузчик студента
@pytest.fixture(scope="session")
def loader():
    spec = importlib.util.spec_from_file_location("student_loader", ART / "student_loader.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture()
def bad_dir():
    """Папка с файлами всех видов «битости» плюс один годный."""
    d = Path(tempfile.mkdtemp())
    good = np.full((80, 60), 255, np.uint8); cv2.putText(good, "A", (5, 50), 0, 1.5, 0, 3)
    cv2.imwrite(str(d / "good.jpg"), good)
    (d / "empty.jpg").write_bytes(b"")
    (d / "text.jpg").write_text("это не картинка")
    full = (d / "good.jpg").read_bytes()
    (d / "truncated.jpg").write_bytes(full[: len(full) // 2])
    return d


def test_loader_rejects_broken_files(loader, bad_dir):
    good, log = loader.validate_files(str(bad_dir), ["good.jpg", "empty.jpg", "text.jpg", "missing.jpg"])
    assert good == ["good.jpg"]
    assert set(log["file"]) == {"empty.jpg", "text.jpg", "missing.jpg"}
    assert {"file", "reason"} <= set(log.columns)


def test_loader_empty_log_keeps_columns(loader, bad_dir):
    _, log = loader.validate_files(str(bad_dir), ["good.jpg"])
    assert len(log) == 0 and {"file", "reason"} <= set(log.columns)


def test_loader_rejects_truncated_jpeg(loader, bad_dir):
    """Усечённый JPEG cv2 читает частично (низ серый); validate_files должна его отбраковать.
    В архиве такой файл scans/batch_2024_02/scan_95004.jpg раньше проходил в clean и train."""
    good, log = loader.validate_files(str(bad_dir), ["good.jpg", "truncated.jpg"])
    assert good == ["good.jpg"]
    assert set(log["file"]) == {"truncated.jpg"}


def test_dataset_tensor_contract(loader, bad_dir):
    ds = loader.DocDataset([("good.jpg", 3)], str(bad_dir), img_size=64)
    x, y = ds[0]
    assert len(ds) == 1 and y == 3
    assert tuple(x.shape) == (3, 64, 64) and str(x.dtype) == "torch.float32"
    assert 0.0 <= float(x.min()) and float(x.max()) <= 1.0


def test_bad_log_matches_reference(art, ref):
    broken = set(ref["pages"].loc[~ref["pages"]["readable"], "file"])
    assert set(art["bad"]["file"]) == broken and len(broken) == 12  # 9 не открываются + 3 усечены


# ---------------------------------------------------------------- Шаг 3: чистка
def test_clean_matches_reference(art, ref):
    mine = dict(zip(art["clean"]["file"], art["clean"]["label"]))
    theirs = dict(zip(ref["clean"]["file"], ref["clean"]["label2"]))
    assert mine == theirs


def test_rejected_log_matches_reference(art, ref):
    assert dict(zip(art["rejected"]["file"], art["rejected"]["reason"])) == ref["journal"]


def test_clean_has_no_content_duplicates(art, ref):
    hashes = art["clean"]["file"].map(ref["hash_of"])
    assert hashes.notna().all() and hashes.is_unique


def test_inherited_labels_are_unambiguous(art, ref):
    in_clean = set(art["clean"]["file"]) & set(ref["inherited"])
    assert len(ref["inherited"]) == 12
    assert len(in_clean) == 9, "из 12 унаследованных 3 оказались дубликатами"


def test_accounting_closes(art, ref):
    n = len(ref["pages"])
    removed_by_label = n - len(art["clean"]) - len(art["rejected"])
    # журнал: не читается 12 + посторонний 393 + дубликат 128 = 533
    assert len(art["rejected"]) == 533 and removed_by_label == 180
    assert n - 533 - 180 == len(art["clean"]) == 3178


# ---------------------------------------------------------------- Шаг 6: разбиение
def test_split_covers_clean_without_overlap(art):
    tr, va = set(art["train"]), set(art["val"])
    assert not tr & va
    assert tr | va == set(art["clean"]["file"])


def test_split_has_no_content_leak(art, ref):
    tr = {ref["hash_of"][f] for f in art["train"]}
    va = {ref["hash_of"][f] for f in art["val"]}
    assert not tr & va


def test_split_is_stratified(art):
    lab = dict(zip(art["clean"]["file"], art["clean"]["label"]))
    tr = pd.Series([lab[f] for f in art["train"]]).value_counts(normalize=True)
    va = pd.Series([lab[f] for f in art["val"]]).value_counts(normalize=True)
    assert set(va.index) == set(ALLOWED)
    assert float((tr - va).abs().max()) < 0.01


# ---------------------------------------------------------------- results.json
def test_results_numbers_match_reference(art, ref):
    r = art["results"]
    vc = ref["clean"]["label2"].value_counts()
    assert r["files_in_archive"] == len(ref["pages"])
    assert r["files_clean"] == len(ref["clean"])
    assert r["class_counts"] == vc.to_dict()
    assert r["imbalance_ratio"] == round(float(vc.max() / vc.min()), 2)
    assert r["train_size"] + r["val_size"] == r["files_clean"]


def test_results_manual_fields_filled(art):
    empty = [k for k in ("binarization", "augmentations", "main_finding")
             if str(art["results"][k]).startswith("заполните")]
    assert not empty, f"в results.json не заполнены поля: {empty}"


# ---------------------------------------------------------------- ноутбук
def test_notebook_has_no_errors(nb_cells):
    _, nb = nb_cells
    errs = [i for i, c in enumerate(nb["cells"]) for o in c.get("outputs", []) if o.get("output_type") == "error"]
    assert not errs, f"ячейки с ошибкой: {errs}"


def test_notebook_reflections_filled(nb_cells):
    src, _ = nb_cells
    unfilled = [i for i, s in enumerate(src) if "✍️ Ваш ответ: ..." in s]
    assert not unfilled, f"не заполнены ответы в ячейках: {unfilled}"


def _aug_source(src):
    cell = next(s for s in src if "def build_train_aug" in s and "TODO 6.1" in s)
    m = re.search(r"def build_train_aug\(\):.*?\]\)\s*\n", cell, re.S)
    return m.group(0)


def test_notebook_noise_matches_report(nb_cells):
    """В отчёте Шага 6 написано per_channel=False: в коде должно быть так же."""
    src, _ = nb_cells
    report_says = any("per_channel=False" in s for s in src if "WRONG_pipeline" in s or "аугментации" in s)
    code = _aug_source(src)
    if report_says:
        assert "per_channel=False" in code, "в отчёте per_channel=False, а в коде build_train_aug его нет"


def test_augmentations_are_physical(nb_cells):
    """Пайплайн студента на сером скане: углы белые, шум одноцветный и слабый."""
    import albumentations as A
    src, _ = nb_cells
    ns = {"A": A, "cv2": cv2}
    exec(_aug_source(src), ns)
    gray = np.full((384, 384), 240, np.uint8)
    img = np.stack([gray] * 3, axis=-1)
    rng_state = np.random.get_state()
    np.random.seed(0)
    import random; random.seed(0)
    outs = [ns["build_train_aug"]()(image=img)["image"] for _ in range(60)]
    np.random.set_state(rng_state)
    channel_gap = max(float(np.abs(o[..., 0].astype(int) - o[..., 1].astype(int)).mean()) for o in outs)
    corners = min(float(o[:8, :8].mean()) for o in outs)
    noise_std = max(float(o[150:230, 150:230, 0].std()) for o in outs)
    assert corners > 150, f"после поворота углы тёмные (минимум {corners:.0f})"
    assert channel_gap < 1.0, f"шум разный по каналам (цветной), расхождение {channel_gap:.1f} градаций"
    assert noise_std < 20, f"шум слишком сильный: std до {noise_std:.0f} градаций из 255"
