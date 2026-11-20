# -*- coding: utf-8 -*-
# Загрузчик данных ДЗ 1. Автор: Дементьев.
import os, cv2, numpy as np, pandas as pd, torch
from torch.utils.data import Dataset


def validate_files(data_dir, rel_paths):
    """Проверяет список файлов. Возвращает (good, bad_log): годные пути и таблицу с причинами отбраковки."""
    good, bad = [], []
    for rel in rel_paths:
        # TODO 2.1 — «пуленепробиваемое» чтение.
        full = os.path.join(data_dir, rel)
        try:
            img = cv2.imread(full, cv2.IMREAD_GRAYSCALE)
        except Exception as e:                       # cv2.imread может бросить, а не вернуть None
            bad.append({"file": rel, "reason": f"exception: {type(e).__name__}"})
            continue
        if img is None:                              # битый/нечитаемый файл
            bad.append({"file": rel, "reason": "imread returned None"})
            continue
        if img.size == 0:                            # прочитался, но пустой (0 пикселей)
            bad.append({"file": rel, "reason": "empty image (size == 0)"})
            continue
        with open(full, "rb") as fh:                 # усечённый JPEG: читается частично, низ серый
            complete = fh.read().rstrip(b"\x00").endswith(b"\xff\xd9")
        if not complete:
            bad.append({"file": rel, "reason": "truncated JPEG (no FFD9 end marker)"})
            continue
        good.append(rel)
    # columns фиксированы, чтобы bad_log["reason"] не падал на пустой таблице
    return good, pd.DataFrame(bad, columns=["file", "reason"])

class DocDataset(Dataset):
    """Страницы документов: читает файл, приводит к одному размеру, отдает тензор и метку."""

    def __init__(self, items, data_dir, img_size=384, transform=None):
        self.items, self.data_dir = items, data_dir
        self.img_size, self.transform = img_size, transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        rel, label = self.items[idx]
        img = cv2.imread(os.path.join(self.data_dir, rel), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = np.stack([img] * 3, axis=-1)                 # серый скан -> три канала
        if self.transform is not None:
            img = self.transform(image=img)["image"]
        tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        return tensor, label
