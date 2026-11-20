"""Независимая проверка Шагов 1-3 ДЗ1: считаем с нуля по сырому архиву (не по коду студента).
Запуск: python -I verify_step123.py <папка с распакованным dataset_dz1_raw>
Печатает ожидаемые числа и сверяет их с тем, что студент получил в ноутбуке."""
import sys, hashlib
from pathlib import Path
import numpy as np, pandas as pd, cv2

D = Path(sys.argv[1])
ALLOWED = ["letter","form","email","invoice","memo","budget","questionnaire","resume"]
labels = pd.read_csv(D/"labels.csv")
assert labels["file"].is_unique

rows=[]
for f in labels["file"]:
    p=D/f; img=cv2.imread(str(p),cv2.IMREAD_GRAYSCALE)
    ok = img is not None and img.size>0
    h = hashlib.md5(p.read_bytes()).hexdigest() if p.exists() and ok else None
    ink = None
    if ok:
        _,b=cv2.threshold(img,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU); ink=float((b>0).mean())
    rows.append((f,ok,h,ink))
P=pd.DataFrame(rows,columns=["file","readable","hash","ink"]).merge(labels,on="file")

exp={}
exp["rows"]=len(P)
exp["broken"]=int((~P.readable).sum())
exp["no_label"]=int(P.label.isna().sum())
foreign=P[P.label.notna() & ~P.label.isin(ALLOWED)]
exp["foreign"]=len(foreign)
valid=P[P.hash.notna()]
exp["dupes"]=int(valid.duplicated("hash",keep="first").sum())

# наследование
lab_ok=valid[valid.label.isin(ALLOWED)].groupby("hash").label.agg(lambda s:frozenset(s))
lab_fr=valid[valid.label.notna() & ~valid.label.isin(ALLOWED)].groupby("hash").label.agg(lambda s:frozenset(s))
inh={}; conf=0; frc=0; nc=0
for _,r in P[P.label.isna()].iterrows():
    h=r.hash
    if h is None or pd.isna(h): nc+=1; continue
    a=lab_ok.get(h,frozenset())
    if len(a)==1: inh[r.file]=next(iter(a))
    elif len(a)>1: conf+=1
    elif h in lab_fr.index: frc+=1
    else: nc+=1
exp.update(inherited=len(inh),conflicts=conf,foreign_copy_only=frc,no_copy=nc)

# clean по порядку фильтров студента
L=P.copy(); L["label2"]=L.label
for f,l in inh.items(): L.loc[L.file==f,"label2"]=l
drop=set(P[~P.readable].file)|set(foreign.file)|set(valid[valid.duplicated("hash",keep="first")].file)
clean=L[~L.file.isin(drop)&L.label2.isin(ALLOWED)]
exp["clean"]=len(clean)
vc=clean.label2.value_counts()
exp["imbalance"]=round(vc.max()/vc.min(),2)
exp["classes"]=vc.to_dict()
exp["inherited_in_clean"]=int(clean.file.isin(inh).sum())
exp["clean_dupe_hashes"]=int(clean.hash.duplicated().sum())
exp["clean_unreadable"]=int((~clean.readable).sum())
# журнал: порядок броken->foreign->dupes
j={}
for f in set(P[~P.readable].file): j.setdefault(f,"broken")
for f in foreign.file: j.setdefault(f,"foreign")
for f in valid[valid.duplicated("hash",keep="first")].file: j.setdefault(f,"dupe")
exp["journal"]=pd.Series(j).value_counts().to_dict()
exp["removed"]=len(P)-len(clean)
for k,v in exp.items(): print(f"{k:20s} {v}")

jf=set(j)
unl_removed=L[~L.file.isin(jf)&~L.label2.isin(ALLOWED)]
print("removed_by_label_filter", len(unl_removed))
print("unlabeled_left_(182)_in_journal", int(L[L.label.isna() & ~L.file.isin(inh)].file.isin(jf).sum()))
print("inherited_in_journal(dupes)", int(pd.Series(list(inh)).isin(jf).sum()))
