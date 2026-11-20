"""Независимая проверка Шага 4 (резкость). Запуск: python -I verify_step4.py <папка архива>"""
import sys, hashlib
from pathlib import Path
import numpy as np, pandas as pd, cv2
D=Path(sys.argv[1]); A=["letter","form","email","invoice","memo","budget","questionnaire","resume"]; SEED=42
l=pd.read_csv(D/"labels.csv"); rows=[]
for f in l.file:
    img=cv2.imread(str(D/f),cv2.IMREAD_GRAYSCALE)
    if img is None or img.size==0: rows.append((f,False,None,None,None)); continue
    _,b=cv2.threshold(img,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    rows.append((f,True,hashlib.md5((D/f).read_bytes()).hexdigest(),float((b>0).mean()),float(img.mean())))
P=pd.DataFrame(rows,columns=["file","readable","hash","ink","brightness"]).merge(l,on="file")
V=P[P.hash.notna()]
lab=V[V.label.isin(A)].groupby("hash").label.agg(lambda s:frozenset(s))
inh={r.file:next(iter(lab[r.hash])) for _,r in P[P.label.isna()&P.hash.notna()].iterrows() if r.hash in lab.index and len(lab[r.hash])==1}
P["label2"]=P.label; 
for f,x in inh.items(): P.loc[P.file==f,"label2"]=x
drop=set(P[~P.readable].file)|set(P[P.label.notna()&~P.label.isin(A)].file)|set(V[V.duplicated("hash")].file)
clean=P[~P.file.isin(drop)&P.label2.isin(A)][["file","label2","ink"]].reset_index(drop=True)
print("clean",len(clean))
sharp=lambda g: cv2.Laplacian(g,cv2.CV_64F).var()
print("первая страница:",round(sharp(cv2.imread(str(D/clean.file[0]),0)),1))
s=clean.sample(300,random_state=SEED).copy()
s["sharp"]=[sharp(cv2.imread(str(D/f),0)) for f in s.file]
print("p5/p50/p95:",[round(x) for x in np.percentile(s.sharp,[5,50,95])],"min",round(s.sharp.min()))
for t in (100,300,500): print("порог",t,int((s.sharp<t).sum()))
print("corr ink~sharp:",round(s.ink.corr(s.sharp),3))
bl=[]
for f in clean.file.sample(6,random_state=1):
    g=cv2.imread(str(D/f),0); bl.append(round(sharp(g)/max(sharp(cv2.GaussianBlur(g,(0,0),2.5)),1e-6),1))
print("во сколько раз упало:",bl)

P2=P.set_index("file")
s["bright"]=s.file.map(P2.brightness)
print("\n10 самых нерезких: ink, brightness"); print(s.nsmallest(10,"sharp")[["sharp","ink","bright"]].round(3).to_string())
print("\nвыбросы ink>0.12 & sharp<5000:"); print(s[(s.ink>0.12)&(s.sharp<5000)][["sharp","ink","bright"]].round(3).to_string())
print("медиана яркости sample:",round(s.bright.median(),1))
