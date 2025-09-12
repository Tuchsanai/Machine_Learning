
# ตารางเปรียบเทียบ: เลือกวิธีแบ่งข้อมูลให้เหมาะกับลักษณะ Data

| วิธีแบ่ง                                | เหมาะกับข้อมูลแบบไหน                                                                                                       | เหตุผลหลัก                                                   | ไม่เหมาะเมื่อ                                                     | ทางเลือก/หมายเหตุ                                                         |
| --------------------------------------- | -------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------ | ----------------------------------------------------------------- | ------------------------------------------------------------------------- |
| **train\_test\_split (สุ่มครั้งเดียว)** | ข้อมูล **IID** (ตัวอย่างอิสระต่อกัน), คลาสค่อนข้างสมดุล, **ไม่มีการซ้ำของหน่วยเดียวกัน** (เช่น คนเดียวกัน/เครื่องเดียวกัน) | เร็ว เรียบง่าย เพียงพอเมื่อข้อมูลมากและกระจายตัวดี           | ข้อมูล **ไม่สมดุลมาก** / มี **กลุ่ม (leakage)** / ขนาดเล็ก        | ถ้าจำเป็นให้ใช้ `stratify=y` สำหรับ classification                        |
| **K-Fold**                              | ข้อมูล IID, ขนาดเล็ก-กลาง, ต้องการ **ประเมิน/จูนไฮเปอร์พารามิเตอร์** อย่างเสถียร                                           | เฉลี่ยผลได้หลายพับ ลดความแปรปรวนจากการสุ่มครั้งเดียว         | คลาสไม่สมดุลมาก, มี **กลุ่มซ้ำ** ในคน/เครื่อง/ผู้เขียน            | ถ้าคลาสไม่สมดุล → ใช้ **Stratified K-Fold**                               |
| **Stratified K-Fold**                   | **Classification ที่ไม่สมดุล** (binary/multi-class), ต้องรักษาสัดส่วนคลาสในทุกพับ                                          | รักษาสัดส่วนคลาสทุกพับ ทำให้เมตริกไม่ลำเอียง                 | **Regression** (ใช้ไม่ได้ตรง ๆ), **มัลติเลเบล** (ต้องเทคนิคพิเศษ) | รีเกรสชันทำแบบ **stratify-บน-bins** ได้ (จัด bin เป้าหมายก่อน)            |
| **Group K-Fold**                        | มี **หลายตัวอย่างต่อหน่วยเดียวกัน** (ผู้ป่วย/ผู้ใช้/อุปกรณ์/ผู้พูด/ร้านค้า/เซสชัน) และต้องการกัน leakage                   | กันข้อมูลของกลุ่มเดียวกันไม่ให้ไปอยู่ทั้ง train และ val/test | ไม่มีโครงสร้างกลุ่มชัดเจน หรือจำนวนกลุ่มน้อยเกิน                  | ถ้าเป็น **อนุกรมเวลา** ควรใช้ **TimeSeriesSplit** (นอกเหนือจาก 4 วิธีนี้) |

> หมายเหตุ: งานอนุกรมเวลา/ตามลำดับ (time dependency) ควรใช้ **TimeSeriesSplit** หรือวิธี holdout ตามเวลา ไม่ควรสุ่มสลับลำดับ

---

# ตัวอย่าง “ประเภท Data” ที่เหมาะกับแต่ละวิธี (5–8 ตัวอย่าง/วิธี พร้อมเหตุผล)

## 1) train\_test\_split (สุ่มครั้งเดียว + อาจ `stratify=y` เมื่อจำเป็น)

1. **Ames / King County House Prices (รีเกรสชัน, หนึ่งแถวต่อบ้าน)**
   **เหตุผล:** ตัวอย่างค่อนข้างอิสระ ไม่มีคน/บ้านเดียวกันซ้ำหลายแถว และคลาส/เป้าหมายต่อเนื่องไม่ต้องรักษาสัดส่วนพิเศษ → สุ่มครั้งเดียวเพียงพอ
   **ระวัง:** ถ้ามีบ้านเดียวกันคนละบันทึก หรือมีช่วงเวลา/ย่านก่อให้เกิด leakage ให้พิจารณา Group/Time split

2. **ตารางคุณสมบัติทางเคมีของโมเลกุล (QSAR) ที่สุ่มคละโมเลกุลอิสระ**
   **เหตุผล:** โมเลกุลเป็นหน่วยอิสระ (ไม่มีหลายแถวต่อโมเลกุล) → สุ่มได้
   **ระวัง:** ถ้ามี “ชุดอนุพันธ์/ตระกูลโมเลกุล” ใกล้กันมาก อาจพิจารณา Group ตามตระกูลเพื่อทดสอบการ generalize

3. **ภาพถ่ายวัตถุที่ไม่มีบุคคล/ไอดีซ้ำ (balanced classes)**
   **เหตุผล:** ตัวอย่างเป็นอิสระ คลาสสมดุล → สุ่มเร็วและพอ
   **ระวัง:** ถ้าภาพหลายมุมของ “วัตถุชิ้นเดียวกัน” → ควร Group

4. **รีวิวสินค้าแบบหนึ่งรีวิวต่อผู้ใช้ (one-row-per-user)**
   **เหตุผล:** ไม่มีผู้ใช้คนเดิมซ้ำหลายแถว ลดความเสี่ยง leakage → สุ่มง่าย
   **ระวัง:** ถ้ามีหลายรีวิว/ผู้ใช้ → ใช้ Group

5. **ผลทดลอง A/B ที่ aggregate เป็นหนึ่งแถวต่อผู้ใช้/หน่วย**
   **เหตุผล:** หนึ่งต่อหนึ่ง ลดการพึ่งพา → สุ่มประหยัดเวลา
   **ระวัง:** ถ้าเก็บหลายรอบต่อผู้ใช้เดียวกัน ให้ Group ตาม user

6. **ชุดข้อมูลคลาสสิกเช่น Iris (สมดุล, ขนาดเล็กแต่เด็มไปด้วย IID)**
   **เหตุผล:** โครงสร้างเรียบง่าย ใช้สุ่มก็เพียงพอเมื่อไม่ได้โฟกัสจูนหนัก
   **ระวัง:** ถ้าต้องประเมินเสถียรภาพของเมตริกมากขึ้น ให้เปลี่ยนเป็น K-Fold

---

## 2) K-Fold (k=5/10 ตามขนาดข้อมูล)

1. **Iris / Wine / Breast Cancer (ชุดเล็ก–กลาง, IID)**
   **เหตุผล:** เฉลี่ยผลหลายพับ ลดความผันผวนจากการสุ่มครั้งเดียว เหมาะกับการ **เลือกโมเดล/จูนพารามิเตอร์**

2. **Digits (ภาพตัวเลข 8×8, \~1,800 ตัวอย่าง) แบบไม่เน้นบาลานซ์มาก**
   **เหตุผล:** ขนาดไม่ใหญ่มาก ใช้ K-Fold เพื่อใช้ข้อมูลคุ้มค่าและได้ค่าประเมินนิ่งขึ้น

3. **รีเกรสชันคุณภาพไวน์/ราคาบ้านฉบับย่อ (ตัวอย่างไม่มาก)**
   **เหตุผล:** การเฉลี่ยหลายพับช่วยให้ RMSE/MAE เสถียรกว่า holdout

4. **ข้อความสั้นอิสระ (เช่น ข่าวพาดหัว) ที่ไม่มีผู้เขียนซ้ำหลายแถว**
   **เหตุผล:** IID สมเหตุสมผล ใช้ K-Fold เพื่อเปรียบเทียบเวกเตอร์ไรเซอร์/โมเดล

5. **ตารางสุขภาพทั่วไป (balanced) เช่น ทำนายความเสี่ยงระดับต่ำ/ปกติ**
   **เหตุผล:** ไม่มีความไม่สมดุลชัด ใช้ K-Fold ได้เต็มประสิทธิภาพ

6. **งานรีเกรสชันตัวเลขต่อเนื่องทั่วไป (ไม่มีคลาส)**
   **เหตุผล:** ไม่มีปัญหาสัดส่วนคลาส → K-Fold มาตรฐาน

> ระวัง: หาก **คลาสไม่สมดุล** ให้เปลี่ยนเป็น **Stratified K-Fold** เพื่อรักษาสัดส่วนในทุกพับ

---

## 3) Stratified K-Fold (classification ที่ไม่สมดุล/หลายคลาส)

1. **Credit Card Fraud (ฉ้อโกง \~0.1–1%)**
   **เหตุผล:** ต้องรักษาสัดส่วน fraud\:non-fraud ให้ใกล้เคียงกันในทุกพับ มิฉะนั้นเมตริกจะผันผวน/ลวงตา

2. **Breast Cancer (malignant เป็นส่วนน้อย)**
   **เหตุผล:** รักษาสัดส่วน malignant/benign เพื่อความเสถียรของ recall/ROC-AUC

3. **Churn Prediction (Positive \~10–30%)**
   **เหตุผล:** ให้แต่ละพับมี churn/non-churn ใกล้เคียงจริง เพื่อวัด F1/PR-AUC อย่างยุติธรรม

4. **Multi-class ไม่สมดุล (เช่น 5 คลาส แต่บางคลาสมีน้อยมาก)**
   **เหตุผล:** ควบคุมสัดส่วนทุกคลาสต่อพับ ป้องกันพับใดพับหนึ่งขาดคลาสหายไป

5. **Defect Detection ในโรงงาน (เสีย \~2–5%)**
   **เหตุผล:** ป้องกัน validation พับที่แทบไม่มี defect ซึ่งทำให้เมตริกสูงผิดจริง

6. **Star Rating (1–5 ดาว) ที่ปลายสุด (1/5) น้อย**
   **เหตุผล:** รักษาสัดส่วนให้ครบทุกระดับดาวในทุกพับ

7. **คลาสโรคหลายระยะ (Stage I–IV) ที่กระจายไม่เท่ากัน**
   **เหตุผล:** ให้ทุกระยะปรากฏในทุกพับ → ประเมิน macro-F1 ได้ตรงกว่า

> หมายเหตุ: งานรีเกรสชันใช้ Stratified โดยตรงไม่ได้ แต่สามารถ **จัด bin เป้าหมาย** (เช่น quantiles) แล้วทำ Stratified K-Fold บน bin แทน

---

## 4) Group K-Fold (กัน leakage ระดับ “หน่วย/เอนทิตี”)

1. **ผู้ป่วยมีหลาย visit/record ต่อคน (Patient-level)**
   **เหตุผล:** ถ้าคนเดียวกันโผล่ทั้ง train และ val จะ leakage (โมเดลจำลักษณะเฉพาะบุคคล) → Group ตาม `patient_id`

2. **เสียงพูดหลายคลิปต่อผู้พูด (Speaker-level)**
   **เหตุผล:** ป้องกันโมเดลเรียนรู้เอกลักษณ์เสียงของ speaker ใน train แล้วไปทาย val ที่เป็นคนเดิม → Group ตาม `speaker_id`

3. **ภาพหลายใบต่อ “วัตถุ/บุคคล” เดียวกัน (Subject-level / product-level)**
   **เหตุผล:** หลายมุม/หลายสภาพแสงของวัตถุเดียวกัน → ถ้าไม่ group จะประเมินสูงเกินจริง

4. **รีวิว/พฤติกรรมผู้ใช้หลายรายการต่อ user (Recommender / CTR)**
   **เหตุผล:** กิจกรรมของ user เดิมซ้ำหลายแถว → Group ตาม `user_id` (หรือทดลอง `item_id` หากอยากทดสอบ generalize ไปยัง item ใหม่)

5. **เซนเซอร์หลายอ่านค่าต่อเครื่อง (Machine-level)**
   **เหตุผล:** ค่าที่อ่านจากเครื่องเดียวกันใกล้กันมาก → Group ตาม `machine_id` เพื่อทดสอบกับเครื่องใหม่จริง ๆ

6. **เอกสาร/โพสต์หลายชิ้นต่อผู้เขียน (Author-level NLP)**
   **เหตุผล:** สำนวนเดียวกันทำให้ leakage → Group ตาม `author_id`

7. **ยอดขายรายบิล/รายการหลายรายการต่อร้าน (Store-level)**
   **เหตุผล:** ร้านเดียวกันมีรูปแบบเฉพาะ → Group ตาม `store_id` เพื่อวัดการย้ายโมเดลไปยังร้านใหม่

8. **ภาพ defect หลายภาพต่อชิ้นงานเดียว (Part-level vision)**
   **เหตุผล:** ชิ้นงานเดียวกันถ่ายหลายมุม → Group ตาม `part_id`

---

## ตารางมองเร็ว: “ตัวอย่างสถานการณ์” vs “วิธีแนะนำ”

| ตัวอย่างสถานการณ์                    | train\_test\_split | K-Fold | Stratified K-Fold | Group K-Fold |
| ------------------------------------ | :----------------: | :----: | :---------------: | :----------: |
| ราคาบ้าน หนึ่งแถวต่อบ้าน (IID)       |          ✓         |    ✓   |                   |              |
| คุณภาพไวน์ (รีเกรสชัน, IID, ชุดเล็ก) |                    |    ✓   |                   |              |
| Fraud Detection (ไม่สมดุลมาก)        |  (กับ `stratify`)  |        |         ✓         |              |
| Breast Cancer (ไม่สมดุล)             |  (กับ `stratify`)  |        |         ✓         |              |
| ภาพหลายมุมต่อวัตถุเดียว              |                    |        |                   |       ✓      |
| ผู้ป่วยหลาย visit                    |                    |        |                   |       ✓      |
| ผู้พูดหลายคลิปเสียง                  |                    |        |                   |       ✓      |
| รีวิวหลายรายการต่อ user              |                    |        |                   |       ✓      |
| ข้อความสั้น IID ไม่มีผู้เขียนซ้ำ     |          ✓         |    ✓   | *ถ้าไม่สมดุล →* ✓ |              |
| รีเกรสชันทั่วไป (ไม่มี bin/กลุ่ม)    |          ✓         |    ✓   |                   |              |

---

## เช็คก่อนเลือกวิธี (Checklist สั้น ๆ)

* **มี “หลายแถวต่อหน่วย” ไหม?** → ใช้ **Group K-Fold**
* **เป็น Classification และคลาสไม่สมดุล?** → **Stratified K-Fold** (หรือ `train_test_split(..., stratify=y)`)
* **ต้องการจูน/ประเมินนิ่ง ๆ บนชุดเล็ก–กลาง?** → **K-Fold**
* **ข้อมูลใหญ่, IID ชัด, ไม่มีปัญหาข้างต้น?** → **train\_test\_split** ก็พอ
* **มีลำดับเวลา?** → ใช้ **TimeSeriesSplit / split ตามเวลา** (อย่าสุ่มปน)





---

## Slide 0 — Learning Goals (What you’ll master)

* Why we split data and how this prevents **data leakage**
* When to use **hold‑out** vs **cross‑validation**
* How **K‑Fold**, **Stratified K‑Fold**, and **Group K‑Fold** differ
* Read **fold diagrams** and map them to real code
* Choose the **right splitter** for your dataset and task

---

## Slide 1 — Why Split at All?

**Objective:** estimate how well the model will generalize to unseen data.

* **Training set** → fit model parameters
* **Validation set** → tune hyperparameters (via CV)
* **Test set** → final, unbiased performance estimate

**If you tune on the test set → leakage → overly optimistic results.**

---

## Slide 2 — Vocabulary & Symbols

* We label samples as **x1, x2, …, xN** (and class labels as **y1, y2, …**)
* **Fold** = one partition; in K‑Fold we cycle through which fold is validation.
* **Stratify** = preserve class proportions in each split.
* **Group** = all samples sharing the same entity (patient, user, house‑ID) must stay together.

---

## Slide 3 — Hold‑Out Split with `train_test_split`

Use for quick baselines or when the dataset is large enough.

**Visual (N = 20, test\_size = 0.2):**

```
All:   x1 x2 x3 x4 x5 x6 x7 x8 x9 x10 x11 x12 x13 x14 x15 x16 x17 x18 x19 x20
Train: [ x? x? x? x? x? x? x? x? x?  x?  x?  x?  x?  x?  x?  x? ]  (80%)
Test:  [ x? x? x? x? ]                                    (20%)
```

* With `shuffle=True` (default) the composition is random; control with `random_state`.
* For **classification**, use `stratify=y` to keep class ratios similar.

**When NOT to use**: time series or grouped data → use specialized splitters.

---

## Slide 4 — `train_test_split` (Regression, quick baseline)

```python
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
import numpy as np

X, y = load_diabetes(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(
    X, y, test_size=0.2, random_state=42 # shuffle by default
)

model = LinearRegression().fit(Xtr, ytr)
y_pred = model.predict(Xte)
rmse = np.sqrt(mean_squared_error(yte, y_pred))
print(f"Test RMSE: {rmse:.3f}")
print(f"Test R2  : {r2_score(yte, y_pred):.3f}")
```

**Tips:**

* Use `train_size`/`test_size` to control proportions.
* Set `random_state` for reproducibility in teaching/demo.

---

## Slide 5 — `train_test_split` (Classification with `stratify`)

**Why?** Keeps class balance similar across train and test (critical for imbalanced data).

**Visual (binary classes A/B, \~50/50):**

```
Indices: x1 x2 x3 x4 x5 x6 x7 x8 x9 x10 x11 x12 x13 x14 x15 x16
Labels :  A  B  A  A  B  B  A  B  A   A   B   A   B   A   B   B
Train :   ~80% with A/B ratio ≈ original
Test  :   ~20% with A/B ratio ≈ original
```

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

X, y = load_breast_cancer(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

clf = LogisticRegression(max_iter=1000).fit(Xtr, ytr)
print("Accuracy:", accuracy_score(yte, clf.predict(Xte)))
```

---

## Slide 6 — K‑Fold Cross‑Validation (Overview)

* Split data into **K** equal folds.
* Train **K times**: each time, hold out one fold as validation and train on the rest.
* Report the **mean/±std** of the metric across K runs.

**Why?** Uses all data for training & validation across runs → more stable estimates than a single hold‑out.

**Typical K**: 5 or 10.

---

## Slide 7 — K‑Fold (Diagram, K=5, N=20)

```
Idx :   1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20
Fold:  [1][2][3][4][5][1][2][3][4][5][1][2][3][4][5][1][2][3][4][5]

Iter 1 (val=Fold 1): val = {x1,x6,x11,x16}
Iter 2 (val=Fold 2): val = {x2,x7,x12,x17}
Iter 3 (val=Fold 3): val = {x3,x8,x13,x18}
Iter 4 (val=Fold 4): val = {x4,x9,x14,x19}
Iter 5 (val=Fold 5): val = {x5,x10,x15,x20}
```

* With `shuffle=True`, fold assignment is randomized (use `random_state`).

---

## Slide 8 — K‑Fold (Code, regression)

```python
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.linear_model import Ridge
import numpy as np

kf = KFold(n_splits=5, shuffle=True, random_state=42)
model = Ridge(alpha=1.0)
rmse_scorer = make_scorer(mean_squared_error, greater_is_better=False)

scores = cross_val_score(model, X, y, cv=kf, scoring=rmse_scorer)
rmse = np.sqrt(-scores)  # scores are negative MSE
print("Fold RMSEs:", np.round(rmse, 3))
print("Mean ± SD RMSE:", f"{rmse.mean():.3f} ± {rmse.std():.3f}")
```

---

## Slide 9 — Stratified K‑Fold (Overview)

**For classification.** Ensures each fold has roughly the same class proportions as the full dataset.

**When to use:** imbalanced classes, small datasets, metrics sensitive to prevalence.

**Not for regression** (there’s `StratifiedKFold` only for classification; for regression see advanced methods like `StratifiedKFold` on binned targets if needed).

---

## Slide 10 — Stratified K‑Fold (Diagram)

Assume 16 samples, labels A/B with counts A=8, B=8.

```
Idx :     1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16
Label:    A  B  A  A  B  B  A  B  A  A  B  A  B  A  B  B

K=4 folds (each fold aims for A=2, B=2):
Fold 1:   A  B  A  B
Fold 2:   A  B  A  B
Fold 3:   A  B  A  B
Fold 4:   A  B  A  B
```

Each iteration: one fold is validation; the rest are training.

---

## Slide 11 — Stratified K‑Fold (Code, classification)

```python
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.linear_model import LogisticRegression

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
clf = LogisticRegression(max_iter=1000)

cv = cross_validate(
    clf, X, y, cv=skf, scoring=["accuracy", "f1_macro"], return_train_score=True
)

print("Val Accuracy:", np.round(cv["test_accuracy"], 3))
print("Val F1-macro:", np.round(cv["test_f1_macro"], 3))
print(
    f"Mean Acc: {cv['test_accuracy'].mean():.3f} ± {cv['test_accuracy'].std():.3f} | "
    f"Mean F1m: {cv['test_f1_macro'].mean():.3f} ± {cv['test_f1_macro'].std():.3f}"
)
```

**Tip:** Always use `shuffle=True` + `random_state` for reproducible teaching demos.

---

## Slide 12 — Group K‑Fold (Overview)

**Goal:** keep all samples from the same **group** entirely in **train** or **validation**, never split across.

**Use‑cases:**

* Multiple records per **patient**, **user**, **household**, **product**
* Multiple images per subject/session

**Why?** If the same group leaks into both train and validation, performance is inflated.

---

## Slide 13 — Group K‑Fold (Diagram)

Example: 10 samples with 5 groups

```
Sample:  x1 x2 x3 x4 x5 x6 x7 x8 x9 x10
Group :  g1 g1 g2 g2 g2 g3 g3 g4 g4 g5

K=3 (one possible assignment):
Iter 1: Val groups {g1, g4} → Val = {x1,x2,x8,x9}; Train = others
Iter 2: Val group  {g2}     → Val = {x3,x4,x5};    Train = others
Iter 3: Val groups {g3, g5} → Val = {x6,x7,x10};   Train = others
```

Groups never appear in both train and validation simultaneously.

---

## Slide 14 — Group K‑Fold (Code)

```python
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

# Fake data (replace with real X, y)
N = 30
X = np.random.randn(N, 5)
y = X[:, 0] * 2.0 + np.random.randn(N)

# Create groups (e.g., 10 users, each appears 3 times)
groups = np.repeat(np.arange(10), 3)  # [0,0,0, 1,1,1, ..., 9,9,9]

model = Ridge(alpha=1.0)
gkf = GroupKFold(n_splits=5)

fold = 0
for train_idx, val_idx in gkf.split(X, y, groups=groups):
    fold += 1
    model.fit(X[train_idx], y[train_idx])
    pred = model.predict(X[val_idx])
    rmse = np.sqrt(mean_squared_error(y[val_idx], pred))
    print(f"Fold {fold}: Val RMSE = {rmse:.3f} | Val groups = {np.unique(groups[val_idx])}")
```

**Note:** Group sizes can be uneven; `GroupKFold` distributes groups across folds to balance counts of groups, not sample counts.

---

## Slide 15 — Grouped Hold‑Out (single split) with `GroupShuffleSplit`

When you need **one train/test split** but must respect groups.

```python
from sklearn.model_selection import GroupShuffleSplit

gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups=groups))
Xtr, Xte = X[train_idx], X[test_idx]
ytr, yte = y[train_idx], y[test_idx]
```

---

## Slide 16 — Visualizing Folds in Code (ASCII helper)

Use this in class to **print** fold maps your students can read.

```python
import numpy as np

def print_fold_map(indices, title=""):
    # indices: list of arrays (each one is the validation indices of that fold)
    all_idx = sorted(np.unique(np.concatenate(indices)))
    asgn = {i: None for i in all_idx}
    for f, val in enumerate(indices, start=1):
        for ix in val:
            asgn[ix] = f
    line_idx = " ".join([f"{i:>2}" for i in all_idx])
    line_fld = " ".join([f"[{asgn[i]:>1}]" for i in all_idx])
    print(title)
    print("Idx :", line_idx)
    print("Fold:", line_fld)

# Example for KFold
from sklearn.model_selection import KFold
X_dummy = np.zeros((20, 3))
kf = KFold(n_splits=5, shuffle=True, random_state=42)
val_lists = [val for _, val in kf.split(X_dummy)]
print_fold_map(val_lists, title="KFold assignment (val fold labels)")
```

This produces a map similar to Slide 7.

---

## Slide 17 — Choosing the Right Splitter (Cheat Sheet)

| Situation                                          | Use                                                    | Why                                     |
| -------------------------------------------------- | ------------------------------------------------------ | --------------------------------------- |
| Quick baseline, large data, no groups              | `train_test_split` (+ `stratify=y` for classification) | Fast, simple                            |
| Hyperparameter tuning (regression)                 | `KFold`                                                | Stable estimate across folds            |
| Hyperparameter tuning (classification, imbalanced) | `StratifiedKFold`                                      | Preserves class ratios                  |
| Same entity appears multiple times                 | `GroupKFold` / `GroupShuffleSplit`                     | Avoid leakage across group              |
| Time‑ordered data                                  | `TimeSeriesSplit`                                      | Respects temporal order (no look‑ahead) |

---

## Slide 18 — Gotchas & Best Practices

* **Always set `random_state`** when teaching/demonstrating.
* **Shuffle** before K‑Fold unless order is meaningful (e.g., time series).
* **Stratify** whenever class imbalance exists.
* For `GroupKFold`, ensure **groups align** with real leakage boundaries.
* Avoid using the **test set** for model selection; keep it sealed until final.

---

## Slide 19 — Mini‑Labs (ready for class)

1. **Hold‑Out + Stratify (Breast Cancer):**

   * Compare accuracy with & without `stratify=y` on small test\_size.
2. **K‑Fold (Diabetes):**

   * Evaluate Ridge with K=3,5,10. Discuss mean vs variance of RMSE.
3. **StratifiedKFold (Iris → binary subset):**

   * Filter to two classes; compare `KFold` vs `StratifiedKFold` stability.
4. **GroupKFold (synthetic groups):**

   * Show inflated accuracy when you ignore groups vs proper GroupKFold.

---

## Slide 20 — Appendix: Quick Patterns

**Scoring with `cross_validate` (multiple metrics):**

```python
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer, f1_macro, accuracy_score

cv = cross_validate(
    clf, X, y, cv=skf,
    scoring={"acc": "accuracy", "f1m": "f1_macro"},
    return_train_score=True
)
print(cv.keys())  # dict of arrays for each metric
```

**Using splitters directly inside GridSearchCV:**

```python
from sklearn.model_selection import GridSearchCV

param_grid = {"alpha": [0.1, 1.0, 10.0]}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
search = GridSearchCV(RidgeClassifier(), param_grid=param_grid, cv=cv)
```

---

## Slide 21 — Summary

* **train\_test\_split** → fast baseline; use `stratify` for classification
* **K‑Fold** → robust estimates; standard for regression
* **StratifiedKFold** → classification with balanced folds
* **GroupKFold** → prevents leakage across entities/groups
* Visualize folds to **explain** and **debug** your evaluation

🎯 *You can copy any code cell into a Jupyter notebook and run immediately.*
