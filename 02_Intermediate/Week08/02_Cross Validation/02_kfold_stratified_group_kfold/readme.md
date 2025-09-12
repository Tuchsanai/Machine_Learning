
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
