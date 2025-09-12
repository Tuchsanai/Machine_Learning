# 📑 Teaching Slides: Grid Search (Easy-to-Understand Version)

---

## **Slide 1 – Introduction to Grid Search**

🔍 **What is Grid Search?**
A technique to find the **best Hyperparameters** by trying every possible value in a “grid”.

⚙️ **What are Hyperparameters?**
Values that must be **set manually before training** (cannot be learned from data).

* KNN → `n_neighbors`
* Decision Tree → `max_depth`
* SVM → `C`, `gamma`
* Neural Network → learning rate, hidden layers, batch size

⭐ **Why are they important?**

* Correct choice ✅ → Higher accuracy
* Wrong choice ❌ → Underfitting / Overfitting

💡 **Analogy:** Like choosing a recipe. Good ingredients = tasty food, wrong ingredients = bad taste.

---

## **Slide 2 – Why Do We Need Grid Search?**

🚫 **Problems with guessing values**

1. **Parameters matter a lot** 🎯

   * Shallow tree → Underfitting
   * Deep tree → Overfitting

2. **Guessing is unreliable** ❌

   * Logistic Regression:

     * `C=1` → Accuracy = 70%
     * `C=100` → Accuracy = 85% 🎯
   * If we never test `C=100`, we miss the best result.

3. **Grid Search = Systematic approach** ✔

   * Tries all combinations
   * Ensures best option is found
   * No risk of skipping good values

👉 From “guessing” → To **scientific searching**

---

## **Slide 3 – How Grid Search Works**

📝 **Main Steps**

1️⃣ Define parameter grid

* SVM: `C=[0.1,1,10]`, `gamma=[0.01,0.1]`
* Decision Tree: `max_depth=[3,5,7]`

2️⃣ Try all combinations

* SVM → 3×2 = 6 models
* Tree → 3×3 = 9 models

3️⃣ Train + Evaluate each

* Tree (depth=7, split=10) → Acc=86% ✅

4️⃣ Pick the best result

* Best SVM = `C=10, gamma=0.01`

---

## **Slide 4 – Example in Scikit-Learn**

```python
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

param_grid = {'C':[0.1,1,10], 'gamma':[0.01,0.1], 'kernel':['rbf']}
model = SVC()

grid = GridSearchCV(model, param_grid, cv=5)
grid.fit(X_train, y_train)

print("Best Parameters:", grid.best_params_)
```

👉 The program tests all values and reports the **best combination automatically**.

---

## **Slide 5 – Key Takeaways**

📌 **Remember this**

* Grid Search = systematic way to find the **best Hyperparameters**
* Tests all values → picks the best
* Works with many models (SVM, KNN, Decision Tree, Logistic Regression)
* Usually combined with **Cross-Validation**
* Prevents random guessing → increases accuracy

🎯 **Shortcut memory:**
“Grid Search = Try every value → Get the best answer”

