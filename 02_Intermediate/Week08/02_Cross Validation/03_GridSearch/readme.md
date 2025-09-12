## **Slide 1 – Introduction to Grid Search**

**What is Grid Search?**

* A **method for hyperparameter tuning** in machine learning
* Works by **trying all possible combinations** of parameters from a predefined set
* Helps us find the **best version of the model**

**Hyperparameters = model settings chosen before training**

* Examples:

  * KNN → number of neighbors
  * Decision Tree → maximum depth
  * SVM → C, gamma

💡 Analogy: Like **testing every recipe** until you find the tastiest dish.

---

## **Slide 2 – Why Do We Need Grid Search?**

**Why not just guess parameters?**

1. **Hyperparameters strongly affect results**

   * Too small depth → underfitting 🌱
   * Too large depth → overfitting 🌳

2. **Random guessing is unreliable**

   * Logistic Regression:

     * `C=1` → 70% accuracy
     * `C=100` → 85% accuracy
   * Without systematic search, we may miss the best setting.

3. **Grid Search = systematic exploration**

   * Ensures all parameter options are tested
   * Provides fair, scientific comparison

👉 **Turns trial & error into a structured process**.

---

## **Slide 3 – How Grid Search Works**

**Step-by-Step Process**

1. **Choose parameter grid**

   * SVM → `C=[0.1, 1, 10]`, `gamma=[0.01, 0.1]`
   * Decision Tree → `max_depth=[3,5,7]`, `min_samples_split=[2,5,10]`
   * KNN → `n_neighbors=[3,5,7,9]`, `weights=[‘uniform’, ‘distance’]`

2. **Try all combinations**

   * SVM: 3×2 = 6 models
   * Decision Tree: 3×3 = 9 models
   * KNN: 4×2 = 8 models

3. **Train and evaluate each model**

   * Example (Decision Tree):

     * depth=5, split=2 → 82% accuracy
     * depth=7, split=10 → 86% ✅

4. **Pick the best parameters**

   * Example: SVM best = `C=10, gamma=0.01`

👉 Result = **most accurate model** trained on best settings.

---

## **Slide 4 – Example in Scikit-Learn**

**Code Example: Grid Search with SVM**

```python
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

# Define parameter grid
param_grid = {
    'C': [0.1, 1, 10],
    'gamma': [0.01, 0.1],
    'kernel': ['rbf']
}

# Create model
model = SVC()

# Apply Grid Search with cross-validation
grid = GridSearchCV(model, param_grid, cv=5)
grid.fit(X_train, y_train)

print("Best Parameters:", grid.best_params_)
```

👉 Grid Search **automatically finds the best parameters**.

---

## **Slide 5 – Key Takeaways**

**Summary**

* Grid Search = **systematic hyperparameter tuning**
* Tries all parameter combinations → picks the best
* Works with many ML models (SVM, KNN, Trees, Logistic Regression)
* Usually combined with **cross-validation** for reliability
* Ensures higher accuracy and avoids “lucky guesses”

🎯 **Remember:** Grid Search = “brute force search for the best settings.”

---

Would you like me to **convert this polished version into a PowerPoint (.pptx) with diagrams (grids, parameter tables, accuracy charts, icons)** so it’s ready for teaching slides?
