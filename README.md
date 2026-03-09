# elastic-net-regression-from-scratch
End-to-end Elastic Net regression implemented from scratch in NumPy with mini-batch gradient descent and model evaluation.
---

# Elastic Net Regression (NumPy Implementation)

This project contains a **from-scratch implementation of Elastic Net Regression using NumPy**, including:

* L1 (Lasso) regularization
* L2 (Ridge) regularization
* Mini-batch gradient descent
* Early stopping
* Feature normalization
* R² score evaluation
* Model saving/loading with pickle

The implementation was **compared with `scikit-learn` using the R² metric** to validate correctness.

---

# Features

* Pure **NumPy implementation**
* **Elastic Net regularization**

Loss function:

[
Loss = MSE + \lambda (\alpha ||w||_1 + (1-\alpha)||w||_2^2)
]

Where:

* **λ (lambda)** → regularization strength
* **α (alpha)** → mix between L1 and L2

| Alpha     | Model Behavior   |
| --------- | ---------------- |
| 1         | Lasso Regression |
| 0         | Ridge Regression |
| 0 < α < 1 | Elastic Net      |

---

# Training Pipeline

The model performs the following steps automatically:

1. Data cleaning (`NaN → 0`)
2. Data shuffling
3. Train/Test split
4. Feature normalization
5. Bias term addition
6. Mini-batch gradient descent
7. Elastic Net regularization
8. Early stopping
9. Learning rate decay

---

# Model Architecture

Prediction equation:

[
y = XW + b
]

Weight update:

[
gradient = \frac{2}{n} X^T (y_{pred}-y)
]

Elastic Net penalty:

[
\lambda(\alpha sign(w) + 2(1-\alpha)w)
]

---

# Usage

## 1. Import

```python
from elastic_net import ElasticNet
```

---

## 2. Train Model

```python
model = ElasticNet(X, y, test_percent=20)

model.training(epochs=1000)
```

---

## 3. Evaluate Model

```python
mse = model.evaluate()
r2 = model.r2_score()

print("MSE:", mse)
print("R2 Score:", r2)
```

---

## 4. Predict

```python
prediction = model.predict([feature1, feature2, feature3])
print(prediction)
```

---

## 5. Save Model

```python
model.save_model("elastic_net.pkl")
```

---

## 6. Load Model

```python
model = ElasticNet.load_model("elastic_net.pkl")
```

---

# Comparison With Scikit-Learn

The implementation was validated by comparing results with:

`sklearn.linear_model.ElasticNet`

Metric used:

* **R² Score**

The results were **close to scikit-learn outputs**, confirming correct implementation of:

* gradient updates
* regularization
* normalization
* prediction pipeline

---

# Hyperparameters

| Parameter     | Default | Description                  |
| ------------- | ------- | ---------------------------- |
| learning_rate | 0.001   | gradient descent step size   |
| lambda        | 0.003   | regularization strength      |
| alpha         | 0.5     | L1/L2 mix                    |
| batch_size    | 4096    | mini batch size              |
| log           | 100     | learning rate decay interval |

---

# Dependencies

```
numpy
pickle (built-in)
```

Install numpy if needed:

```bash
pip install numpy
```

---

# Project Goal

This project was built to:

* understand **Elastic Net regression internals**
* implement **gradient descent optimizers**
* explore **regularization effects**
* compare custom ML models with **scikit-learn**

---

# Author

Krish Kumar

---
