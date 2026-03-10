# ElasticNet Regression — Mathematical Explanation

## 1. Linear Model

For a dataset with **n samples** and **d features**

$$
X \in \mathbb{R}^{n \times d}, \quad y \in \mathbb{R}^{n \times 1}
$$

Prediction equation

$$
\hat{y} = Xw + b
$$

Where

* (w) = weight vector
* (b) = bias

To simplify computation, bias is merged into the weight vector.

$$
X' = [X \ \ 1]
$$

$$
\theta =
\begin{bmatrix}
w \
b
\end{bmatrix}
$$

Final model

$$
\hat{y} = X'\theta
$$

---

# 2. Feature Normalization

To stabilize training, features are standardized.

$$
X_{norm} = \frac{X - \mu_X}{\sigma_X}
$$

Where

* ( \mu_X ) = feature mean
* ( \sigma_X ) = feature standard deviation

Target normalization:

$$
y_{norm} = \frac{y - \mu_y}{\sigma_y}
$$

---

# 3. Mean Squared Error

Prediction error

$$
e = \hat{y} - y
$$

Mean Squared Error (MSE)

$$
MSE = \frac{1}{n} \sum_{i=1}^{n}(y_i - \hat{y}_i)^2
$$

Vector form

$$
MSE = \frac{1}{n} ||X'\theta - y||^2
$$

---

# 4. ElasticNet Regularization

ElasticNet combines **L1 (Lasso)** and **L2 (Ridge)** penalties.

### L1 penalty

$$
L1 = \sum_{j=1}^{d} |w_j|
$$

### L2 penalty

$$
L2 = \sum_{j=1}^{d} w_j^2
$$

---

# 5. ElasticNet Loss Function

Total objective

$$
J(\theta) =
\frac{1}{n} ||X'\theta - y||^2
+
\lambda
(
\alpha \sum |w|
+
(1-\alpha) \sum w^2
)
$$

Where

* ( \lambda ) → regularization strength
* ( \alpha ) → L1/L2 mixing ratio

Special cases

* ( \alpha = 1 ) → Lasso
* ( \alpha = 0 ) → Ridge

---

# 6. Gradient of MSE

$$
\nabla_{\theta} =
\frac{2}{n} X'^T (X'\theta - y)
$$

---

# 7. Gradient of Regularization

L1 gradient

$$
\nabla L1 = sign(w)
$$

L2 gradient

$$
\nabla L2 = 2w
$$

---

# 8. Final Gradient

For weights

$$
\nabla_w =
\frac{2}{n} X'^T (X'\theta - y)
+
\lambda
(
\alpha \cdot sign(w)
+
2(1-\alpha)w
)
$$

Bias is **not regularized**.

---

# 9. Gradient Descent Update

Weights are updated using

$$
\theta = \theta - \eta \nabla_{\theta}
$$

Where

* ( \eta ) = learning rate

---

# 10. Mini-Batch Gradient Descent

Instead of using all data, batches are used.

For batch (B)

$$
\nabla =
\frac{2}{|B|}
X_B^T (X_B\theta - y_B)
$$

Parameters update after each batch.

Advantages

* faster training
* less memory
* smoother optimization

---

# 11. Prediction

For a new input (x)

Normalize

$$
x_{norm} = \frac{x-\mu_X}{\sigma_X}
$$

Add bias

$$
x' = [x_{norm},1]
$$

Prediction

$$
\hat{y}_{norm} = x'\theta
$$

Convert back to original scale

$$
\hat{y} = \hat{y}_{norm}\sigma_y + \mu_y
$$

---

# 12. R² Score

Model performance metric

$$
R^2 =
1 -
\frac{\sum (y - \hat{y})^2}
{\sum (y - \bar{y})^2}
$$

Where

* residual sum of squares
* total variance

Interpretation

| R² | Meaning                 |
| -- | ----------------------- |
| 1  | perfect model           |
| 0  | same as predicting mean |
| <0 | worse than mean         |

---

# Final Objective

Training minimizes

$$
\min_{\theta}
\left[
\frac{1}{n} ||X'\theta - y||^2
+
\lambda
(
\alpha ||w||_1
+
(1-\alpha)||w||_2^2
)
\right]
$$
