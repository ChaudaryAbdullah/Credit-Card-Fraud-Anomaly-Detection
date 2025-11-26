# 🔱 Time Series Anomaly Detection (TS-AD) using a Contamination-Robust Transformer-GAN Framework

This project presents a robust framework for anomaly detection in multivariate time series data, specifically designed to mitigate the challenges posed by **training data contamination** (where anomalies are mixed with normal data). The core architecture combines a **Transformer Autoencoder** with **Contrastive Learning** and a **Generative Adversarial Network (GAN)** for enhanced representation learning and generalization.

## 📝 Implementation Details

- **File:** `experiment1.ipynb`
- **Dataset Source:** Local CSV file, expected at ./data/creditcard.csv`.
- **Core Libraries:** PyTorch, Pandas, NumPy, Scikit-learn.

---

## 📊 1. Dataset and Preprocessing

### Dataset Used

The model is trained and evaluated on the **Kaggle Credit Card Fraud Detection Dataset**. This dataset is suitable for TS-AD because it represents sequential transactional data (ordered by `Time`) and features a highly imbalanced classification task (`Class` label).

### Preprocessing Steps

1.  **Feature Standardization:** All sensitive features (V1-V28, Amount) are scaled using `StandardScaler` to ensure features contribute equally to the distance metrics and aid stable training.
2.  **Time Series Windowing:** The core requirement of TS-AD is met by applying a **sliding window technique** to convert the sequential data into fixed-length sequence samples.
    - **Window Size:** $W = 10$ time steps.
    - **Step Size:** $S = 1$.
3.  **Label Aggregation:** A sequence window is labeled as **anomalous (1)** if **any** transaction within that 10-step window is fraudulent; otherwise, it is labeled **normal (0)**.
4.  **Train/Test Split:** The dataset is split (80/20) and training is performed **exclusively** on the **Normal windows** ($y_{\text{train}} = 0$) to learn the uncontaminated distribution.

---

## 🧠 2. Model Architecture and Components

The framework is built around three jointly trained components:

### A. Transformer Autoencoder (AE)

- **Role:** Feature extraction and sequence reconstruction.
- **Components:**
  - **Encoder:** Maps the input sequence (10 timesteps $\times$ 29 features) to a compact **latent representation** ($z \in \mathbb{R}^{64}$). It uses a `TransformerEncoder` preceded by **Positional Encoding** to capture temporal dependencies.
  - **Decoder:** Reconstructs the original input sequence from the latent vector $z$.

### B. Geometric Masking (Augmentation)

This technique is applied to generate two corrupted views (`view1`, `view2`) of the same input batch, crucial for the contrastive learning stage.

- **Time Masking:** Randomly zeroes out entire time steps (rows) with a probability of 10%.
- **Feature Masking:** Randomly zeroes out specific features across all time steps (columns) with a probability of 10%.
- **Noise Injection:** Adds small Gaussian noise ($\sigma=0.01$) to the input, enhancing robustness (reducing overfitting).

### C. Joint Contamination-Robust Training (AE-GAN-Contrastive)

This training loop combines three specific loss terms:

| Component             | Loss Function                                      | Purpose                                                                                                                                                                                                                                                                                                                                                 |
| :-------------------- | :------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Reconstruction**    | **MSE Loss** ($\mathcal{L}_{\text{recon}}$)        | Forces the AE to accurately reconstruct the normal input sequence.                                                                                                                                                                                                                                                                                      |
| **Contrastive**       | **NT-Xent Loss** ($\mathcal{L}_{\text{contrast}}$) | Pulls the latent representations of augmented views ($z_1, z_2$) closer together, forcing the encoder to learn robust, invariant features of the normal distribution.                                                                                                                                                                                   |
| **Adversarial (GAN)** | **Wasserstein Loss** ($\mathcal{L}_{\text{adv}}$)  | Trains the **Generator** (`gen`) to synthesize realistic normal latent vectors and the **Encoder** to ensure its real latent vectors ($z_{\text{orig}}$) are indistinguishable from the synthetic ones, effectively regularizing the latent space to a known, simple distribution (noise from `gen`). This mitigates the impact of minor contamination. |

---

## ⚙️ 3. Training Procedure

The model is trained jointly using an adversarial approach, optimizing three separate modules:

- **AE Optimizer:** Updates the Autoencoder (Encoder + Decoder).
- **Disc Optimizer:** Updates the Discriminator.
- **Gen Optimizer:** Updates the Generator.

### Key Training Parameters

| Parameter                           | Value                                                                                                  | Rationale                                                                                                                                                                        |
| :---------------------------------- | :----------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Epochs**                          | 20                                                                                                     | Sufficient for convergence.                                                                                                                                                      |
| **Learning Rate**                   | $5 \times 10^{-4}$                                                                                     | Reduced for stable GAN convergence.                                                                                                                                              |
| **GAN Ratio** ($N_{\text{critic}}$) | 5                                                                                                      | Discriminator is updated 5 times more frequently than the Generator to ensure convergence balance (WGAN setup).                                                                  |
| **Gradient Clipping**               | $\text{max\_norm}=1.0$                                                                                 | Applied to all parameters to prevent exploding gradients and stabilize the Transformer's training.                                                                               |
| **WGAN Clipping**                   | $\text{clip\_value}=0.01$                                                                              | Applied to the Discriminator's weights for a stable Wasserstein GAN formulation.                                                                                                 |
| **Loss Weights**                    | $\mathcal{L}_{\text{recon}}=1.0$, $\mathcal{L}_{\text{contrast}}=0.3$, $\mathcal{L}_{\text{adv}}=0.01$ | $\mathcal{L}_{\text{recon}}$ dominates, $\mathcal{L}_{\text{contrast}}$ provides representation stability, and $\mathcal{L}_{\text{adv}}$ is kept small for mild regularization. |

---

## 📈 4. Evaluation and Results

### Anomaly Scoring

The final anomaly score for a test window is a composite measure, emphasizing both the inability to reconstruct the input and the deviation from the normal latent distribution:

$$
\text{Score} = \underbrace{\frac{1}{T \cdot F} \sum_{t,f} (x_{t,f} - \hat{x}_{t,f})^2}_{\text{Reconstruction Error}} + \underbrace{0.001 \times (z - \mu)^\top \Sigma^{-1} (z - \mu)}_{\text{Mahalanobis Distance}}
$$

Where:

- $\hat{x}$ is the reconstructed sequence, and $x$ is the original.
- $z$ is the latent vector.
- $\mu$ and $\Sigma^{-1}$ are the mean and inverse covariance of the latent space, calculated using the normal training data.

### Evaluation Metrics

The performance is measured using standard metrics for imbalanced classification:

| Metric      | Code Value | Interpretation                                                                                              |
| :---------- | :--------- | :---------------------------------------------------------------------------------------------------------- |
| **ROC AUC** | **0.9575** | The model's ability to rank anomalies above normal instances across all thresholds.                         |
| **PR AUC**  | **0.7841** | Measures the trade-off between Precision and Recall, crucial for highly imbalanced anomaly detection tasks. |

> **Eval ROC AUC: 0.9575, PR AUC: 0.7841**

### Effectiveness Summary

The results demonstrate **strong effectiveness** in anomaly detection:

- **High ROC AUC ($\approx 0.96$)** indicates excellent separability between normal and anomalous windows.
- The **combined framework** proves successful in achieving **robust generalization** by:
  - **Transformer:** Effectively learning complex temporal dependencies.
  - **Contrastive Learning:** Ensuring the learned latent space is robust to minor data perturbations.
  - **GAN (WGAN):** Regularizing the latent space to prevent the encoder from mapping potential in-lier anomalies to the 'normal' center, thereby enhancing the sensitivity of the **Mahalanobis Distance** score.
