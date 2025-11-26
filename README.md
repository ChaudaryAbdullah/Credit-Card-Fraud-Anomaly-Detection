# 🔱 Time Series Anomaly Detection (TS-AD) using a Contamination-Robust Transformer-GAN Framework

This project presents a robust framework for anomaly detection in multivariate time series data, specifically designed to mitigate the challenges posed by **training data contamination** (where anomalies are mixed with normal data). The core architecture combines a **Transformer Autoencoder** with **Contrastive Learning** and a **Generative Adversarial Network (GAN)** for enhanced representation learning and generalization.

---

## 📝 Implementation Details

- **File:** `experiment3.ipynb`
- **Dataset Source:** Local CSV file, expected at `./data/creditcard.csv` (Path used: `/content/drive/MyDrive/Credit-Card-Fraud-Anomaly-Detection/data/creditcard.csv`).
- **Core Libraries:** PyTorch, Pandas, NumPy, Scikit-learn.

---

## 📊 1. Dataset and Preprocessing

### Dataset Used

The model is trained and evaluated on the **Kaggle Credit Card Fraud Detection Dataset**. This dataset is suitable for TS-AD because it represents sequential transactional data (ordered by `Time`) and features a highly imbalanced classification task (`Class` label).

### Preprocessing Steps

1. **Feature Standardization:** All sensitive features (V1-V28, Amount) are scaled using `StandardScaler` to ensure features contribute equally to the distance metrics and aid stable training.
2. **Time Series Windowing:** The core requirement of TS-AD is met by applying a **sliding window technique** to convert the sequential data into fixed-length sequence samples.
    - **Window Size:** $W = 10$ time steps.
    - **Step Size:** $S = 1$.
3. **Label Aggregation:** A sequence window is labeled as **anomalous (1)** if **any** transaction within that 10-step window is fraudulent; otherwise, it is labeled **normal (0)**.
4. **Train/Test Split:** The dataset is split (80/20) and training is performed on a **SMOTE-balanced training set** to leverage supervised learning for maximum discriminative power.

---

## 🧠 2. Model Architecture and Components

The framework is built around three jointly trained components:

### A. Transformer Autoencoder (AE)

- **Role:** Feature extraction and sequence reconstruction. A **Classification Head** is added to the AE, utilizing the latent vector $z$ for direct binary prediction.
- **Components:**
    - **Encoder:** Maps the input sequence (10 timesteps $\times$ 29 features) to a compact **latent representation** ($z \in \mathbb{R}^{64}$). It uses a `TransformerEncoder` preceded by **Positional Encoding** to capture temporal dependencies.
    - **Decoder:** Reconstructs the original input sequence from the latent vector $z$.

### B. Geometric Masking (Augmentation)

This technique is applied to generate two corrupted views (`view1`, `view2`) of the same input batch, crucial for the contrastive learning stage.

- **Time Masking:** Randomly zeroes out entire time steps (rows) with a probability of $10\%$.
- **Feature Masking:** Randomly zeroes out specific features across all time steps (columns) with a probability of $10\%$.
- **Noise Injection:** Adds small Gaussian noise ($\sigma=0.01$) to the input, enhancing robustness (reducing overfitting).

### C. Joint Contamination-Robust Training (AE-GAN-Contrastive)

This training loop combines four specific loss terms (including a new classification loss term, $\mathcal{L}_{\text{class}}$, used on the SMOTE-balanced data):

| Component | Loss Function | Purpose |
| :--- | :--- | :--- |
| **Reconstruction** | **MSE Loss** ($\mathcal{L}_{\text{recon}}$) | Forces the AE to accurately reconstruct the normal input sequence. |
| **Contrastive** | **NT-Xent Loss** ($\mathcal{L}_{\text{contrast}}$) | Pulls the latent representations of augmented views ($z_1, z_2$) closer together, forcing the encoder to learn robust, invariant features. |
| **Classification** | **BCEWithLogitsLoss** ($\mathcal{L}_{\text{class}}$) | Directly supervises the prediction on the SMOTE-balanced training data. |
| **Adversarial (GAN)** | **Wasserstein Loss** ($\mathcal{L}_{\text{adv}}$) | Regularizes the latent space by training the Encoder to fool the Discriminator, mitigating minor contamination. |

---

## ⚙️ 3. Training Procedure

The model is trained jointly using an adversarial approach, optimizing three separate modules (Autoencoder, Discriminator, and Generator).

### Key Training Parameters

| Parameter | Value |
| :--- | :--- |
| **Epochs** | 20 |
| **Learning Rate** | $5 \times 10^{-4}$ |
| **GAN Ratio** ($N_{\text{critic}}$) | 5 (Discriminator updated 5x more often) |
| **Loss Weights** | $\mathcal{L}_{\text{recon}}=0.3$, $\mathcal{L}_{\text{contrast}}=0.2$, $\mathcal{L}_{\text{class}}=1.0$, $\mathcal{L}_{\text{adv}}=0.01$ |

---

## 📈 4. Evaluation and Results

### Anomaly Scoring

The final **Anomaly Score** for a test window is the **raw logit output from the Classification Head**.

### Evaluation Metrics

The performance is measured on the highly imbalanced original test set (Anomaly ratio: $\approx 1.56\%$).

| Metric | Code Value | Interpretation |
| :--- | :--- | :--- |
| **ROC AUC** | **0.9993** | Excellent separability between normal and anomalous instances. |
| **PR AUC** | **0.9954** | Excellent measure of the Precision/Recall trade-off, critical for fraud detection. |
| **Best F1 Score** | **0.9859** | Achieved at the optimal threshold of $0.6700$. |

### Classification Report (at Optimal Threshold $0.6700$)

| Class | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| Normal | 1.00 | 1.00 | 1.00 | 56073 |
| Anomaly | 0.99 | 0.99 | 0.99 | 887 |

### Effectiveness Summary

The results demonstrate **outstanding effectiveness** in anomaly detection. The supervised training on SMOTE-balanced sequences, combined with the feature learning capabilities of the Transformer-GAN architecture, resulted in exceptional performance metrics for the credit card fraud detection task.
