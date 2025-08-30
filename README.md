
# Credit Card Fraud Detection

This project builds machine learning models to detect fraudulent credit card transactions using the popular **creditcard.csv** dataset. The dataset is highly imbalanced, so techniques like **SMOTE oversampling** are applied to rebalance the classes before training.

---

## 📂 Project Structure

```
├── creditcard.csv        # Dataset (not included here, download separately)
├── fraud_detection.py    # Main Python script (your code)
├── README.md             # Project documentation
```

---

## 📊 Dataset

* **Source**: [Kaggle - Credit Card Fraud Detection Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud)
* **Description**:

  * `Time`: Seconds elapsed between this transaction and the first transaction in the dataset.
  * `Amount`: Transaction amount.
  * `V1 ... V28`: PCA-transformed features to protect confidentiality.
  * `Class`: Target label (0 → normal, 1 → fraud).

The dataset is highly imbalanced with **fraudulent transactions < 0.2%** of the data.

---

## ⚙️ Preprocessing

1. **Scaling**: StandardScaler is applied to `Time` and `Amount`.
2. **Train-Test Split**: Stratified 80/20 split to preserve class distribution.
3. **Rebalancing**:

   * Used **SMOTE (Synthetic Minority Oversampling Technique)** to generate synthetic samples for the minority class (fraud).

---

## 🤖 Models Used

1. **Logistic Regression**

   * Class weight balanced.
   * `max_iter=1000`.

2. **XGBoost Classifier**

   * 200 estimators.
   * Max depth = 5.
   * Learning rate = 0.1.
   * Subsample & column sampling = 0.8.
   * `scale_pos_weight=1` (since SMOTE balances classes).

---

## 📈 Evaluation Metrics

* **Classification Report**: Precision, Recall, F1-score for fraud and non-fraud classes.
* **ROC-AUC Score**: Evaluates discrimination ability of models.
* **Confusion Matrix**: Heatmap for visualizing classification performance.

---

## 🔎 Feature Importance (XGBoost Only)

The project includes **feature importance analysis** using XGBoost’s feature importances.
The results are plotted as a bar chart to highlight the most important predictors of fraud.

---

## 🚀 How to Run

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/credit-card-fraud-detection.git
   cd credit-card-fraud-detection
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Add the dataset `creditcard.csv` in the project directory.

4. Run the script:

   ```bash
   python fraud_detection.py
   ```

---

## 📦 Requirements

* Python 3.7+
* pandas
* numpy
* matplotlib
* seaborn
* scikit-learn
* imbalanced-learn
* xgboost
* tqdm

Install all with:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn xgboost tqdm
```

---

## 📊 Results (Example)

* Logistic Regression:

  * ROC-AUC ≈ 0.96
* XGBoost:

  * ROC-AUC ≈ 0.99

*(Exact results may vary depending on random seed and dataset version.)*

---

## 📌 Future Work

* Hyperparameter tuning with GridSearch/RandomSearch.
* Trying other algorithms (Random Forest, LightGBM, Neural Networks).
* Cost-sensitive learning instead of oversampling.
