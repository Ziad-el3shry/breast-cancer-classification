# 🧬 Breast Cancer Classification

An interactive **machine learning web app** for predicting whether a breast tumor is **Benign** or **Malignant**. Built using **Streamlit**, **Scikit-learn**, and **Plotly**, this tool allows real-time classification with model comparison and feature insights.

---

## 🔗 GitHub Repository

📂 [https://github.com/Ziad-el3shry/breast-cancer-classification](https://github.com/Ziad-el3shry/breast-cancer-classification)

---

## 🧠 Project Highlights

- 🧪 Compare **Logistic Regression**, **Random Forest**, and **SVM** classifiers
- 📉 Visualize results using **interactive confusion matrix**
- 🧾 Manually input tumor data for live prediction
- 🔍 View **top feature importances** for Random Forest
- 📊 Built-in preprocessing, model training & evaluation

---

## 📁 Project Structure

```
breast-cancer-classification/
│
├── Data/
│   └── data.csv
│
├── breast_cancer_app.py               # Main Streamlit application
├── Breast Cancer Classification.ipynb # Jupyter Notebook (EDA & model dev)
├── requirements.txt                   # Python dependencies
└── README.md                          # Project documentation
```

---

## 📘 Notebook: `Breast Cancer Classification.ipynb`

The notebook includes:

- ✅ Exploratory Data Analysis (EDA)
- ✅ Correlation heatmaps
- ✅ Model training and evaluation
- ✅ Feature importance exploration
- ✅ Accuracy, precision, recall metrics

**Note:** The notebook is intended for analysis, experimentation, and insight — not deployment.

---

## ▶️ Run the Streamlit App Locally

### 1. Clone the repo

```bash
git clone https://github.com/Ziad-el3shry/breast-cancer-classification.git
cd breast-cancer-classification
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Launch the app

```bash
streamlit run breast_cancer_app.py
```

---

## 🧪 Dataset Info

- 📦 Source: [`sklearn.datasets.load_breast_cancer`](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html)
- 🎯 Target: Diagnosis (`0 = Malignant`, `1 = Benign`)
- 🧬 Features: 30 numeric features (e.g., radius, texture, smoothness)

---

## 🧾 Example Features for Prediction

```text
mean radius: 17.99
mean texture: 10.38
mean perimeter: 122.8
mean area: 1001.0
mean smoothness: 0.1184
...
```

You can input these values manually in the Streamlit app to see predictions in real time.

---

## 📊 Sample Model Output (Confusion Matrix)

| Actual / Predicted | Benign | Malignant |
|--------------------|--------|-----------|
| **Benign**         | 108    | 2         |
| **Malignant**      | 3      | 59        |

---

## 📌 Requirements

```
streamlit
pandas
numpy
scikit-learn
plotly
```

Install with:

```bash
pip install -r requirements.txt
```

---

## 🌱 Future Enhancements

- Add ROC & Precision-Recall curves
- Support exporting results as CSV
- Enable dataset upload via UI
- Improve UI styling and feedback

---

## 👤 Author

**Ziad Attia**  
AI Engineer & Machine Learning Enthusiast  
🌐 [GitHub](https://github.com/Ziad-el3shry)
