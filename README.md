# 🌍 Landslide Susceptibility Prediction System

A **machine learning–powered landslide prediction system** that analyzes **topographic and geological features** to estimate the probability of **landslide occurrence**.
This project uses **stacked ensemble learning**, **advanced feature engineering**, and **threshold optimization**, paired with an interactive **Gradio UI** for real-time predictions.

---

## 📌 Features

* 🧠 **Stacked Ensemble Model** — XGBoost + LightGBM + Random Forest with Logistic Regression as meta-learner
* ⚡ Automated hyperparameter tuning using **Optuna**
* 🧮 Advanced **feature engineering** from 25-point terrain attributes
* 🧪 SMOTE-ENN balancing to handle imbalanced datasets
* 📊 Auto-generated training visualizations (ROC, PR curve, Confusion Matrix, Probability Distribution)
* 🌐 **Gradio Web UI** for interactive prediction

---

## 🏗️ Project Structure

```
.
├── app2.py                                # Gradio web app
├── train2.py                              # Model training & hyperparameter tuning
├── ultimate_optimized_model.joblib        # Trained model
│
├── Data Set/
│   └── Train.csv                          # Training dataset
│
├── Training Output/
│   ├── ultimate_precision_recall_curve.png
│   ├── ultimate_probability_distribution.png
│   ├── ultimate_roc_curve.png
│   └── ultimate_stacked_confusion_matrix.png
│
└── README.md
```

---

## ⚙️ Installation

```bash
git clone https://github.com/Monish21072004/Land-Slide-Detection-.git
cd Land-Slide-Detection-
```

```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows
```

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install pandas numpy scikit-learn xgboost lightgbm imbalanced-learn optuna matplotlib seaborn joblib gradio plotly kaleido
```

---

## 🧠 Model Training

To train and optimize the model:

```bash
python train2.py
```

✅ This will:

* Load and process `Train.csv`
* Run **feature engineering**
* Tune hyperparameters with **Optuna**
* Train a **stacked ensemble model**
* Save the trained model as `ultimate_optimized_model.joblib`
* Generate training evaluation plots in `Training Output/`.

---

## 📊 Training Output Visualizations

### 📈 Precision–Recall Curve

![Precision Recall Curve](https://github.com/Monish21072004/Land-Slide-Detection-/blob/main/Training%20Output/ultimate_precision_recall_curve.png?raw=true)

---

### 📊 Probability Distribution

![Probability Distribution](https://github.com/Monish21072004/Land-Slide-Detection-/blob/main/Training%20Output/ultimate_probability_distribution.png?raw=true)

---

### 🧭 ROC Curve

![ROC Curve](https://github.com/Monish21072004/Land-Slide-Detection-/blob/main/Training%20Output/ultimate_roc_curve.png?raw=true)

---

### 🧮 Confusion Matrix

![Confusion Matrix](https://github.com/Monish21072004/Land-Slide-Detection-/blob/main/Training%20Output/ultimate_stacked_confusion_matrix.png?raw=true)

---

## 🧾 Dataset Preview

The dataset is located in the
👉 [`/Data Set`](https://github.com/Monish21072004/Land-Slide-Detection-/tree/main/Data%20Set) folder.

Below is a **sample preview of the `Train.csv`** file structure:

| Sample_ID | 1_elevation | 2_elevation | ... | 25_slope | 1_twi | ... | Label |
| --------- | ----------: | ----------: | --- | -------: | ----: | --- | ----: |
| 1001      |      215.32 |      218.10 | ... |    12.53 |  4.22 | ... |     1 |
| 1002      |      194.88 |      199.67 | ... |     8.11 |  3.78 | ... |     0 |
| 1003      |      223.41 |      226.02 | ... |    11.92 |  4.11 | ... |     1 |
| 1004      |      202.64 |      205.19 | ... |     7.80 |  3.65 | ... |     0 |

📌 **Columns:**

* `Sample_ID` — Unique identifier for each sample
* `1_elevation` ... `25_elevation` — Elevation features (25 points)
* `1_slope` ... `25_slope` — Slope values (25 points)
* Additional terrain features: `aspect`, `placurv`, `procurv`, `lsfactor`, `twi`, `geology`, `sdoif` (each with 25 points)
* `Label` — 1 = Landslide, 0 = Not Landslide (used only for training)

✅ For prediction, upload the same format **without the `Label` column**.

---

## 🧮 Example Output

```json
{
  "Sample_1001": {
    "Prediction": "Landslide",
    "Landslide_Probability": "0.9821"
  },
  "Sample_1002": {
    "Prediction": "Not a Landslide",
    "Landslide_Probability": "0.0413"
  }
}
```

---

## 📊 Evaluation Metrics

* ✅ Accuracy
* 📈 ROC AUC Score
* 🧮 MCC (Matthews Correlation Coefficient)
* 📊 Confusion Matrix
* 📈 Precision–Recall Curve
* 🧭 Optimal Threshold based on F1-score

---

## 🌐 Running the Web App

Launch the interactive Gradio interface:

```bash
python app2.py
```

Then open: 👉 `http://127.0.0.1:7860`

With this interface, you can:

* 📤 Upload your own CSV
* 🧭 Adjust the decision threshold
* 🧪 Test a real landslide scenario

---

## 🧭 Future Enhancements

* [ ] Cloud deployment (AWS / HuggingFace Spaces)
* [ ] Real-time GIS visualization
* [ ] Batch processing for large geospatial regions
* [ ] REST API integration

---

## 🤝 Contributing

1. Fork this repository
2. Create a feature branch
3. Commit your changes
4. Submit a Pull Request 🚀

---

## 📜 License

This project is licensed under the **MIT License**.

---

## 👨‍💻 Author

**Monish V**
📧 [monishv217@gmail.com](mailto:monishv217@gmail.com)

⭐ If you find this project helpful, consider giving it a **star** on GitHub!

---

✅ All visualizations are pulled directly from [`/Training Output`](https://github.com/Monish21072004/Land-Slide-Detection-/tree/main/Training%20Output)
✅ Dataset is available at [`/Data Set`](https://github.com/Monish21072004/Land-Slide-Detection-/tree/main/Data%20Set)
