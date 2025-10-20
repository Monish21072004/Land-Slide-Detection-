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
│   └── Train.csv                          # Dataset file
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

## 🌐 Running the Web App

To launch the interactive Gradio interface:

```bash
python app2.py
```

This will open the browser at something like:
👉 `http://127.0.0.1:7860`

With this interface, you can:

* 📤 Upload your own CSV files for prediction
* 🧭 Adjust the **decision threshold**
* 🧪 Run a **real landslide scenario**

---

## 🧾 Dataset Format

Your dataset (`Train.csv`) must include:

* `Sample_ID`
* 25 columns for each feature (e.g., `1_elevation` … `25_elevation`, `1_slope` … `25_slope`, etc.)
* `Label` (1 = Landslide, 0 = Not Landslide) → *used only during training*

✅ For predictions, the `Label` column should be removed.

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

## 🧭 Future Enhancements

* [ ] Cloud deployment (AWS / HuggingFace Spaces)
* [ ] Real-time GIS visualization
* [ ] Batch prediction for large datasets
* [ ] REST API integration

---

## 🤝 Contributing

1. Fork this repository
2. Create your feature branch
3. Commit your changes
4. Submit a Pull Request 🚀

---

## 📜 License

This project is licensed under the **MIT License**.

---

## 👨‍💻 Author

**Monish V**
📧 *[your email here]*
⭐ If you find this project helpful, consider giving it a **star** on GitHub!

---

✅ All image links above use the correct path from:
[`/Training Output`](https://github.com/Monish21072004/Land-Slide-Detection-/tree/main/Training%20Output)

