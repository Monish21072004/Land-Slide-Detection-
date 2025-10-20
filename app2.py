import pandas as pd
import numpy as np
import gradio as gr
import joblib
import warnings

# We need to import all the custom classes and functions used in the pipeline
# even if they are not directly called, so that joblib can unpickle the model.
from sklearn.ensemble import StackingClassifier
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from imblearn.combine import SMOTEENN
from imblearn.pipeline import Pipeline as ImbPipeline
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings('ignore', category=UserWarning)

# --- 1. Load the Saved Model Pipeline ---
MODEL_PATH = 'ultimate_optimized_model.joblib'
print("Loading the trained model pipeline...")
try:
    pipeline = joblib.load(MODEL_PATH)
    print("Model loaded successfully.")
except FileNotFoundError:
    print(f"Error: Model file not found at '{MODEL_PATH}'. Make sure the model is in the same folder as this script.")
    pipeline = None
except Exception as e:
    print(f"An error occurred while loading the model: {e}")
    pipeline = None


# --- 2. Advanced Feature Engineering Function ---
def create_features(df):
    """
    Engineers new summary features from the raw 25-point data. This must be
    identical to the function used during training.
    """
    df_features = df.copy()
    prefixes = ['elevation', 'slope', 'aspect', 'placurv', 'procurv', 'lsfactor', 'twi', 'geology', 'sdoif']

    for prefix in prefixes:
        cols = [col for col in df.columns if col.startswith(f'1_{prefix}') or col[1:].startswith(f'_{prefix}')]
        if not cols: continue

        df_features[f'{prefix}_mean'] = df[cols].mean(axis=1)
        df_features[f'{prefix}_std'] = df[cols].std(axis=1)
        df_features[f'{prefix}_min'] = df[cols].min(axis=1)
        df_features[f'{prefix}_max'] = df[cols].max(axis=1)
        df_features[f'{prefix}_range'] = df_features[f'{prefix}_max'] - df_features[f'{prefix}_min']

    df_features.fillna(0, inplace=True)
    return df_features


# --- 3. Load and Store a Real Landslide Example ---
REAL_LANDSLIDE_RAW_DATA = None
TRAINING_COLUMNS = None
try:
    train_df_path = r"C:\Users\Monish V\OneDrive\Documents\RANDOM_PROJECTS\GIS\MY Gis 2\Train.csv"
    train_df = pd.read_csv(train_df_path)

    # Store the full raw data row for a real landslide
    real_landslide_df = train_df[train_df['Label'] == 1].head(1)
    if not real_landslide_df.empty:
        REAL_LANDSLIDE_RAW_DATA = real_landslide_df.drop(columns=['Label'])
        print("Successfully stored a real landslide data scenario.")

    # Get the correct column order from the full engineered training set
    train_df_engineered = create_features(train_df.drop(columns=['Label']))
    TRAINING_COLUMNS = train_df_engineered.drop(['Sample_ID'], axis=1).columns.tolist()
    print("Successfully captured the training column order.")

except Exception as e:
    print(f"Could not load or process training data. Error: {e}")


# --- 4. Define the Prediction Function ---
def predict_from_df(df, threshold):
    """
    A generic prediction function that takes a DataFrame and a threshold.
    """
    if pipeline is None: return "Error: Model not loaded.", ""
    if TRAINING_COLUMNS is None: return "Error: Could not get training columns.", ""

    try:
        # Store Sample_ID if it exists
        sample_ids = df['Sample_ID'] if 'Sample_ID' in df.columns else df.index

        # Apply the exact same feature engineering
        df_engineered = create_features(df)

        # Ensure the column order and names match the training data
        df_to_predict = df_engineered[TRAINING_COLUMNS]

        print(f"Making predictions on {len(df_to_predict)} sample(s)...")
        probabilities = pipeline.predict_proba(df_to_predict)[:, 1]
        predictions = (probabilities >= threshold).astype(int)
        print("Predictions generated.")

        # Format the output
        results = {}
        for sid, prob, pred in zip(sample_ids, probabilities, predictions):
            prediction_label = "Landslide" if pred == 1 else "Not a Landslide"
            results[f"Sample_{sid}"] = {
                "Prediction": prediction_label,
                "Landslide_Probability": f"{prob:.4f}"
            }
        return results

    except Exception as e:
        return f"An error occurred during prediction: {e}"


# --- 5. Gradio Handler Functions ---
def predict_from_file(input_csv, threshold):
    if input_csv is None:
        return "Please upload a CSV file."
    temp_file_path = input_csv.name
    df = pd.read_csv(temp_file_path)
    return predict_from_df(df, threshold)


def predict_real_scenario(threshold):
    if REAL_LANDSLIDE_RAW_DATA is None:
        return "Real landslide scenario not loaded. Check console for errors."
    return predict_from_df(REAL_LANDSLIDE_RAW_DATA, threshold)


# --- 6. Create and Launch the Gradio Interface ---
print("Creating Gradio interface...")

with gr.Blocks(theme=gr.themes.Soft()) as iface:
    gr.Markdown("# Landslide Prediction Model")
    gr.Markdown(
        "This application uses a highly trained model to predict the likelihood of landslides. You can either upload your own data in a CSV file or run a pre-loaded scenario from a confirmed landslide.")

    with gr.Tabs():
        with gr.TabItem("Predict from CSV File"):
            gr.Markdown(
                "Upload a CSV file with the same format as the original `Train.csv` (the 'Label' column is not required).")
            file_input = gr.File(label="Upload Test CSV")
            threshold_slider_1 = gr.Slider(minimum=0.0, maximum=1.0, value=0.94, step=0.01,
                                           label="Decision Threshold (Confidence Level)")
            file_predict_btn = gr.Button("Predict from File", variant="primary")
            file_output = gr.JSON(label="Prediction Results")

        with gr.TabItem("Run a Real Scenario"):
            gr.Markdown(
                "Click the button below to run a prediction on a pre-loaded data sample from a real, confirmed landslide. This is used to verify that the model is working correctly.")
            threshold_slider_2 = gr.Slider(minimum=0.0, maximum=1.0, value=0.94, step=0.01,
                                           label="Decision Threshold (Confidence Level)")
            scenario_predict_btn = gr.Button("Run Real Landslide Scenario", variant="primary")
            scenario_output = gr.JSON(label="Prediction Result")

    # Connect button events to functions
    file_predict_btn.click(
        fn=predict_from_file,
        inputs=[file_input, threshold_slider_1],
        outputs=file_output
    )

    scenario_predict_btn.click(
        fn=predict_real_scenario,
        inputs=[threshold_slider_2],
        outputs=scenario_output
    )

print("Launching Gradio app... Open the URL in your browser.")
iface.launch()
