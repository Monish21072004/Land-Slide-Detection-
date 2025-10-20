import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             roc_auc_score, roc_curve, precision_recall_curve, auc,
                             matthews_corrcoef, cohen_kappa_score, f1_score)
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from imblearn.combine import SMOTEENN
from imblearn.pipeline import Pipeline as ImbPipeline
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import optuna
import warnings

# Suppress warnings for a cleaner output
warnings.filterwarnings('ignore', category=UserWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)


# --- 1. Advanced Feature Engineering ---
def create_features(df):
    """
    Engineers new summary features from the raw 25-point data.
    """
    print("Starting feature engineering...")
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
    print("Feature engineering completed.")
    return df_features


# --- 2. Load Data and Create Features ---
try:
    file_path = r"C:\Users\Monish V\OneDrive\Documents\RANDOM_PROJECTS\GIS\MY Gis 2\Train.csv"
    df = pd.read_csv(file_path)

    df_engineered = create_features(df)

    X = df_engineered.drop(['Sample_ID', 'Label'], axis=1)
    y = df_engineered['Label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print("Data loaded and split successfully.")


    # --- 3. Optuna Objective Function for Hyperparameter Search ---
    def objective(trial):
        xgb_params = {
            'n_estimators': trial.suggest_int('xgb_n_estimators', 200, 600),
            'max_depth': trial.suggest_int('xgb_max_depth', 4, 8),
            'learning_rate': trial.suggest_float('xgb_learning_rate', 0.01, 0.1),
            'subsample': trial.suggest_float('xgb_subsample', 0.7, 1.0),
            'colsample_bytree': trial.suggest_float('xgb_colsample_bytree', 0.7, 1.0),
            'gamma': trial.suggest_float('xgb_gamma', 0.1, 0.5),
        }
        rf_params = {
            'n_estimators': trial.suggest_int('rf_n_estimators', 200, 600),
            'max_depth': trial.suggest_int('rf_max_depth', 10, 25),
            'min_samples_split': trial.suggest_int('rf_min_samples_split', 2, 8),
            'min_samples_leaf': trial.suggest_int('rf_min_samples_leaf', 1, 4),
        }
        lgbm_params = {
            'n_estimators': trial.suggest_int('lgbm_n_estimators', 200, 600),
            'learning_rate': trial.suggest_float('lgbm_learning_rate', 0.01, 0.1),
            'num_leaves': trial.suggest_int('lgbm_num_leaves', 20, 40),
        }

        estimators = [
            ('xgb', xgb.XGBClassifier(**xgb_params, objective='binary:logistic', eval_metric='logloss', random_state=42,
                                      use_label_encoder=False)),
            ('rf', RandomForestClassifier(**rf_params, random_state=42, class_weight='balanced', n_jobs=-1)),
            ('lgbm',
             lgb.LGBMClassifier(**lgbm_params, random_state=42, class_weight='balanced', n_jobs=-1, verbosity=-1))
        ]
        final_estimator = LogisticRegression(solver='liblinear')
        stacking_model = StackingClassifier(estimators=estimators, final_estimator=final_estimator, cv=3,
                                            passthrough=True, n_jobs=-1)

        pipeline = ImbPipeline([
            ('scaler', StandardScaler()),
            ('transformer', QuantileTransformer(output_distribution='normal', random_state=42)),
            ('resampler', SMOTEENN(random_state=42)),
            ('stacking_classifier', stacking_model)
        ])

        pipeline.fit(X_train, y_train)
        y_pred_proba = pipeline.predict_proba(X_test)[:, 1]

        precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
        f1_scores = 2 * (precision * recall) / (precision + recall)
        return np.nanmax(f1_scores)


    # --- 4. Run the Optimization ---
    print("\nStarting final, targeted hyperparameter optimization with Optuna...")
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=25)
    print("Optimization completed.")
    print(f"Best F1-score found during search: {study.best_value:.4f}")
    print("Best parameters found:")
    print(study.best_params)

    # --- 5. Generate Optuna Visualizations ---
    print("\nGenerating Optuna visualizations...")
    try:
        fig = optuna.visualization.plot_optimization_history(study)
        fig.write_image("optuna_optimization_history.png")
        print("Generated optuna_optimization_history.png")

        fig = optuna.visualization.plot_param_importances(study)
        fig.write_image("optuna_param_importances.png")
        print("Generated optuna_param_importances.png")
    except ImportError:
        print("\nCould not generate Optuna plots. Please install kaleido: 'pip install --upgrade kaleido'")
    except Exception as e:
        print(f"An error occurred during Optuna plot generation: {e}")

    # --- 6. Train the Final Model with the Best Parameters ---
    print("\nTraining the final model with the absolute best parameters...")
    best_params = study.best_params

    final_xgb_params = {k.replace('xgb_', ''): v for k, v in best_params.items() if k.startswith('xgb_')}
    final_rf_params = {k.replace('rf_', ''): v for k, v in best_params.items() if k.startswith('rf_')}
    final_lgbm_params = {k.replace('lgbm_', ''): v for k, v in best_params.items() if k.startswith('lgbm_')}

    final_estimators = [
        ('xgb',
         xgb.XGBClassifier(**final_xgb_params, objective='binary:logistic', eval_metric='logloss', random_state=42,
                           use_label_encoder=False)),
        ('rf', RandomForestClassifier(**final_rf_params, random_state=42, class_weight='balanced', n_jobs=-1)),
        ('lgbm',
         lgb.LGBMClassifier(**final_lgbm_params, random_state=42, class_weight='balanced', n_jobs=-1, verbosity=-1))
    ]
    final_final_estimator = LogisticRegression(solver='liblinear')
    final_stacking_model = StackingClassifier(estimators=final_estimators, final_estimator=final_final_estimator, cv=5,
                                              passthrough=True, n_jobs=-1)

    final_pipeline = ImbPipeline([
        ('scaler', StandardScaler()),
        ('transformer', QuantileTransformer(output_distribution='normal', random_state=42)),
        ('resampler', SMOTEENN(random_state=42)),
        ('stacking_classifier', final_stacking_model)
    ])

    final_pipeline.fit(X_train, y_train)
    print("Final model training completed.")

    # --- 7. Save and Evaluate the Final, Optimized Model ---
    model_filename = 'ultimate_optimized_model.joblib'
    joblib.dump(final_pipeline, model_filename)
    print(f"\nUltimate optimized model saved successfully as '{model_filename}'")

    y_pred_proba = final_pipeline.predict_proba(X_test)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
    f1_scores = 2 * (precision * recall) / (precision + recall)
    optimal_idx = np.nanargmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]
    y_pred_optimal = (y_pred_proba >= optimal_threshold).astype(int)

    print("\n--- Final Evaluation with Optimal Threshold ---")
    print(f"Optimal Threshold found: {optimal_threshold:.4f}")
    final_accuracy = accuracy_score(y_test, y_pred_optimal)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    mcc = matthews_corrcoef(y_test, y_pred_optimal)
    print(f"Final Accuracy: {final_accuracy:.4f}")
    print(f"ROC AUC Score: {roc_auc:.4f}")
    print(f"MCC Score: {mcc:.4f}")
    print(classification_report(y_test, y_pred_optimal))

    # --- 8. Generate Final Model Visualizations ---
    print("\nGenerating final model visualizations...")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred_optimal)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='viridis',
                xticklabels=['Predicted 0', 'Predicted 1'],
                yticklabels=['Actual 0', 'Actual 1'])
    plt.title('Confusion Matrix for Optimized Model')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.savefig('ultimate_stacked_confusion_matrix.png')
    plt.close()
    print("Generated ultimate_stacked_confusion_matrix.png")

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig('ultimate_roc_curve.png')
    plt.close()
    print("Generated ultimate_roc_curve.png")

    # Precision-Recall Curve
    pr_auc = auc(recall, precision)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (area = {pr_auc:.2f})')
    plt.scatter(recall[optimal_idx], precision[optimal_idx], marker='o', color='red',
                label=f'Optimal Threshold ({optimal_threshold:.2f})', zorder=5)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.savefig('ultimate_precision_recall_curve.png')
    plt.close()
    print("Generated ultimate_precision_recall_curve.png")

except FileNotFoundError:
    print(f"Error: The file was not found at the path: {file_path}")
    print("Please make sure the file path is correct and the file exists.")
except ImportError as e:
    print(f"An error occurred. One or more required libraries are not installed: {e}")
    print(
        "Please run 'pip install pandas numpy xgboost lightgbm scikit-learn imbalanced-learn matplotlib seaborn joblib optuna plotly kaleido'")
except Exception as e:
    print(f"An error occurred: {e}")
