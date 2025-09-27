import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import RandomOverSampler
import joblib
import matplotlib.pyplot as plt
import os
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from sklearn.preprocessing import LabelEncoder, StandardScaler
import seaborn as sns

# === 1. Load Dataset ===
print("Loading dataset...")
df = pd.read_csv("Crop Recommendation using Soil Properties and Weather Prediction.csv")
# 2. Kenya soil analysis (with county/sub-county columns)
soil_df = pd.read_csv("kenya_soil_crop_prediction_dataset.csv")

# Quick look
print(crop_df.head())
print(soil_df.head())
# Preserve Soilcolor columns before encoding
soilcolor_col = df["Soilcolor"].unique().tolist()

# One-hot encode categorical
print("Preprocessing data...")
df_encoded = pd.get_dummies(df, columns=["Soilcolor"])
joblib.dump(soilcolor_col, "models/soilcolor_col.pkl")

# Encode label
le = LabelEncoder()
y = le.fit_transform(df["label"])
X = df_encoded.drop("label", axis=1)

# Normalize numeric features
print("Scaling numeric features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, "models/scaler.pkl")

# === 2. Handle Class Imbalance ===
print("Handling class imbalance...")
ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X_scaled, y)

# === 3. Train-Test Split ===
print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, stratify=y_resampled, random_state=42
)
# Create models directory if it doesn't exist
os.makedirs('models', exist_ok=True)

# === 1. Load Dataset ===
print("Loading dataset...")
df = pd.read_csv("Crop Recommendation using Soil Properties and Weather Prediction.csv")

# One-hot encode categorical
print("Preprocessing data...")
df_encoded = pd.get_dummies(df, columns=["Soilcolor"])

# Features & Target
le = LabelEncoder()
y = le.fit_transform(df["label"])
X = df_encoded.drop("label", axis=1)

# === 2. Handle Class Imbalance ===
print("Handling class imbalance...")
ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X, y)

# === 3. Train-Test Split ===
print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, stratify=y_resampled, random_state=42
)

# === 4. Train Random Forest & XGBoost with tuned params ===
print("Training models...")
rf_params = {
    'n_estimators': 200,
    'max_depth': 20,
    'min_samples_split': 2,
    'min_samples_leaf': 1,
    'max_features': 'sqrt',
    'n_jobs': -1,
    'random_state': 42
}

xgb_params = {
    'n_estimators': 200,
    'max_depth': 10,
    'learning_rate': 0.05,
    'subsample':.9,
    'colsample_bytree': .8,
    'eval_metric': 'mlogloss',
    'random_state': 42
}

rf_model = RandomForestClassifier(**rf_params)
xgb_model = XGBClassifier(**xgb_params)

rf_model.fit(X_train, y_train)
xgb_model.fit(X_train, y_train)

# === 5. Evaluate Both Models ===
print("Evaluating models...")
rf_pred = rf_model.predict(X_test)
xgb_pred = xgb_model.predict(X_test)

rf_acc = accuracy_score(y_test, rf_pred)
xgb_acc = accuracy_score(y_test, xgb_pred)

rf_f1 = f1_score(y_test, rf_pred, average='macro')
xgb_f1 = f1_score(y_test, xgb_pred, average='macro')

print(f"✅ RF Accuracy: {rf_acc*100:.2f}% | Macro F1: {rf_f1:.4f}")
print(f"✅ XGB Accuracy: {xgb_acc*100:.2f}% | Macro F1: {xgb_f1:.4f}")

if xgb_f1 > rf_f1:
    final_model = xgb_model
    final_name = "xgb"
    final_acc = xgb_acc
    final_pred = xgb_pred
else:
    final_model = rf_model
    final_name = "rf"
    final_acc = rf_acc
    final_pred = rf_pred

cm = confusion_matrix(y_test, final_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
disp.plot(xticks_rotation=90)
plt.title("🌾 Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.close()

# Correlation Heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(pd.DataFrame(X_resampled, columns=X.columns).corr(), cmap='coolwarm')
plt.title("📊 Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig("correlation_heatmap.png")
plt.close()

# Feature Importances
# (already in your script)

# === 8. Save Artifacts ===
print("Saving models and artifacts...")
joblib.dump(final_model, f"models/{final_name}_crop_model.pkl")
joblib.dump(le, "models/label_encoder.pkl")
joblib.dump(list(X.columns), "models/feature_names.pkl")
print("📦 Model, scaler, label encoder, features, and visual assets saved.")
# === 6. Print Summary ===
print("\n📋 Classification Report:")
print(classification_report(y_test, final_pred, target_names=le.classes_))

# === 7. Plot Feature Importances ===
importances = final_model.feature_importances_
features = list(X.columns)
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
plt.title("🌾 Feature Importances")
plt.bar(range(len(importances)), importances[indices], align="center")
plt.xticks(range(len(importances)), [features[i] for i in indices], rotation=90)
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.close()

# === 8. Save Artifacts ===
print("Saving models and artifacts...")
joblib.dump(final_model, f"models/{final_name}_crop_model.pkl")
joblib.dump(le, "models/label_encoder.pkl")
joblib.dump(list(X.columns), "models/feature_names.pkl")

print(f"\n✅ Best Model: {final_name.upper()} with {final_acc*100:.2f}% accuracy")
print("📦 Model, label encoder, and feature list saved.")
print("✅ Model training and evaluation completed successfully.")   