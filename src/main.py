import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from xgboost import XGBClassifier

from src.data_preprocessing import load_and_clean_data
from src.feature_engineering import get_preprocessor, encode_target
from src.model_training import train_model
from src.evaluation import evaluate_model

RANDOM_STATE = 42

# -------------------- LOAD DATA --------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(BASE_DIR, "data", "raw", "synthetic_food_dataset_imbalanced.csv")

df = load_and_clean_data(data_path)

# -------------------- FEATURES & TARGET --------------------
X = df.drop("Food_Name", axis=1)
y = df["Food_Name"]

y_encoded, label_encoder = encode_target(y)

# -------------------- FEATURE TYPES --------------------
numerical_features = [
    "Calories","Protein","Fat","Carbs","Sugar","Fiber",
    "Sodium","Cholesterol","Glycemic_Index",
    "Water_Content","Serving_Size"
]

categorical_features = [
    "Meal_Type","Preparation_Method",
    "Is_Vegan","Is_Gluten_Free"
]

preprocessor = get_preprocessor(numerical_features, categorical_features)

# -------------------- SPLIT --------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded,
    test_size=0.2,
    stratify=y_encoded,
    random_state=RANDOM_STATE
)

# -------------------- MODELS --------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "KNN": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(random_state=RANDOM_STATE),
    "SVM": SVC(),
    "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
    "XGBoost": XGBClassifier(eval_metric="mlogloss", random_state=RANDOM_STATE)
}

results = {}
trained_models = {}

# -------------------- TRAIN & EVALUATE --------------------
for name, model in models.items():
    trained_model = train_model(model, preprocessor, X_train, y_train)
    acc, y_pred = evaluate_model(trained_model, X_test, y_test, name)

    results[name] = acc
    trained_models[name] = trained_model

# -------------------- BEST MODEL --------------------
best_model_name = max(results, key=results.get)
best_model = trained_models[best_model_name]

print("\n✅ Best Model:", best_model_name)

# -------------------- BUSINESS USE CASE PREDICTION --------------------
sample_input = pd.DataFrame([{
    "Calories": 250,
    "Protein": 30,
    "Fat": 5,
    "Carbs": 20,
    "Sugar": 3,
    "Fiber": 4,
    "Sodium": 400,
    "Cholesterol": 50,
    "Glycemic_Index": 45,
    "Water_Content": 60,
    "Serving_Size": 150,
    "Meal_Type": "Dinner",
    "Preparation_Method": "Grilled",
    "Is_Vegan": 0,
    "Is_Gluten_Free": 1
}])

pred_class = best_model.predict(sample_input)
pred_food = label_encoder.inverse_transform(pred_class)

print("🍽️ Predicted Food:", pred_food[0])

# -------------------- SINGLE DASHBOARD VISUAL --------------------
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

# 1️⃣ Model Accuracy Comparison
axes[0].bar(results.keys(), results.values())
axes[0].set_title("Model Accuracy Comparison")
axes[0].tick_params(axis='x', rotation=45)
axes[0].set_ylabel("Accuracy")

# 2️⃣ Confusion Matrix
cm = confusion_matrix(y_test, best_model.predict(X_test))
sns.heatmap(cm, ax=axes[1], cmap="Blues")
axes[1].set_title(f"Confusion Matrix - {best_model_name}")

# 3️⃣ Feature Importance (if tree-based)
if best_model_name in ["Random Forest", "Gradient Boosting", "XGBoost"]:
    model = best_model.named_steps['model']
    preprocessor = best_model.named_steps['preprocessor']

    feature_names = (
        numerical_features +
        list(preprocessor.named_transformers_['cat']
             .named_steps['onehot']
             .get_feature_names_out(categorical_features))
    )

    importances = model.feature_importances_
    indices = np.argsort(importances)[-10:]  # top 10

    axes[2].barh(range(len(indices)), importances[indices])
    axes[2].set_yticks(range(len(indices)))
    axes[2].set_yticklabels([feature_names[i] for i in indices])
    axes[2].set_title("Top 10 Important Features")

plt.tight_layout()
plt.show()
