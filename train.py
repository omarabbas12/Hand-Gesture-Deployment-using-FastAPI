import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from transformer import LandmarkNormalizer, map_labels
import joblib
import mlflow
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import lightgbm as lgb

# Load your dataset
df = pd.read_csv('hand_landmarks_data.csv')
df = map_labels(df)

# Split features and labels
X = df.drop("label", axis=1)
y = df["label"]

# Apply transformation
transformer = LandmarkNormalizer()
X_transformed = transformer.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.2)

mlflow.set_experiment("Hand_Landmarks_Classification")

# Train & log
with mlflow.start_run(run_name='Logistic_Regression') as run:
    model = lgb.LGBMClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Compute metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    # Log metrics
    mlflow.log_param("model", "Logistic_Regression")
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", prec)
    mlflow.log_metric("recall", rec)
    mlflow.log_metric("f1_score", f1)

    # Save model
    joblib.dump(model, "model.pkl")
    mlflow.log_artifact("model.pkl")


    # Save classification report
    report = classification_report(y_test, y_pred)
    with open("classification_report.txt", "w") as f:
        f.write(report)
    mlflow.log_artifact("classification_report.txt")

    # Save and log confusion matrix plot
    cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=model.classes_, yticklabels=model.classes_, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    mlflow.log_artifact("confusion_matrix.png")

    download_url = mlflow.get_artifact_uri("model.pkl")
    print(f"Model saved to: {download_url}")