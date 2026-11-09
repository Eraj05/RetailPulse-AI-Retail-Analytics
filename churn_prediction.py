
import os
import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


os.makedirs("models", exist_ok=True)
os.makedirs("data", exist_ok=True)


def create_churn_dataset(df):
    df = df.copy()

    df["InvoiceDate"] = pd.to_datetime(
        df["InvoiceDate"], errors="coerce"
    )
    df = df.dropna(subset=["CustomerID", "InvoiceDate"])

    reference_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

    customer_data = df.groupby("CustomerID").agg(
        FirstPurchase=("InvoiceDate", "min"),
        LastPurchase=("InvoiceDate", "max"),
        Frequency=("InvoiceNo", "nunique"),
        Monetary=("TotalSales", "sum"),
        TotalQuantity=("Quantity", "sum")
    ).reset_index()

    customer_data["Recency"] = (
        reference_date - customer_data["LastPurchase"]
    ).dt.days

    customer_data["Tenure"] = (
        customer_data["LastPurchase"] -
        customer_data["FirstPurchase"]
    ).dt.days

    customer_data["AverageOrderValue"] = (
        customer_data["Monetary"] /
        customer_data["Frequency"].clip(lower=1)
    )

    # Inactivity greater than 90 days is treated as churn
    customer_data["Churn"] = (
        customer_data["Recency"] > 90
    ).astype(int)

    return customer_data


def train_churn_model(customer_data):

    features = [
        "Frequency",
        "Monetary",
        "TotalQuantity",
        "Tenure",
        "AverageOrderValue"
    ]

    X = customer_data[features]
    y = customer_data["Churn"]

    if y.nunique() < 2:
        raise ValueError(
            "Churn dataset has only one class. "
            "Cannot train a classification model."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight="balanced"
        ))
    ])

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print("\nModel Accuracy:", round(accuracy * 100, 2), "%")
    print("\nClassification Report:")
    print(classification_report(
        y_test,
        predictions,
        zero_division=0
    ))

    joblib.dump(
        {
            "model": model,
            "features": features
        },
        "models/churn_prediction.pkl"
    )

    return model


def predict_churn_risk(customer_data, model):

    features = [
        "Frequency",
        "Monetary",
        "TotalQuantity",
        "Tenure",
        "AverageOrderValue"
    ]

    customer_data = customer_data.copy()

    customer_data["ChurnPrediction"] = model.predict(
        customer_data[features]
    )

    customer_data["ChurnProbability"] = (
        model.predict_proba(customer_data[features])[:, 1]
    )

    customer_data["ChurnRisk"] = customer_data[
        "ChurnProbability"
    ].apply(
        lambda x: "High Risk" if x >= 0.70
        else "Medium Risk" if x >= 0.40
        else "Low Risk"
    )

    return customer_data


if __name__ == "__main__":

    print("Loading RetailPulse dataset...")

    df = pd.read_csv("data/retail_cleaned.csv")

    print("Creating customer churn dataset...")

    customer_data = create_churn_dataset(df)

    print("Training Random Forest model...")

    model = train_churn_model(customer_data)

    print("Predicting customer churn risk...")

    results = predict_churn_risk(customer_data, model)

    results.to_csv(
        "data/customer_churn_predictions.csv",
        index=False
    )

    print("\nCUSTOMER CHURN PREDICTION COMPLETED!")

    print("\nChurn Risk Summary:")
    print(results["ChurnRisk"].value_counts())

    print("\nGenerated files:")
    print("1. data/customer_churn_predictions.csv")
    print("2. models/churn_prediction.pkl")
