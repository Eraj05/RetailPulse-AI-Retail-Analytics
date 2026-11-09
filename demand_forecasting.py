import os
import joblib
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


os.makedirs("models", exist_ok=True)
os.makedirs("data", exist_ok=True)


def prepare_demand_data(df):

    df = df.copy()

    df["InvoiceDate"] = pd.to_datetime(
        df["InvoiceDate"], errors="coerce"
    )

    df = df.dropna(subset=["InvoiceDate", "Quantity"])

    # Consider only actual sales, not returned quantities
    df = df[df["Quantity"] > 0]

    # Daily total demand
    daily = (
        df.groupby(df["InvoiceDate"].dt.normalize())["Quantity"]
        .sum()
        .asfreq("D", fill_value=0)
        .rename("Demand")
        .reset_index()
    )

    daily.columns = ["Date", "Demand"]

    return daily


def create_features(daily):

    data = daily.copy()

    data["Lag_1"] = data["Demand"].shift(1)
    data["Lag_7"] = data["Demand"].shift(7)

    data["Rolling_7"] = (
        data["Demand"].shift(1).rolling(7).mean()
    )

    data["DayOfWeek"] = data["Date"].dt.dayofweek
    data["Month"] = data["Date"].dt.month
    data["Day"] = data["Date"].dt.day

    data = data.dropna().reset_index(drop=True)

    return data


def train_demand_model(data):

    features = [
        "Lag_1",
        "Lag_7",
        "Rolling_7",
        "DayOfWeek",
        "Month",
        "Day"
    ]

    X = data[features]
    y = data["Demand"]

    split = int(len(data) * 0.8)

    if split == 0 or split >= len(data):
        raise ValueError("Not enough data for model training.")

    X_train = X.iloc[:split]
    X_test = X.iloc[split:]

    y_train = y.iloc[:split]
    y_test = y.iloc[split:]

    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        max_depth=12,
        min_samples_leaf=2,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))

    print("\nMODEL PERFORMANCE")
    print("-------------------------")
    print("MAE:", round(mae, 2))
    print("RMSE:", round(rmse, 2))

    return model


def forecast_next_30_days(model, daily):

    history = daily["Demand"].astype(float).tolist()

    last_date = daily["Date"].max()

    forecasts = []

    for i in range(1, 31):

        forecast_date = last_date + pd.Timedelta(days=i)

        features = pd.DataFrame([{
            "Lag_1": history[-1],
            "Lag_7": history[-7],
            "Rolling_7": np.mean(history[-7:]),
            "DayOfWeek": forecast_date.dayofweek,
            "Month": forecast_date.month,
            "Day": forecast_date.day
        }])

        predicted_demand = max(
            0, float(model.predict(features)[0])
        )

        forecasts.append({
            "Date": forecast_date,
            "PredictedDemand": round(predicted_demand, 2)
        })

        history.append(predicted_demand)

    return pd.DataFrame(forecasts)


def run_demand_forecasting():

    print("\n===================================")
    print(" RETAILPULSE DEMAND FORECASTING")
    print("===================================")

    print("\nLoading dataset...")

    df = pd.read_csv("data/retail_cleaned.csv")

    print("Preparing daily demand data...")

    daily = prepare_demand_data(df)

    if len(daily) < 40:
        raise ValueError(
            "At least 40 days of daily data are required."
        )

    daily.to_csv(
        "data/daily_demand.csv", index=False
    )

    print("Creating forecasting features...")

    data = create_features(daily)

    print("Training Random Forest model...")

    model = train_demand_model(data)

    joblib.dump(
        model, "models/demand_forecasting.pkl"
    )

    print("\nGenerating next 30 days forecast...")

    forecast = forecast_next_30_days(model, daily)

    forecast.to_csv(
        "data/demand_forecast.csv", index=False
    )

    print("\nNEXT 30 DAYS DEMAND FORECAST")
    print(forecast.to_string(index=False))

    print("\nDEMAND FORECASTING COMPLETED!")

    print("\nGenerated files:")
    print("1. data/daily_demand.csv")
    print("2. data/demand_forecast.csv")
    print("3. models/demand_forecasting.pkl")


if __name__ == "__main__":
    run_demand_forecasting()