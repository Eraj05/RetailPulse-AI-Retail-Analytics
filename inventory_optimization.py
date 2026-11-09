
import os
import pandas as pd
import numpy as np

os.makedirs("data", exist_ok=True)

LEAD_TIME_DAYS = 7
REVIEW_PERIOD_DAYS = 7
SERVICE_LEVEL_Z = 1.645

# Confirmed exceptional bulk transactions
EXCEPTION_TRANSACTIONS = {
    ("581483", "23843", 80995),
    ("541431", "23166", 74215),
}


def load_and_clean_data():
    print("\nLoading cleaned retail dataset...")

    df = pd.read_csv("data/retail_cleaned.csv", low_memory=False)

    df["InvoiceDate"] = pd.to_datetime(
        df["InvoiceDate"], errors="coerce"
    )

    df["Quantity"] = pd.to_numeric(
        df["Quantity"], errors="coerce"
    )

    df["InvoiceNo"] = (
        df["InvoiceNo"].astype(str)
        .str.replace(r"\.0$", "", regex=True)
        .str.strip()
    )

    df["StockCode"] = (
        df["StockCode"].astype(str).str.strip()
    )

    return df


def separate_inventory_exceptions(df):
    df = df.copy()

    exception_mask = pd.Series(False, index=df.index)

    for invoice, stock_code, quantity in EXCEPTION_TRANSACTIONS:
        match = (
            (df["InvoiceNo"] == invoice)
            & (df["StockCode"] == stock_code)
            & (df["Quantity"] == quantity)
        )
        exception_mask |= match

    exceptions = df[exception_mask].copy()
    normal_data = df[~exception_mask].copy()

    if not exceptions.empty:
        exceptions["ExceptionReason"] = (
            "Exceptional bulk transaction - excluded from routine inventory planning"
        )

    exceptions.to_csv(
        "data/inventory_exceptions.csv",
        index=False
    )

    print(f"Exceptional transactions identified: {len(exceptions)}")
    print(
        "Exceptional quantity excluded:",
        int(exceptions["Quantity"].sum()) if not exceptions.empty else 0
    )

    return normal_data, exceptions


def prepare_inventory_data(df):
    df = df.copy()

    df = df.dropna(
        subset=["InvoiceDate", "StockCode", "Quantity"]
    )

    # Exclude returns and cancelled quantities
    df = df[df["Quantity"] > 0]

    return df


def calculate_inventory_metrics(df):

    df = prepare_inventory_data(df)

    if df.empty:
        raise ValueError("No valid sales data found.")

    start_date = df["InvoiceDate"].min().normalize()
    end_date = df["InvoiceDate"].max().normalize()

    date_range = pd.date_range(
        start=start_date,
        end=end_date,
        freq="D"
    )

    recommendations = []

    print("\nCalculating product-wise demand...")

    for stock_code, group in df.groupby("StockCode"):

        daily_demand = (
            group.groupby(
                group["InvoiceDate"].dt.normalize()
            )["Quantity"]
            .sum()
            .reindex(date_range, fill_value=0)
        )

        avg_daily_demand = daily_demand.mean()
        std_daily_demand = daily_demand.std(ddof=1)

        if pd.isna(std_daily_demand):
            std_daily_demand = 0

        max_daily_demand = daily_demand.max()

        demand_cv = (
            std_daily_demand / avg_daily_demand
            if avg_daily_demand > 0 else 0
        )

        safety_stock = (
            SERVICE_LEVEL_Z
            * std_daily_demand
            * np.sqrt(LEAD_TIME_DAYS)
        )

        reorder_point = (
            avg_daily_demand * LEAD_TIME_DAYS
            + safety_stock
        )

        target_stock = (
            avg_daily_demand
            * (LEAD_TIME_DAYS + REVIEW_PERIOD_DAYS)
            + SERVICE_LEVEL_Z
            * std_daily_demand
            * np.sqrt(LEAD_TIME_DAYS + REVIEW_PERIOD_DAYS)
        )

        suggested_order_qty = max(
            0, target_stock - reorder_point
        )

        description = group["Description"].dropna()

        product_name = (
            description.mode().iloc[0]
            if not description.empty
            else "Unknown Product"
        )

        recommendations.append({
            "StockCode": stock_code,
            "Description": product_name,
            "TotalDemand": group["Quantity"].sum(),
            "AverageDailyDemand": round(avg_daily_demand, 2),
            "StdDailyDemand": round(std_daily_demand, 2),
            "MaxDailyDemand": round(max_daily_demand, 2),
            "DemandVariability": round(demand_cv, 2),
            "SafetyStock": round(safety_stock, 2),
            "ReorderPoint": round(reorder_point, 2),
            "TargetStock": round(target_stock, 2),
            "SuggestedOrderQtyAtROP": round(
                suggested_order_qty, 2
            ),
            "LeadTimeDays": LEAD_TIME_DAYS
        })

    result = pd.DataFrame(recommendations)

    result = result.sort_values(
        by="TotalDemand",
        ascending=False
    ).reset_index(drop=True)

    return result


def create_inventory_summary(inventory, exceptions):

    summary = {
        "TotalProducts": len(inventory),
        "TotalDemand": inventory["TotalDemand"].sum(),
        "AverageDailyDemand": round(
            inventory["AverageDailyDemand"].mean(), 2
        ),
        "AverageSafetyStock": round(
            inventory["SafetyStock"].mean(), 2
        ),
        "TotalSuggestedOrderQuantity": round(
            inventory["SuggestedOrderQtyAtROP"].sum(), 2
        ),
        "ExcludedExceptionTransactions": len(exceptions),
        "ExcludedExceptionQuantity": (
            int(exceptions["Quantity"].sum())
            if not exceptions.empty else 0
        )
    }

    return summary


def run_inventory_optimization():

    print("\n======================================")
    print(" RETAILPULSE INVENTORY OPTIMIZATION")
    print("======================================")

    df = load_and_clean_data()

    normal_data, exceptions = separate_inventory_exceptions(df)

    inventory = calculate_inventory_metrics(normal_data)

    inventory.to_csv(
        "data/inventory_recommendations.csv",
        index=False
    )

    summary = create_inventory_summary(
        inventory, exceptions
    )

    summary_df = pd.DataFrame(
        list(summary.items()),
        columns=["Metric", "Value"]
    )

    summary_df.to_csv(
        "data/inventory_summary.csv",
        index=False
    )

    print("\nINVENTORY SUMMARY")
    print("------------------------------")

    for key, value in summary.items():
        print(f"{key}: {value:,.2f}")

    print("\nTOP 10 PRODUCTS BY DEMAND")
    print(inventory.head(10).to_string(index=False))

    print("\nINVENTORY OPTIMIZATION COMPLETED!")

    print("\nGenerated files:")
    print("1. data/inventory_recommendations.csv")
    print("2. data/inventory_summary.csv")
    print("3. data/inventory_exceptions.csv")


if __name__ == "__main__":
    run_inventory_optimization()
