
import pandas as pd


def get_sales_summary(df):

    summary = {
        "total_revenue": df["TotalSales"].sum(),

        "total_orders": df["InvoiceNo"].nunique(),

        "total_customers": df["CustomerID"].nunique(),

        "total_products": df["StockCode"].nunique(),

        "total_quantity": df["Quantity"].sum(),

        "average_order_value": (
            df.groupby("InvoiceNo")["TotalSales"].sum().mean()
        )
    }

    return summary


def monthly_sales(df):

    monthly = (
        df.set_index("InvoiceDate")
        .resample("MS")["TotalSales"]
        .sum()
        .reset_index()
    )

    return monthly


def daily_sales(df):

    daily = (
        df.groupby(
            df["InvoiceDate"].dt.date
        )["TotalSales"]
        .sum()
        .reset_index()
    )

    daily.columns = ["Date", "Revenue"]

    return daily


def top_products(df, limit=10):

    products = (
        df.groupby("Description")
        .agg(
            TotalSales=("TotalSales", "sum"),
            TotalQuantity=("Quantity", "sum")
        )
        .sort_values(
            by="TotalSales",
            ascending=False
        )
        .head(limit)
        .reset_index()
    )

    return products


def country_sales(df):

    countries = (
        df.groupby("Country")
        .agg(
            TotalRevenue=("TotalSales", "sum"),
            TotalOrders=("InvoiceNo", "nunique")
        )
        .sort_values(
            by="TotalRevenue",
            ascending=False
        )
        .reset_index()
    )

    return countries


def customer_sales(df):

    customers = (
        df.dropna(subset=["CustomerID"])
        .groupby("CustomerID")
        .agg(
            TotalOrders=("InvoiceNo", "nunique"),
            TotalSpent=("TotalSales", "sum"),
            TotalQuantity=("Quantity", "sum")
        )
        .sort_values(
            by="TotalSpent",
            ascending=False
        )
        .reset_index()
    )

    return customers
