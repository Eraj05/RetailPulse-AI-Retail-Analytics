
import streamlit as st
import pandas as pd
import plotly.express as px
import os

# ==============================
# PAGE CONFIGURATION
# ==============================
st.set_page_config(
    page_title="RetailPulse",
    page_icon="📊",
    layout="wide"
)

# ==============================
# PROFESSIONAL THEME + SIDEBAR FIX
# ==============================
st.markdown("""
<style>

/* Main application background */
.stApp {
    background-color: #F5F7FB;
}

/* Main content spacing */
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* Headings */
.stApp h1,
.stApp h2,
.stApp h3,
.stApp h4,
.stApp h5,
.stApp h6 {
    color: #1E293B !important;
    font-weight: 700 !important;
}

/* Main content text */
.stApp .stMarkdown,
.stApp p,
.stApp label {
    color: #334155;
}

/* Sidebar background */
section[data-testid="stSidebar"] {
    background-color: #172554 !important;
    min-width: 250px;
}

/* Sidebar content scrolling */
section[data-testid="stSidebar"] > div {
    height: 100vh !important;
    overflow-y: auto !important;
    overflow-x: hidden !important;
}

/* Sidebar text */
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] span {
    color: #FFFFFF !important;
}

/* Navigation radio group */
section[data-testid="stSidebar"] [data-testid="stRadio"] {
    width: 100% !important;
    overflow: visible !important;
}

/* Keep all navigation options visible */
section[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] {
    display: flex !important;
    flex-direction: column !important;
    gap: 10px !important;
    height: auto !important;
    max-height: none !important;
    overflow: visible !important;
    opacity: 1 !important;
    filter: none !important;
}

/* Individual navigation option */
section[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label {
    display: flex !important;
    visibility: visible !important;
    opacity: 1 !important;
    filter: none !important;
    color: #FFFFFF !important;
    background-color: transparent !important;
    padding: 10px 12px !important;
    border-radius: 8px !important;
    width: 100% !important;
    cursor: pointer !important;
}

/* Navigation option text */
section[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label p,
section[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label span {
    color: #FFFFFF !important;
    opacity: 1 !important;
}

/* Selected navigation option */
section[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label:has(input:checked) {
    background-color: #2563EB !important;
}

/* Hover effect */
section[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label:hover {
    background-color: #263B70 !important;
}

/* Sidebar scrollbar */
section[data-testid="stSidebar"] > div::-webkit-scrollbar {
    width: 6px;
}

section[data-testid="stSidebar"] > div::-webkit-scrollbar-thumb {
    background-color: #64748B;
    border-radius: 10px;
}

/* Metric cards */
[data-testid="stMetric"] {
    background-color: #FFFFFF;
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #E2E8F0;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
}

[data-testid="stMetricLabel"] {
    color: #64748B !important;
}

[data-testid="stMetricValue"] {
    color: #1E40AF !important;
    font-weight: 800 !important;
}

/* Dataframes */
[data-testid="stDataFrame"] {
    border-radius: 10px;
}

/* Download buttons */
.stDownloadButton button {
    background-color: #2563EB;
    color: #FFFFFF !important;
    border-radius: 8px;
    border: none;
    padding: 8px 18px;
}

.stDownloadButton button:hover {
    background-color: #1D4ED8;
    color: #FFFFFF !important;
}

/* Dividers */
hr {
    border-color: #CBD5E1;
}

</style>
""", unsafe_allow_html=True)


# ==============================
# APPLICATION HEADER
# ==============================
st.title("📊 RetailPulse")
st.caption("AI-Powered Retail Analytics & Prediction System")


# ==============================
# LOAD DATA
# ==============================
@st.cache_data
def load_data():
    data = pd.read_csv(
        "data/retail_cleaned.csv",
        low_memory=False
    )

    data["InvoiceDate"] = pd.to_datetime(
        data["InvoiceDate"],
        errors="coerce"
    )

    return data


df = load_data()


# ==============================
# SIDEBAR NAVIGATION
# ==============================
st.sidebar.title("RetailPulse")
st.sidebar.caption("AI-Powered Retail Analytics")

page = st.sidebar.radio(
    "Navigation",
    [
        "Executive Dashboard",
        "Sales Analytics",
        "Customer Segmentation",
        "Churn Prediction",
        "Demand Forecasting",
        "Inventory Optimization"
    ],
    key="retailpulse_navigation"
)


# ==================================================
# 1. EXECUTIVE DASHBOARD
# ==================================================
if page == "Executive Dashboard":

    st.header("Executive Overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Revenue",
        f"£{df['TotalSales'].sum():,.2f}"
    )

    col2.metric(
        "Total Orders",
        f"{df['InvoiceNo'].nunique():,}"
    )

    col3.metric(
        "Total Customers",
        f"{df['CustomerID'].nunique():,}"
    )

    col4.metric(
        "Total Products",
        f"{df['StockCode'].nunique():,}"
    )

    st.divider()

    st.subheader("Monthly Revenue Trend")

    monthly = (
        df.set_index("InvoiceDate")
        .resample("MS")["TotalSales"]
        .sum()
        .reset_index()
    )

    fig = px.line(
        monthly,
        x="InvoiceDate",
        y="TotalSales",
        markers=True,
        title="Monthly Sales Performance"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.subheader("Top 10 Products")

    products = (
        df.groupby("Description")["TotalSales"]
        .sum()
        .nlargest(10)
        .reset_index()
    )

    fig2 = px.bar(
        products,
        x="TotalSales",
        y="Description",
        orientation="h",
        title="Top Selling Products"
    )

    st.plotly_chart(
        fig2,
        use_container_width=True
    )

    st.subheader("Recent Transactions")

    st.dataframe(
        df.tail(10),
        use_container_width=True
    )


# ==================================================
# 2. SALES ANALYTICS
# ==================================================
elif page == "Sales Analytics":

    st.header("📈 Sales Analytics")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Total Revenue",
            f"£{df['TotalSales'].sum():,.2f}"
        )

    with col2:
        average_order = (
            df.groupby("InvoiceNo")["TotalSales"]
            .sum()
            .mean()
        )

        st.metric(
            "Average Order Value",
            f"£{average_order:,.2f}"
        )

    st.subheader("Monthly Revenue")

    monthly = (
        df.set_index("InvoiceDate")
        .resample("MS")["TotalSales"]
        .sum()
        .reset_index()
    )

    fig = px.line(
        monthly,
        x="InvoiceDate",
        y="TotalSales",
        markers=True,
        title="Monthly Revenue Trend"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.subheader("Country-wise Revenue")

    country = (
        df.groupby("Country")["TotalSales"]
        .sum()
        .nlargest(10)
        .reset_index()
    )

    fig2 = px.bar(
        country,
        x="Country",
        y="TotalSales",
        title="Top 10 Countries by Revenue"
    )

    st.plotly_chart(
        fig2,
        use_container_width=True
    )

    st.subheader("Top 10 Products")

    products = (
        df.groupby("Description")["TotalSales"]
        .sum()
        .nlargest(10)
        .reset_index()
    )

    fig3 = px.bar(
        products,
        x="TotalSales",
        y="Description",
        orientation="h",
        title="Top Selling Products"
    )

    st.plotly_chart(
        fig3,
        use_container_width=True
    )


# ==================================================
# 3. CUSTOMER SEGMENTATION
# ==================================================
elif page == "Customer Segmentation":

    st.header("👥 Customer Segmentation")
    st.caption("RFM Analysis + K-Means Clustering")

    file_path = "data/customer_segments.csv"

    if not os.path.exists(file_path):

        st.error("Customer segmentation file not found!")

        st.info(
            "Run: python -m modules.customer_segmentation"
        )

    else:

        customers = pd.read_csv(file_path)

        segment_col = next(
            (
                col for col in [
                    "Segment",
                    "CustomerSegment",
                    "Cluster",
                    "ClusterName"
                ]
                if col in customers.columns
            ),
            None
        )

        if segment_col is None:

            st.error("Segment column not found in CSV.")

            st.write(
                "Available columns:",
                customers.columns.tolist()
            )

        else:

            total = len(customers)
            counts = customers[segment_col].value_counts()

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Total Customers",
                f"{total:,}"
            )

            col2.metric(
                "Customer Segments",
                f"{customers[segment_col].nunique()}"
            )

            col3.metric(
                "Largest Segment",
                str(counts.index[0])
            )

            st.divider()

            st.subheader("Customer Segment Distribution")

            distribution = (
                customers[segment_col]
                .value_counts()
                .reset_index()
            )

            distribution.columns = [
                "Segment",
                "Customers"
            ]

            fig = px.pie(
                distribution,
                names="Segment",
                values="Customers",
                hole=0.45,
                title="Customer Distribution"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            st.subheader("Segment Comparison")

            fig2 = px.bar(
                distribution,
                x="Segment",
                y="Customers",
                color="Segment",
                text="Customers",
                title="Customers by Segment"
            )

            st.plotly_chart(
                fig2,
                use_container_width=True
            )

            st.subheader("Customer Segment Details")

            st.dataframe(
                customers,
                use_container_width=True
            )

            st.download_button(
                "Download Customer Segments",
                customers.to_csv(index=False),
                file_name="customer_segments.csv",
                mime="text/csv"
            )


# ==================================================
# 4. CHURN PREDICTION
# ==================================================
elif page == "Churn Prediction":

    st.header("🔍 Customer Churn Prediction")

    st.caption(
        "Random Forest Model | Inactivity-based Churn Analysis"
    )

    file_path = "data/customer_churn_predictions.csv"

    if not os.path.exists(file_path):

        st.error("Churn prediction file not found!")

        st.code(
            "python -m modules.churn_prediction"
        )

    else:

        churn_df = pd.read_csv(file_path)

        st.subheader("Customer Churn Overview")

        prediction_col = next(
            (
                col for col in [
                    "ChurnPrediction",
                    "PredictedChurn",
                    "Churn_Prediction",
                    "Prediction",
                    "Churn_Risk",
                    "RiskLevel"
                ]
                if col in churn_df.columns
            ),
            None
        )

        probability_col = next(
            (
                col for col in [
                    "ChurnProbability",
                    "Churn_Probability",
                    "Probability",
                    "ChurnRiskProbability"
                ]
                if col in churn_df.columns
            ),
            None
        )

        if prediction_col is None and probability_col is None:

            st.warning(
                "Prediction columns were not found in the CSV."
            )

            st.write(
                "Available columns:",
                churn_df.columns.tolist()
            )

        else:

            if prediction_col is None:

                churn_df["ChurnRiskCategory"] = (
                    pd.to_numeric(
                        churn_df[probability_col],
                        errors="coerce"
                    )
                    .apply(
                        lambda x: (
                            "High Risk"
                            if pd.notna(x) and x >= 0.5
                            else "Low Risk"
                        )
                    )
                )

                display_col = "ChurnRiskCategory"

            else:

                display_col = prediction_col

            counts = churn_df[display_col].value_counts()

            total_customers = len(churn_df)

            high_risk_values = [
                "1",
                "1.0",
                "true",
                "yes",
                "churn",
                "high risk",
                "high"
            ]

            risk_customers = sum(
                str(value).strip().lower() in high_risk_values
                for value in churn_df[display_col]
            )

            risk_percentage = (
                risk_customers / total_customers * 100
                if total_customers > 0 else 0
            )

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Total Customers",
                f"{total_customers:,}"
            )

            col2.metric(
                "Churn Risk Customers",
                f"{risk_customers:,}"
            )

            col3.metric(
                "Churn Risk Percentage",
                f"{risk_percentage:.2f}%"
            )

            st.divider()

            st.subheader("Churn Risk Distribution")

            distribution = (
                counts.rename_axis("Churn Status")
                .reset_index(name="Customers")
            )

            fig = px.pie(
                distribution,
                names="Churn Status",
                values="Customers",
                hole=0.45,
                title="Customer Churn Distribution"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            st.subheader("Churn Status Comparison")

            fig2 = px.bar(
                distribution,
                x="Churn Status",
                y="Customers",
                color="Churn Status",
                text="Customers"
            )

            st.plotly_chart(
                fig2,
                use_container_width=True
            )

            if probability_col is not None:

                st.subheader("Churn Probability Analysis")

                churn_df[probability_col] = pd.to_numeric(
                    churn_df[probability_col],
                    errors="coerce"
                )

                fig3 = px.histogram(
                    churn_df.dropna(subset=[probability_col]),
                    x=probability_col,
                    nbins=20,
                    title="Customer Churn Probability"
                )

                st.plotly_chart(
                    fig3,
                    use_container_width=True
                )

            st.subheader("Customer Churn Details")

            st.dataframe(
                churn_df,
                use_container_width=True
            )

            st.download_button(
                "Download Churn Predictions",
                churn_df.to_csv(index=False),
                file_name="customer_churn_predictions.csv",
                mime="text/csv"
            )

            st.info(
                "Note: Churn is defined using customer inactivity "
                "of more than 90 days in this project. This is an "
                "inactivity-based risk indicator, not confirmed "
                "future customer behavior."
            )


# ==================================================
# 5. DEMAND FORECASTING
# ==================================================
elif page == "Demand Forecasting":

    st.header("📈 Demand Forecasting")

    st.caption(
        "Machine Learning-Based 30-Day Demand Prediction"
    )

    file_path = "data/demand_forecast.csv"

    if not os.path.exists(file_path):

        st.error("Demand forecast file not found!")

        st.code(
            "python -m modules.demand_forecasting"
        )

    else:

        forecast_df = pd.read_csv(file_path)

        date_col = next(
            (
                col for col in [
                    "Date",
                    "InvoiceDate",
                    "ForecastDate"
                ]
                if col in forecast_df.columns
            ),
            None
        )

        demand_col = next(
            (
                col for col in [
                    "PredictedDemand",
                    "ForecastDemand",
                    "PredictedQuantity",
                    "Forecast",
                    "Demand"
                ]
                if col in forecast_df.columns
            ),
            None
        )

        if date_col is None or demand_col is None:

            st.warning(
                "Expected forecast columns were not found."
            )

            st.write(
                "Available columns:",
                forecast_df.columns.tolist()
            )

        else:

            forecast_df[date_col] = pd.to_datetime(
                forecast_df[date_col],
                errors="coerce"
            )

            forecast_df[demand_col] = pd.to_numeric(
                forecast_df[demand_col],
                errors="coerce"
            )

            forecast_df = (
                forecast_df.dropna(
                    subset=[date_col, demand_col]
                )
                .sort_values(date_col)
            )

            if forecast_df.empty:

                st.warning(
                    "No valid forecast records available."
                )

            else:

                total_demand = forecast_df[demand_col].sum()
                average_demand = forecast_df[demand_col].mean()

                col1, col2, col3 = st.columns(3)

                col1.metric(
                    "Forecast Period",
                    f"{len(forecast_df)} Days"
                )

                col2.metric(
                    "Total Predicted Demand",
                    f"{total_demand:,.0f}"
                )

                col3.metric(
                    "Average Daily Demand",
                    f"{average_demand:,.2f}"
                )

                st.divider()

                st.subheader("Future Demand Trend")

                fig = px.line(
                    forecast_df,
                    x=date_col,
                    y=demand_col,
                    markers=True,
                    title="Predicted Daily Demand"
                )

                fig.update_layout(
                    xaxis_title="Date",
                    yaxis_title="Predicted Quantity"
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

                st.subheader("Daily Forecast Details")

                st.dataframe(
                    forecast_df,
                    use_container_width=True
                )

                st.download_button(
                    "Download Demand Forecast",
                    forecast_df.to_csv(index=False),
                    file_name="demand_forecast.csv",
                    mime="text/csv"
                )

                st.info(
                    "Note: This forecast represents aggregate daily "
                    "store demand, not individual product-level demand."
                )


# ==================================================
# 6. INVENTORY OPTIMIZATION
# ==================================================
elif page == "Inventory Optimization":

    st.header("📦 Inventory Optimization")

    st.caption(
        "AI-Based Stock Monitoring and Reorder Recommendations"
    )

    file_path = "data/inventory_recommendations.csv"

    if not os.path.exists(file_path):

        st.error(
            "Inventory recommendations file not found!"
        )

        st.code(
            "python -m modules.inventory_optimization"
        )

    else:

        inventory_df = pd.read_csv(file_path)

        required_columns = [
            "Description",
            "AverageDailyDemand",
            "SafetyStock",
            "ReorderPoint",
            "SuggestedOrderQtyAtROP"
        ]

        missing_columns = [
            col for col in required_columns
            if col not in inventory_df.columns
        ]

        if missing_columns:

            st.warning(
                "Some expected columns are missing."
            )

            st.write(
                "Missing columns:",
                missing_columns
            )

            st.write(
                "Available columns:",
                inventory_df.columns.tolist()
            )

        else:

            numeric_columns = [
                "AverageDailyDemand",
                "SafetyStock",
                "ReorderPoint",
                "SuggestedOrderQtyAtROP"
            ]

            for col in numeric_columns:

                inventory_df[col] = pd.to_numeric(
                    inventory_df[col],
                    errors="coerce"
                )

            inventory_df = inventory_df.dropna(
                subset=numeric_columns
            )

            # KPI calculations
            total_products = len(inventory_df)

            reorder_products = (
                inventory_df["SuggestedOrderQtyAtROP"] > 0
            ).sum()

            total_order_qty = inventory_df[
                "SuggestedOrderQtyAtROP"
            ].sum()

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Total Products",
                f"{total_products:,}"
            )

            col2.metric(
                "Products Requiring Reorder",
                f"{reorder_products:,}"
            )

            col3.metric(
                "Suggested Order Quantity",
                f"{total_order_qty:,.0f}"
            )

            st.divider()

            # Top products by demand
            st.subheader(
                "Top 10 Products by Daily Demand"
            )

            top_products = inventory_df.nlargest(
                10,
                "AverageDailyDemand"
            )

            fig = px.bar(
                top_products,
                x="AverageDailyDemand",
                y="Description",
                orientation="h",
                title="Products with Highest Average Daily Demand",
                text="AverageDailyDemand"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            # Reorder recommendations
            st.subheader(
                "Stock Reorder Recommendations"
            )

            reorder_df = (
                inventory_df[
                    inventory_df["SuggestedOrderQtyAtROP"] > 0
                ]
                .sort_values(
                    "SuggestedOrderQtyAtROP",
                    ascending=False
                )
            )

            if reorder_df.empty:

                st.success(
                    "No products currently require a reorder recommendation."
                )

            else:

                fig2 = px.bar(
                    reorder_df.head(15),
                    x="Description",
                    y="SuggestedOrderQtyAtROP",
                    title="Top 15 Suggested Reorder Quantities"
                )

                st.plotly_chart(
                    fig2,
                    use_container_width=True
                )

            # Complete inventory table
            st.subheader(
                "Complete Inventory Analysis"
            )

            st.dataframe(
                inventory_df,
                use_container_width=True
            )

            # Download recommendations
            st.download_button(
                "Download Inventory Recommendations",
                inventory_df.to_csv(index=False),
                file_name="inventory_recommendations.csv",
                mime="text/csv"
            )

            # Exception transactions information
            exception_file = "data/inventory_exceptions.csv"

            if os.path.exists(exception_file):

                exceptions_df = pd.read_csv(
                    exception_file,
                    low_memory=False
                )

                with st.expander(
                    "View Excluded Exceptional Transactions"
                ):

                    st.write(
                        "These exceptional bulk transactions "
                        "were excluded from routine inventory calculations."
                    )

                    st.dataframe(
                        exceptions_df,
                        use_container_width=True
                    )

            st.info(
                "Note: Recommendations use historical demand, "
                "a 7-day lead time and safety-stock assumptions. "
                "Actual available stock is not included in the dataset. "
                "Suggested order quantity is an order-up-to scenario, "
                "not a live stock shortage calculation."
            )
