
import os
import joblib
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# Create folders
os.makedirs("models", exist_ok=True)
os.makedirs("data", exist_ok=True)


def create_rfm(df):

    df = df.copy()

    # Remove customers without CustomerID
    df = df.dropna(subset=["CustomerID"])

    df["InvoiceDate"] = pd.to_datetime(
        df["InvoiceDate"],
        errors="coerce"
    )

    df = df.dropna(subset=["InvoiceDate"])

    # Reference date
    reference_date = (
        df["InvoiceDate"].max() +
        pd.Timedelta(days=1)
    )

    # Calculate RFM metrics
    rfm = df.groupby("CustomerID").agg(
        Recency=(
            "InvoiceDate",
            lambda x: (reference_date - x.max()).days
        ),
        Frequency=("InvoiceNo", "nunique"),
        Monetary=("TotalSales", "sum")
    ).reset_index()

    return rfm


def train_customer_segmentation(rfm, n_clusters=5):

    features = ["Recency", "Frequency", "Monetary"]

    # Log transformation
    X = rfm[features].clip(lower=0)
    X = np.log1p(X)

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train K-Means
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10
    )

    rfm["Cluster"] = kmeans.fit_predict(X_scaled)

    # Silhouette Score evaluation
    if len(set(rfm["Cluster"])) > 1 and len(set(rfm["Cluster"])) < len(rfm):
        score = silhouette_score(X_scaled, rfm["Cluster"])

        print("\nMODEL EVALUATION")
        print("-------------------------")
        print(f"Number of Clusters: {n_clusters}")
        print(f"Silhouette Score: {score:.4f}")
    else:
        print("\nSilhouette Score cannot be calculated.")

    # Calculate cluster characteristics
    centroids = pd.DataFrame(
        scaler.inverse_transform(kmeans.cluster_centers_),
        columns=features
    )

    centroids = np.expm1(centroids)

    # Business-based cluster ranking
    centroids["BusinessScore"] = (
        -centroids["Recency"].rank(pct=True) +
        centroids["Frequency"].rank(pct=True) +
        centroids["Monetary"].rank(pct=True)
    )

    ranked_clusters = (
        centroids["BusinessScore"]
        .sort_values(ascending=False)
        .index.tolist()
    )

    segment_names = [
        "Champions",
        "Loyal Customers",
        "Potential Loyalists",
        "At Risk Customers",
        "Lost Customers"
    ]

    cluster_mapping = {
        cluster: segment_names[index]
        for index, cluster in enumerate(ranked_clusters)
    }

    rfm["Segment"] = rfm["Cluster"].map(cluster_mapping)

    # Save model and scaler
    joblib.dump(
        {
            "model": kmeans,
            "scaler": scaler,
            "features": features,
            "cluster_mapping": cluster_mapping
        },
        "models/customer_segmentation.pkl"
    )

    return rfm


def segment_summary(rfm):

    summary = (
        rfm.groupby("Segment")
        .agg(
            Customers=("CustomerID", "count"),
            AverageRecency=("Recency", "mean"),
            AverageFrequency=("Frequency", "mean"),
            AverageSpending=("Monetary", "mean"),
            TotalRevenue=("Monetary", "sum")
        )
        .reset_index()
    )

    return summary


if __name__ == "__main__":

    print("Loading RetailPulse dataset...")

    df = pd.read_csv(
        "data/retail_cleaned.csv",
        low_memory=False
    )

    print("Creating RFM analysis...")

    rfm = create_rfm(df)

    print("Training K-Means model...")

    rfm = train_customer_segmentation(
        rfm,
        n_clusters=5
    )

    # Save customer segments
    rfm.to_csv(
        "data/customer_segments.csv",
        index=False
    )

    # Generate summary
    summary = segment_summary(rfm)

    summary.to_csv(
        "data/segment_summary.csv",
        index=False
    )

    print("\nCUSTOMER SEGMENTATION COMPLETED!")

    print("\nCustomer Segment Summary:")
    print(summary.to_string(index=False))

    print("\nFiles generated:")
    print("1. data/customer_segments.csv")
    print("2. data/segment_summary.csv")
    print("3. models/customer_segmentation.pkl")
