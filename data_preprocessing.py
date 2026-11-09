
import os
import io
import zipfile
import requests
import pandas as pd

DATA_DIR = "data"

RAW_FILE = os.path.join(DATA_DIR, "retail_original.csv")
CLEAN_FILE = os.path.join(DATA_DIR, "retail_cleaned.csv")

ZIP_URL = (
    "https://archive.ics.uci.edu/"
    "static/public/352/online+retail.zip"
)



def download_dataset():

    import os
    import io
    import zipfile
    import requests
    import pandas as pd

    os.makedirs(DATA_DIR, exist_ok=True)

    required = {
        "InvoiceNo", "StockCode", "Description",
        "Quantity", "InvoiceDate", "UnitPrice",
        "CustomerID", "Country"
    }

    # Check whether existing CSV is complete
    if os.path.exists(RAW_FILE):

        df = pd.read_csv(
            RAW_FILE,
            encoding="ISO-8859-1"
        )

        if required.issubset(set(df.columns)):
            print("Complete existing dataset found!")
            return df

        print("Existing dataset is incomplete.")
        print("Downloading complete dataset...")

    url = (
        "https://archive.ics.uci.edu/"
        "static/public/352/online+retail.zip"
    )

    response = requests.get(url, timeout=180)
    response.raise_for_status()

    with zipfile.ZipFile(
        io.BytesIO(response.content)
    ) as archive:

        excel_files = [
            name for name in archive.namelist()
            if name.lower().endswith(".xlsx")
        ]

        if not excel_files:
            raise FileNotFoundError(
                "Excel dataset not found inside ZIP"
            )

        with archive.open(excel_files[0]) as file:
            df = pd.read_excel(file, engine="openpyxl")

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Downloaded dataset missing columns: {missing}"
        )

    df.to_csv(RAW_FILE, index=False)

    print("Complete dataset downloaded!")
    print("Columns:", df.columns.tolist())
    print("Records:", len(df))

    return df


def clean_dataset(df):

    df = df.copy()

    df = df.drop_duplicates()

    df["InvoiceDate"] = pd.to_datetime(
        df["InvoiceDate"],
        errors="coerce"
    )

    df["Quantity"] = pd.to_numeric(
        df["Quantity"],
        errors="coerce"
    )

    df["UnitPrice"] = pd.to_numeric(
        df["UnitPrice"],
        errors="coerce"
    )

    df = df.dropna(
        subset=[
            "InvoiceNo",
            "StockCode",
            "InvoiceDate",
            "Quantity",
            "UnitPrice"
        ]
    )

    df = df[
        ~df["InvoiceNo"].astype(str).str.startswith("C")
    ]

    df = df[
        (df["Quantity"] > 0) &
        (df["UnitPrice"] > 0)
    ].copy()

    df["TotalSales"] = (
        df["Quantity"] * df["UnitPrice"]
    )

    df["Year"] = df["InvoiceDate"].dt.year
    df["Month"] = df["InvoiceDate"].dt.month
    df["Day"] = df["InvoiceDate"].dt.day
    df["DayOfWeek"] = df["InvoiceDate"].dt.dayofweek

    df.to_csv(CLEAN_FILE, index=False)

    print("\nData cleaning completed!")
    print("Cleaned records:", len(df))

    return df


if __name__ == "__main__":

    df = download_dataset()

    retail = clean_dataset(df)

    print("\nDataset Preview:")
    print(retail.head())

    print("\nDataset Shape:")
    print(retail.shape)

    print("\nMissing Values:")
    print(retail.isnull().sum())
