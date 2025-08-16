import os
import io
import pandas as pd
import numpy as np
from datetime import timedelta
import joblib
from .config import settings

CLUSTER_MAP = {
    0: "At-Risk Customers",
    1: "Regular Customers",
    2: "VIP Customers"
}
REQUIRED_COLUMNS = [
    "InvoiceNo","StockCode","Description","Quantity","InvoiceDate","UnitPrice",
    "CustomerID","Name","Email","Country"
]

def load_models():
    scaler_path=os.path.join(settings.MODELS_DIR, settings.SCALER_FILE)
    kmeans_path = os.path.join(settings.MODELS_DIR, settings.KMEANS_FILE)
    if not (os.path.exists(scaler_path) and os.path.exists(kmeans_path)):
        raise FileNotFoundError("Missing rfm_scaler.pkl or rfm_kmeans.pkl in app/models/")
    scaler = joblib.load(scaler_path)
    kmeans = joblib.load(kmeans_path)
    return scaler, kmeans

def clean_transactions(df: pd.DataFrame) -> pd.DataFrame:
    # Keep only required columns if present; ignore extra columns
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    # Convert dtypes
    df = df.copy()
    df["InvoiceNo"] = df["InvoiceNo"].astype(str)
    df["StockCode"] = df["StockCode"].astype(str)
    df["Description"] = df["Description"].astype(str)
    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
    df["UnitPrice"] = pd.to_numeric(df["UnitPrice"], errors="coerce")
    df["CustomerID"] = pd.to_numeric(df["CustomerID"], errors="coerce").astype("Int64")
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    # Basic cleaning: drop cancelled ('C' prefix), non-positive values, nulls
    df = df[~df["InvoiceNo"].str.startswith("C", na=False)]
    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]
    df = df.dropna(subset=["CustomerID","InvoiceDate","Email"])
    # Compute line total
    df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]
    return df

def derive_customer_features(df: pd.DataFrame):
    # Snapshot date = day after max invoice date
    snapshot_date = df["InvoiceDate"].max() + timedelta(days=1)
    # Last purchase date per customer
    last_purchase = df.groupby("CustomerID")["InvoiceDate"].max().rename("Last_Purchase")
    # Recency / Frequency / Monetary
    recency = (snapshot_date - last_purchase).dt.days.rename("Recency")
    frequency = df.groupby("CustomerID")["InvoiceNo"].nunique().rename("Frequency")
    monetary = df.groupby("CustomerID")["TotalPrice"].sum().rename("Monetary")
    total_spent = monetary.rename("Total_Spent")
    days_ago = recency.rename("Last_Purchase_Days_Ago")
    # Fav category (most purchased Description by quantity; tie-break by revenue)
    prod_agg = df.groupby(["CustomerID","Description"]).agg(
        Qty=("Quantity","sum"),
        Rev=("TotalPrice","sum")
    ).reset_index().sort_values(["CustomerID","Qty","Rev"], ascending=[True,False,False], inplace=True)
    fav = prod_agg.groupby("CustomerID").first()["Description"].rename("Fav_Category")
    # Name/Email: take the most recent known pair per customer
    latest_rows = df.sort_values("InvoiceDate").groupby("CustomerID").tail(1)
    id_name = latest_rows.set_index("CustomerID")[["Name","Email"]]
    # Merge all features
    features = (
        pd.concat([last_purchase, days_ago, recency, frequency, monetary, total_spent, fav], axis=1)
        .join(id_name, how="left")
        .reset_index()
    )
    # Order columns
    cols = ["CustomerID","Name","Email","Last_Purchase","Fav_Category","Total_Spent",
            "Last_Purchase_Days_Ago","Recency","Frequency","Monetary"]
    # Sort by Monetary desc for convenience
    features = features[cols].sort_values("Monetary", ascending=False).reset_index(drop=True)
    return features, snapshot_date

def score_segments(features: pd.DataFrame, scaler, kmeans) -> pd.DataFrame:
    rfm = features[["Recency","Frequency","Monetary"]]
    # Ensure DataFrame with correct column names for sklearn to avoid warnings
    rfm_scaled = pd.DataFrame(
        scaler.transform(rfm),
        columns=rfm.columns,
        index=rfm.index
    )
    clusters = kmeans.predict(rfm_scaled)
    out = features.copy()
    out["Cluster"] = clusters
    out["Segment"] = out["Cluster"].map(CLUSTER_MAP).fillna("Unknown")
    return out

def generate_email(name, segment, last_purchase_days, fav_category):
    name = (name or "there").strip() or "there"
    fav = (fav_category or "Your favourites").strip() or "your favorites"
    # Simple templating; can be upgraded to HTML
   
    if segment == "VIP Customers":
        subject = f"Exclusive early access just for you"
        body = (
            f"Hey {name},\n\n"
            f"As one of our VIPs, enjoy early access to new arrivals in {fav}.\n"
            f"Thanks for being with us!"
        )
    elif segment == "Regular Customers":
        subject = f"A little thank-you: 15% off {fav}"
        body = (
            f"Hi {name},\n\n"
            f"You've been loving our {fav}. Here's 15% off your next order!\n"
            f"See what's new and recommended for you."
        )
    else:  # At-Risk or Unknown
        subject = f"We miss you — {fav} waiting for you"
        days = int(last_purchase_days) if pd.notna(last_purchase_days) else "many"
        body = (
            f"Hey {name},\n\n"
            f"It's been {int(days)} days since your last purchase. "
            f"Come back for 25% off {fav}!"
        )
    return subject, body