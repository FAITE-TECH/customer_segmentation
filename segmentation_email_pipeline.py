import os
import argparse
import pandas as pd
import numpy as np
from datetime import timedelta
from dotenv import load_dotenv
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import smtplib
import joblib
import re

CLUSTER_MAP = {
    0: "At-Risk Customers",
    1: "Regular Customers",
    2: "VIP Customers"
}

REQUIRED_COLUMNS = [
    "InvoiceNo", "StockCode", "Description", "Quantity", "InvoiceDate", "UnitPrice",
    "CustomerID", "Name", "Email", "Country"
]


def debug(df, name):
    print(f"[DEBUG] {name}: shape={df.shape}")
    return df


def load_models(scaler_path="rfm_scaler.pkl", kmeans_path="rfm_kmeans.pkl"):
    if not os.path.exists(scaler_path) or not os.path.exists(kmeans_path):
        raise FileNotFoundError("rfm_scaler.pkl and/or rfm_kmeans.pkl not found in working directory.")
    scaler = joblib.load(scaler_path)
    kmeans = joblib.load(kmeans_path)
    return scaler, kmeans


def clean_transactions(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    df = df.copy()
    df["InvoiceNo"] = df["InvoiceNo"].astype(str)
    df["StockCode"] = df["StockCode"].astype(str)
    df["Description"] = df["Description"].astype(str)
    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
    df["UnitPrice"] = pd.to_numeric(df["UnitPrice"], errors="coerce")
    df["CustomerID"] = pd.to_numeric(df["CustomerID"], errors="coerce").astype("Int64")
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")


    df = df[~df["InvoiceNo"].str.startswith("C", na=False)]
    df = df[(df["Quantity"] > 0) & (df["UnitPrice"]>0)]
    df = df.dropna(subset=["CustomerID", "InvoiceDate", "Email"])


    df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]
    return df


def derive_customer_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Timestamp]:

    snapshot_date = df["InvoiceDate"].max() + timedelta(days =1)

    last_purchase = df.groupby("CustomerID")["InvoiceDate"].max().rename("Last_Purchase")

    recency = (snapshot_date - last_purchase).dt.days.rename("Recency")
    frequency = df.groupby("CustomerID")["InvoiceNo"].nunique().rename("Frequency")
    monetary = df.groupby("CustomerID")["TotalPrice"].sum().rename("Monetary")
    total_spent = monetary.rename("Total_Spent")
    days_ago = recency.rename("Last_Purchase_Days_Ago")

    prod_agg = df.groupby(["CustomerID","Description"]).agg(
        Qty=("Quantity", "sum"),
        Rev=("TotalPrice", "sum")
    ).reset_index()
    prod_agg.sort_values(["CustomerID", "Qty", "Rev"], ascending=[True,False,False], inplace=True)
    fav = prod_agg.groupby("CustomerID").first()["Description"].rename("Fav_Category")

    latest_rows = df.sort_values("InvoiceDate").groupby("CustomerID").tail(1)
    id_name = latest_rows.set_index("CustomerID")[["Name", "Email"]]


    features = (
        pd.concat([last_purchase, days_ago, recency, frequency, monetary, total_spent, fav], axis=1)
        .join(id_name, how="left")
        .reset_index()
    )


    cols = ["CustomerID", "Name", "Email", "Last_Purchase","Fav_category","Total_Spent",
            "Last_Purchase_Days_Ago", "Recency", "Frequency", "Monetary"]
    features = features[cols]

    features = features.sort_values("Monetary", ascending=False).reset_index(drop=True)
    return features, snapshot_date



