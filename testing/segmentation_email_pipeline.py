
#!/usr/bin/env python3
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

# ---------------------------
# Config / Constants
# ---------------------------
CLUSTER_MAP = {
    0: "At-Risk Customers",
    1: "Regular Customers",
    2: "VIP Customers"
}
REQUIRED_COLUMNS = [
    "InvoiceNo","StockCode","Description","Quantity","InvoiceDate","UnitPrice",
    "CustomerID","Name","Email","Country"
]

# ---------------------------
# Utility
# ---------------------------
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

def derive_customer_features(df: pd.DataFrame) -> pd.DataFrame:
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
    ).reset_index()
    prod_agg.sort_values(["CustomerID","Qty","Rev"], ascending=[True,False,False], inplace=True)
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
    features = features[cols]
    # Sort by Monetary desc for convenience
    features = features.sort_values("Monetary", ascending=False).reset_index(drop=True)
    return features, snapshot_date

def score_segments(features: pd.DataFrame, scaler, kmeans) -> pd.DataFrame:
    rfm_df = features[["Recency","Frequency","Monetary"]].copy()
    # Ensure DataFrame with correct column names for sklearn to avoid warnings
    rfm_scaled = pd.DataFrame(
        scaler.transform(rfm_df),
        columns=rfm_df.columns,
        index=rfm_df.index
    )
    clusters = kmeans.predict(rfm_scaled)
    out = features.copy()
    out["Cluster"] = clusters
    out["Segment"] = out["Cluster"].map(CLUSTER_MAP).fillna("Unknown")
    return out

def generate_email(name, segment, last_purchase_days, fav_category):
    # Simple templating; can be upgraded to HTML
    if pd.isna(name) or not str(name).strip():
        name = "there"
    fav_text = fav_category if isinstance(fav_category,str) and fav_category.strip() else "your favorites"
    if segment == "VIP Customers":
        subject = f"Exclusive early access just for you"
        body = (
            f"Hey {name},\n\n"
            f"As one of our VIPs, enjoy early access to new arrivals in {fav_text}.\n"
            f"Thanks for being with us!"
        )
    elif segment == "Regular Customers":
        subject = f"A little thank-you: 15% off {fav_text}"
        body = (
            f"Hi {name},\n\n"
            f"You've been loving our {fav_text}. Here's 15% off your next order!\n"
            f"See what's new and recommended for you."
        )
    else:  # At-Risk or Unknown
        subject = f"We miss you — {fav_text} waiting for you"
        body = (
            f"Hey {name},\n\n"
            f"It's been {int(last_purchase_days)} days since your last purchase. "
            f"Come back for 25% off {fav_text}!"
        )
    return subject, body

def send_email_gmail(to_email, subject, body, from_email, app_password):
    msg = MIMEMultipart()
    msg["From"] = from_email
    msg["To"] = to_email
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(from_email, app_password)
        server.sendmail(from_email, to_email, msg.as_string())

def main():
    parser = argparse.ArgumentParser(description="Segment customers from transactional CSV and send personalized emails.")
    parser.add_argument("--input", required=True, help="Path to input CSV with transactions + Name + Email.")
    parser.add_argument("--output", default="customers_segmented.csv", help="Where to save the customer-level output CSV.")
    parser.add_argument("--dry-run", action="store_true", help="Compute segments and save CSV, but do not send emails.")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of emails (useful for testing).")
    args = parser.parse_args()

    # Load env
    load_dotenv()
    EMAIL_USER = os.getenv("EMAIL_USER")
    EMAIL_PASS = os.getenv("EMAIL_PASS")
    if not args.dry_run and (not EMAIL_USER or not EMAIL_PASS):
        raise RuntimeError("EMAIL_USER/EMAIL_PASS not set. Create a .env or set env vars.")

    # Load pickled models
    scaler, kmeans = load_models()

    # Read & clean
    raw = pd.read_csv(args.input)
    tx = clean_transactions(raw)
    if tx.empty:
        raise ValueError("No valid transactions to process after cleaning.")

    # Aggregate
    features, snapshot_date = derive_customer_features(tx)

    # Score
    scored = score_segments(features, scaler, kmeans)

    # Save output CSV
    scored.to_csv(args.output, index=False)
    print(f"[OK] Saved segmented customers to: {args.output}")
    print(scored.head())

    # Send emails unless dry-run
    if not args.dry_run:
        sent = 0
        for _, row in scored.iterrows():
            # Basic validation
            email = str(row["Email"]).strip()
            if not email or "@" not in email:
                continue
            subject, body = generate_email(row["Name"], row["Segment"], row["Last_Purchase_Days_Ago"], row["Fav_Category"])
            try:
                send_email_gmail(email, subject, body, EMAIL_USER, EMAIL_PASS)
                sent += 1
                print(f"  -> Sent to {email} [{row['Segment']}]")
                if args.limit and sent >= args.limit:
                    break
            except Exception as e:
                print(f"  !! Failed for {email}: {e}")
        print(f"[OK] Emails attempted: {sent}")

if __name__ == "__main__":
    main()
