from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import pandas as pd
from io import StringIO, BytesIO
from .config import settings
from .schemas import SegmentResponse, SegmentSummary, CustomerOut
from .pipeline import (
    load_models, clean_transactions, derive_customer_features,
    score_segments, generate_email
)
from .emailer import send_email_gmail

app = FastAPI(
    title="Customer Segmentation API",
    description="Upload a transactional CSV, compute RFM-based segments, and (optionally) send personalized emails.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

SCALER, KMEANS = load_models()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/segment", response_model=SegmentResponse)
async def segment_customers(
    file: UploadFile = File(..., description="csv with transactions"),
    send_emails: bool = Query(False, description="Set true to send emails after segmentation"),
    limit: Optional[int] = Query(None, description="Max emails to send (for testing)"),
):
    
    raw_bytes = await file.read()
    try:
        df = pd.read_csv(BytesIO(raw_bytes))
    except Exception:
        df = pd.read_csv(BytesIO(raw_bytes), encoding="latin-1")

    tx = clean_transactions(df)
    features, snapshot_date = derive_customer_features(tx)

    scored = score_segments(features, SCALER, KMEANS)

    summary = scored["Segment"].value_counts().to_dict()

    rows = []

    for _, r in scored.iterrows():
        rows.append({
            "CustomerID": int(r["CustomerID"]),
            "Name": r.get("Name"),
            "Email": r.get("Email"),
            "Last_Purchase": pd.to_datetime(r["Last_Purchase"]).strftime("%Y-%m-%d"),
            "Fav_Category": r.get("Fav_Category"),
            "Total_Spent": float(r["Total_Spent"]),
            "Last_Purchase_Days_Ago": int(r["Last_Purchase_Days_Ago"]),
            "Recency": int(r["Recency"]),
            "Frequency": int(r["Frequency"]),
            "Monetary": float(r["Monetary"]),
            "Cluster": int(r["Cluster"]),
            "Segment": r["Segment"]
        })

    sent = 0
    if send_emails:
        if not (settings.EMAIL_USER and settings.EMAIL_PASS):
            return JSONResponse(
                status_code=400,
                content={"error": "Email_USER/EMAIL_PASS not configured on server."},

            
            )
        for r in rows:
            email = r["Email"]
            if not email or "@" not in email:
                continue
            subject, body = generate_email(
                r["Name"], r["Segment"], r["Last_Purchase_Days_Ago"], r["Fav_Category"]

            )
            try:
                send_email_gmail(email, subject, body, settings.EMAIL_USER, settings.EMAIL_PASS)
                sent += 1
                if limit and sent >= limit:
                    break
            except Exception as e:
                print(f"Email failed for {email}: {e}")

    out_path = f"customers_segmented.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)

    return{
        "snapshot_date": pd.to_datetime(snapshot_date).strftime("%Y-%m-%d"),
        "summary": {"counts": summary},
        "rows": rows,
    }