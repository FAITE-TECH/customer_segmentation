from pydantic import BaseModel, EmailStr
from typing import List, Dict, Any, Optional

class CustomerOut(BaseModel):
    CustomerID: int
    Name: Optional[str] = None
    Email: Optional[EmailStr] = None
    Last_Purchase: str
    Fav_Category: str | None = None
    Total_Spent: float
    Last_Purchase_Days_ago: int
    Recency: int
    Frequency: int
    Monetary: float
    Cluster: int
    Segment: str

class SegmentSummary(BaseModel):
    counts: Dict[str, int]

class SegmentResponse(BaseModel):
    snapshot_date:str
    summary: SegmentSummary
    rows: List[CustomerOut]