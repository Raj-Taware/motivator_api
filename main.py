import os
import httpx
import datetime
from zoneinfo import ZoneInfo
from typing import List
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr
from sqlalchemy import create_engine, Column, Integer, String, Text, Time, Date, Boolean, JSON
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.exc import IntegrityError

# Load environment variables first
load_dotenv()

# --- FastAPI App and CORS Middleware Initialization ---
# Initialize the app before anything else
app = FastAPI()

origins = [
    "https://dailymotivator.netlify.app",
    "https://www.dailymotivator.netlify.app",
    "http://localhost",
    "http://127.0.0.1",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Environment Variable Loading ---
DATABASE_URL = os.getenv("DATABASE_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SENDGRID_API_KEY = os.getenv("SENDGRID_API_KEY")
CRON_SECRET_KEY = os.getenv("CRON_SECRET_KEY")
SENDER_EMAIL = os.getenv("SENDER_EMAIL")

# --- Database Setup (SQLAlchemy) ---
Base = declarative_base()
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# --- SQLAlchemy Table Models ---
class Subscription(Base):
    __tablename__ = "subscriptions"
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    subject = Column(Text, nullable=False)
    preferred_time = Column(Time, nullable=False)
    timezone = Column(String, nullable=False)
    last_sent_date = Column(Date, nullable=True)
    status = Column(String, default="active", nullable=False)

class ActionLog(Base):
    __tablename__ = "action_logs"
    id = Column(Integer, primary_key=True, index=True)
    subscription_id = Column(Integer, nullable=True)
    action_type = Column(String, nullable=False)
    actor = Column(String, nullable=False)
    details = Column(JSON, nullable=True)
    success = Column(Boolean, nullable=False)

# --- Pydantic Models for API data validation ---
class SubscriptionCreate(BaseModel):
    email: EmailStr
    subject: str
    preferred_time: str
    timezone: str

# --- Dependency Functions ---
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

async def verify_cron_secret(x_cron_secret: str = Header(None)):
    if not x_cron_secret or x_cron_secret != CRON_SECRET_KEY:
        raise HTTPException(status_code=403, detail="Invalid cron secret key")
    return True

# --- Logging Helper ---
def log_action(db: Session, sub_id: int, action: str, actor: str, details: dict, success: bool):
    log_entry = ActionLog(
        subscription_id=sub_id,
        action_type=action,
        actor=actor,
        details=details,
        success=success
    )
    db.add(log_entry)
    db.commit()

# --- External Service Functions ---
def get_motivational_quote(prompt: str) -> str:
    try:
        with httpx.Client() as client:
            response = client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
                json={"model": "gpt-3.5-turbo", "messages": [{"role": "user", "content": prompt}], "temperature": 0.7, "max_tokens": 100},
                timeout=20,
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"Error calling OpenAI: {e}")
        return None

def send_email(to_email: str, subject: str, body: str) -> (bool, int):
    from sendgrid import SendGridAPIClient
    from sendgrid.helpers.mail import Mail
    message = Mail(from_email=SENDER_EMAIL, to_emails=to_email, subject=subject, plain_text_content=body)
    try:
        sg = SendGridAPIClient(SENDGRID_API_KEY)
        response = sg.send(message)
        return (response.status_code in [200, 202], response.status_code)
    except Exception as e:
        status_code = getattr(e, 'status_code', 500)
        return (False, status_code)

# --- API Endpoints ---
@app.post("/subscribe")
def create_subscription(sub: SubscriptionCreate, db: Session = Depends(get_db)):
    try:
        new_sub = Subscription(email=sub.email, subject=sub.subject, preferred_time=datetime.time.fromisoformat(sub.preferred_time), timezone=sub.timezone)
        db.add(new_sub)
        db.commit()
        db.refresh(new_sub)
        log_action(db, new_sub.id, "SUBSCRIPTION_CREATED", "API_ENDPOINT", sub.dict(), True)
        return {"message": "Subscription successful!"}
    except IntegrityError:
        db.rollback()
        raise HTTPException(status_code=400, detail="Email address already subscribed.")
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Could not process subscription: {str(e)}")

@app.get("/send-reminders", dependencies=[Depends(verify_cron_secret)])
def send_reminders(db: Session = Depends(get_db)):
    now_utc = datetime.datetime.now(datetime.timezone.utc)
    today_utc = now_utc.date()
    subs_to_check: List[Subscription] = db.query(Subscription).filter(Subscription.status == 'active', (Subscription.last_sent_date == None) | (Subscription.last_sent_date < today_utc)).all()
    sent_count = 0
    for sub in subs_to_check:
        try:
            user_tz = ZoneInfo(sub.timezone)
            preferred_dt_local = datetime.datetime.combine(today_utc, sub.preferred_time, tzinfo=user_tz)
            preferred_dt_utc = preferred_dt_local.astimezone(datetime.timezone.utc)
            time_difference = (now_utc - preferred_dt_utc).total_seconds()
            if 0 <= time_difference < 600:
                quote = get_motivational_quote(f"Given this subject: '{sub.subject}', return a single concise motivational quote (max 2 sentences, 50 words).") or "Here is your daily note as requested."
                send_success, status_code = send_email(sub.email, sub.subject, f"{quote}\n\n---\nYour daily dose of motivation.")
                if send_success:
                    sub.last_sent_date = today_utc
                    db.commit()
                    log_action(db, sub.id, "EMAIL_SENT", "SCHEDULER", {"quote": quote, "status_code": status_code}, True)
                    sent_count += 1
                else:
                    if status_code == 400:
                        sub.status = 'invalid_email'
                        db.commit()
                        log_action(db, sub.id, "EMAIL_SEND_FAILED_INVALID", "SCHEDULER", {"status_code": status_code}, False)
                    else:
                        log_action(db, sub.id, "EMAIL_SEND_FAILED", "SCHEDULER", {"status_code": status_code}, False)
        except Exception as e:
            log_action(db, sub.id, "PROCESSING_ERROR", "SCHEDULER", {"error": str(e)}, False)
    return {"message": f"Checked {len(subs_to_check)} subscriptions. Sent {sent_count} emails."}