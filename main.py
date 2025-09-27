import os
import httpx
import datetime
from zoneinfo import ZoneInfo
from typing import List
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException, Depends, Header
from pydantic import BaseModel, EmailStr
from sqlalchemy import create_engine, Column, Integer, String, Text, Time, Date, Boolean, JSON, func
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.exc import IntegrityError
from dotenv import load_dotenv

# --- Load Environment Variables ---
load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SENDGRID_API_KEY = os.getenv("SENDGRID_API_KEY")
CRON_SECRET_KEY = os.getenv("CRON_SECRET_KEY")
SENDER_EMAIL = os.getenv("SENDER_EMAIL")

app = FastAPI()

# --- Add this CORS middleware block ---
origins = [
    "https://dailymotivator.netlify.app", # <--- Your correct live URL
    "http://localhost",
    "http://127.0.0.1",               # Optional: for local testing
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"], # Allows all methods (GET, POST, etc.)
    allow_headers=["*"], # Allows all headers
)

# --- Database Setup (SQLAlchemy) ---
Base = declarative_base()
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

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
    # Note: The 'timestamp' column is set by default in the database (DEFAULT NOW())

# --- Pydantic Models for API data validation ---
class SubscriptionCreate(BaseModel):
    email: EmailStr
    subject: str
    preferred_time: str # e.g., "09:30"
    timezone: str

# --- FastAPI App Initialization ---
app = FastAPI()

# Dependency to get a DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Dependency to verify the cron secret key
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

# --- API Endpoints ---
@app.post("/subscribe")
def create_subscription(sub: SubscriptionCreate, db: Session = Depends(get_db)):
    """Endpoint to receive new subscriptions from the Netlify form."""
    try:
        new_sub = Subscription(
            email=sub.email,
            subject=sub.subject,
            preferred_time=datetime.time.fromisoformat(sub.preferred_time),
            timezone=sub.timezone,
            status="active"
        )
        db.add(new_sub)
        db.commit()
        db.refresh(new_sub)
        log_action(db, new_sub.id, "SUBSCRIPTION_CREATED", "API_ENDPOINT", sub.dict(), True)
        return {"message": "Subscription successful!"}
    except IntegrityError:
        # This happens if the email is a duplicate (due to the UNIQUE constraint)
        db.rollback()
        raise HTTPException(status_code=400, detail="Email address already subscribed.")
    except Exception as e:
        db.rollback()
        log_action(db, None, "SUBSCRIPTION_FAILED", "API_ENDPOINT", {"error": str(e)}, False)
        raise HTTPException(status_code=500, detail="Could not process subscription.")

@app.get("/send-reminders", dependencies=[Depends(verify_cron_secret)])
def send_reminders(db: Session = Depends(get_db)):
    """
    The main logic triggered by the scheduler. Finds and emails eligible users.
    """
    now_utc = datetime.datetime.now(datetime.timezone.utc)
    today_utc = now_utc.date()
    
    # 1. Query for active users who haven't been sent an email today.
    subs_to_check: List[Subscription] = db.query(Subscription).filter(
        Subscription.status == 'active',
        (Subscription.last_sent_date == None) | (Subscription.last_sent_date < today_utc)
    ).all()

    sent_count = 0
    for sub in subs_to_check:
        try:
            # 2. Convert user's preferred time to UTC for today.
            user_tz = ZoneInfo(sub.timezone)
            preferred_dt_local = datetime.datetime.combine(today_utc, sub.preferred_time, tzinfo=user_tz)
            preferred_dt_utc = preferred_dt_local.astimezone(datetime.timezone.utc)

            # 3. Check if it's time to send (within a 10-minute window).
            time_difference = (now_utc - preferred_dt_utc).total_seconds()
            if 0 <= time_difference < 600: # 10-minute window
                
                # 4a. Call LLM for a quote
                prompt = f"Given this subject: '{sub.subject}', return a single concise motivational quote (max 2 sentences, 50 words)."
                quote = get_motivational_quote(prompt)
                
                if not quote:
                    log_action(db, sub.id, "LLM_API_CALL_FAILED", "SCHEDULER", {"prompt": prompt}, False)
                    continue # Skip to next user

                # 4b. Send email
                email_body = f"{quote}\n\n---\nYour daily dose of motivation."
                send_success = send_email(sub.email, sub.subject, email_body)

                if send_success:
                    # 4c. Update DB and log success
                    sub.last_sent_date = today_utc
                    db.commit()
                    log_action(db, sub.id, "EMAIL_SENT", "SCHEDULER", {"quote": quote}, True)
                    sent_count += 1
                else:
                    log_action(db, sub.id, "EMAIL_SEND_FAILED", "SCHEDULER", {}, False)
        
        except Exception as e:
            # Log any unexpected errors during processing for a single user
            log_action(db, sub.id, "PROCESSING_ERROR", "SCHEDULER", {"error": str(e)}, False)

    return {"message": f"Checked {len(subs_to_check)} subscriptions. Sent {sent_count} emails."}


# --- External Service Functions ---
def get_motivational_quote(prompt: str) -> str:
    """Calls OpenAI API to get a quote."""
    try:
        with httpx.Client() as client:
            response = client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
                json={
                    "model": "gpt-3.5-turbo",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.7,
                    "max_tokens": 100,
                },
                timeout=20,
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"Error calling OpenAI: {e}")
        return None

def send_email(to_email: str, subject: str, body: str) -> bool:
    """Sends an email using the SendGrid API."""
    from sendgrid import SendGridAPIClient
    from sendgrid.helpers.mail import Mail
    
    message = Mail(
        from_email=SENDER_EMAIL,
        to_emails=to_email,
        subject=subject,
        plain_text_content=body
    )
    try:
        sg = SendGridAPIClient(SENDGRID_API_KEY)
        response = sg.send(message)
        return response.status_code in [200, 202]
    except Exception as e:
        print(f"Error calling SendGrid: {e}")
        return False
import os
import httpx
import datetime
from zoneinfo import ZoneInfo
from typing import List

from fastapi import FastAPI, HTTPException, Depends, Header
from pydantic import BaseModel, EmailStr
from sqlalchemy import create_engine, Column, Integer, String, Text, Time, Date, Boolean, JSON, func
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.exc import IntegrityError
from dotenv import load_dotenv

# --- Load Environment Variables ---
load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SENDGRID_API_KEY = os.getenv("SENDGRID_API_KEY")
CRON_SECRET_KEY = os.getenv("CRON_SECRET_KEY")
SENDER_EMAIL = os.getenv("SENDER_EMAIL")

# --- Database Setup (SQLAlchemy) ---
Base = declarative_base()
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

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
    # Note: The 'timestamp' column is set by default in the database (DEFAULT NOW())

# --- Pydantic Models for API data validation ---
class SubscriptionCreate(BaseModel):
    email: EmailStr
    subject: str
    preferred_time: str # e.g., "09:30"
    timezone: str

# --- FastAPI App Initialization ---
app = FastAPI()

# Dependency to get a DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Dependency to verify the cron secret key
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

# --- API Endpoints ---
@app.post("/subscribe")
def create_subscription(sub: SubscriptionCreate, db: Session = Depends(get_db)):
    """Endpoint to receive new subscriptions from the Netlify form."""
    try:
        new_sub = Subscription(
            email=sub.email,
            subject=sub.subject,
            preferred_time=datetime.time.fromisoformat(sub.preferred_time),
            timezone=sub.timezone,
            status="active"
        )
        db.add(new_sub)
        db.commit()
        db.refresh(new_sub)
        log_action(db, new_sub.id, "SUBSCRIPTION_CREATED", "API_ENDPOINT", sub.dict(), True)
        return {"message": "Subscription successful!"}
    except IntegrityError:
        # This happens if the email is a duplicate (due to the UNIQUE constraint)
        db.rollback()
        raise HTTPException(status_code=400, detail="Email address already subscribed.")
    except Exception as e:
        db.rollback()
        log_action(db, None, "SUBSCRIPTION_FAILED", "API_ENDPOINT", {"error": str(e)}, False)
        raise HTTPException(status_code=500, detail="Could not process subscription.")

@app.get("/send-reminders", dependencies=[Depends(verify_cron_secret)])
def send_reminders(db: Session = Depends(get_db)):
    """
    The main logic triggered by the scheduler. Finds and emails eligible users.
    """
    now_utc = datetime.datetime.now(datetime.timezone.utc)
    today_utc = now_utc.date()
    
    # 1. Query for active users who haven't been sent an email today.
    subs_to_check: List[Subscription] = db.query(Subscription).filter(
        Subscription.status == 'active',
        (Subscription.last_sent_date == None) | (Subscription.last_sent_date < today_utc)
    ).all()

    sent_count = 0
    for sub in subs_to_check:
        try:
            # 2. Convert user's preferred time to UTC for today.
            user_tz = ZoneInfo(sub.timezone)
            preferred_dt_local = datetime.datetime.combine(today_utc, sub.preferred_time, tzinfo=user_tz)
            preferred_dt_utc = preferred_dt_local.astimezone(datetime.timezone.utc)

            # 3. Check if it's time to send (within a 10-minute window).
            time_difference = (now_utc - preferred_dt_utc).total_seconds()
            if 0 <= time_difference < 600: # 10-minute window
                
                # 4a. Call LLM for a quote
                prompt = f"Given this subject: '{sub.subject}', return a single concise motivational quote (max 2 sentences, 50 words)."
                quote = get_motivational_quote(prompt)
                
                if not quote:
                    log_action(db, sub.id, "LLM_API_CALL_FAILED", "SCHEDULER", {"prompt": prompt}, False)
                    continue # Skip to next user

                # 4b. Send email
                email_body = f"{quote}\n\n---\nYour daily dose of motivation."
                send_success = send_email(sub.email, sub.subject, email_body)

                if send_success:
                    # 4c. Update DB and log success
                    sub.last_sent_date = today_utc
                    db.commit()
                    log_action(db, sub.id, "EMAIL_SENT", "SCHEDULER", {"quote": quote}, True)
                    sent_count += 1
                else:
                    log_action(db, sub.id, "EMAIL_SEND_FAILED", "SCHEDULER", {}, False)
        
        except Exception as e:
            # Log any unexpected errors during processing for a single user
            log_action(db, sub.id, "PROCESSING_ERROR", "SCHEDULER", {"error": str(e)}, False)

    return {"message": f"Checked {len(subs_to_check)} subscriptions. Sent {sent_count} emails."}


# --- External Service Functions ---
def get_motivational_quote(prompt: str) -> str:
    """Calls OpenAI API to get a quote."""
    try:
        with httpx.Client() as client:
            response = client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
                json={
                    "model": "gpt-3.5-turbo",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.7,
                    "max_tokens": 100,
                },
                timeout=20,
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"Error calling OpenAI: {e}")
        return None

def send_email(to_email: str, subject: str, body: str) -> bool:
    """Sends an email using the SendGrid API."""
    from sendgrid import SendGridAPIClient
    from sendgrid.helpers.mail import Mail
    
    message = Mail(
        from_email=SENDER_EMAIL,
        to_emails=to_email,
        subject=subject,
        plain_text_content=body
    )
    try:
        sg = SendGridAPIClient(SENDGRID_API_KEY)
        response = sg.send(message)
        return response.status_code in [200, 202]
    except Exception as e:
        print(f"Error calling SendGrid: {e}")
        return False