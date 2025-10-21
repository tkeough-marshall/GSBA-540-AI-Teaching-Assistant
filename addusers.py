import os
import csv
from tkinter import Tk, filedialog
from supabase import create_client, Client
from dotenv import load_dotenv
from tqdm import tqdm  # progress bar

# --- Load environment variables ---
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

COMMON_PASSWORD = "FTMBA2027!" # common password for all users

# --- GUI file picker ---
Tk().withdraw()
csv_path = filedialog.askopenfilename(
    title="Select CSV with student emails",
    filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
)

if not csv_path:
    print("No file selected.")
    exit(0)

# --- Read emails ---
with open(csv_path, "r", newline="", encoding="utf-8") as f:
    reader = csv.reader(f)
    emails = [row[0].strip() for row in reader if row and row[0].strip()]

print(f"\nLoaded {len(emails)} emails from {os.path.basename(csv_path)}")
print("Starting user creation...\n")

# --- Create users with progress bar ---
success, failed = 0, 0

for email in tqdm(emails, desc="Creating users", unit="user"):
    try:
        supabase.auth.admin.create_user({
            "email": email,
            "password": COMMON_PASSWORD,
            "email_confirm": True,
            "user_metadata": {"course": "GSBA540"}
        })
        success += 1
    except Exception as e:
        failed += 1
        tqdm.write(f"❌ Failed: {email} — {e}")

print(f"\n✅ Done. {success} succeeded, {failed} failed.")