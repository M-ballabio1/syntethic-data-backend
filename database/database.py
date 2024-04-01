import os
from sqlalchemy import Column, Integer, String, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import os
from datetime import datetime
from supabase import Client, create_client

# Initialize Supabase client
supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_KEY")

supabase = Client( 
	supabase_url = supabase_url,
    supabase_key = supabase_key
)

# Function to write transaction data to the database
def write_transaction(model_id: str, method: str, url: str, status_code: int):
    timestamp = datetime.utcnow().isoformat()
    data = {'model_id': model_id, 'method': method, 'url': url, 'timestamp': timestamp, 'status_code': status_code}
    response = supabase.table('transactions').upsert(data, returning='minimal').execute()
    return response.data

# Function to retrieve all transaction data from the database
def get_transactions():
    response = supabase.table('transactions').select('*').execute()
    return response.data