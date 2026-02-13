import os

from supabase import Client, create_client

url: str | None = os.environ.get("SUPABASE_URL")
key: str | None = os.environ.get("SUPABASE_KEY")
supabase: Client = create_client(str(url), str(key))
