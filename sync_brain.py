import os
import glob
import requests
from supabase import create_client, Client
from dotenv import load_dotenv

# 1. Setup Supabase
load_dotenv()
url: str = os.getenv("SUPABASE_URL")
key: str = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(url, key)

# 2. Point to your Obsidian Vault
VAULT_PATH = r"C:\Users\diner\.openclaw\workspace\Dalpazor-Vault"

def get_local_embedding(text):
    """Turns Markdown text into a 768-dimension vector using local Ollama."""
    url = "http://localhost:11434/api/embeddings"
    payload = {
        "model": "nomic-embed-text",
        "prompt": text
    }
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        return response.json()["embedding"]
    else:
        raise Exception(f"Ollama Error: {response.text}")

def sync_vault():
    print("🧠 Initiating Sovereign Brain Sync (Ollama)...")
    
    # Clear the old knowledge table first
    try:
        supabase.table("dpc_knowledge").delete().neq("file_name", "placeholder").execute()
    except Exception as e:
        pass 
    
    # 3. Find all Markdown files
    search_pattern = os.path.join(VAULT_PATH, "**/*.md")
    md_files = glob.glob(search_pattern, recursive=True)
    
    if not md_files:
        print("⚠️ No markdown files found. Check your VAULT_PATH.")
        return

    # 4. Read, Vectorize Locally, and Upload
    for file_path in md_files:
        file_name = os.path.basename(file_path)
        
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            
        if not content.strip():
            continue
            
        print(f"⚡ Vectorizing: {file_name}...")
        
        try:
            embedding = get_local_embedding(content)
            
            # Push to Supabase
            supabase.table("dpc_knowledge").insert({
                "file_name": file_name,
                "content": content,
                "embedding": embedding
            }).execute()
            print(f"  └─ Uploaded {file_name} successfully.")
        except Exception as e:
            print(f"  └─ ❌ Failed to upload {file_name}: {str(e)}")

    print("\n✅ Sovereign Brain Sync Complete. No external APIs used.")

if __name__ == "__main__":
    sync_vault()