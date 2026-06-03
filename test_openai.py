# test_openai.py
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

try:
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input="Test embedding"
    )
    print(f"✅ Success! Embedding dimensions: {len(response.data[0].embedding)}")
    print(f"   First 5 values: {response.data[0].embedding[:5]}")
except Exception as e:
    print(f"❌ Error: {str(e)}")
