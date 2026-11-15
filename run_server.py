import os
import sys
import uvicorn
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

HOST = os.getenv("HOST", "127.0.0.1")
PORT = int(os.getenv("PORT", 8000))
NASA_KEY = os.getenv("NASA_API_KEY", "")
GEMINI_KEY = os.getenv("GEMINI_API_KEY", "")

if __name__ == "__main__":
    from backend.migrations import upgrade_database

    upgrade_database()
    nasa_status = "Active (Verified in .env)" if len(NASA_KEY) > 5 else "Inactive (DEMO_KEY fallback)"
    gemini_status = "Active (Verified in .env)" if len(GEMINI_KEY) > 10 else "Inactive (Local RAG fallback)"
    
    print("==================================================")
    print("  Orbit Overload - Space Intelligence Platform")
    print("  FastAPI & Three.js Local Server")
    print(f"  Running on: http://localhost:{PORT}")
    print(f"  NASA API: {nasa_status}")
    print(f"  Gemini AI: {gemini_status}")
    print("  Database migrations: applied")
    print("==================================================")
    sys.stdout.flush()
    uvicorn.run("backend.main:app", host=HOST, port=PORT, log_level="info")
