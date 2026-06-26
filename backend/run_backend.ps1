$env:PYTHONIOENCODING = "utf-8"
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload --port 8000
