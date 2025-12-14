npm install
  
cd backend  
python -m venv .venv  
.\.venv\Scripts\Activate.ps1  
pip install --upgrade pip  
pip install -r requirements  
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000  
  
[ShipDetector] Loaded classifier + segmenter  
Application startup complete.  
  
Repo root:  
npm run dev  
http://localhost:5173  
  
http://127.0.0.1:8000/docs  
