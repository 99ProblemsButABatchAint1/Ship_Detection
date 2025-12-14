from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routes.health import router as health_router
from app.routes.detect import router as detect_router

from app.deps import detector 

app = FastAPI(title="Ship Detection API", version="0.1.0")

# Dev-friendly CORS (later restrict origins)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",  # Vite dev server
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def startup():
    detector.load()

app.include_router(health_router)
app.include_router(detect_router)
