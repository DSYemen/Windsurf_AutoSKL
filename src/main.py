from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.api import ml_routes
from src.auth import routes as auth_routes
from src.database import create_tables
from fastapi.staticfiles import StaticFiles
from src.config import settings
from fastapi.templating import Jinja2Templates
from fastapi import Request

app = FastAPI(title="AutoSKL API", version="2.0.0")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static_files")

# Initialize templates
templates = Jinja2Templates(directory="src/frontend/templates")
templates.env.globals["static_url"] = lambda path: f"/static/{path}"

# Include routers
app.include_router(auth_routes.router, prefix="/api/v1/auth", tags=["auth"])
app.include_router(ml_routes.router, prefix="/api/v1/ml", tags=["machine-learning"])

# Create database tables on startup
@app.on_event("startup")
async def startup_event():
    create_tables()
    
    # Create required directories
    settings.MODEL_STORE_PATH.mkdir(parents=True, exist_ok=True)
    settings.UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
    settings.EXPORT_FOLDER.mkdir(parents=True, exist_ok=True)

# Frontend routes
@app.get("/dashboard")
async def dashboard(request: Request):
    # Mock stats data (replace with real data from database later)
    stats = {
        "total_models": 42,
        "model_growth": 15,
        "active_experiments": 8,
        "total_predictions": 12500,
        "prediction_growth": 25,
        "system_health": 98,
        "model_performance_data": {
            "labels": ["Model A", "Model B", "Model C", "Model D"],
            "accuracy": [0.95, 0.87, 0.92, 0.89],
            "f1_score": [0.94, 0.86, 0.91, 0.88]
        },
        "prediction_volume_data": {
            "labels": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
            "values": [1200, 1900, 1500, 1800, 2100, 1700, 1600]
        }
    }
    return templates.TemplateResponse("dashboard.html", {"request": request, "stats": stats})

@app.get("/models")
async def models_page(request: Request):
    return templates.TemplateResponse("models.html", {"request": request})

@app.get("/experiments")
async def experiments_page(request: Request):
    return templates.TemplateResponse("experiments.html", {"request": request})

@app.get("/")
async def root():
    return {
        "message": "Welcome to AutoSKL API",
        "version": "2.0.0",
        "docs_url": "/docs"
    }
