import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from app.api.router import router
from app.models.qwen_omni_wrapper import QwenOmniWrapper

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Qwen 2.5 Omni WebRTC API",
    description="API for real-time inference with Qwen 2.5 Omni multimodal model",
    version="0.1.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development; restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model instance (initialized on startup)
model_instance = None

@app.on_event("startup")
async def startup_event():
    global model_instance
    try:
        logger.info("Initializing Qwen 2.5 Omni model...")
        model_instance = QwenOmniWrapper()
        logger.info("Model initialization complete")
    except Exception as e:
        logger.error(f"Failed to initialize model: {str(e)}")
        raise RuntimeError(f"Model initialization failed: {str(e)}")

@app.get("/health")
async def health_check():
    if model_instance is not None and model_instance.is_ready:
        return {"status": "healthy", "model_loaded": True}
    return {"status": "unhealthy", "model_loaded": False}

# Include API router
app.include_router(router, prefix="/api")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
