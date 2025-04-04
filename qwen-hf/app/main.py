import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from app.api.router import router
from app.models.qwen_omni_wrapper import QwenOmniWrapper
from app.dependencies import model_instance

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
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    global model_instance
    try:
        logger.info("Initializing Qwen 2.5 Omni model...")
        from app.dependencies import model_instance as deps_model_instance
        global model_instance
        model_instance = QwenOmniWrapper()
        deps_model_instance = model_instance
        logger.info("Model initialization complete")
    except Exception as e:
        logger.error(f"Failed to initialize model: {str(e)}")
        raise RuntimeError(f"Model initialization failed: {str(e)}")

@app.get("/health")
async def health_check():
    from app.dependencies import model_instance
    if model_instance is not None and model_instance.is_ready:
        return {"status": "healthy", "model_loaded": True}
    return {"status": "unhealthy", "model_loaded": False}

# Include API router
app.include_router(router, prefix="/api")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
