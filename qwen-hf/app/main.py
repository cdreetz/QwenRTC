import os
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from app.api.router import router
from app.models.qwen_omni_wrapper import QwenOmniWrapper
from app.dependencies import model_instance, initialize_model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_MODEL_TYPE = os.getenv("MODEL_TYPE", "qwen")

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
    try:
        model_type = DEFAULT_MODEL_TYPE
        logger.info("Initializing {model_type} model...")
        
        # Initialize the model
        initialize_model(model_type)
        
        logger.info("Model initialization complete {model_type}")
    except Exception as e:
        logger.error(f"Failed to initialize model: {str(e)}")
        raise RuntimeError(f"Model initialization failed: {str(e)}")

@app.get("/health")
async def health_check():
    from app.dependencies import model_instance
    if model_instance is not None and model_instance.is_ready:
        return {"status": "healthy", "model_loaded": True}
    return {"status": "unhealthy", "model_loaded": False}

@app.get("/live")
async def live_check():
    return {"status": "healthy"}

# Include API router
app.include_router(router, prefix="/api")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
