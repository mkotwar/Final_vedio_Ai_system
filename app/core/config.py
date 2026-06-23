"""Configuration module for the AI Video Search Engine.

This module defines the central configuration settings using pydantic-settings,
loading variables from environment variables and an optional .env file.
"""

import os
from pathlib import Path
from typing import Literal
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Determine the project root directory (video-search-engine/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    """Application-wide settings loaded from environment variables and/or .env file."""

    model_config = SettingsConfigDict(
        env_file=os.path.join(PROJECT_ROOT, ".env"),
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    # General App Settings
    APP_NAME: str = Field(default="AI Video Search Engine", description="Name of the application")
    ENV: Literal["development", "production", "testing"] = Field(
        default="development", description="Current execution environment"
    )
    DEBUG: bool = Field(default=True, description="Enable or disable debug mode")
    HOST: str = Field(default="127.0.0.1", description="Host binding address")
    PORT: int = Field(default=8000, description="Port number to bind the application")

    # Logging Settings
    LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO", description="Global logging level"
    )
    LOG_FORMAT: Literal["text", "json"] = Field(
        default="text", description="Format of log outputs"
    )

    # VLM Model Settings
    VLM_ENGINE_TYPE: Literal["native_hf", "native_vllm", "mock"] = Field(
        default="native_hf", description="Which VLM engine backend to use"
    )
    QWEN_MODEL_ID: str = Field(
        default="Qwen/Qwen2.5-VL-7B-Instruct", description="VLM model identifier"
    )
    MOCK_MODEL: bool = Field(
        default=False,
        description="Enable mock model response mode for fast testing and local development",
    )
    BATCH_SIZE: int = Field(default=4, description="Frame processing batch size")
    QWEN_MAX_NEW_TOKENS: int = Field(
        default=256,
        description="Max new tokens to generate during Qwen VLM inference",
    )

    # Adaptive Frame Sampling Settings
    ENABLE_ADAPTIVE_SAMPLING: bool = Field(
        default=True,
        description="Enable adaptive frame sampling to filter visually similar frames",
    )
    ENABLE_MOTION_WINDOWING: bool = Field(
        default=True,
        description="Enable dense motion windowing preprocessing pass to filter out static video segments entirely",
    )
    MOTION_THRESHOLD_PERCENT: float = Field(
        default=0.01,
        description="Percentage of pixels (0.0 to 1.0) that must change to trigger a motion frame",
    )
    MOTION_CONSECUTIVE_FRAMES: int = Field(
        default=3,
        description="Number of consecutive motion frames needed to open a motion window",
    )
    PRE_EVENT_BUFFER_SECONDS: int = Field(
        default=2,
        description="Padding added before a motion window begins",
    )
    POST_EVENT_BUFFER_SECONDS: int = Field(
        default=2,
        description="Padding added after a motion window ends",
    )
    MAX_FRAMES_PER_WINDOW: int = Field(
        default=30,
        description="Maximum length (in seconds) of a single motion window",
    )
    SSIM_THRESHOLD: float = Field(
        default=0.92,
        description="SSIM similarity threshold below which frames are accepted",
    )
    HISTOGRAM_THRESHOLD: float = Field(
        default=0.25,
        description="Histogram correlation difference threshold above which frames are accepted",
    )
    MOTION_THRESHOLD: float = Field(
        default=0.15,
        description="Pixel difference threshold ratio above which frames are accepted",
    )
    TEMPORAL_INTERVAL_SECONDS: float = Field(
        default=10.0,
        description="Temporal safeguard interval to retain a frame regardless of similarity metrics",
    )
    EVENT_SIMILARITY_THRESHOLD: float = Field(
        default=0.70,
        description="Similarity ratio threshold above which consecutive frames are grouped into the same event",
    )
    EVENT_CONTINUITY_THRESHOLD: float = Field(
        default=0.35,
        description="Weighted semantic continuity score below which an event cluster is split",
    )
    EVENT_CONTEXT_WINDOW: int = Field(
        default=5,
        description="Number of recent frames used when evaluating continuity",
    )
    MAX_EVENT_DURATION_SECONDS: int = Field(
        default=15,
        description="Safety boundary: Maximum duration in seconds for a single event cluster",
    )

    # Qdrant & Embeddings Settings
    QDRANT_HOST: str = Field(default="localhost", description="Qdrant service host")
    QDRANT_PORT: int = Field(default=6333, description="Qdrant service port")
    QDRANT_COLLECTION: str = Field(default="video_events", description="Collection name for events")
    EMBEDDING_MODEL_ID: str = Field(
        default="BAAI/bge-m3", description="Text embedding model identifier"
    )
    USE_LOCAL_QDRANT: bool = Field(
        default=True,
        description="Fallback to local in-memory Qdrant instance for development and tests",
    )

    # Narrative Reasoner Settings
    GEMINI_API_KEY: str = Field(
        default="",
        description="API Key for Gemini models (Narrative Reasoner)",
    )
    NARRATIVE_MODEL_ID: str = Field(
        default="gemini-2.5-flash",
        description="LLM to use for narrative building",
    )

    # Directory Paths (Relative to project root, resolved to absolute paths)
    DATA_DIR: Path = Field(
        default=PROJECT_ROOT / "data",
        description="Path to the base data directory",
    )
    VIDEOS_DIR: Path = Field(
        default=PROJECT_ROOT / "data" / "videos",
        description="Path to the video storage directory",
    )
    FRAMES_DIR: Path = Field(
        default=PROJECT_ROOT / "data" / "frames",
        description="Path to the extracted frames storage directory",
    )
    METADATA_DIR: Path = Field(
        default=PROJECT_ROOT / "data" / "metadata",
        description="Path to the metadata storage directory",
    )
    LOGS_DIR: Path = Field(
        default=PROJECT_ROOT / "data" / "logs",
        description="Path to the logs directory",
    )
    EVENTS_DIR: Path = Field(
        default=PROJECT_ROOT / "data" / "events",
        description="Path to the events storage directory",
    )

    def __init__(self, **values):
        super().__init__(**values)
        # Convert relative paths to absolute paths relative to PROJECT_ROOT if needed
        self.DATA_DIR = (PROJECT_ROOT / self.DATA_DIR).resolve()
        self.VIDEOS_DIR = (PROJECT_ROOT / self.VIDEOS_DIR).resolve()
        self.FRAMES_DIR = (PROJECT_ROOT / self.FRAMES_DIR).resolve()
        self.METADATA_DIR = (PROJECT_ROOT / self.METADATA_DIR).resolve()
        self.LOGS_DIR = (PROJECT_ROOT / self.LOGS_DIR).resolve()
        self.EVENTS_DIR = (PROJECT_ROOT / self.EVENTS_DIR).resolve()


# Singleton settings instance
settings = Settings()
