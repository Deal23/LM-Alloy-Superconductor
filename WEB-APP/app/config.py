import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class Config:
    """Application configuration"""

    # Paths
    PROJECT_ROOT: str
    MODEL_DIR: str
    DATA_DIR: str
    RESULTS_DIR: str
    UPLOADS_DIR: str

    # Flask
    SECRET_KEY: str
    MAX_CONTENT_LENGTH: int = 16 * 1024 * 1024  # 16MB

    # Model defaults
    DEFAULT_MODEL: str = "ET"
    DEFAULT_TEST_SIZE: float = 0.2
    DEFAULT_N_ESTIMATORS: int = 300

    # Task settings
    TASK_TIMEOUT: int = 3600  # 1 hour
    MAX_WORKERS: int = 4

    # Reliability scoring weights
    RELIABILITY_STD_WEIGHT: float = 0.35
    RELIABILITY_PROXIMITY_WEIGHT: float = 0.30
    RELIABILITY_MODEL_WEIGHT: float = 0.25
    RELIABILITY_PLAUSIBILITY_WEIGHT: float = 0.10

    @classmethod
    def from_env(cls) -> "Config":
        """Load from environment variables with defaults"""
        project_root = os.environ.get(
            "PROJECT_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        return cls(
            PROJECT_ROOT=project_root,
            MODEL_DIR=os.environ.get(
                "MODEL_DIR", os.path.join(project_root, "ml", "models")
            ),
            DATA_DIR=os.environ.get("DATA_DIR", os.path.join(project_root, "data")),
            RESULTS_DIR=os.environ.get(
                "RESULTS_DIR", os.path.join(project_root, "results")
            ),
            UPLOADS_DIR=os.environ.get(
                "UPLOADS_DIR", os.path.join(project_root, "data", "uploads")
            ),
            SECRET_KEY=os.environ.get("SECRET_KEY", "dev-secret-change-in-prod"),
        )


# Global config instance
config = Config.from_env()
