"""
Application configuration loaded from environment variables via pydantic-settings.

Settings include Redis connection details, Ollama base URL, and feature flags
for enabling or disabling individual detection strategies.
"""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Top-level application settings."""

    # Redis
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0

    # Ollama
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "mistral:latest"

    # Great Expectations
    ge_expectation_suite_path: str = "expectations/suite.json"

    # One-Class SVM
    oc_svm_training_data_path: str = "data/normal_samples.npy"
    oc_svm_model_path: str = "models/svm.joblib"
    oc_svm_kernel: str = "rbf"
    oc_svm_nu: float = 0.1
    oc_svm_gamma: float | str = 0.001
    svm_llm_threshold: float = 0.5

    # Caching
    detection_cache_ttl_seconds: int = 300

    # Feature flags
    enable_ge_strategy: bool = True
    enable_svm_strategy: bool = True
    enable_llm_strategy: bool = True

    class Config:
        env_file = ".env"


settings = Settings()
