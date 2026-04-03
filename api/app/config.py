from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=(".env", "../.env"), env_prefix="APP_", extra="ignore")

    data_dir: Path = Path("data")
    workspace_dir: Path = Path("data/workspaces")
    memory_dir: Path = Path("data/memories")
    artifacts_dir: Path = Path("artifacts")
    cors_origins: str = "http://localhost:3000,http://127.0.0.1:3000"
    model_name: str = "distilbert-base-uncased"
    default_agent_model: str = Field(
        default="gpt-5.4-mini",
        validation_alias=AliasChoices("APP_AGENT_MODEL", "AGENT_MODEL", "APP_DEFAULT_AGENT_MODEL", "DEFAULT_AGENT_MODEL"),
    )
    responses_generation_model: str = Field(
        default="gpt-5.4-mini",
        validation_alias=AliasChoices(
            "APP_RESPONSES_GENERATION_MODEL",
            "RESPONSES_GENERATION_MODEL",
            "APP_GENERATION_MODEL",
            "GENERATION_MODEL",
        ),
    )
    max_sequence_length: int = 256
    batch_size: int = 4
    epochs: int = 3
    learning_rate: float = 2e-5
    generated_examples_per_label: int = 24
    eval_holdout_ratio: float = 0.2
    database_url: str = Field(
        default="",
        validation_alias=AliasChoices("APP_DATABASE_URL", "DATABASE_URL"),
    )
    runloop_api_key: str | None = Field(default=None, validation_alias=AliasChoices("APP_RUNLOOP_API_KEY", "RUNLOOP_API_KEY"))
    openai_api_key: str | None = Field(default=None, validation_alias=AliasChoices("APP_OPENAI_API_KEY", "OPENAI_API_KEY"))

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


settings = Settings()
