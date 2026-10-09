import os
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):

    aws_region: str = "us-east-1"
    anthropic_api_key: str | None = os.getenv("ANTHROPIC_API_KEY")
    bedrock_model_id: str = (
        "eu.anthropic.claude-sonnet-4-5-20250929-v1:0"
    )

    model_config = SettingsConfigDict(
        env_file=".env"
    )

settings = Settings()
