from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    OPENAI_API_KEY: str
    DB_URL: str
    EMBEDDING_MODEL: str = "text-embedding-ada-002"

    class Config:
        env_file = ".env"

settings = Settings()
