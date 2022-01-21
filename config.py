from pydantic import BaseSettings
import os

ROOT_DIR = os.path.join("..")


class MlApiSettings(BaseSettings):
    ML_API_USERNAME: str = ''
    ML_API_PASSWORD: str = ''

    class Config:
        env_file = ".env"