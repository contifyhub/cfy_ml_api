from pydantic import BaseSettings
import os

ROOT_DIR = os.path.join("..")


class MlApiSettings(BaseSettings):
    api_username: str = ''
    api_password: str = ''

    class Config:
        env_file = ".env"