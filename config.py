from pydantic import BaseSettings


class Settings(BaseSettings):
    api_username: str = ''
    api_password: str = ''

    class Config:
        env_file = ".env"
