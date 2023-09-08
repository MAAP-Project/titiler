"""app settings"""

from pydantic_settings import BaseSettings


class MosaicSettings(BaseSettings):
    """Application settings"""

    backend: str = "file://"
    host: str = "/tmp"
    format: str = ".json.gz"  # format will be ignored for dynamodb backend

    class Config:
        """model config"""

        env_prefix = "MOSAIC_"
        env_file = ".env"
