from typing import Literal

from pydantic import BaseModel, Field, field_validator


class CameraCreate(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    source_type: Literal["webcam", "rtsp"]
    source: str = Field(min_length=1, max_length=2048)

    @field_validator("name", "source")
    @classmethod
    def strip_whitespace(cls, value: str) -> str:
        return value.strip()

    @field_validator("source")
    @classmethod
    def validate_source(cls, value: str) -> str:
        if not value:
            raise ValueError("Source cannot be empty")
        return value


class CameraUpdate(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    enabled: bool

    @field_validator("name")
    @classmethod
    def strip_name(cls, value: str) -> str:
        return value.strip()


class CameraResponse(BaseModel):
    id: int
    name: str
    source_type: str
    source: str
    enabled: bool
    created_at: str
