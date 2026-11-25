from pydantic import BaseModel, ConfigDict, Field


class GenerationRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    profile: str = Field(min_length=1, max_length=128)
    input: dict


class GenerationResponse(BaseModel):
    profile: str
    result: dict
    model: str
    usage: dict
    response_time_ms: float
