"""Trusted, server-side generation profiles; independent of retrieval profiles."""
import json
from pathlib import Path
from typing import Annotated

from jsonschema import Draft202012Validator
from pydantic import BaseModel, ConfigDict, Field


class GenerationProfile(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    profile: Annotated[str, Field(pattern=r'^[a-z][a-z0-9_]{0,127}$')]
    version: str = Field(min_length=1)
    model: str = Field(min_length=1)
    prompt_file: str
    schema_file: str
    temperature: float = Field(default=0, ge=0, le=2)
    max_tokens: int = Field(default=500, gt=0, le=4096)
    max_input_length: int = Field(default=8000, gt=0, le=100000)
    timeout_seconds: float = Field(default=15, gt=0, le=60)
    rate_limit_per_minute: int = Field(default=10, gt=0)
    log_content: bool = False


def load_generation_profiles(directory):
    if not directory:
        return {}
    root = Path(directory).resolve()
    if not root.is_dir():
        raise ValueError('Generation profiles directory does not exist')
    profiles = {}
    for path in sorted(root.glob('*.json')):
        try:
            profile = GenerationProfile.model_validate_json(path.read_text(encoding='utf-8-sig'))
            prompt = (root / profile.prompt_file).read_text(encoding='utf-8-sig').strip()
            schema = json.loads((root / profile.schema_file).read_text(encoding='utf-8-sig'))
            Draft202012Validator.check_schema(schema)
            if not prompt or schema.get('type') != 'object':
                raise ValueError('Expected nonempty prompt and object schema')
            if profile.profile in profiles:
                raise ValueError('Duplicate generation profile')
            # No external references: validation must never fetch remote schemas.
            def check_refs(node):
                if isinstance(node, dict):
                    for key, value in node.items():
                        if key in ('$ref', '$dynamicRef') and (not isinstance(value, str) or not value.startswith('#')):
                            raise ValueError('External schema references are forbidden')
                        check_refs(value)
                elif isinstance(node, list):
                    for value in node:
                        check_refs(value)
            check_refs(schema)
            profiles[profile.profile] = (profile, prompt, schema)
        except Exception as exc:
            raise ValueError(f'Invalid generation profile: {path.name}') from exc
    return profiles
