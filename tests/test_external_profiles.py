import sys
from pathlib import Path
import json
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "assistant_api"))
from profiles import load_profiles

def test_external_profile(tmp_path):
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs/a.md").write_text("test")
    (tmp_path / "prompt.md").write_text("prompt")
    p = dict(mode="custom", collection="custom_kb", data_dir="docs", prompt_file="prompt.md", model="test", temperature=0.2, max_tokens=800, top_k=5, max_query_length=2000, rate_limit_per_minute=5, cache_enabled=False, log_content=False)
    (tmp_path / "custom.json").write_text(json.dumps(p))
    result = load_profiles(tmp_path, {})
    assert result["custom"]["system_prompt"] == "prompt"
    assert result["custom"]["data_dir"] == tmp_path / "docs"
    with pytest.raises(ValueError):
        load_profiles(tmp_path, {"custom": {"collection": "old"}})
