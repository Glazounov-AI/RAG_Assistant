import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'assistant_api'))
from config import KNOWLEDGE_BASES
from query_router import route


def test_portfolio_count_alias():
    assert route('Сколько проектов в портфолио?', KNOWLEDGE_BASES['portfolio']['catalog'])[0] == 'count'
