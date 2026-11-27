import time
import pytest
from observability import deadline, remaining

def test_provider_reserve():
    token = deadline.set(time.monotonic()+1)
    try:
        assert 0 < remaining(15, reserve=.25) <= .75
    finally:
        deadline.reset(token)

def test_provider_reserve_exhausted():
    token = deadline.set(time.monotonic()+.1)
    try:
        with pytest.raises(TimeoutError):
            remaining(15, reserve=.25)
    finally:
        deadline.reset(token)
