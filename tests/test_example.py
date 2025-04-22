import warnings
import pytest 


warnings.filterwarnings(action="ignore", message="unclosed", category=ResourceWarning)

def test_example():
    assert 1==1

@pytest.mark.asyncio
async def test_example_async():
    assert 1==1