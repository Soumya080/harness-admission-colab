import pytest
from runner import cache
def test_roundtrip_and_append_only(tmp_path):
    rows = [dict(harness="h", split="dev", item=1, temp=0.0, sample=0, text="x", correct=1)]
    cache.write(rows, "h", "dev", 0.0, root=tmp_path)
    assert cache.has("h", "dev", 0.0, root=tmp_path)
    with pytest.raises(FileExistsError):
        cache.write(rows, "h", "dev", 0.0, root=tmp_path)
    assert len(cache.load_all(root=tmp_path)) == 1
