from runner.extract import extract_gsm8k_answer as ex
def test_hash():        assert ex("work...\n#### 1,234") == 1234.0
def test_negative():    assert ex("#### -7") == -7.0
def test_boxed():       assert ex("so \\boxed{42}") == 42.0
def test_last_number(): assert ex("first 3 then 18") == 18.0
def test_none():        assert ex("no idea") is None
