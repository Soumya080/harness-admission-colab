from proposer.parse import parse_reply
OLD = "You are a careful math assistant."
def test_fenced():   assert parse_reply('```\n{"op":"add","rationale":"x","new_system_prompt":"You are a careful math assistant. Check units."}\n```', OLD)
def test_garbage():  assert parse_reply("Sure! Here is a better prompt...", OLD) is None
def test_same():     assert parse_reply('{"op":"add","rationale":"x","new_system_prompt":"You are a careful math assistant."}', OLD) is None
def test_bad_op():   assert parse_reply('{"op":"replace","rationale":"x","new_system_prompt":"Something long enough here."}', OLD) is None
