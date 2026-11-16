from proposer.manual import all_manual, CLEAN, HANDICAP
def test_tags_and_ids():
    hs = all_manual()
    tags = [h.tag for h in hs]
    assert tags.count("base") == 1 and tags.count("planted") == 1 and tags.count("noop") >= 5
    assert len({h.id for h in hs}) == len(hs)
def test_base_is_handicapped_and_noops_keep_handicap():
    hs = all_manual()
    assert HANDICAP in hs[0].system_prompt
    assert all(HANDICAP in h.system_prompt for h in hs if h.tag == "noop")
    assert [h for h in hs if h.tag == "planted"][0].system_prompt == CLEAN
