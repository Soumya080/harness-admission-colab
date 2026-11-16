"""Hand-made harnesses. python -m proposer.manual  saves: base (handicapped), planted, no-ops.
Tags: base = the START of every run (handicapped). planted = the known-real edit. noop = known-null edits."""
from harness.core import Harness
from harness.store import save_harness

CLEAN = ("You are a careful math assistant. Solve the problem step by step, "
         "then write the final answer on its own line after ####.")
HANDICAP = " Do not show any working; reply with only the final number."

# Same meaning as CLEAN, different words. READ EACH TWICE: meaning must not change.
NOOP_REWORDINGS = [
    "You are a careful mathematics assistant. Work through the problem step by step, then give the final answer on its own line after ####.",
    "Act as a careful math helper. Reason step by step and put the final answer on a separate line after ####.",
    "You are a meticulous assistant for math problems. Go step by step, and finish with the final answer on its own line after ####.",
    "As a careful math assistant, solve each problem one step at a time, then write the final answer alone on a line after ####.",
    "You help with math carefully. Think through the solution step by step and state the final answer on its own line following ####.",
]

def base():
    return Harness(CLEAN + HANDICAP, tag="base", note="handicapped start")

def planted(b):
    h = Harness(b.system_prompt.replace(HANDICAP, ""), parent=b.id, tag="planted", note="removes the handicap")
    assert h.system_prompt == CLEAN
    return h

def noops(b):
    return [Harness(p + HANDICAP, parent=b.id, tag="noop", note="rewording") for p in NOOP_REWORDINGS]

def all_manual():
    b = base()
    hs = [b, planted(b), *noops(b)]
    assert len({h.id for h in hs}) == len(hs), "two manual harnesses share an id"
    return hs

if __name__ == "__main__":
    for h in all_manual():
        save_harness(h); print(f"{h.tag:8s} {h.id}")
