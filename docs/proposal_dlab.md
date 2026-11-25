# Proposal: Are the prompts an optimizer keeps really better?

*Soumy Sawant · Summer@EPFL 2027 · for Robert West's Data Science Lab (dlab)*

## Question
Iterative prompt optimizers keep a candidate (a prompt, a prompt-model checkpoint, a textual reward) when it
scores higher on a validation set. When candidates are close and the validation set is small, which of
those choices are real improvements, and do the chosen candidates keep their gain on fresh data and under shift?

## Why it matters, with evidence
TRPrompt (Nica et al., 2025) selects its best checkpoint and its optimal textual reward by comparing candidates on a
10% validation split, about 100 questions on GSMHard. In my own setting (Qwen2.5-3B on GSM8K, 30 prompt edits from a 7B
proposer, 1,000 optimization runs replayed from real outputs; [results](../README.md),
`results/selection_resolution.json`, `results/claimed_vs_real.csv`):

- Good prompts differ by a median of **1.8 points**; a 100-item paired comparison resolves only about **±6 points**
  and picks the worse of two prompts that differ by ≥2 points on a 400-item audit **30%** of the time.
- Greedy keep-if-better acceptance **claimed +112 points** of improvement where the audit confirmed **+69**; after the
  first real fix it claimed +55, of which about 6 were real. An anytime-valid gate (PACE) claimed what the audit confirmed.

My setting differs from TRPrompt's (a smaller model; prompt edits rather than a trained prompt model). Whether the
same gap appears in TRPrompt's selection steps is the open question.

## Plan (12 weeks)
| Weeks | Work | Output |
|---|---|---|
| 1–2 | Reproduce TRPrompt's loop on GSMHard at small scale; log every candidate's per-question outcomes on the validation split and on a held-out audit split. | Paired outcome matrices for checkpoints and rewards |
| 3–5 | At each selection step, compare argmax selection with a paired anytime-valid gate (PACE) and McNemar: claimed vs audited gain, and how often the selected candidate is truly better. | Claimed-vs-real table for TRPrompt-style selection |
| 6–8 | Transfer: evaluate selected vs rejected candidates on MATH (and GSM8K). Does validation-selected gain survive a shift? Split the gap into selection noise and distribution shift. | Transfer table with paired intervals |
| 9–10 | Cost: how many validation questions does each gate need to make the same decisions (sequential stopping)? | Cost–accuracy curve |
| 11–12 | Write-up; release code and outcome matrices. | Short paper draft and public repository |

## What already exists
A tested pipeline (vLLM executor, proposer, cached outcome matrices, greedy/McNemar/PACE gates, replay simulator, 20 tests):
[github.com/Soumya080/harness-admission](https://github.com/Soumya080/harness-admission).

## Risks and fallbacks
- **TRPrompt is too expensive to reproduce at full scale.** Fall back to fewer iterations and a smaller prompt model; the
  selection question only needs the candidates' outcomes, not a fully trained system.
- **No real gap appears.** That is a positive result for TRPrompt's design and worth reporting, with the minimum detectable effect stated.
- **The gate is too conservative.** Report the trade-off between false keeps and missed improvements explicitly.
