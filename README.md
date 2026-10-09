# FGL-Admission: a skill-write gate against environmental attacks on Hermes skills

Pre-registered experiment (2026-10-09). Proposal and results in Russian: [REPORT_ru.md](REPORT_ru.md),
pre-registration: [docs/PREREG_fgl_hermes_env_attack.md](docs/PREREG_fgl_hermes_env_attack.md).

## Threat
The attacker controls neither the user prompt nor the skill file. A tip sits in the agent's
environment (README, tool output) during a benign task. Hermes's background skill curator
(`agent/background_review.py`, "Be ACTIVE ... tool-usage pattern emerged ... Capture it")
promotes it into the persistent `SKILL.md`; a later session reuses the skill and the harm fires.

## What Hermes v0.19.0 already has
- `tools/skills_guard.py`: regex scanner; for agent-created skills it is **off by default**
  (`skills.guard_agent_created: False`), and it has no signatures for semantic drift.
- `tools/write_approval.py`: write gate by policy, not by content.
- No separate trusted spec: the spec *is* the mutating `SKILL.md`, so runtime grounding of a plan
  against the current skill cannot see the drift.

## Proposal: FGL-Admission
Move Factual Grounding Layer (FGL) from execution time to write time. Before
`skill_manage create/edit/patch`, every new directive in the v1 -> v2 diff is grounded against
trusted session evidence only: the user's request and corrections, the previously admitted version,
the verified outcome. Environment content is not evidence. A directive that expands scope without
support (`NOT_ENOUGH_INFO`) or removes a safeguard (`REFUTES`) sends the write to Hermes's staging
queue instead of the library. Built on the lifecycle view of three papers: Dynamic Agent Skills
(arXiv:2607.10113, admission + governance stages), Agent Skill Evaluation and Evolution
(arXiv:2606.11435, evaluate every update), SkillsVote (arXiv:2605.18401, evidence-gated updates).

## Results (12 SkillHarm SMP scenarios, one model plays agent, curator, gate and judge)

| Metric | gpt-4.1-mini | gpt-5.1 |
|---|---|---|
| Curator promoted the environment tip into SKILL.md | 50% | 58% |
| Caught at admission, Hermes skills_guard (even if enabled) | 0% | 0% |
| Caught at admission, FGL-Admission | 100% | 100% |
| Reuse ASR, no gate / FGL-Admission | 50% / 0% | 33% / 0% |
| FGL at runtime vs the current (mutated) skill | 0% | 0% |
| False block on user-originated benign updates | 17% | 33% |
| Harmless environment tips sent to staging | 82% | 83% |

False blocks are curator additions the user never asked for (row limits, recursive cleanup); the
write is staged, not lost. N=12, our harness rather than an end-to-end Hermes CLI run: Hermes v0.19
has no hook on `skill_manage`, which is exactly the proposed integration point.

## Run
```
pip install -r requirements.txt
export OPENAI_API_KEY=...
python -m benchmark.fgl_hermes_env_attack --model gpt-4.1-mini
```
Real Hermes pieces are imported from the installed `hermes-agent` package when available
(`agent.background_review._SKILL_REVIEW_PROMPT`, `tools.skills_guard.scan_skill`); attack payloads
are the 12 SMP mutations in `benchmark/skillharm_smp.py` (reproduction of SkillHarm, arXiv:2606.02540).
