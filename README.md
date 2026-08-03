# Anamnesis AGI Benchmark Runner

This repository runs reproducible benchmark evaluations for the frozen Anamnesis AGI Candidate.

## Scientific rules

- Public benchmark runs are evidence about capability, not proof of AGI.
- Every run records the model provider, model ID, dataset commit, task IDs, prompts, raw predictions, exact-match scores, errors, and limitations.
- The public ARC-AGI-2 evaluation set has contamination risk and cannot satisfy the candidate's independent/private-test requirement.
- No score may be relabeled as independent evidence when the developer controls the harness.

Initial target: a rate-limited ARC-AGI-2 public-evaluation pilot using the candidate's solver → adversarial critic → revision loop.
