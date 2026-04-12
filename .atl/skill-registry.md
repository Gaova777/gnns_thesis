# Skill Registry — gnns_thesis

Generated: 2026-04-11
Stack: Python 3.12, PyTorch, PyTorch Geometric, Optuna, MLflow

## User Skills

| Skill | Trigger Context |
|-------|----------------|
| `branch-pr` | Creating PRs, preparing branches for review |
| `issue-creation` | Filing GitHub issues, bug reports, feature requests |
| `judgment-day` | Deep adversarial review before merging; when high-confidence review is needed |
| `skill-creator` | Creating new AI agent skills or documenting patterns |
| `skill-registry` | After installing/removing skills, updating this registry |
| `sdd-init` | Initialize SDD context for this project |
| `sdd-explore` | Investigate ideas or codebase areas before committing |
| `sdd-propose` | Draft a change proposal with intent and scope |
| `sdd-spec` | Write specifications with requirements and scenarios |
| `sdd-design` | Create technical design with architecture decisions |
| `sdd-tasks` | Break down a change into implementation tasks |
| `sdd-apply` | Implement tasks from the change |
| `sdd-verify` | Validate implementation against specs |
| `sdd-archive` | Close a change and persist final state |

## SDD Skills (orchestrator-managed)

`sdd-explore`, `sdd-propose`, `sdd-spec`, `sdd-design`, `sdd-tasks`, `sdd-apply`, `sdd-verify`, `sdd-archive`, `sdd-onboard`

## Project Conventions

- Package manager: uv (pyproject.toml)
- Python: 3.12
- ML stack: PyTorch ≥2.6 + PyTorch Geometric ≥2.7 + Optuna ≥4.0 + MLflow ≥2.0
- Source layout: `src/` package with sub-modules (models, data, training, explainability, stability, analysis, balancing)
- Configs: YAML files in `configs/` (main + per-machine variants: A/B/C)
- Scripts: Pipeline entrypoints in `scripts/` (run_training, run_explain, run_stability, run_full_pipeline, merge_results)
- No test runner: pytest not configured — Strict TDD Mode disabled
- Git: conventional commits, no AI attribution in commit messages

## Compact Rules

### Python scientific ML conventions
- Use `uv` for dependency management (not pip directly)
- MLflow for experiment tracking (`mlflow.log_*`)
- Optuna for hyperparameter optimization
- PyTorch Geometric data objects (`Data`, `DataLoader`) for graphs
- Follow existing module structure under `src/` when adding new code
- Configs live in `configs/` as YAML — don't hardcode experiment params in scripts

### PR & Issue workflow
- Use `branch-pr` skill for all PR creation
- Use `issue-creation` skill for all GitHub issues
- Follow conventional commits (no AI attribution in commit messages)

### Distributed execution context
- Experiments run across 3 machines (A/B/C), each with separate config variants
- Results are merged via `scripts/merge_results.py`
- Machine launchers: `scripts/run_machine{A,B,C}.{sh,bat}`
