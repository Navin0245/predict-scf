# Contributing to scf-surrogate

## Commit convention
Conventional Commits: https://www.conventionalcommits.org

Format: <type>(<scope>): <short description>

Types: feat | fix | docs | test | refactor | chore | perf

Examples:
    feat(joints): add KT joint gap validation
    fix(api): return 422 when beta exceeds DNV upper bound
    test(pipeline): add integration test for labeller filter

## Branch naming
    feature/<short-description>
    fix/<short-description>
    docs/<short-description>

## Pull request checklist
- [ ] pytest tests/ passes
- [ ] mypy --strict src/ passes
- [ ] ruff check src/ passes
- [ ] New code has docstrings and type hints
- [ ] CHANGELOG.md updated
- [ ] ADR written if architectural decision was made
