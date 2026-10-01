# Contributing and Push Guidelines

## One-time setup
```bash
git clone https://github.com/TheRollerBlader/frc-vision-scout.git && cd frc-vision-scout
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install
cp .env.example .env   # fill in your own keys; never commit .env
```

## Golden rules
1. **Never push directly to `main`.** All changes go through a pull request.
2. **Never commit** secrets, `.env`, CAD files, match video, datasets, or model weights. Link to them or host them elsewhere (see `data/README.md`).
3. **Every PR must pass CI** (lint, format, type check, tests) before merging.
4. **Keep PRs small** and focused on one thing.

## Branch naming
`type/short-description`, for example:
- `feat/onshape-export`
- `fix/botsort-id-swap`
- `docs/data-card`
- `exp/hue-augmentation-off` (experiments)

Types: `feat`, `fix`, `docs`, `refactor`, `test`, `chore`, `exp`.

## Commit messages (Conventional Commits)
`type(scope): summary in present tense`, for example:
- `feat(ingest): add STEP to glTF converter`
- `fix(track): prevent alliance switching between tracks`
- `docs(adr): add ADR for BoT-SORT over ByteTrack`

## Daily workflow
```bash
git switch main && git pull
git switch -c feat/my-change
# ...work...
ruff check . --fix && ruff format . && pytest
git add <specific files>        # avoid `git add .` blindly
git commit -m "feat(scope): what changed"
git push -u origin feat/my-change
# open a Pull Request on GitHub
```

## Before opening a PR
- [ ] `ruff check .`, `ruff format --check .`, and `pytest` pass locally
- [ ] No secrets or large/binary files staged (`git status`, `git diff --staged`)
- [ ] Docs, configs, and CHANGELOG updated if behavior changed
- [ ] For model or data changes: metrics before and after are in the PR description

## Review and merge
- At least 1 review from the project lead (see CODEOWNERS).
- Use **Squash and merge** to keep history clean.
- Delete the branch after merging.

## Releases
Semantic versioning (`v0.1.0`). Tag from `main`, add a CHANGELOG entry, publish a GitHub Release.

## If you accidentally commit a secret
Rotate/revoke the key immediately, then tell the project lead. Deleting the commit is not enough; the key must be considered leaked.

## Credit
All contributions are made under the project's LICENSE. Contributors are listed in NOTICE.
