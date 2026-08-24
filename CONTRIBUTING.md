# Contributing

PRs are welcome. By submitting a PR you agree to license your contribution under the MIT license.

## Setup

Rust 1.85+ and [uv](https://docs.astral.sh/uv/) for Python.

```bash
cargo fmt --all -- --check
cargo clippy --all-targets -- -D warnings
cargo test --lib --bin kryo

# CRIU integration tests (Linux, needs CRIU + usually root)
sudo -E env "PATH=$PATH" cargo test --test criu_integration -- --nocapture

cd python
uv sync
uv run ruff check .
uv run mypy src
uv run python -m unittest discover -s tests
```

## Commits and PRs

This repo uses [Conventional Commits](https://www.conventionalcommits.org/). PR titles must match, because [release-please](https://github.com/googleapis/release-please) cuts versions from them:

- `feat:` minor bump
- `fix:` patch bump
- `feat!:` / `BREAKING CHANGE:` major bump
- `docs:`, `ci:`, `chore:` do not release

One version is shared across `Cargo.toml` and every `pyproject.toml`. Do not bump versions by hand; merge the release-please PR instead.

## Release publishing

On each GitHub release the workflow uploads Linux binaries, then publishes the crate to crates.io and the Python package to PyPI.

You need:

1. Repo secret `CARGO_REGISTRY_TOKEN` (crates.io API token)
2. PyPI [trusted publisher](https://docs.pypi.org/trusted-publishers/) for `akhileshsharma99/kryo`, workflow `release.yml`
3. GitHub Actions allowed to create pull requests (Settings → Actions → General) so release-please can open the Release PR
