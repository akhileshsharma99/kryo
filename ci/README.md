# CI

| Path | Role |
|------|------|
| [actions/](actions/) | Composite actions (Python lint, path filters) |
| [benchmark/](benchmark/) | Lambda Cloud GPU benches (orchestrator, VM bootstrap, result formatting) |
| [release-please/](release-please/) | Versioning config for the Release workflow |

GPU benches are **not** on pull requests. The Release workflow dispatches them after a GitHub Release exists. See [benchmark/README.md](benchmark/README.md).
