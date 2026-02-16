# Contributing to CourtIQ

## Branching Strategy
- Primary working branch: dev
- Create feature branches for all new work:
  - feature/<name>-<short-description>
  - Example: feature/erwin-model-weighted-avg

## Development Workflow
1. Pull the latest dev
2. Create a feature branch
3. Make changes
4. Commit with a clear message
5. Push your branch
6. Open a Pull Request into dev

## Commit Message Style
Use short, descriptive messages:
- Add Streamlit dashboard skeleton
- Fix predict endpoint validation
- Improve model weighted last N games

## Pre-PR Checklist
Before opening a Pull Request:
- API runs locally: uvicorn api.main:app --reload
- /docs loads successfully
- /predict works for at least one player
- No secrets or tokens are committed

## Role Focus Areas
- Gregory: Backend, deployment, repository structure
- Erwin: Data pipeline, model tuning, evaluation
- Heather: Frontend UI, documentation, presentation materials
