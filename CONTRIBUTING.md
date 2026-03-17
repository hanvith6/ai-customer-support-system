# Contributing

Contributions to the AI Customer Support System are welcome.

## Getting Started

1. Fork the repository.
2. Create a feature branch from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. Install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

## Development Workflow

1. Make your changes in the feature branch.
2. Run the test suite to ensure nothing is broken:
   ```bash
   pytest tests/test_api.py -v
   ```
3. Start the server locally to verify your changes:
   ```bash
   python -m backend.app
   ```

## Code Style

- Follow existing project conventions.
- Use type hints for function signatures.
- Keep functions focused and well-named.
- Add structured logging for new pipeline stages.

## Pull Requests

1. Ensure all tests pass before submitting.
2. Write a clear PR description explaining what changed and why.
3. Keep PRs focused on a single change.

## Reporting Issues

Open a GitHub issue with:

- A clear description of the problem.
- Steps to reproduce.
- Expected vs. actual behavior.

## Adding New Intents

To add new customer support intent categories:

1. Edit `model/intents.json` with new tags, patterns, and responses.
2. Delete `model/data.pth` to trigger automatic retraining.
3. Run the server and verify the new intents are classified correctly.
