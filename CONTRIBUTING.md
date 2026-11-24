# Contributing

Use English for code, comments, documentation, issues, and commit messages. Keep clinical notes, credentials, vector indexes, and generated reports outside Git.

Install the UI and development extras, then run Ruff, the unittest suite, and the synthetic CLI demo. Add regression coverage for changes to identity handling, evidence validation, failure states, or export formats.

New model providers should implement the interface in providers.py. Keep external calls at explicit runtime entry points. For retrieval changes, verify that document and query embeddings use the appropriate task type and that all queries remain patient-scoped.

For UI changes, exercise file upload, validation, processing, and download with synthetic data. Check narrow and desktop layouts. Use real screenshots of the application in documentation.
