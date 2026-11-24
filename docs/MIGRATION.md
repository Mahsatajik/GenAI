# Migrating from the Kaggle notebook

The original notebook is preserved at [commit 4ced28d](https://github.com/Mahsatajik/GenAI/blob/4ced28d/google_kaggle_capstone_project.ipynb). The Python package becomes the executable source of truth.

| Notebook behavior | Python application |
| --- | --- |
| Installation and uninstallation inside cells | Package dependencies and optional UI/dev extras |
| Kaggle Secrets and fixed input/output paths | Environment variables, CLI arguments, and uploaded files |
| Automatic tuning and a hardcoded tuned-model ID | Configurable generation, embedding, and existing summary-model IDs |
| API probing and UI launch during import | Explicit CLI/UI entry points and client creation |
| Parallel lists for notes and successful summaries | Identity-bearing Summary records, including failures |
| Shared collection queried across patients | Independent run index and patient metadata filters |
| Criterion-name-only grouping | Trial plus criterion identity |
| Missing results or API errors reported as No | Unknown with error metadata |
| Free-form JSON parsing and long reasoning requests | Structured responses and concise evidence-based justifications |
| Reverse retrieval across all patients | Reverse retrieval scoped to the same patient |
| Repeated global logging and stdout capture | Standard logging and run-local progress |
| Shared output filenames | Unique run directories and collision-resistant trial/criterion paths |
| Global delete/clear action | Retained run artifacts and explicit new runs |

The original CSV column names remain supported. The application adds optional patient IDs and keeps the distinction between inclusion and exclusion criteria. Yes describes whether a criterion is present, including an exclusion criterion.

ROUGE, fuzzy matching, and cosine similarity remain evidence-alignment diagnostics. The original notebook did not establish clinical eligibility accuracy, and the migrated application makes no validated clinical performance claim.

Fine-tuning is no longer initiated as a side effect of running a screening request. Set GEMINI_SUMMARY_MODEL to an existing tuned model when supported by the configured provider.

Synthetic mode is new and uses fictional examples with literal matching. It is intended for reproducible execution and development without API credentials.
