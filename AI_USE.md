# AI Use Disclosure

## Research and coding contributions

I used OpenAI Codex throughout this project. The research ideas, questions, experimental priorities, and responsibility for data collection were mine. I supplied data and context, identified sources, and directed the collection and comparisons. Codex performed most of the coding, including reconstruction and evaluation pipelines, data retrieval and matching utilities, tests, repository consolidation, and documentation. AI therefore also assisted with the implementation of data collection; it was not solely a writing or debugging tool.

I set the scientific constraints: preserve the original author's methods, avoid holdout leakage, compare identical eligible populations, distinguish raw and calibrated results, and prioritize classification accuracy and batter/pitch-group tendencies over precise XYZ deviations.

## Important prompts and directions

The following summarize my main prompts rather than reproduce the full conversation:

- Inspect and reproduce the original author's complete validation and calibration pipeline, documenting unavailable models and artifacts.
- Collect Baseball Savant's pitch-level whiff-tail labels, preserve their provenance, match them to our Statcast data, and audit missing values, reporting caps, and selection bias.
- Compare global, batter-median, and EV50 references using development-only calibration and unchanged held-out evaluation.
- Consolidate the experiments into one configurable pipeline while preserving all tested methods, classifications, populations, and evaluation metrics.
- Produce exactly four outputs: `metrics.csv`, `players.parquet`, `swings.parquet`, and `report.md`, preserving the exact 974,886-swing population.
- Add paired bootstrap confidence intervals clustered by batter for improvements over the author and null baselines, especially classification-rate and within-batter pitch-group contrast MAE.
- Preserve source datasets, frozen parameters, the original author's submodule, and the scratch notebook, and verify changes against existing regression benchmarks.

## Written report

AI was also used to clarify wording and improve readability in the written report. It was particularly helpful with LaTeX formatting and presentation. This assistance supported the communication of the research; the research ideas and direction remained mine.
