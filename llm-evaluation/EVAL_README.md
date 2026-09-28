# LLM Evaluation — Crave It · Search It · Cook It

`eval_llm.py` is an **LLM-as-a-judge evaluation harness** for the recipe RAG chatbot. It imports the app directly (`from app.app import chain, vs_text, ...`), so it scores the same retrieval and generation pipeline the app uses, not a mock.

Latest results (30 cases, one run each) are in [`../eval-results/`](../eval-results) and summarized in the main [README](../README.md).

## Metrics

| Metric | Type | How it is measured |
| --- | --- | --- |
| **Faithfulness (1–5)** | LLM judge | Is the answer grounded only in the retrieved recipe context? |
| **Relevancy (1–5)** | LLM judge | Does the answer address the user's question? |
| **Completeness (0/1)** | Deterministic (regex) | Does the answer contain both an ingredient list and steps? |
| **Retrieval hit rate** | Deterministic | Does any retrieved document contain an expected keyword? |
| **Citation accuracy** | Deterministic | Do the `[Recipe Name]` tags in the answer match retrieved recipe titles? |
| **Latency** | Deterministic | Wall-clock time for retrieval + generation per query |

## Pipeline

```text
Query → Lingua language detection + translation to English
      → FAISS text retrieval (top 5, all-MiniLM-L6-v2)
      → Answer generation (Groq, gpt-oss-120b, temperature 0)
      → Deterministic checks: completeness, retrieval hit, citations
      → Judge (Groq, gpt-oss-20b): faithfulness + relevancy as JSON
```

| Component | Model |
| --- | --- |
| Answer generation | `openai/gpt-oss-120b` |
| Judge | `openai/gpt-oss-20b` (override with `EVAL_JUDGE_MODEL`) |
| Text embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Image embeddings (app only) | OpenCLIP ViT-B-32 |

The generator and the judge are different models but from the same family, so judge scores may be somewhat lenient toward the generator's style.

## Files

```text
llm-evaluation/
├── eval_llm.py        # the harness
├── eval_dataset.json  # 30 test cases
└── EVAL_README.md     # this file
eval-results/
├── eval_report.json   # full per-case traces and summary
└── partial_results.csv
```

`eval_llm.py` writes `eval_report.json` and `partial_results.csv` next to itself (`llm-evaluation/`) by default. The committed copies live in `eval-results/`.

## Dataset (30 cases)

| Group | Count | Purpose |
| --- | --- | --- |
| Baseline | 25 | Dishes present in the corpus, including 5 non-English queries (Hindi, French, Spanish, Russian, Chinese) |
| Not in dataset | 2 | `beef rendang`, `churros with chocolate dipping sauce`. The correct answer is "No relevant recipes found in context." |
| Ambiguous near-miss | 3 | `sushi egg roll`, `seafood carbonara`, `mango cheesecake`. The dish family exists but the exact dish does not, so these test honest refusal vs. stitching a made-up answer |

Each case is a JSON object with a `query` and optional `expected_keywords`.

## Setup and running

Install dependencies and set your Groq key (via `.env` or the environment):

```bash
pip install -r requirements.txt
export GROQ_API_KEY=your_key_here          # PowerShell: $env:GROQ_API_KEY="your_key_here"
```

Run from the **project root**:

```bash
python llm-evaluation/eval_llm.py                        # all cases, 1 run each
python llm-evaluation/eval_llm.py --limit 3              # quick smoke test
python llm-evaluation/eval_llm.py --repeats 3            # average 3 runs per case
python llm-evaluation/eval_llm.py --dataset my_cases.json --out my_report.json
```

| Flag | Default | Meaning |
| --- | --- | --- |
| `--dataset` | `llm-evaluation/eval_dataset.json` | Test cases to run |
| `--out` | `llm-evaluation/eval_report.json` | Where the JSON report is written |
| `--limit` | all | Only run the first N cases |
| `--repeats` | 1 | Runs per case, averaged (generation and judging are non-deterministic) |

To stay under Groq's free-tier rate limit, the script sleeps 180 s between cases (`SLEEP_BETWEEN_CASES_S`), so a full 30-case run takes roughly 90 minutes. The judge retries after a 429. Progress is appended to `partial_results.csv` so an interrupted run is not lost. Judge limits can be tuned with the `EVAL_JUDGE_*` environment variables in the script.

## Output

The script prints a Markdown summary table and writes `eval_report.json` containing the summary, per-case aggregates, and raw per-run data (answer, retrieved docs, scores, judge reasoning, latency, invalid citation tags).

## Deterministic checks

- **Completeness:** a regex check for ingredient and step markers. Correct refusals ("No relevant recipes found") score 0 by design, and non-English answers can be missed (e.g. "Ingrédients"), so read the rate alongside the refusal cases.
- **Retrieval hit rate:** only defined for cases with `expected_keywords`. Cases with an empty list (the refusal cases) are not a meaningful retrieval test.
- **Citation accuracy:** extracts `[Tag]` citations from the answer and compares them with the retrieved recipe titles, so a fabricated but well-formatted citation is still caught.

## Isolation

Each case calls the app's `chain` directly and does not append to the chat history, so cases do not influence each other.

## Limitations

- One run per case in the committed results, so no variance estimate. Use `--repeats 3` or more for regression tracking.
- LLM-judge scores can vary between runs and the judge shares a model family with the generator.
- 30 cases is a small set. Treat the numbers as a sanity check, not a benchmark.
- For comparable runs, keep the dataset, retrieval settings (`TOP_K_TEXT`), embedding model, generation model, judge model, prompt and temperature constant.
