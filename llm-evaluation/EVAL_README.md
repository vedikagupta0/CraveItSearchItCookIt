# LLM Evaluation — Crave It · Search It · Cook It

`eval\_llm.py` adds an LLM-as-judge evaluation harness on top of the existing
RAG chatbot. It imports `app.py` directly, so it runs the exact same
retrieval + generation pipeline the Space serves — no mocking.

## What it measures

|Metric|How|
|-|-|
|Faithfulness (1-5)|Groq LLM judge checks the answer is grounded only in retrieved recipe context|
|Relevancy (1-5)|Groq LLM judge checks the answer addresses the user's question|
|Completeness (0/1)|Deterministic regex check for ingredients + step-by-step instructions|
|Retrieval hit rate|Deterministic keyword check against retrieved docs (optional, per test case)|
|Latency|Wall-clock seconds per query|

## Files added

* `eval\_llm.py` — the harness
* `eval\_dataset.json` — 25 vvariety test queries

## Run it

```bash
export GROQ\_API\_KEY=your\_key\_here      # same secret the app already uses
pip install -r requirements.txt
python eval\_llm.py
```

Output: a Markdown summary table printed to stdout, plus a full
`eval\_report.json` with per-query scores and reasoning, for tracking
regressions over time (e.g. run in CI on every `app.py` change).

## 

