# CraveIt, Search It, Cook It

A chat-based recipe assistant that accepts **different language queries**, retrieves the closest matching recipes by meaning, surfaces visually similar dish photos, and
generates a grounded, chef-style answer — with every factual claim tagged back to its source recipe.

## Evaluation dataset (30 cases)

The eval set (`eval_dataset.json`) has **30 queries**:

| Group | Count | Purpose |
|---|---|---|
| Baseline | 25 | Dishes present in the corpus, including 5 non-English queries (Hindi, French, Spanish, Russian, Chinese) |
| Not in dataset | 2 | `beef rendang`, `churros with chocolate dipping sauce`. The correct behaviour is to say no relevant recipe was found |
| Ambiguous near-miss | 3 | `sushi egg roll`, `seafood carbonara`, `mango cheesecake`. The dish family exists in the corpus but the exact dish does not, so these test whether the model refuses honestly or stitches a near-match into a made-up answer |

## Evaluation results (30-case run)

One run per case, judged by `openai/gpt-oss-20b`. Full traces: [`eval-results/eval_report.json`](eval-results/eval_report.json)
and [`eval-results/partial_results.csv`](eval-results/partial_results.csv).

| Metric | Score | What it measures |
|---|---|---|
| Faithfulness | **4.93 / 5** | Does the answer only use facts present in the retrieved recipes? (28 of 30 cases scored 5) |
| Relevancy | **5.0 / 5** | Does the answer address the user's query? |
| Retrieval hit rate | **93%** | Did retrieval surface a recipe matching the expected keywords? (28 / 30) |
| Citation accuracy | **99.1%** | Of all inline citation tags, how many match a retrieved recipe title? |
| Completeness rate | **67%** | Did the answer contain both an ingredient list and steps? (20 / 30) |
| Cases flagged for citations | **1 / 30** | Queries with a citation tag that did not match a retrieved title |
| Model errors | **0 / 30** | Generation or judge failures |
| Avg. end-to-end latency | **5.68s** | Retrieval + generation time per query |

### Negative and ambiguous cases

All 5 added cases were answered correctly with "No relevant recipes found in context." and scored 5/5:

| Case type | Queries | Outcome |
|---|---|---|
| Not in dataset | `beef rendang`, `churros with chocolate dipping sauce` | Refused, no made-up recipe |
| Ambiguous near-miss | `sushi egg roll`, `seafood carbonara`, `mango cheesecake` | Refused, even though related recipes exist in the corpus |

### What the eval surfaced, honestly

- **Faithfulness slips are small.** The 2 answers scoring 4 were `chicken tikka masala` (an invented
  45-minute cooking time) and `pad thai noodles` (tamarind paste listed for a recipe that does not
  contain it).
- **Completeness (67%) understates answer quality.** The 10 cases scoring 0 are mostly correct
  refusals, which have no ingredients or steps by design, plus the French crêpes answer, which the regex missed because it says "Ingrédients". Of the 21 answered cases, 20 were complete.

## Translation layer updated
- The language is detected with Lingua, the query is translated to English, and it is matched against the recipe corpus by meaning, not by keyword overlap. The accuracy and latency is approximately the same so the results remains the same.


## What it does

- **Cross-lingual retrieval** — a query in Hindi, French, or another language is detected and translated and matched against the recipe corpus by meaning.
- **Multimodal similarity** — surfaces visually similar dish photos alongside text matches.
- **Grounded generation** — every ingredient, quantity, and step in the generated answer is tagged
  `[Recipe Name]` back to its source, so claims can be traced and checked (this is what the
  faithfulness/citation-accuracy metrics above are measuring).

## Architecture

See [`architecture/`](architecture) for the retrieval + generation pipeline diagram.

## Repo structure

```
app/              application entrypoint
architecture/     system design docs / diagrams
eval-results/     eval_report.json, partial_results.csv — full eval traces
indexes/          vector indexes for retrieval
llm-evaluation/   eval harness — LLM-as-judge scoring for faithfulness/relevancy/citation
ppt/              slide summary
src/              core RAG pipeline
```

## Setup

```
pip install -r requirements.txt
```

Create a `.env` file with your Groq API key:

```
GROQ_API_KEY=your_key_here
```

```
python llm-evaluation/eval_llm.py              # all 30 cases
python llm-evaluation/eval_llm.py --limit 3    # quick smoke test
```

## Stack

Python · LangChain · FAISS · OpenCLIP · Groq (gpt-oss) · Gradio · LLM-as-judge evaluation · Lingua · deep-translator

## License

MIT — see [LICENSE](LICENSE).