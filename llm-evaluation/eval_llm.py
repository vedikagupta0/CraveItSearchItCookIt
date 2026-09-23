"""
eval_llm.py — LLM-judge evaluation harness for "Crave It · Search It · Cook It"

WHAT THIS EVALUATES
--------------------
The app is a RAG pipeline: user query -> (translate) -> FAISS text retrieval
-> Groq openai/gpt-oss-20b generates a chef-style answer from the
retrieved recipe context.

This script runs each test query through the *real* pipeline (imported
directly from app.py, so it uses the same indexes/model the Space uses)
--repeats times each ,
and scores each answer on three axes using a Groq LLM as judge
(default: openai/gpt-oss-20b):

  1. Faithfulness   (1-5) - is the answer grounded ONLY in the retrieved
                             recipe context, with no invented ingredients/
                             steps/facts?
  2. Relevancy      (1-5) - does the answer actually address the user's
                             question / requested dish?
  3. Completeness   (0/1) - deterministic heuristic: does the answer contain
                             an ingredient list AND step-by-step instructions?
                             (checked with regex, not the LLM, so it's cheap
                             and 100% reproducible)

It also records retrieval + generation latency and, when the dataset entry
provides `expected_keywords`, a deterministic retrieval-hit check (did any
of the retrieved documents mention an expected keyword).


CITATION VERIFICATION 
----------------------------
The app's prompt now asks the model to tag every factual claim with the
recipe it came from, e.g. "3 whole eggs [Carbonara]". This is useful, but a
citation tag alone doesn't guarantee the citation is real — a model can
still hallucinate a recipe and wrap it in a confident-looking tag. So this
script ALSO runs a deterministic, non-LLM check: every [Tag] in the answer
is compared against the titles of the recipes that were actually retrieved.
This gives a `citation_accuracy` metric that can't be fooled by an LLM
judge failing to notice a fabricated-but-well-formatted citation.

USAGE
-----
    export GROQ_API_KEY=...             
    pip install -r requirements.txt
    python eval_llm.py                                  # uses eval_dataset.json, 1 run/case
    python eval_llm.py --dataset my_cases.json --out my_report.json
    python eval_llm.py --limit 3        # quick smoke test on first 3 cases
    python eval_llm.py --repeats 1      # single run per case (faster, noisier)

By default the judge is openai/gpt-oss-20b (free tier: 8000 tokens/min).
The script pauses EVAL_SLEEP_BETWEEN_CASES_S between every generation/judge 
call to stay under that limit, and retries on a 429 rate-limit error 
after sleeping EVAL_JUDGE_RETRY_SLEEP_S. 

OUTPUT
------
Writes a JSON report (default: eval_report.json) with per-case aggregated
scores (mean/std across repeats), the raw per-run data, and overall summary
stats, plus prints a Markdown summary table to stdout.
"""

from __future__ import annotations

import argparse
import csv
import difflib
import json
import os
from pathlib import Path
import re
import statistics
import sys
import time
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()  # load GROQ_API_KEY from .env 

EVAL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EVAL_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

JUDGE_MODEL = os.environ.get("EVAL_JUDGE_MODEL", "openai/gpt-oss-20b")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

JUDGE_RETRY_SLEEP_S = 180
JUDGE_MAX_RETRIES = 1
SLEEP_BETWEEN_CASES_S = 180
DEFAULT_REPEATS = 1

JUDGE_MAX_CONTEXT_CHARS = int(os.environ.get("EVAL_JUDGE_MAX_CONTEXT_CHARS", "8000")) # max chars of retrieved context to send to judge
JUDGE_MAX_ANSWER_CHARS = int(os.environ.get("EVAL_JUDGE_MAX_ANSWER_CHARS", "4000")) # max chars of generated answer to send to judge
JUDGE_MAX_TOKENS = int(os.environ.get("EVAL_JUDGE_MAX_TOKENS", "2048")) # max tokens for judge's response (default 2048, since we only want a short JSON object back)

try:
    from app.app import chain, vs_text, translate_to_english, TOP_K_TEXT
except Exception as e:
    print(
        "ERROR: could not import app.py pipeline. Run this script with "
        f"dependencies installed and GROQ_API_KEY set.\nOriginal error: {e}",
        file=sys.stderr,
    )
    raise


JUDGE_SYSTEM_PROMPT = """You are a strict evaluator of a recipe RAG chatbot's answer.
You will be given: the user's question, the recipe context the chatbot retrieved,
and the chatbot's answer.

Score the answer on two axes:

- "faithfulness" (integer 1-5): Is the answer grounded ONLY in the given recipe
  context? 5 = every claim (ingredients, quantities, steps) traces back to the
  context. 1 = the answer invents ingredients/steps/facts not present in context,
  or contradicts it. The answer may contain [Recipe Name] citation tags after
  claims — a tag does NOT automatically mean the claim is grounded; still check
  whether the tagged recipe and its details genuinely appear in the context.
- "relevancy" (integer 1-5): Does the answer actually address what the user
  asked for? 5 = directly and fully answers the question. 1 = off-topic or
  ignores the question.

Respond with ONLY a JSON object, no prose, no markdown fences, in this exact
shape:
{"faithfulness": <int 1-5>, "relevancy": <int 1-5>, "reasoning": "<one short sentence>"}
"""

JUDGE_USER_TEMPLATE = """Question:
{question}

Retrieved recipe context:
{context}

Chatbot answer:
{answer}
"""

_INGREDIENT_HINTS = re.compile(r"\bingredient", re.IGNORECASE) # this is a simple heuristic to check if the answer contains an ingredient list
_STEP_HINTS = re.compile(
    r"(\bstep\s*\d|\binstructions?\b|\bmethod\b|^\s*\d+[\.\)]\s+|\bdirections?\b)",
    re.IGNORECASE | re.MULTILINE,
) # this is a simple heuristic to check if the answer contains step-by-step instructions

def check_completeness(answer: str) -> int:
    """Deterministic heuristic: 1 if answer looks like it has both an
    ingredient list and numbered/step-like instructions, else 0."""
    has_ingredients = bool(_INGREDIENT_HINTS.search(answer))
    has_steps = bool(_STEP_HINTS.search(answer))
    return int(has_ingredients and has_steps)


def check_retrieval_hit(context_docs: list, expected_keywords: list[str]) -> bool:
    if not expected_keywords:
        return True  # nothing to check against
    joined = " ".join(d.page_content.lower() for d in context_docs)
    return any(kw.lower() in joined for kw in expected_keywords)


# ── Citation verification (programmatic, non-LLM) ──────────────────────────
_TITLE_FIELD_RE = re.compile(r"title:\s*(.+?)\s*;;", re.IGNORECASE) # extracts the title from a recipe's rag_text field
_CITATION_TAG_RE = re.compile(r"\[([^\[\]]{1,80})\]")# matches [Recipe Name] tags in the answer, up to 80 chars long (to avoid accidental matches on long prose in the answer)

def extract_titles(docs: list) -> set[str]:
    titles = set()
    for d in docs:
        m = _TITLE_FIELD_RE.search(d.page_content)
        if m:
            titles.add(m.group(1).strip().lower())
    return titles


def _tag_matches_title(tag: str, titles: set[str]) -> bool:
    # Strip trailing parenthetical IDs like "(1501)" or "(9516)" that the
    # model sometimes appends to an otherwise-real title.
    tag_norm = re.sub(r"\s*\(.*?\)\s*$", "", tag.strip().lower()).strip()
    if not tag_norm:
        return False
    for t in titles:
        if tag_norm == t or tag_norm in t or t in tag_norm:
            return True
        if difflib.SequenceMatcher(None, tag_norm, t).ratio() > 0.82:
            return True
    return False


def check_citation_faithfulness(answer: str, docs: list) -> dict:
    """Deterministic faithfulness check independent of the LLM judge: every
    [Recipe Name] tag in the answer is verified against the titles of
    recipes that were genuinely retrieved. Catches confident-looking but
    fabricated citations that an LLM judge might not notice."""
    titles = extract_titles(docs)
    tags = _CITATION_TAG_RE.findall(answer)
    if not tags:
        return {
            "num_tags": 0,
            "num_valid_tags": 0,
            "num_invalid_tags": 0,
            "citation_accuracy": None,
            "invalid_tags": [],
        }
    valid = [t for t in tags if _tag_matches_title(t, titles)]
    invalid = [t for t in tags if not _tag_matches_title(t, titles)]
    return {
        "num_tags": len(tags),
        "num_valid_tags": len(valid),
        "num_invalid_tags": len(invalid),
        "citation_accuracy": round(len(valid) / len(tags), 3),
        "invalid_tags": sorted(set(invalid)),
    }


def build_judge():
    if not GROQ_API_KEY:
        raise ValueError(
            "GROQ_API_KEY is not set. Export it before running the eval script."
        )
    return ChatGroq(
        model=JUDGE_MODEL,
        api_key=GROQ_API_KEY,
        temperature=0,
        max_tokens=JUDGE_MAX_TOKENS,
    )


def _extract_content(response) -> str:
    """Pull the judge's text out of a langchain_groq response robustly.
    """
    content = (getattr(response, "content", "") or "").strip()
    if content:
        return content

    additional = getattr(response, "additional_kwargs", {}) or {}
    for key in ("reasoning_content", "reasoning"):
        val = additional.get(key)
        if val:
            return str(val).strip()

    meta = getattr(response, "response_metadata", {}) or {}
    for key in ("reasoning_content", "reasoning"):
        val = meta.get(key)
        if val:
            return str(val).strip()

    return ""


def _is_rate_limit_error(e: Exception) -> bool:
    msg = str(e).lower()

    if "429" in msg or "rate_limit" in msg or "rate limit" in msg:
        return True

    status_code = getattr(e, "status_code", None)
    if status_code == 429:
        return True

    response = getattr(e, "response", None)
    if response is not None and getattr(response, "status_code", None) == 429:
        return True

    return False


def _call_judge_with_retry(judge, messages) -> str:
    """Call judge, retrying on 429 rate limits."""
    last_err = None

    for attempt in range(JUDGE_MAX_RETRIES + 1):
        try:
            response = judge.invoke(messages)
            return _extract_content(response)

        except Exception as e:
            last_err = e

            if _is_rate_limit_error(e) and attempt < JUDGE_MAX_RETRIES:
                retry_num = attempt + 1
                print(
                    f"  rate limited, sleeping {JUDGE_RETRY_SLEEP_S}s "
                    f"before retry {retry_num}/{JUDGE_MAX_RETRIES}..."
                )
                time.sleep(JUDGE_RETRY_SLEEP_S)
                continue

            raise

    raise last_err


def judge_answer(judge, question: str, context_doc_texts: list[str], answer: str) -> dict:
    # context = _truncate_context_docs(context_doc_texts, JUDGE_MAX_CONTEXT_CHARS)
    # answer_for_judge = _truncate(answer, JUDGE_MAX_ANSWER_CHARS)

    messages = [
        ("system", JUDGE_SYSTEM_PROMPT),
        ("user", JUDGE_USER_TEMPLATE.format(question=question, context=context_doc_texts, answer=answer)),
    ]

    try:
        raw = _call_judge_with_retry(judge, messages)
    except Exception as e:
        # One bad case shouldn't kill the whole eval run.
        return {
            "faithfulness": 0,
            "relevancy": 0,
            "reasoning": f"JUDGE_CALL_FAILED: {type(e).__name__}: {str(e)[:200]}",
        }

    if not raw:
        return {"faithfulness": 0, "relevancy": 0, "reasoning": "EMPTY_JUDGE_OUTPUT"}

    # strip accidental markdown fences
    raw = re.sub(r"^```(json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
    try:
        parsed = json.loads(raw)
        return {
            "faithfulness": int(parsed.get("faithfulness", 0)),
            "relevancy": int(parsed.get("relevancy", 0)),
            "reasoning": str(parsed.get("reasoning", "")),
        }
    except Exception:
        return {"faithfulness": 0, "relevancy": 0, "reasoning": f"UNPARSEABLE_JUDGE_OUTPUT: {raw[:200]}"}


def run_case_once(judge, case: dict) -> dict:
    """Run a single case exactly once: retrieve, generate, judge."""
    question = case["query"]
    expected_keywords = case.get("expected_keywords", [])

    t0 = time.perf_counter()
    english_q = translate_to_english(question)

    retriever = vs_text.as_retriever(search_kwargs={"k": TOP_K_TEXT})
    retrieved_docs = retriever.invoke(english_q)
    t_retrieval = time.perf_counter() - t0

    try:
        answer = chain.invoke(english_q)
        error = None
    except Exception as e:
        answer = ""
        error = str(e)
    t_total = time.perf_counter() - t0

    result = {
        "answer": answer,
        "error": error,
        "retrieval_hit": check_retrieval_hit(retrieved_docs, expected_keywords),
        "num_docs_retrieved": len(retrieved_docs),
        "completeness": check_completeness(answer) if answer else 0,
        "retrieval_latency_s": round(t_retrieval, 3),
        "total_latency_s": round(t_total, 3),
    }

    if answer:
        citation = check_citation_faithfulness(answer, retrieved_docs)
    else:
        citation = {"num_tags": 0, "num_valid_tags": 0, "num_invalid_tags": 0,
                    "citation_accuracy": None, "invalid_tags": []}
    result.update(citation)

    if answer and not error:
        judged = judge_answer(judge, question, [d.page_content for d in retrieved_docs], answer)
    else:
        judged = {"faithfulness": 0, "relevancy": 0, "reasoning": "no answer generated (error)"}

    result.update(judged)
    result["translated_query"] = english_q
    return result


def _mean_std(runs: list[dict], key: str) -> tuple[float, float]:
    vals = [r[key] for r in runs if isinstance(r.get(key), (int, float))]
    if not vals:
        return 0.0, 0.0
    mean = round(statistics.mean(vals), 2)
    std = round(statistics.stdev(vals), 2) if len(vals) > 1 else 0.0
    return mean, std


def run_case(judge, case: dict, repeats: int) -> dict:
    """Run a case `repeats` times and aggregate into mean ± std, so a single
    noisy run doesn't get reported as THE score for this query."""
    runs = []
    for r in range(repeats):
        runs.append(run_case_once(judge, case))
        time.sleep(JUDGE_RETRY_SLEEP_S)

    faith_mean, faith_std = _mean_std(runs, "faithfulness")
    rel_mean, rel_std = _mean_std(runs, "relevancy")
    lat_mean, lat_std = _mean_std(runs, "total_latency_s")
    n = len(runs)

    citation_vals = [r["citation_accuracy"] for r in runs if r.get("citation_accuracy") is not None]
    citation_accuracy_mean = round(statistics.mean(citation_vals), 3) if citation_vals else None
    all_invalid_tags = sorted({t for r in runs for t in r.get("invalid_tags", [])})

    return {
        "query": case["query"],
        "num_runs": n,
        "faithfulness_mean": faith_mean,
        "faithfulness_std": faith_std,
        "relevancy_mean": rel_mean,
        "relevancy_std": rel_std,
        "completeness_rate": round(sum(r["completeness"] for r in runs) / n, 2),
        "retrieval_hit_rate": round(sum(1 for r in runs if r["retrieval_hit"]) / n, 2),
        "error_rate": round(sum(1 for r in runs if r["error"]) / n, 2),
        "citation_accuracy": citation_accuracy_mean,
        "invalid_tags_seen": all_invalid_tags,
        "avg_total_latency_s": lat_mean,
        "latency_std_s": lat_std,
        "runs": runs,
    }


def summarize(case_results: list[dict]) -> dict:
    def avg(key):
        vals = [r[key] for r in case_results if isinstance(r.get(key), (int, float))]
        return round(statistics.mean(vals), 2) if vals else 0.0

    n = len(case_results)
    citation_vals = [r["citation_accuracy"] for r in case_results if r.get("citation_accuracy") is not None]
    avg_citation_accuracy = round(statistics.mean(citation_vals), 3) if citation_vals else None
    cases_with_bad_citations = sum(1 for r in case_results if r.get("invalid_tags_seen"))

    return {
        "num_cases": n,
        "runs_per_case": case_results[0]["num_runs"] if case_results else 0,
        "avg_faithfulness": avg("faithfulness_mean"),
        "avg_faithfulness_std": avg("faithfulness_std"),
        "avg_relevancy": avg("relevancy_mean"),
        "avg_relevancy_std": avg("relevancy_std"),
        "completeness_rate": avg("completeness_rate"),
        "retrieval_hit_rate": avg("retrieval_hit_rate"),
        "error_rate": avg("error_rate"),
        "avg_citation_accuracy": avg_citation_accuracy,
        "cases_with_fabricated_citations": cases_with_bad_citations,
        "avg_total_latency_s": avg("avg_total_latency_s"),
    }


def print_markdown_table(case_results: list[dict]) -> None:
    print("\n| Query | Faithfulness | Relevancy | Citation Acc. | Complete | Retrieval Hit | Latency (s) |")
    print("|---|---|---|---|---|---|---|")
    for r in case_results:
        q = (r["query"][:35] + "…") if len(r["query"]) > 35 else r["query"]
        cit = "—" if r["citation_accuracy"] is None else f"{r['citation_accuracy']*100:.0f}%"
        print(
            f"| {q} | {r['faithfulness_mean']}±{r['faithfulness_std']}/5 | "
            f"{r['relevancy_mean']}±{r['relevancy_std']}/5 | "
            f"{cit} | "
            f"{r['completeness_rate']*100:.0f}% | "
            f"{r['retrieval_hit_rate']*100:.0f}% | "
            f"{r['avg_total_latency_s']}±{r['latency_std_s']} |"
        )


def main():
    parser = argparse.ArgumentParser(description="LLM-judge eval for Crave It · Search It · Cook It")
    parser.add_argument("--dataset", default=str(EVAL_DIR / "eval_dataset.json"))
    parser.add_argument("--out", default=str(EVAL_DIR / "eval_report.json"))
    parser.add_argument("--limit", type=int, default=None, help="only run first N cases")
    parser.add_argument(
        "--repeats", type=int, default=DEFAULT_REPEATS,
        help="how many times to run each case and average (default 3, since generation "
             "and judging are both non-deterministic)",
    )
    args = parser.parse_args()

    with open(args.dataset, "r", encoding="utf-8") as f:
        cases: list[dict] = json.load(f)
    if args.limit:
        cases = cases[: args.limit]

    judge = build_judge()

    results = []
    for i, case in enumerate(cases, 1):
        print(f"[{i}/{len(cases)}] Running {args.repeats}x: {case['query']!r}")
        data = run_case(judge, case, args.repeats)

        # Append progress to a CSV so partial results survive an interruption.
        with open(EVAL_DIR / "partial_results.csv", "a", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([i] + list(data.values()))

            #print data values with their keys for debugging
            print(f"data : {data}")

        results.append(data)
        time.sleep(SLEEP_BETWEEN_CASES_S)

    summary = summarize(results)

    report = {"summary": summary, "cases": results}
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print_markdown_table(results)
    print("\n### Summary")
    for k, v in summary.items():
        print(f"- **{k}**: {v}")
    print(f"\nFull report written to {args.out}")


if __name__ == "__main__":
    main()