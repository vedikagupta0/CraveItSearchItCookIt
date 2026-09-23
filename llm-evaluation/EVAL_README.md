# LLM Evaluation — Crave It · Search It · Cook It

`eval_llm.py` provides an **LLM-as-a-judge evaluation harness** for the *Crave It · Search It · Cook It* RAG chatbot.

The evaluation harness imports `app.py` directly, so it uses the same retrieval and generation pipeline as the deployed application rather than mocking the chatbot.

## What It Measures

| Metric                 | Type          | How It Is Measured                                                                               |
| ---------------------- | ------------- | ------------------------------------------------------------------------------------------------ |
| **Faithfulness (1–5)** | LLM judge     | Evaluates whether the generated answer is grounded only in the retrieved recipe context          |
| **Relevancy (1–5)**    | LLM judge     | Evaluates whether the answer directly addresses the user's question                              |
| **Completeness (0/1)** | Deterministic | Checks whether the answer contains ingredient information and step-by-step instructions          |
| **Retrieval Hit Rate** | Deterministic | Checks whether expected recipe keywords are present in the retrieved documents                   |
| **Citation Accuracy**  | Deterministic | Verifies that recipe citation tags in the generated answer correspond to retrieved recipe titles |
| **Latency**            | Deterministic | Measures wall-clock generation time for each query                                               |

## Evaluation Pipeline

The evaluation follows the same RAG architecture as the application:

```text
User Query
    │
    ▼
Translate to English
    │
    ▼
FAISS Recipe Retrieval
    │
    ▼
Top-K Retrieved Recipe Documents
    │
    ▼
GPT-OSS-120B
    │
    ▼
Generated Recipe Answer
    │
    ├──► Completeness Check
    ├──► Retrieval Hit Check
    ├──► Citation Accuracy Check
    │
    ▼
GPT-OSS-20B Judge
    │
    ├──► Faithfulness (1–5)
    └──► Relevancy (1–5)
```

### Models

| Component         | Model                                    |
| ----------------- | ---------------------------------------- |
| Answer Generation | `openai/gpt-oss-120b`                    |
| LLM Judge         | `openai/gpt-oss-20b`                     |
| Text Embeddings   | `sentence-transformers/all-MiniLM-L6-v2` |
| Text Retrieval    | FAISS                                    |
| Image Retrieval   | OpenCLIP ViT-B-32                        |

The **120B model generates the answer**. The **20B model evaluates the generated answer**. They are not interchangeable parts of the pipeline.

## Files

```text
.
├── app/
│   └── app.py
├── eval_llm.py
├── eval_dataset.json
├── eval_report.json
└── requirements.txt
```

### `eval_llm.py`

The evaluation harness. It:

1. Loads the existing RAG application.
2. Translates test questions when necessary.
3. Retrieves relevant recipe documents from FAISS.
4. Generates an answer using the same generation chain as the application.
5. Checks retrieval and citation accuracy deterministically.
6. Checks answer completeness.
7. Sends the answer and retrieved context to the LLM judge.
8. Records faithfulness, relevancy, latency, and other evaluation results.

### `eval_dataset.json`

Contains the evaluation queries and their expected retrieval information.

The dataset contains **25 varied test queries** covering different recipe requests and conversational scenarios.

### `eval_report.json`

Contains the detailed results for each test case, including generated answers, retrieved documents, scores, reasoning, latency, and deterministic evaluation results.

## Installation

Install the project dependencies:

```bash
pip install -r requirements.txt
```

Set the same Groq API key used by the application.

### macOS / Linux

```bash
export GROQ_API_KEY=your_key_here
```

### Windows PowerShell

```powershell
$env:GROQ_API_KEY="your_key_here"
```

## Run the Evaluation

From the project root:

```bash
python eval_llm.py
```

To run multiple repetitions of each test case:

```bash
python eval_llm.py --repeats 3
```

If the script supports selecting a different dataset:

```bash
python eval_llm.py --dataset eval_dataset.json
```

## Output

The evaluation prints a Markdown summary table to stdout and writes detailed results to:

```text
eval_report.json
```

The exact values depend on the current model responses, retrieval results, dataset, and evaluation run.

## Deterministic Checks

Some metrics do not require an LLM judge.

### Completeness

The evaluator checks whether the generated answer contains:

* ingredient information
* preparation/instruction information
* step-by-step directions

This produces a binary result:

```text
1 = complete
0 = incomplete
```

The check is intentionally deterministic so that this metric remains stable across evaluation runs.

### Retrieval Hit Rate

For test cases that define expected keywords, the evaluator checks whether those keywords occur in the retrieved recipe documents.

This helps identify cases where the generator may receive poor or irrelevant retrieval context.

### Citation Accuracy

The application can generate recipe citation tags such as:

```text
Add the eggs and cheese [Carbonara].
```

The evaluator extracts these tags and compares them with the recipe titles contained in the retrieved context.

This allows citation correctness to be evaluated independently from the LLM judge.

## LLM-as-a-Judge

The judge receives:

* the original user question
* the retrieved recipe context
* the generated answer

It evaluates two dimensions.

### Faithfulness

**1–5 scale**

Measures whether the generated answer is supported by the retrieved recipe context.

A high score means the response stays grounded in the retrieved recipes and does not introduce unsupported recipe facts.

### Relevancy

**1–5 scale**

Measures whether the generated response actually answers the user's question.

The judge returns structured JSON:

```json
{
  "faithfulness": 5,
  "relevancy": 5,
  "reasoning": "The answer is grounded in the retrieved recipe context and directly addresses the user's request."
}
```

## Evaluation Isolation

Each evaluation case should use a **fresh conversation history**.

This prevents one test case from influencing another through the chatbot's conversational memory.

Conceptually:

```python
eval_history = InMemoryChatMessageHistory()
eval_chain = build_chain(vs_text, eval_history)

answer = eval_chain.invoke(question)
```

rather than reusing the application's global conversation history across all evaluation cases.

This is important because otherwise:

```text
Test Case 1
    ↓
conversation history
    ↓
Test Case 2
    ↓
conversation history from Test 1 + Test 2
    ↓
Test Case 3
```

could introduce cross-test contamination.

## Error Handling

If answer generation fails, the evaluation records the generation error rather than attempting to score an empty answer.

For example:

```text
Generation Error:
NameError: name 'chain' is not defined
```

The evaluator should report the error in the per-case results so that generation failures can be distinguished from low-quality model answers.

## Reproducibility

For meaningful regression tracking, keep the following consistent between evaluation runs:

* evaluation dataset
* retrieval configuration
* embedding model
* generation model
* judge model
* prompt
* `TOP_K_TEXT`
* temperature
* evaluation code

Changes to any of these can affect the reported metrics.

## Regression Testing

The evaluation harness can be used after changes to:

* `app.py`
* retrieval configuration
* prompts
* embedding models
* generation models
* recipe datasets
* citation logic

For example:

```bash
python eval_llm.py --repeats 3
```

The resulting `eval_report.json` can be compared against previous runs to identify changes in:

* faithfulness
* relevancy
* completeness
* retrieval quality
* citation accuracy
* latency

## Notes

The evaluation is designed to measure the chatbot's **retrieval and generation quality**, not to determine whether a recipe is objectively good or bad.

LLM-judge metrics are model-based evaluations and can therefore contain evaluator variability. Deterministic metrics such as retrieval-hit checks, citation matching, completeness checks, and latency are included to provide additional signals alongside the LLM-based scores.
