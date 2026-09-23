"""
Crave It · Search It · Cook It
Multimodal Recipe Chatbot — Gradio + LangChain + CLIP + Groq
"""

import os
import io
import sys
import warnings
import logging
from pathlib import Path

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import pandas as pd
import torch
import gradio as gr
import re

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.build_indexes import build_rag_column, OpenCLIPEmbeddings

from deep_translator import GoogleTranslator
from langdetect import detect, LangDetectException

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# ── Config ───────────────────────────────────────────────────────────────────
from dotenv import load_dotenv
load_dotenv()
GROQ_API_KEY   = os.environ.get("GROQ_API_KEY", "")
DATA_PATH      = os.environ.get("DATA_PATH", "data/recipes_images.json")

# Local cache folder where HF images will be stored after download.
HF_IMAGE_REPO  = "vedikagupta0/recipe-images"
HF_IMAGE_CACHE = os.environ.get("IMAGE_FOLDER", "data/hf_images")

# Constants for the app
TEXT_INDEX_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TEXT_INDEX_DIR = "indexes/recipe_text_store"
CLIP_INDEX_DIR = "indexes/recipe_clip_store"
TOP_K_IMAGES   = 3
TOP_K_TEXT     = 5
HF_IMAGE_BASE_URL = (
    "https://huggingface.co/datasets/vedikagupta0/recipe-images/resolve/main/images")
GROQ_MODEL     = "openai/gpt-oss-120b"  # Groq LLM model for RAG answer generation
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"

logger.info(f"Device: {DEVICE}")

# ── HuggingFace image download ────────────────────────────────────────────────

def ensure_hf_images(local_dir: str = HF_IMAGE_CACHE) -> str:
    """
    Download the recipe-images dataset from HuggingFace into `local_dir`
    if it hasn't been downloaded yet.

    The repo layout is:
        images/<filename>.jpg   (stored as Git-LFS / xet pointers)

    After snapshot_download the images live at:
        <local_dir>/images/

    Returns the path to the `images/` subfolder so the rest of the
    code can treat it exactly like the old `data/images` folder.
    """
    images_subfolder = os.path.join(local_dir, "images")

    if os.path.isdir(images_subfolder) and os.listdir(images_subfolder):
        logger.info(f"HF image cache found at {images_subfolder} — skipping download.")
        return images_subfolder

    logger.info(
        f"Downloading recipe images from HuggingFace ({HF_IMAGE_REPO}) "
        f"into {local_dir} … (this may take a while for 5.6 GB)"
    )
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id=HF_IMAGE_REPO,
            repo_type="dataset",
            local_dir=local_dir,
            # Only pull the images/ folder — skip .gitattributes etc.
            allow_patterns=["images/*"],
        )
        logger.info(f"Download complete. Images at {images_subfolder}")
    except Exception as e:
        logger.error(f"Failed to download images from HuggingFace: {e}")
        logger.warning("Continuing without images — CLIP index will be skipped.")
        return ""

    return images_subfolder

# ── Translation ───────────────────────────────────────────────────────────────

def translate_to_english(text: str) -> str:
    try:
        lang = detect(text)
        if lang == "en":
            return text
        return GoogleTranslator(source=lang, target="en").translate(text)
    except (LangDetectException, Exception):
        return text

# ── Index builder / loader ────────────────────────────────────────────────────

def build_or_load_indexes():
    text_emb  = HuggingFaceEmbeddings(model_name=TEXT_INDEX_MODEL)
    clip_emb  = OpenCLIPEmbeddings()

    # ── Text index ──
    if os.path.exists(TEXT_INDEX_DIR):
        logger.info("Loading existing text index …")
        vs_text = FAISS.load_local(TEXT_INDEX_DIR, text_emb, allow_dangerous_deserialization=True)
    else:
        logger.info("Building text index …")
        if not os.path.exists(DATA_PATH):
            raise FileNotFoundError(f"Dataset not found at {DATA_PATH}. See README.")
        df = pd.read_json(DATA_PATH)
        df["servings"].fillna("4-5", inplace=True)
        df["ratings"].fillna({"rating": 0.0, "count": 0}, inplace=True)
        df["description"].fillna("Description unavailable.", inplace=True)
        df["image_filename"].fillna("", inplace=True)
        df = build_rag_column(df, list(df.columns))
        docs = [
            Document(page_content=row["rag_text"], metadata={"recipe_id": idx + 1})
            for idx, row in df.iterrows()
        ]
        os.makedirs(TEXT_INDEX_DIR, exist_ok=True)
        vs_text = FAISS.from_documents(docs, text_emb)
        vs_text.save_local(TEXT_INDEX_DIR)
        logger.info(f"Text index saved — {len(docs):,} docs")

    # ── CLIP index ──
    if os.path.exists(CLIP_INDEX_DIR):
        logger.info("Loading existing CLIP index …")
        vs_clip = FAISS.load_local(CLIP_INDEX_DIR, clip_emb, allow_dangerous_deserialization=True)
    else:
        logger.info("Building CLIP index …")

        # Resolve image folder: download from HF if not already on disk.
        image_folder = ensure_hf_images(HF_IMAGE_CACHE)

        if not image_folder or not os.path.isdir(image_folder):
            logger.warning("Image folder unavailable — CLIP index will be empty.")
            vs_clip = None
        else:
            valid_ext = {".jpg", ".jpeg", ".png", ".webp"}
            image_paths = [
                os.path.join(image_folder, f)
                for f in os.listdir(image_folder)
                if os.path.splitext(f)[1].lower() in valid_ext
            ]
            if not image_paths:
                logger.warning(f"No images found in {image_folder} — CLIP index will be empty.")
                vs_clip = None
            else:
                image_docs = [
                    Document(page_content=path, metadata={"image_path": path})
                    for path in image_paths
                ]
                os.makedirs(CLIP_INDEX_DIR, exist_ok=True)
                vs_clip = FAISS.from_documents(image_docs, clip_emb)
                vs_clip.save_local(CLIP_INDEX_DIR)
                logger.info(f"CLIP index saved — {len(image_docs):,} images")

    return vs_text, vs_clip

# ── LangChain chain ───────────────────────────────────────────────────────────

PROMPT = ChatPromptTemplate.from_template("""\
You are a recipe assistant that answers strictly from retrieved recipe context. You are \
being evaluated on FAITHFULNESS: every ingredient, quantity, step, and recipe name you \
write must be traceable to the context you were given — never to your own training data, \
even when you're confident about what the "real" or "traditional" version of a dish is.

CITATION RULE (the main defense against hallucination):
Tag every factual claim with the recipe it came from, like this: "3 whole eggs [Carbonara]" \
or "simmer 10 minutes [Paella Vegetariana]". If you cannot attach a tag because nothing in \
the context supports the claim, DELETE THE CLAIM — do not write it untagged, and do not \
write it "generally" without a source. Only recipes that appear in the context below may \
ever be tagged or mentioned — never introduce a recipe name that isn't in the context.

--- EXAMPLE OF WRONG BEHAVIOR (do not do this) ---
Context contains only: "Carbonara (Guanciale, Egg, and Pecorino Romano)" with 3 eggs, guanciale, Pecorino.
Question: "spaghetti carbonara"

BAD answer (hallucinated): "Comparing this to the American bacon-and-onion version and the
zucchini variation, the classic recipe uses guanciale..." — WRONG: those other two recipes
were never in the context. They came from training data, not retrieval. This is exactly the
failure this system is graded on.

GOOD answer: "Only one carbonara recipe is available: Carbonara (Guanciale, Egg, and Pecorino
Romano) [Carbonara]. It uses guanciale [Carbonara], 3 whole eggs [Carbonara], and Pecorino
Romano [Carbonara]. Here is the full recipe: ..." — CORRECT: only claims traceable to the
single retrieved recipe, no invented comparison.
--- END EXAMPLES ---

TASK:
- If multiple context recipes are relevant, compare ONLY those, tagging every claim, then
  recommend the best match and present its full ingredients and steps with tags.
- If only one context recipe is relevant, present it directly with tags.
- If none of the context is relevant, say so plainly and stop — do not answer from memory.
- Stay completely faithful to the context. Do not invent any ingredients, steps, or recipe names.
- Assume all you know is the context. If the context is empty, say "No relevant recipes found in context." If context seems unrelated to the question, say "No relevant recipes found in context." Do not answer from memory.
- NEVER make up a recipe, it's ingredients, steps or name. Only use what is in the context.
- IF the exxact match of the recipe is not in the context, say "No relevant recipes found in context." Do not answer from memory or manipulate the context to answer user query.
Before finalizing, silently re-read your draft and delete any untagged or unsupported claim.
- IMPORTANT - Do not say things like "(all as per given recipe, listed in source, etc.)", the tags seems sufficient.
---
Recipe context:
{context}

Conversation history:
{chat_history}

Question:
{question}
""")

def build_chain(vs_text, history):
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY is not set. Add it in Space secrets.")
    llm = ChatGroq(model=GROQ_MODEL, api_key=GROQ_API_KEY, temperature=0)
    retriever = vs_text.as_retriever(search_kwargs={"k": TOP_K_TEXT})
    chain = (
        {
            "chat_history": RunnableLambda(lambda _: history.messages),
            "context": retriever,
            "question": lambda x: x,
        }
        | PROMPT
        | llm
        | StrOutputParser()
    )
    return chain

# ── Image helpers ───────────────────────────────────────────────────────────────────

def _path_to_hf_url(path: str) -> str:
    """Convert any stored image path (old local Windows/Linux or new) to the HF CDN URL.
    Handles both forward-slash and backslash separators."""
    # Replace backslashes so basename works correctly on Linux
    filename = path.replace("\\", "/").split("/")[-1]
    return f"{HF_IMAGE_BASE_URL}/{filename}"

def _filename_to_caption(path_or_url: str) -> str:
    filename = path_or_url.replace("\\", "/").split("/")[-1]
    name = filename.split(".")[0]                       # drop extension
    name = re.sub(r"[-_]?\d+$", "", name)                # strip trailing id, e.g. "-56389846"
    return name.replace("-", " ").replace("_", " ").strip().title()

def clip_gallery_images(query: str, vs_clip, k=TOP_K_IMAGES) -> list[tuple]:
    """Return (hf_url, caption) pairs for the top-k CLIP results."""
    if vs_clip is None:
        return []
    try:
        docs = vs_clip.similarity_search(query, k=k)
    except Exception as e:
        logger.warning(f"CLIP search error: {e}")
        return []
    gallery = []
    for doc in docs:
        path = doc.metadata.get("image_path", "") or doc.page_content
        if not path:
            continue
        url = _path_to_hf_url(path)
        logger.info(f"CLIP result: stored_path={path!r} → url={url}")
        gallery.append((url, _filename_to_caption(path)))
    return gallery



# ── removing citations and tags for display  ──────────────────────────────────
def remove_recipe_tags(text: str) -> str:
    """
    Remove recipe citation tags such as:
    [Carbonara]
    [Paella Vegetariana]
    [Chicken Tikka Masala]
    
    while keeping the actual answer unchanged.
    """
    return re.sub(r"\s*\[[^\[\]]+\]", "", text).strip()


# ── Gradio state init ─────────────────────────────────────────────────────────
logger.info("Initialising indexes …")
vs_text, vs_clip = build_or_load_indexes()
history = InMemoryChatMessageHistory()
chain   = build_chain(vs_text, history)

# ── Recipe image lookup (recipe_id → HF CDN URL) ─────────────────────────────
recipe_image_map: dict[int, str] = {}
try:
    if os.path.exists(DATA_PATH):
        _df = pd.read_json(DATA_PATH)
        for _idx, _row in _df.iterrows():
            _fname = str(_row.get("image_filename", "")).strip()
            if _fname:
                # image_filename may be a bare name or include a path prefix
                _fname = _fname.replace("\\", "/").split("/")[-1]
                recipe_image_map[_idx + 1] = f"{HF_IMAGE_BASE_URL}/{_fname}"
        logger.info(f"Recipe image map built — {len(recipe_image_map):,} entries")
except Exception as e:
    logger.warning(f"Could not build recipe image map: {e}")

logger.info("Ready.")

# ── Chat callback ─────────────────────────────────────────────────────────────
def respond(user_msg: str, chat_history: list):
    if not user_msg.strip():
        return chat_history, [], [], ""

    # Translate
    english = translate_to_english(user_msg)
    translated_note = f"*(Translated: {english})*\n\n" if english != user_msg else ""

    # RAG retrieval — get top docs to build the recipe-photo gallery
    retriever = vs_text.as_retriever(search_kwargs={"k": TOP_K_TEXT})
    top_docs  = retriever.invoke(english)

    recipe_gallery_imgs = []
    for doc in top_docs:
        recipe_id = doc.metadata.get("recipe_id")
        img_url = recipe_image_map.get(recipe_id)
        if img_url and 'nan' not in img_url:
            img_url = img_url + ".jpg"
            recipe_gallery_imgs.append((img_url, _filename_to_caption(img_url)))

    # CLIP gallery
    gallery_imgs0 = clip_gallery_images(english, vs_clip)
    recipe_urls = [img[0] for img in recipe_gallery_imgs]
    gallery_imgs = []
    for url, caption in gallery_imgs0:
        if url not in recipe_urls:
            gallery_imgs.append((url, caption))


    # RAG answer
    try:
        answer = chain.invoke(english)
    except Exception as e:
        answer = f"⚠️ Error: {e}"

    # Keep tagged answer in internal conversation history
    history.add_user_message(english)
    history.add_ai_message(answer)

    # Remove citation tags only from what the user sees.
    display_answer = remove_recipe_tags(answer)

    full_answer = translated_note + display_answer
    chat_history = chat_history + [
        {"role": "user",      "content": user_msg},
        {"role": "assistant", "content": full_answer},
    ]
    return chat_history, recipe_gallery_imgs, gallery_imgs, ""

def clear_session():
    history.clear()
    return [], [], [], ""

# ── UI ────────────────────────────────────────────────────────────────────────

CSS = """
#chatbox { height: 520px; overflow-y: auto; }
.gallery-row { margin-top: 8px; }
footer { display: none !important; }
"""

with gr.Blocks(css=CSS, title="👨🏻‍🍳 Crave It · Search It · Cook It") as demo:
    gr.Markdown(
        """
# 👨🏻‍🍳 Crave It · Search It · Cook It
**Multimodal Recipe Chatbot** — semantic text retrieval · CLIP image search · Llama-3.3-70B on Groq

Ask anything in **any language** — the bot translates, finds the closest recipes visually and semantically, then gives you a chef-style answer.
        """
    )

    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(elem_id="chatbox", label="Chat")
            with gr.Row():
                msg_box = gr.Textbox(
                    value="healthy tea recipe for daily consumption",
                    placeholder="Ask a recipe question (e.g. 'healthy tea', 'pasta carbonara') …",
                    show_label=False,
                    scale=7,
                )
                send_btn = gr.Button("Send 🍴", variant="primary", scale=1)
            clear_btn = gr.Button("🗑️ Clear session", variant="secondary")

        with gr.Column(scale=2):

            gr.Markdown("### 🍽️ Recipe Images")
            recipe_gallery = gr.Gallery(
                columns=5,
                height=220,
                object_fit="cover",
                show_label=False,
                elem_classes=["gallery-row"],
                selected_index=0
            )

            gr.Markdown("### 🍜 Visually Similar Dishes")
            gallery = gr.Gallery(
                columns=3,
                height=260,
                object_fit="cover",
                show_label=False,
                elem_classes=["gallery-row"],
                selected_index=0,
            )

    gr.Markdown(
        """
--- 
**TechStack:** LangChain · FAISS · OpenCLIP ViT-B-32 · MiniLM-L6-v2 · Groq Llama-3.3-70B
        """
    )

    # Wiring
    send_btn.click(
        respond,
        [msg_box, chatbot],
        [chatbot, recipe_gallery, gallery, msg_box]
    )

    msg_box.submit(
        respond,
        [msg_box, chatbot],
        [chatbot, recipe_gallery, gallery, msg_box]
    )

    clear_btn.click(
        clear_session,
        [],
        [chatbot, recipe_gallery, gallery, msg_box]
    )

if __name__ == "__main__":
    demo.launch()