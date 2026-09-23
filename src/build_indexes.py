"""Usage:
    python build_indexes.py \
        --data   data/recipes_images.json \
        --images data/images \
        --out    indexes
"""

import argparse
import os
import logging
import pandas as pd
import open_clip
import torch
from PIL import Image
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

device = "cuda" if torch.cuda.is_available() else "cpu"
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def build_rag_column(df, cols, new_col="rag_text"):
    """building a common rag_text coloumn for the recipes"""
    def concat_row(row):
        parts = []
        for c in cols:
            val = row[c]
            if isinstance(val, list):
                val = ", ".join(map(str, val))
            parts.append(f"{c}: {val}")
        return " ;; ".join(parts)
    df[new_col] = df.apply(concat_row, axis=1)
    df[new_col] = df[new_col].str.replace("\\", "", regex=False)
    return df


class OpenCLIPEmbeddings(Embeddings):
    """Embeddings class that uses OpenCLIP to generate embeddings for images."""
    def __init__(self, device_name=device):
        self.device = device_name
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="openai"
        )
        self.model = self.model.to(self.device).eval()
        self.tokenizer = open_clip.get_tokenizer("ViT-B-32")

    def embed_documents(self, paths):
        """Generate embeddings for a list of image file paths."""
        embeddings = []
        with torch.no_grad():
            for path in paths:
                try:
                    """generate embeddings for a list of image file paths."""
                    image = Image.open(path).convert("RGB")
                    image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
                    features = self.model.encode_image(image_tensor)
                    features = features / features.norm(dim=-1, keepdim=True)
                    embeddings.append(features.cpu().numpy()[0].tolist())
                except Exception as error:
                    logger.warning(f"Skipping {path}: {error}")
                    embeddings.append([0.0] * 512)
        return embeddings

    def embed_query(self, text):
        """Generate an embedding for a text query."""
        with torch.no_grad():
            tokens = self.tokenizer([text]).to(self.device)
            features = self.model.encode_text(tokens)
            features = features / features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy()[0].tolist()


def collect_image_paths(image_root: str):
    """ Return a list of all image file paths under image_root."""
    valid_ext = {".jpg", ".jpeg", ".png", ".webp"}
    paths = []
    for root, _, files in os.walk(image_root):
        for file_name in files:
            if os.path.splitext(file_name)[1].lower() in valid_ext:
                paths.append(os.path.join(root, file_name))
    return paths


def build_text_index(data_path: str, out_dir: str):
    """ data preprocessing -> build rag text -> build text index"""
    logger.info(f"Loading dataset from {data_path} …")
    df = pd.read_json(data_path)
    df['servings'] = df['servings'].astype(str)
    df["servings"].fillna("4-5", inplace=True)
    df["ratings"].fillna({"rating": 0.0, "count": 0}, inplace=True)
    df["description"].fillna("Description unavailable.", inplace=True)
    df["image_filename"].fillna("", inplace=True)
    df = build_rag_column(df, list(df.columns))

    docs = [
        Document(page_content=row["rag_text"], metadata={"recipe_id": idx + 1})
        for idx, row in df.iterrows()
    ]
    """docs example : Document(page_content="Recipe 1", metadata={"recipe_id": 1})"""
    logger.info(f"Built {len(docs):,} documents")

    emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vs = FAISS.from_documents(docs, emb) # got embeddings generated and stored in FAISS vector store and FAISS takes document and generate embeddings using the model and store them in the vector store
    save_path = os.path.join(out_dir, "recipe_text_store")
    os.makedirs(save_path, exist_ok=True)
    vs.save_local(save_path)
    logger.info(f"Text index saved to {save_path}")


def build_clip_index(image_folder: str, out_dir: str):
    """building image indexes """
    logger.info(f"Building CLIP index on {device} …")

    paths = collect_image_paths(image_folder)
    logger.info(f"Found {len(paths):,} images")

    image_docs = [
        Document(page_content=p, metadata={"image_path": p}) for p in paths
    ]
    """ image_docs example : Document(page_content="path/to/image.jpg", metadata={"image_path": "path/to/image.jpg"}) """
    clip_emb = OpenCLIPEmbeddings()
    vs = FAISS.from_documents(image_docs, clip_emb)
    save_path = os.path.join(out_dir, "recipe_clip_store")
    os.makedirs(save_path, exist_ok=True)
    vs.save_local(save_path)
    logger.info(f"CLIP index saved to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",   default=r"C:\Users\gvedi\.cache\kagglehub\datasets\seungyeonhan1\recipe-dataset-with-images-tags-and-ratings\versions\4\recipes_images.json")
    parser.add_argument("--images", default="data/images")
    parser.add_argument("--out",    default="indexes")
    parser.add_argument("--skip-clip", action="store_true",
                        help="Build only the text index (faster, no GPU needed)")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    build_text_index(args.data, args.out)
    if not args.skip_clip:
        build_clip_index(args.images, args.out)
    logger.info("Indexing complete!")
