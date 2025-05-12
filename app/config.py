import chromadb
import langchain
from pathlib import Path
from datetime import datetime
from typing import Set


chroma_version = chromadb.__version__.replace(".", "_")
lc_version = langchain.__version__.replace(".", "_")
SOURCE_ROOT = Path(".").resolve().name  # or pass this as an env/config param
# VECTORSTORE_DIR = str(Path(f"vectorstore_{SOURCE_ROOT}_lc{lc_version}_chroma{chroma_version}_{datetime.now():%Y%m%d}"))
EMBEDDING_MODEL = "intfloat/e5-base-v2"
LLM_ENDPOINT = "http://localhost:8080/generate"  # Local LLM endpoint
ALLOWED_EXTENSIONS = [
    ".py",
    ".json", ".yaml", ".yml",
    ".txt"
    ".md", ".ts", ".ipynb", ".sh", ".ps1"
]
EXCLUDED_DIRS: Set[str] = {
    ".git", "node_modules", "__pycache__", ".venv", "venv", "env", ".mypy_cache", "models", "build", "neurobase.egg-info", ".idea", ".vscode", ".pytest_cache",
}

def get_vectorstore_dir(code_path) -> str:
    """
    Get the vectorstore directory path.
    """
    if code_path:
        return str(Path(f"vectorstore_{Path(code_path).resolve().name}_lc{lc_version}_chroma{chroma_version}_{datetime.now():%Y%m%d}"))
    else:
        return str(Path(f"vectorstore_{SOURCE_ROOT}_lc{lc_version}_chroma{chroma_version}_{datetime.now():%Y%m%d}"))