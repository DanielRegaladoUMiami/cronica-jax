"""Push the Gradio Space to Hugging Face Spaces.

Validates BEFORE push:
  - app.py is parseable Python
  - requirements.txt lists jax + tokenizers + gradio
  - README.md has YAML frontmatter
  - cronica/ package can be assembled from src/

Then assembles a staging dir containing:
  - app.py
  - requirements.txt
  - README.md (with HF Space metadata)
  - cronica/  (copied from src/cronica so the Space can import it)

Usage:
    python -m scripts.push_space \
        --repo DanielRegaladoCardoso/cronica-jax-space
"""
from __future__ import annotations

import argparse
import ast
import logging
import shutil
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

SPACE_DIR = Path("space")
PKG_SRC = Path("src/cronica")


def validate() -> None:
    assert SPACE_DIR.exists(), f"FAIL: {SPACE_DIR} missing"
    app = SPACE_DIR / "app.py"
    req = SPACE_DIR / "requirements.txt"
    readme = SPACE_DIR / "README.md"
    for p in (app, req, readme):
        assert p.exists(), f"FAIL: {p} missing"

    # app.py is valid Python
    ast.parse(app.read_text())
    logger.info("app.py: valid Python syntax")

    # requirements has critical deps
    text = req.read_text()
    for pkg in ("jax", "jaxlib", "tokenizers", "gradio", "huggingface-hub"):
        assert pkg in text, f"FAIL: requirements.txt missing {pkg}"
    logger.info("requirements.txt: jax + jaxlib + tokenizers + gradio + huggingface-hub")

    # README has frontmatter
    rl = readme.read_text()
    assert rl.startswith("---\n") and "sdk: gradio" in rl, \
        "FAIL: README.md missing HF Spaces YAML frontmatter"
    logger.info("README.md: HF Spaces frontmatter present")

    # cronica package importable from src
    assert PKG_SRC.exists(), f"FAIL: {PKG_SRC} missing"
    for module in ("__init__.py", "model.py", "tokenizer.py", "sample.py", "train.py"):
        assert (PKG_SRC / module).exists(), f"FAIL: {PKG_SRC/module} missing"
    logger.info("cronica/ package complete (model, tokenizer, sample, train)")


def assemble(staging: Path) -> None:
    """Copy app + cronica pkg into a staging dir."""
    shutil.copy2(SPACE_DIR / "app.py", staging / "app.py")
    shutil.copy2(SPACE_DIR / "requirements.txt", staging / "requirements.txt")
    shutil.copy2(SPACE_DIR / "README.md", staging / "README.md")
    pkg_dst = staging / "cronica"
    pkg_dst.mkdir()
    for f in PKG_SRC.glob("*.py"):
        shutil.copy2(f, pkg_dst / f.name)
    logger.info("Staged in %s: %s", staging, sorted(p.name for p in staging.iterdir()))


def push(staging: Path, repo: str) -> None:
    from huggingface_hub import HfApi
    api = HfApi()
    try:
        whoami = api.whoami()
    except Exception as e:
        raise SystemExit(f"FAIL: HF auth not configured. Run `hf auth login`. {e}")
    logger.info("HF auth OK as %s", whoami.get("name"))

    api.create_repo(repo_id=repo, repo_type="space", space_sdk="gradio",
                    exist_ok=True)
    api.upload_folder(folder_path=str(staging), repo_id=repo, repo_type="space")
    logger.info("Pushed to https://huggingface.co/spaces/%s", repo)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default="DanielRegaladoCardoso/cronica-jax-space")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    validate()
    with tempfile.TemporaryDirectory() as td:
        staging = Path(td)
        assemble(staging)
        if args.dry_run:
            logger.info("DRY RUN: validations passed, staged at %s. Skipping push.", staging)
            return
        push(staging, args.repo)


if __name__ == "__main__":
    main()
