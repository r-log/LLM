"""Build a corpus of C-family documentation (language references, library docs, guidelines).

    python scripts/build_docs_corpus.py                       # sources marked "open" only
    python scripts/build_docs_corpus.py --allow-research-only # also sources whose license
                                                              # allows only personal/internal use

Writes data/docs-raw/docs.jsonl, one line per document:
    {"source", "license", "use", "path", "text"}
plus each source's LICENSE file under data/docs-raw/licenses/ and a summary.json.

Sources are fetched with a shallow, sparse git clone (only the listed paths), so git and
network access to github.com are needed. `use` is "open" (license allows reuse with
attribution) or "research-only" (do not use for a model you publish or sell). The license
notes below are from the repositories' stated terms; read the saved LICENSE files yourself
before any use beyond private research.

Mixing into training: prepare_data.py --source code --add-docs data/docs-raw/docs.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

SOURCES = [
    # The original repository is no longer public; up-to-date public forks carry the same
    # license. The first one that can be cloned is used.
    {"name": "msvc-cpp-docs", "repo": "MicrosoftDocs/cpp-docs",
     "mirrors": ["gcampbell-msft/cpp-docs", "MasoudAmiriBatmanghlnj/cpp-docs", "kermitdafrog8/cpp-docs"],
     "paths": ["docs/c-language", "docs/cpp", "docs/standard-library", "docs/c-runtime-library",
               "docs/preprocessor", "docs/build", "docs/code-quality"],
     "license": "CC-BY-4.0 (documentation), MIT (code samples)", "use": "open"},
    {"name": "dotnet-csharp-docs", "repo": "dotnet/docs",
     "paths": ["docs/csharp"],
     "license": "CC-BY-4.0 (documentation), MIT (code samples)", "use": "open"},
    {"name": "cpp-core-guidelines", "repo": "isocpp/CppCoreGuidelines",
     "paths": ["CppCoreGuidelines.md"],
     "license": "Standard C++ Foundation license: personal or internal business use only",
     "use": "research-only"},
]
TEXT_SUFFIXES = {".md", ".txt"}
FRONT_MATTER = re.compile(r"\A---\n.*?\n---\n", re.DOTALL)


GIT_ENV = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}  # never ask for a password: fail instead


def sparse_clone(repo: str, paths: list[str], dest: Path, url: str | None = None) -> None:
    """Shallow clone that only checks out `paths` (files or folders) and root LICENSE files."""
    if dest.exists():
        shutil.rmtree(dest)
    url = url or f"https://github.com/{repo}.git"
    run = lambda *a: subprocess.run(list(a), check=True, env=GIT_ENV, timeout=3600)  # noqa: E731
    run("git", "clone", "--quiet", "--depth", "1", "--filter=blob:none", "--no-checkout", url, str(dest))
    patterns = ["/LICENSE*", "/license*"] + [f"/{p.strip('/')}" + ("" if Path(p).suffix else "/") for p in paths]
    run("git", "-C", str(dest), "sparse-checkout", "set", "--no-cone", *patterns)
    run("git", "-C", str(dest), "checkout", "--quiet")


def clean(text: str) -> str:
    text = FRONT_MATTER.sub("", text.replace("\r\n", "\n"))
    return text.strip() + "\n"


def collect(src: dict, root: Path):
    for p in sorted(root.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in TEXT_SUFFIXES or ".git" in p.parts:
            continue
        rel = p.relative_to(root).as_posix()
        if rel.lower().startswith("license"):
            continue
        try:
            text = clean(p.read_text(encoding="utf-8"))
        except UnicodeDecodeError:
            continue
        if len(text) >= 200:
            yield {"source": src["name"], "license": src["license"], "use": src["use"], "path": rel, "text": text}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/docs-raw")
    ap.add_argument("--allow-research-only", action="store_true")
    ap.add_argument("--keep-clones", action="store_true")
    ap.add_argument("--url-override", nargs=2, action="append", metavar=("REPO", "URL"), default=[],
                    help="clone REPO from URL instead of GitHub (for testing)")
    args = ap.parse_args()
    overrides = dict(args.url_override)

    out = Path(args.out)
    (out / "licenses").mkdir(parents=True, exist_ok=True)
    stats = {}
    tmp = out / "docs.jsonl.part"
    with open(tmp, "w", encoding="utf-8") as fh:
        for src in SOURCES:
            if src["use"] != "open" and not args.allow_research_only:
                print(f"skipping {src['name']} ({src['use']}; pass --allow-research-only to include)")
                continue
            clone = out / "clones" / src["name"]
            fetched = None
            for repo in [src["repo"], *src.get("mirrors", [])]:
                print(f"fetching {src['name']} from {repo} ...", flush=True)
                try:
                    sparse_clone(repo, src["paths"], clone, overrides.get(repo))
                    fetched = repo
                    break
                except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
                    print(f"  could not fetch {repo}: {type(e).__name__}", flush=True)
            if fetched is None:
                print(f"  skipping {src['name']}: no source could be fetched", flush=True)
                continue
            for lic in clone.glob("[Ll][Ii][Cc][Ee][Nn][Ss][Ee]*"):
                shutil.copy(lic, out / "licenses" / f"{src['name']}--{lic.name}")
            stats[f"{src['name']} fetched from"] = fetched
            n = 0
            for doc in collect(src, clone):
                fh.write(json.dumps(doc, ensure_ascii=False) + "\n")
                n += 1
                stats[f"{src['name']} docs"] = stats.get(f"{src['name']} docs", 0) + 1
                stats[f"{src['name']} bytes"] = stats.get(f"{src['name']} bytes", 0) + len(doc["text"].encode("utf-8"))
            print(f"{src['name']}: {n} documents")
            if not args.keep_clones:
                shutil.rmtree(clone)
    tmp.rename(out / "docs.jsonl")  # only a complete file counts as done
    (out / "summary.json").write_text(json.dumps(dict(sorted(stats.items())), indent=2))
    print(json.dumps(dict(sorted(stats.items())), indent=2))


if __name__ == "__main__":
    main()
