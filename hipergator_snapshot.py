#!/usr/bin/env python3
"""Capture a single-file text snapshot of a HiPerGator project tree.

Designed to be run on HiPerGator (e.g. on a login node or inside a SLURM job)
to give enough context about the pieces of the project that do not live in the git repo:

  - conda environments under ``conda_envs/`` (package listings, not contents)
  - trained LoRA adapters and their configs
  - RAG Chroma persistence files
  - any custom scripts, SLURM jobs, YAML configs, READMEs sitting on /blue

Usage:
    python hipergator_snapshot.py \\
        [--root /blue/jasondeanarnold/SPARCP] \\
        [--output sparcp_snapshot_<host>_<date>.txt] \\
        [--max-content-bytes 200000] \\
        [--include-notebooks]

The script is stdlib-only (no pip deps) so it runs on the base conda env or
system Python. It is read-only.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import io
import os
import platform
import socket
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_ROOT = "/blue/jasondeanarnold/SPARCP"

# Directory names we never recurse into.
SKIP_DIR_NAMES = {
    "__pycache__",
    ".git",
    ".github",
    ".ipynb_checkpoints",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".venv",
    "node_modules",
    ".cache",
    ".conda",
    ".jupyter",
    ".vscode",
    ".idea",
    "__MACOSX",
}

# Directories we list but do not descend into; we emit a summary line instead.
# Full site-packages listings would easily add 100k+ lines to the snapshot.
SUMMARIZE_DIR_NAMES = {
    "site-packages",
    "conda-meta",  # huge pile of per-package json manifests; keep summarized
    "pkgs",        # conda pkgs cache, binary-heavy
    "share",       # typically docs / locales, not useful for this snapshot
    "include",     # C headers from conda deps
}

# File suffixes we inline in full (size-capped).
TEXT_SUFFIXES = {
    ".py", ".sh", ".bash", ".zsh", ".slurm", ".sbatch",
    ".yml", ".yaml", ".json", ".toml", ".ini", ".cfg", ".conf",
    ".txt", ".md", ".rst", ".csv", ".tsv",
    ".env", ".example",
    ".dockerfile", ".containerfile",
    ".sql", ".xml", ".html", ".css", ".js", ".ts",
    ".co", ".colang",  # NeMo Guardrails
    ".jinja", ".j2", ".tmpl", ".template",
    ".gitignore", ".gitattributes",
    ".log",
}

# Suffixes we never inline; we report name + size only.
BINARY_SUFFIXES = {
    ".bin", ".safetensors", ".pt", ".pth", ".ckpt", ".onnx", ".gguf", ".h5",
    ".pkl", ".pickle", ".npy", ".npz", ".parquet", ".arrow", ".feather",
    ".whl", ".tar", ".tar.gz", ".tgz", ".zip", ".gz", ".bz2", ".xz", ".7z",
    ".wav", ".mp3", ".mp4", ".flac", ".m4a", ".ogg", ".webm",
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".pdf", ".svg",
    ".so", ".dylib", ".dll", ".a", ".o",
    ".sqlite", ".sqlite3", ".db",
}

# Specific basenames that are text-ish even without a useful suffix.
TEXT_BASENAMES = {
    "Dockerfile", "Containerfile", "Makefile", "GNUmakefile",
    "LICENSE", "COPYING", "NOTICE", "README", "CHANGELOG", "AUTHORS",
    "requirements", "requirements.in",
}

# Per-file content cap (bytes). Overridable via --max-content-bytes.
DEFAULT_MAX_CONTENT = 200_000

# Global cap on total inlined content bytes to keep the snapshot shareable.
DEFAULT_TOTAL_CONTENT_CAP = 20 * 1024 * 1024  # 20 MB

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def human_size(n: int) -> str:
    f = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if f < 1024.0 or unit == "TB":
            return f"{f:,.1f} {unit}" if unit != "B" else f"{int(f)} B"
        f /= 1024.0
    return f"{f:,.1f} TB"


def is_text_file(path: Path) -> bool:
    name = path.name
    if name in TEXT_BASENAMES:
        return True
    suffix = path.suffix.lower()
    if suffix in BINARY_SUFFIXES:
        return False
    if suffix in TEXT_SUFFIXES:
        return True
    # Composite suffixes (.tar.gz)
    for composite in (".tar.gz", ".tar.bz2", ".tar.xz"):
        if name.lower().endswith(composite):
            return False
    return False


def run_cmd(cmd: List[str], timeout: int = 60) -> str:
    """Run a command and return stdout+stderr as a single string, never raising."""
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        out = proc.stdout or ""
        err = proc.stderr or ""
        combined = out
        if err.strip():
            combined += ("\n[stderr]\n" + err)
        if proc.returncode != 0:
            combined += f"\n[exit code {proc.returncode}]"
        return combined.rstrip() + "\n"
    except FileNotFoundError:
        return f"[command not found: {cmd[0]}]\n"
    except subprocess.TimeoutExpired:
        return f"[timeout after {timeout}s: {' '.join(cmd)}]\n"
    except Exception as exc:
        return f"[error running {' '.join(cmd)}: {exc}]\n"


def dir_summary(path: Path) -> Tuple[int, int]:
    """Return (file_count, total_bytes) for everything under path, shallow walk."""
    count = 0
    total = 0
    try:
        for root, _dirs, files in os.walk(path):
            for name in files:
                fp = os.path.join(root, name)
                try:
                    total += os.path.getsize(fp)
                    count += 1
                except OSError:
                    pass
    except OSError:
        pass
    return count, total


# ---------------------------------------------------------------------------
# Snapshot pieces
# ---------------------------------------------------------------------------


def write_header(out: io.StringIO, root: Path) -> None:
    out.write("=" * 78 + "\n")
    out.write(f"HIPERGATOR SNAPSHOT -- {root}\n")
    out.write("=" * 78 + "\n\n")
    out.write(f"host:      {socket.gethostname()}\n")
    out.write(f"user:      {os.environ.get('USER', os.environ.get('USERNAME', '?'))}\n")
    out.write(f"date:      {_dt.datetime.now().isoformat(timespec='seconds')}\n")
    out.write(f"root:      {root}\n")
    out.write(f"python:    {sys.version.split()[0]} ({sys.executable})\n")
    out.write(f"platform:  {platform.platform()}\n")
    out.write("\n")


def write_system_section(out: io.StringIO) -> None:
    out.write("-" * 78 + "\n")
    out.write("SYSTEM\n")
    out.write("-" * 78 + "\n\n")
    for label, cmd in [
        ("uname -a",            ["uname", "-a"]),
        ("cat /etc/os-release", ["cat", "/etc/os-release"]),
        ("hostname -I",         ["hostname", "-I"]),
        ("whoami / id",         ["id"]),
        ("groups",              ["groups"]),
        ("free -h",             ["free", "-h"]),
        ("lscpu (head)",        ["lscpu"]),
        ("nvidia-smi",          ["nvidia-smi"]),
        ("module list",         ["bash", "-lc", "module list 2>&1"]),
        ("conda --version",     ["conda", "--version"]),
        ("which conda",         ["which", "conda"]),
        ("df -h /blue",         ["df", "-h", "/blue"]),
    ]:
        out.write(f"$ {label}\n")
        out.write(run_cmd(cmd, timeout=30))
        out.write("\n")


def write_disk_usage(out: io.StringIO, root: Path) -> None:
    out.write("-" * 78 + "\n")
    out.write(f"DISK USAGE (du -h --max-depth=2 {root})\n")
    out.write("-" * 78 + "\n\n")
    out.write(run_cmd(["du", "-h", "--max-depth=2", str(root)], timeout=180))
    out.write("\n")


def iter_visible(path: Path) -> Iterable[Path]:
    """Sorted immediate children of path (dirs first, then files)."""
    try:
        entries = list(path.iterdir())
    except (PermissionError, OSError) as exc:
        return []
    entries.sort(key=lambda p: (not p.is_dir(), p.name.lower()))
    return entries


def write_tree(
    out: io.StringIO,
    root: Path,
    max_depth: int,
) -> List[Path]:
    """Write the directory tree and return the list of text files to inline later."""
    out.write("-" * 78 + "\n")
    out.write(f"DIRECTORY TREE (max-depth {max_depth})\n")
    out.write("-" * 78 + "\n\n")

    text_files: List[Path] = []

    def walk(p: Path, prefix: str, depth: int) -> None:
        if depth > max_depth:
            return
        entries = list(iter_visible(p))
        if not entries:
            return
        for i, child in enumerate(entries):
            last = (i == len(entries) - 1)
            branch = "`-- " if last else "|-- "
            next_prefix = prefix + ("    " if last else "|   ")
            try:
                is_dir = child.is_dir() and not child.is_symlink()
            except OSError:
                is_dir = False

            if is_dir:
                if child.name in SKIP_DIR_NAMES:
                    out.write(f"{prefix}{branch}{child.name}/  [skipped]\n")
                    continue
                if child.name in SUMMARIZE_DIR_NAMES:
                    count, total = dir_summary(child)
                    out.write(
                        f"{prefix}{branch}{child.name}/  [{count:,} files, "
                        f"{human_size(total)} -- contents omitted]\n"
                    )
                    continue
                out.write(f"{prefix}{branch}{child.name}/\n")
                walk(child, next_prefix, depth + 1)
            else:
                try:
                    size = child.stat().st_size
                except OSError:
                    size = 0
                marker = ""
                if child.is_symlink():
                    try:
                        target = os.readlink(child)
                        marker = f" -> {target}"
                    except OSError:
                        marker = " -> ?"
                out.write(
                    f"{prefix}{branch}{child.name}  ({human_size(size)}){marker}\n"
                )
                if is_text_file(child):
                    text_files.append(child)

    out.write(f"{root}/\n")
    walk(root, "", 1)
    out.write("\n")
    return text_files


def find_conda_envs(root: Path) -> List[Path]:
    """Return candidate conda env prefixes under <root>/conda_envs/ and <root>."""
    envs: List[Path] = []
    candidates = [root / "conda_envs", root]
    for container in candidates:
        if not container.is_dir():
            continue
        for child in container.iterdir():
            if not child.is_dir():
                continue
            # An env typically has bin/python and conda-meta/
            if (child / "bin" / "python").exists() or (child / "conda-meta").is_dir():
                envs.append(child)
    # Deduplicate preserving order
    seen = set()
    uniq: List[Path] = []
    for e in envs:
        key = str(e.resolve())
        if key in seen:
            continue
        seen.add(key)
        uniq.append(e)
    return uniq


def write_conda_envs(out: io.StringIO, root: Path) -> None:
    out.write("-" * 78 + "\n")
    out.write("CONDA ENVIRONMENTS\n")
    out.write("-" * 78 + "\n\n")
    envs = find_conda_envs(root)
    if not envs:
        out.write(f"[no conda envs found under {root} or {root}/conda_envs]\n\n")
        return
    for env in envs:
        out.write(f"### {env}\n\n")
        py = env / "bin" / "python"
        if py.exists():
            out.write("$ <env>/bin/python --version\n")
            out.write(run_cmd([str(py), "--version"], timeout=15))
            out.write("\n")
        out.write("$ conda list -p <env>\n")
        out.write(run_cmd(["conda", "list", "-p", str(env)], timeout=120))
        out.write("\n")
        pip = env / "bin" / "pip"
        if pip.exists():
            out.write("$ <env>/bin/pip freeze\n")
            out.write(run_cmd([str(pip), "freeze"], timeout=60))
            out.write("\n")
        out.write("\n")


def write_model_inventory(out: io.StringIO, root: Path) -> None:
    out.write("-" * 78 + "\n")
    out.write("TRAINED MODEL INVENTORY\n")
    out.write("-" * 78 + "\n\n")
    candidates = [
        root / "trained_models",
        root / "models",
        root / "adapters",
    ]
    found_any = False
    for container in candidates:
        if not container.is_dir():
            continue
        found_any = True
        out.write(f"### {container}\n\n")
        for child in sorted(container.iterdir()):
            if not child.is_dir():
                continue
            count, total = dir_summary(child)
            out.write(f"- {child.name}/  ({count:,} files, {human_size(total)})\n")
            for fname in ("adapter_config.json", "config.json",
                          "tokenizer_config.json", "training_args.bin",
                          "README.md"):
                fp = child / fname
                if fp.is_file() and is_text_file(fp):
                    try:
                        text = fp.read_text(encoding="utf-8", errors="replace")
                    except OSError as exc:
                        text = f"[read error: {exc}]"
                    out.write(f"\n  --- {fname} ---\n")
                    for line in text.splitlines():
                        out.write(f"  {line}\n")
            # List large weight artifacts by name+size
            for weight in sorted(child.rglob("*")):
                if weight.is_file() and weight.suffix.lower() in {
                    ".safetensors", ".bin", ".pt", ".pth", ".gguf", ".onnx",
                }:
                    try:
                        sz = weight.stat().st_size
                    except OSError:
                        sz = 0
                    out.write(
                        f"  [weight] {weight.relative_to(child)}  ({human_size(sz)})\n"
                    )
            out.write("\n")
        out.write("\n")
    if not found_any:
        out.write("[no trained_models/ or models/ or adapters/ dir found]\n\n")


def write_rag_inventory(out: io.StringIO, root: Path) -> None:
    out.write("-" * 78 + "\n")
    out.write("RAG (Chroma) INVENTORY\n")
    out.write("-" * 78 + "\n\n")
    candidates = list(root.rglob("chroma*")) + list(root.rglob("rag"))
    rag_dirs = [p for p in candidates if p.is_dir()]
    if not rag_dirs:
        out.write("[no chroma* or rag/ directory found under root]\n\n")
        return
    seen = set()
    for rag in rag_dirs:
        key = str(rag.resolve())
        if key in seen:
            continue
        seen.add(key)
        count, total = dir_summary(rag)
        out.write(f"### {rag}  ({count:,} files, {human_size(total)})\n")
        for child in sorted(rag.iterdir()):
            try:
                sz = child.stat().st_size if child.is_file() else 0
            except OSError:
                sz = 0
            kind = "dir" if child.is_dir() else "file"
            out.write(f"  - {child.name} ({kind}, {human_size(sz)})\n")
        out.write("\n")


def write_file_contents(
    out: io.StringIO,
    files: List[Path],
    max_content_bytes: int,
    total_cap: int,
    include_notebooks: bool,
) -> None:
    out.write("-" * 78 + "\n")
    out.write(f"FILE CONTENTS (per-file cap {human_size(max_content_bytes)}, "
              f"global cap {human_size(total_cap)})\n")
    out.write("-" * 78 + "\n\n")
    total = 0
    skipped_big = 0
    skipped_nb = 0
    for f in sorted(set(files)):
        if f.suffix.lower() == ".ipynb" and not include_notebooks:
            skipped_nb += 1
            continue
        try:
            size = f.stat().st_size
        except OSError:
            continue
        if total + min(size, max_content_bytes) > total_cap:
            skipped_big += 1
            continue
        out.write("=" * 78 + "\n")
        out.write(f"### {f}  ({human_size(size)})\n")
        out.write("=" * 78 + "\n")
        try:
            with open(f, "rb") as fh:
                raw = fh.read(max_content_bytes + 1)
        except OSError as exc:
            out.write(f"[read error: {exc}]\n\n")
            continue
        truncated = len(raw) > max_content_bytes
        if truncated:
            raw = raw[:max_content_bytes]
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            text = raw.decode("latin-1", errors="replace")
        # Heuristic: if the decoded text looks binary (NUL-heavy), skip body.
        if text.count("\x00") > 16:
            out.write("[looks binary; body omitted]\n\n")
            continue
        out.write(text)
        if not text.endswith("\n"):
            out.write("\n")
        if truncated:
            out.write(f"... [truncated at {human_size(max_content_bytes)}] ...\n")
        out.write("\n")
        total += len(raw)
    out.write(
        f"\n[inlined total: {human_size(total)}; skipped due to global cap: "
        f"{skipped_big}; notebooks skipped: {skipped_nb}]\n"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=DEFAULT_ROOT,
                        help=f"Project root to snapshot (default: {DEFAULT_ROOT})")
    parser.add_argument("--output", default=None,
                        help="Output .txt file (default: sparcp_snapshot_<host>_<YYYYMMDD-HHMM>.txt)")
    parser.add_argument("--max-depth", type=int, default=6,
                        help="Max tree depth to descend (default: 6)")
    parser.add_argument("--max-content-bytes", type=int, default=DEFAULT_MAX_CONTENT,
                        help=f"Per-file content cap in bytes (default: {DEFAULT_MAX_CONTENT})")
    parser.add_argument("--total-content-cap", type=int, default=DEFAULT_TOTAL_CONTENT_CAP,
                        help=f"Total inlined content cap in bytes (default: {DEFAULT_TOTAL_CONTENT_CAP})")
    parser.add_argument("--include-notebooks", action="store_true",
                        help="Also inline .ipynb files (usually large; off by default)")
    parser.add_argument("--skip-conda", action="store_true",
                        help="Skip `conda list` / `pip freeze` calls")
    parser.add_argument("--skip-system", action="store_true",
                        help="Skip the SYSTEM section (uname, nvidia-smi, etc.)")
    parser.add_argument("--skip-du", action="store_true",
                        help="Skip the du disk-usage sweep (can be slow on /blue)")
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    if not root.is_dir():
        print(f"ERROR: root path does not exist or is not a directory: {root}",
              file=sys.stderr)
        return 2

    if args.output:
        output = Path(args.output).expanduser().resolve()
    else:
        stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M")
        host = socket.gethostname().split(".")[0]
        output = Path.cwd() / f"sparcp_snapshot_{host}_{stamp}.txt"

    buf = io.StringIO()
    write_header(buf, root)
    if not args.skip_system:
        write_system_section(buf)
    if not args.skip_du:
        write_disk_usage(buf, root)

    text_files = write_tree(buf, root, max_depth=args.max_depth)

    if not args.skip_conda:
        write_conda_envs(buf, root)
    write_model_inventory(buf, root)
    write_rag_inventory(buf, root)

    write_file_contents(
        buf,
        text_files,
        max_content_bytes=args.max_content_bytes,
        total_cap=args.total_content_cap,
        include_notebooks=args.include_notebooks,
    )

    output.write_text(buf.getvalue(), encoding="utf-8")
    total_bytes = output.stat().st_size
    print(f"Wrote {output}  ({human_size(total_bytes)})")
    print(f"Share it with:  scp {output} <dest>")
    return 0


if __name__ == "__main__":
    sys.exit(main())
