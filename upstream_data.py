"""Access data from the upstream Anthropic Jacobian Lens repository.

No checkout is required inside this project. By default, the upstream repository
archive is downloaded once to the user's cache directory. If a checkout already
exists, set JLENS_UPSTREAM_DIR to reuse it instead.
"""

from __future__ import annotations

import os
import shutil
import tarfile
import tempfile
import urllib.request
from pathlib import Path

UPSTREAM_ARCHIVE_URL = (
    "https://github.com/anthropics/jacobian-lens/archive/refs/heads/main.tar.gz"
)
CACHE_ROOT = Path(
    os.environ.get(
        "JLENS_PROBE_CACHE",
        Path.home() / ".cache" / "jlens-probe",
    )
).expanduser()


def get_upstream_root() -> Path:
    """Return a usable upstream jacobian-lens repository tree.

    If JLENS_UPSTREAM_DIR is set, that checkout is used. Otherwise the upstream
    main-branch archive is downloaded and cached outside this repository.
    """
    override = os.environ.get("JLENS_UPSTREAM_DIR")
    if override:
        root = Path(override).expanduser().resolve()
        if not root.exists():
            raise FileNotFoundError(
                f"JLENS_UPSTREAM_DIR does not exist: {root}"
            )
        return root

    root = CACHE_ROOT / "jacobian-lens-main"
    if root.exists():
        return root

    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"Downloading upstream Jacobian Lens data to {CACHE_ROOT} ...")

    with tempfile.TemporaryDirectory(dir=CACHE_ROOT) as tmpdir:
        tmpdir = Path(tmpdir)
        archive = tmpdir / "jacobian-lens-main.tar.gz"
        try:
            urllib.request.urlretrieve(UPSTREAM_ARCHIVE_URL, archive)
        except Exception as exc:
            raise RuntimeError(
                "Could not download the upstream Jacobian Lens repository. "
                "Connect to the internet and retry, or set JLENS_UPSTREAM_DIR "
                "to an existing anthropics/jacobian-lens checkout."
            ) from exc

        with tarfile.open(archive, "r:gz") as tar:
            # The archive comes from the fixed official GitHub repository URL.
            # Still reject entries that would escape the temporary directory.
            base = tmpdir.resolve()
            for member in tar.getmembers():
                target = (tmpdir / member.name).resolve()
                if base not in target.parents and target != base:
                    raise RuntimeError("Unsafe path in upstream archive.")
            tar.extractall(tmpdir)

        extracted = tmpdir / "jacobian-lens-main"
        if not extracted.exists():
            raise RuntimeError("Unexpected upstream archive layout.")
        shutil.move(str(extracted), str(root))

    return root


def get_evaluation_dir() -> Path:
    """Return the upstream lens-evaluation data directory."""
    path = get_upstream_root() / "data" / "evaluations"
    if not path.exists():
        raise FileNotFoundError(
            f"Upstream evaluation data directory not found: {path}"
        )
    return path
