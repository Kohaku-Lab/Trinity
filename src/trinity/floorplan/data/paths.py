"""Locate (and, when missing, download) FloorSet and the bookshelf benchmarks.

Every dataset resolves the same way: an explicit ``override`` argument, then its environment
variable, then the project data directory ``<repo>/data`` (``TRINITY_DATA`` overrides it).
When none holds the data and downloading is allowed, it is fetched into the project data
directory.

| dataset              | environment variable  | layout under the root                         |
|----------------------|-----------------------|-----------------------------------------------|
| FloorSet validation  | ``TRINITY_FLOORSET``  | ``LiteTensorDataTest/config_<21..120>/``      |
| FloorSet train       | ``TRINITY_TRAIN_LANCE`` | ``floorset_lite_mibfix.lance`` (a script builds it) |
| GSRC                 | ``TRINITY_GSRC``      | ``gsrc/{HARD,SOFT}/n<size>.{blocks,nets,pl}`` |
| MCNC                 | ``TRINITY_MCNC``      | ``mcnc/{HARD,SOFT}/<name>.{blocks,nets,pl}``  |
"""

import os
import ssl
import tarfile
import urllib.request
from collections.abc import Callable
from pathlib import Path

from tqdm import tqdm

VALIDATION_DIRNAME = "LiteTensorDataTest"
VALIDATION_URL = (
    "https://huggingface.co/datasets/IntelLabs/FloorSet/resolve/main/"
    "LiteTensorDataTest.tar.gz"
)

TRAIN_DIRNAME = "floorset_lite"
TRAIN_LANCE_DIRNAME = "floorset_lite_mibfix.lance"
TRAIN_URL = (
    "https://huggingface.co/datasets/IntelLabs/FloorSet/resolve/main/"
    "LiteTensorData_v2.tar.gz"
)

BOOKSHELF_SUFFIXES = ("blocks", "nets", "pl")

GSRC_DIRNAME = "gsrc"
GSRC_BASE_URL = "http://vlsicad.eecs.umich.edu/BK/GSRCbench"
GSRC_VARIANTS = ("HARD", "SOFT")
GSRC_SIZES = (10, 30, 50, 100, 200, 300)

MCNC_DIRNAME = "mcnc"
MCNC_BASE_URL = "https://vlsicad.eecs.umich.edu/BK/MCNCbench"
MCNC_VARIANTS = ("HARD", "SOFT")
MCNC_NAMES = ("apte", "xerox", "hp", "ami33", "ami49")


def project_data_root() -> Path:
    """The project data directory: ``TRINITY_DATA`` or ``<repo>/data``."""
    env = os.environ.get("TRINITY_DATA")
    if env:
        return Path(env)
    repo_root = Path(__file__).resolve().parents[4]
    return repo_root / "data"


def _candidate_roots(override: str | None, env_var: str) -> list[Path]:
    """The roots to probe, in order: ``override``, ``$env_var``, the project data dir."""
    roots = [Path(override)] if override else []
    if os.environ.get(env_var):
        roots.append(Path(os.environ[env_var]))
    roots.append(project_data_root())
    return roots


def _find_root(
    override: str | None,
    env_var: str,
    has_data: Callable[[Path], bool],
    download: Callable[[Path], None],
    allow_download: bool,
    name: str,
) -> Path:
    """The first candidate root holding the data, downloading into the project dir if allowed."""
    candidates = _candidate_roots(override, env_var)
    for root in candidates:
        if has_data(root):
            return root
    data_root = project_data_root()
    if allow_download:
        download(data_root)
        if has_data(data_root):
            return data_root
    probed = [str(root) for root in candidates]
    raise FileNotFoundError(
        f"{name} not found. Set {env_var}, place it under {data_root}, "
        f"or pass allow_download=True. Probed: {probed}"
    )


def _download_tar(url: str, root: Path, archive_name: str, desc: str) -> None:
    """Stream ``url`` to ``root/archive_name``, extract it into ``root``, then delete it."""
    root.mkdir(parents=True, exist_ok=True)
    archive = root / archive_name
    if not archive.exists():
        response = urllib.request.urlopen(url)
        total = int(response.info()["Content-Length"])
        with (
            open(archive, "wb") as f,
            tqdm(total=total, unit="B", unit_scale=True, desc=desc) as bar,
        ):
            while block := response.read(1 << 20):
                f.write(block)
                bar.update(len(block))
    with tarfile.open(archive) as tar:
        tar.extractall(root)
    archive.unlink()


def _download_files(urls_and_paths: list[tuple[str, Path]], desc: str) -> None:
    """Fetch every ``(url, path)`` pair from the UMich bookshelf mirror."""
    # The mirror's certificate chain is stale; these are public, read-only text files.
    context = ssl.create_default_context()
    context.check_hostname = False
    context.verify_mode = ssl.CERT_NONE
    for url, path in tqdm(urls_and_paths, desc=desc, unit="file"):
        path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url, context=context) as response:
            path.write_bytes(response.read())


# ---- FloorSet ----------------------------------------------------------------


def _has_validation(root: Path) -> bool:
    config_dir = root / VALIDATION_DIRNAME
    return config_dir.is_dir() and sum(1 for _ in config_dir.glob("config_*")) >= 100


def _download_validation(root: Path) -> None:
    _download_tar(VALIDATION_URL, root, "LiteTensorDataTest.tar.gz", "FloorSet val")


def find_floorset_root(
    override: str | None = None, allow_download: bool = True
) -> Path:
    """The directory that contains ``LiteTensorDataTest/``."""
    return _find_root(
        override,
        "TRINITY_FLOORSET",
        _has_validation,
        _download_validation,
        allow_download,
        "FloorSet validation set",
    )


def validation_case_path(
    root: Path, n_blocks: int, identifier: int = 1
) -> tuple[Path, Path]:
    """The ``(litedata, litelabel)`` paths of one validation case (``config_<n_blocks>``)."""
    config_dir = root / VALIDATION_DIRNAME / f"config_{n_blocks}"
    data = config_dir / f"litedata_{identifier}.pth"
    label = config_dir / f"litelabel_{identifier}.pth"
    return data, label


def find_train_lance(override: str | None = None) -> Path:
    """The transcoded 1M-layout train set ``floorset_lite_mibfix.lance``.

    Raises ``FileNotFoundError`` with the build command when it is absent; the Lance set is
    produced by ``scripts/data/transcode_floorset.py``, never downloaded.
    """
    for candidate in (override, os.environ.get("TRINITY_TRAIN_LANCE")):
        if candidate and Path(candidate).is_dir():
            return Path(candidate)
    local = project_data_root() / TRAIN_LANCE_DIRNAME
    if local.is_dir():
        return local
    raise FileNotFoundError(
        f"train Lance dataset not found at {local}. Build it with:\n"
        "  kogine run scripts/data/transcode_floorset.py"
    )


def download_train_raw(root: Path | None = None) -> Path:
    """Fetch and extract the raw 1M ``.th`` tree (``LiteTensorData_v2``, ~6.6 GB).

    Returns the ``floorset_lite/`` directory under ``root`` (default: the project data dir).
    """
    root = project_data_root() if root is None else Path(root)
    tree = root / TRAIN_DIRNAME
    if not tree.is_dir():
        _download_tar(TRAIN_URL, root, "LiteTensorData_v2.tar.gz", "FloorSet train")
    return tree


# ---- Bookshelf benchmarks (GSRC, MCNC) ---------------------------------------


def _bookshelf_files(variants, names) -> list[tuple[str, str, str]]:
    """Every ``(variant, name, suffix)`` of one benchmark suite."""
    return [(v, n, s) for v in variants for n in names for s in BOOKSHELF_SUFFIXES]


def _gsrc_files():
    return _bookshelf_files(GSRC_VARIANTS, [f"n{n}" for n in GSRC_SIZES])


def _mcnc_files():
    return _bookshelf_files(MCNC_VARIANTS, MCNC_NAMES)


def _has_gsrc(root: Path) -> bool:
    base = root / GSRC_DIRNAME
    return all((base / v / f"{n}.{s}").is_file() for v, n, s in _gsrc_files())


def _download_gsrc(root: Path) -> None:
    base = root / GSRC_DIRNAME
    pairs = [
        (f"{GSRC_BASE_URL}/{v}/{n}.{s}", base / v / f"{n}.{s}")
        for v, n, s in _gsrc_files()
    ]
    _download_files(pairs, "GSRC")


def find_gsrc_root(override: str | None = None, allow_download: bool = True) -> Path:
    """The ``gsrc/`` directory (``{HARD,SOFT}/n<size>.<suffix>``)."""
    root = _find_root(
        override,
        "TRINITY_GSRC",
        _has_gsrc,
        _download_gsrc,
        allow_download,
        "GSRC benchmark",
    )
    return root / GSRC_DIRNAME


def _has_mcnc(root: Path) -> bool:
    base = root / MCNC_DIRNAME
    return all((base / v / f"{n}.{s}").is_file() for v, n, s in _mcnc_files())


def _download_mcnc(root: Path) -> None:
    base = root / MCNC_DIRNAME
    pairs = [
        (f"{MCNC_BASE_URL}/{v}/{n}.{s}", base / v / f"{n}.{s}")
        for v, n, s in _mcnc_files()
    ]
    _download_files(pairs, "MCNC")


def find_mcnc_root(override: str | None = None, allow_download: bool = True) -> Path:
    """The ``mcnc/`` directory (``{HARD,SOFT}/<name>.<suffix>``)."""
    root = _find_root(
        override,
        "TRINITY_MCNC",
        _has_mcnc,
        _download_mcnc,
        allow_download,
        "MCNC benchmark",
    )
    return root / MCNC_DIRNAME
