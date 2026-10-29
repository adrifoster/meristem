"""Utility methods"""

import os
import re
import tempfile
from pathlib import Path

# identifiers (run IDs, ensemble IDs, stage names) end up in PBS job names,
# file and directory names, and job scripts, so only these characters are safe
_ID_RE = re.compile(r"^[A-Za-z0-9_\-]+$")

# one address in a #PBS -M list; deliberately loose, but with nothing that can
# break out of the #PBS line (whitespace, newlines, quotes, shell characters)
_EMAIL_RE = re.compile(r"^[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+$")


def check_id(name: str, value: str, where: str = "") -> None:
    """Check an identifier (e.g. a run ID or stage name) is a non-empty string of
    letters, digits, hyphens, and underscores

    Args:
        name (str): field name, for the error message
        value (str): identifier to check
        where (str, optional): appended to the field in the error message, e.g.
            " in ensemble 'my_ens'". Defaults to "".

    Raises:
        ValueError: if value isn't a string, is empty, or has other characters
    """
    if not isinstance(value, str):
        raise ValueError(
            f"{name}{where} must be a string, got {type(value).__name__} {value!r}."
        )
    if not value:
        raise ValueError(f"{name}{where} must not be empty.")
    if not _ID_RE.match(value):
        raise ValueError(
            f"{name} {value!r}{where} contains invalid characters. "
            "Only letters, digits, hyphens, and underscores are allowed."
        )


def check_email(name: str, value: str) -> None:
    """Check an email setting is one address, or a comma-separated list of them
    (as PBS's -M takes), of the form name@host

    Args:
        name (str): field name, for the error message
        value (str): email address(es) to check

    Raises:
        ValueError: if value isn't a string, or any entry isn't an address
    """
    if not isinstance(value, str):
        raise ValueError(
            f"{name} must be a string, got {type(value).__name__} {value!r}."
        )
    bad = [entry for entry in value.split(",") if not _EMAIL_RE.match(entry)]
    if bad:
        raise ValueError(
            f"{name} {value!r} must be an email address, or a comma-separated list "
            f"of them with no spaces (e.g. name@ucar.edu); not valid: {bad}."
        )


def to_absolute_path(
    name: str, path: Path | str | None, where: str = ""
) -> Path | None:
    """Convert a path field to a Path, checking it's absolute

    Args:
        name (str): field name, for the error message
        path (Path | str | None): path; None (an unset optional field) passes
        where (str, optional): appended to the field in the error message, e.g.
            " for stage 'spinup_ad'". Defaults to "".

    Raises:
        ValueError: if path isn't a str or Path, is empty, or is relative
        (hinting at an unset environment variable if it still contains a $)

    Returns:
        Path | None: path as a Path, or None
    """
    if path is None:
        return None
    if not isinstance(path, (str, Path)):
        raise ValueError(
            f"{name}{where} must be a path, got {type(path).__name__} {path!r}."
        )
    # checked before the Path conversion, which would turn "" into "."
    if not str(path):
        raise ValueError(f"{name}{where} must not be empty.")
    path = Path(path)
    if not path.is_absolute():
        hint = " (is an environment variable unset?)" if "$" in str(path) else ""
        raise ValueError(f"{name} {str(path)!r}{where} must be an absolute path{hint}.")
    return path


def write_atomically(path: Path, text: str) -> None:
    """Write text to path so that readers only ever see the old or the new
    contents, even if the process is killed or another process writes the same
    file at the same time

    The text goes to a uniquely named temporary file in the same directory,
    which then replaces path in one rename. A fixed temporary name would let two
    writers overwrite each other's temporary file mid-write.

    Args:
        path (Path): file to write
        text (str): its new contents

    Raises:
        OSError: if the file can't be written; path is left as it was
    """
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        # mkstemp makes the file private (0600); use the usual permissions
        os.chmod(tmp, 0o666 & ~_umask())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def _umask() -> int:
    """The process's current umask (os.umask can only be read by setting it)"""
    current = os.umask(0)
    os.umask(current)
    return current
