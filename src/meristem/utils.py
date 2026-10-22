"""Utility methods"""

from pathlib import Path


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
        ValueError: if path is relative, hinting at an unset environment
        variable if it still contains a $

    Returns:
        Path | None: path as a Path, or None
    """
    if path is None:
        return None
    path = Path(path)
    if not path.is_absolute():
        hint = " (is an environment variable unset?)" if "$" in str(path) else ""
        raise ValueError(f"{name} {str(path)!r}{where} must be an absolute path{hint}.")
    return path
