"""Helper module for top-level commands"""

import shlex
import sys


def meristem_cmd() -> str:
    """Get the shell command that runs this meristem install.

    Uses the absolute path of the running Python interpreter with
    ``-m meristem``, so generated job scripts call the same meristem that
    wrote them, without depending on PATH or an activated environment.

    Returns:
        str: shell-quoted command, e.g. ``/path/to/python -m meristem``
    """
    return shlex.join([sys.executable, "-m", "meristem"])
