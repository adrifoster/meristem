"""
Allow running meristem as `python -m meristem`.

Equivalent to the `meristem` command. Job scripts use this form, called
through the absolute path of the Python that wrote them, so jobs run the same
meristem install without activating an environment.
"""

import sys

from meristem.cli import main

sys.exit(main())
