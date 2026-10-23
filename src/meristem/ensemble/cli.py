#!/usr/bin/env python3
"""
CLI for a ensemble run (the `meristem ensemble` command).
"""

import argparse
from pathlib import Path

from .model_ensemble import Ensemble

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="meristem ensemble",
        description="Drive an ensemble of model runs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    return parser
    
  