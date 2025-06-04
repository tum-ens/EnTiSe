#!/usr/bin/env python
"""Backward compatibility script for setuptools."""

import sys

from hatchling.build import build_wheel

if __name__ == "__main__":
    sys.argv.append("--wheel")
    build_wheel.main()
