"""unit-fix: single-turn Python repair with a sandboxed unit-test grader.

The model is given a small broken function and the failing tests that
expose the bug. Hidden tests also run. The answer is the patched
function inside ``<answer>`` tags.

Gold is the in-repo repair for that family. Dataset construction
executes gold (must pass) and the buggy original (must fail). Reward
is behavioral — all tests pass — plus a small XML-format bonus and
partial credit when the visible tests pass but a hidden edge still
fails.

The naive policy (echo the original function) must not score 1.2.
"""

from __future__ import annotations

import ast
import copy
import json
import random
import re
import sys
import time
from dataclasses import dataclass, field
from textwrap import dedent
from typing import Any, Literal

# NOTE: This is a truncated stub for the tool call. Full content is in the local package and zip. The full file will be pushed via user PowerShell from the zip to avoid message size limits.

def load_environment(**kwargs):
    raise NotImplementedError('Full source is in the package zip; re-push from local.')
