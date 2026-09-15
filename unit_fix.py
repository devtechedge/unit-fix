"""unit-test: single-turn Python repair with a sandboxed exec grader.

The model receives a small broken function plus failing visible unit tests
and must put a patched function body in ``<answer>...</answer>``. Reward is
behavioral: every visible + hidden test must pass. Dataset construction
executes gold (must pass) and the original buggy function (must fail a
visible test), so naive echo never farms partial credit.
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
from typing import Any, Callable, Literal

Difficulty = Literal["easy", "medium", "hard"]
FamilyName = Literal[
    "nth_item",
    "sum_through",
    "mean_or_none",
    "codepoint_count",
    "with_item",
    "closed_slice",
    "rotate_left",
    "safe_ratio",
    "window_count",
    "same_letters",
    "field_count",
    "unique_keep_order",
    "clamp",
    "chunks",
    "first_index",
]

ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)
FENCE_RE = re.compile(r"^```(?:python|py)?\s*|\s*```$", re.IGNORECASE | re.MULTILINE)

DIFFICULTIES: tuple[Difficulty, ...] = ("easy", "medium", "hard")
DIFFICULTY_WEIGHTS = {"easy": 0.40, "medium": 0.35, "hard": 0.25}

FAMILIES: tuple[FamilyName, ...] = (
    "nth_item",
    "sum_through",
    "mean_or_none",
    "codepoint_count",
    "with_item",
    "closed_slice",
    "rotate_left",
    "safe_ratio",
    "window_count",
    "same_letters",
    "field_count",
    "unique_keep_order",
    "clamp",
    "chunks",
    "first_index",
)

MAX_STEPS = 50_000
MAX_WALL_SECONDS = 0.5
MAX_RANGE_LEN = 10_000
MAX_SUM_LEN = 10_000

SYSTEM_PROMPT = """You are repairing a small Python function.

Rules:
- You may reason before answering.
- Put the complete repaired function (def ... including the body) inside <answer>...</answer>.
- Do not use imports, dunder attributes, file/network I/O, or unbounded loops.
- Make the function pass the visible unit tests. Hidden tests also apply.
- Prefer a correct general fix over hardcoding the visible example.

Respond in this format:
<answer>
def function_name(...):
    ...
</answer>
"""


# ---------------------------------------------------------------------------
# Family implementations - buggy body (prompt) + gold repair
# ---------------------------------------------------------------------------


FAMILY_IMPL: dict[str, dict[str, Any]] = {
    'nth_item': {
        'func_name': 'nth_item',
        'signature': 'def nth_item(items, n):',
        'doc': 'Return the n-th item using 1-based indexing (n=1 is the first element).',
        'buggy': 'def nth_item(items, n):\n    """Return the n-th item using 1-based indexing (n=1 is the first element)."""\n    return items[n]\n',
        'gold': 'def nth_item(items, n):\n    """Return the n-th item using 1-based indexing (n=1 is the first element)."""\n    return items[n - 1]\n',
    },
    'sum_through': {
        'func_name': 'sum_through',
        'signature': 'def sum_through(lo, hi):',
        'doc': 'Return the inclusive sum of integers from lo through hi.',
        'buggy': 'def sum_through(lo, hi):\n    """Return the inclusive sum of integers from lo through hi."""\n    return sum(range(lo, hi))\n',
        'gold': 'def sum_through(lo, hi):\n    """Return the inclusive sum of integers from lo through hi."""\n    return sum(range(lo, hi + 1))\n',
    },
    'mean_or_none': {
        'func_name': 'mean_or_none',
        'signature': 'def mean_or_none(xs):',
        'doc': 'Return the arithmetic mean of xs, or None when xs is empty.',
        'buggy': 'def mean_or_none(xs):\n    """Return the arithmetic mean of xs, or None when xs is empty."""\n    return sum(xs) / len(xs)\n',
        'gold': 'def mean_or_none(xs):\n    """Return the arithmetic mean of xs, or None when xs is empty."""\n    if not xs:\n        return None\n    return sum(xs) / len(xs)\n',
    },
    'codepoint_count': {
        'func_name': 'codepoint_count',
        'signature': 'def codepoint_count(s):',
        'doc': 'Return the number of Unicode code points in s (not UTF-8 bytes).',
        'buggy': 'def codepoint_count(s):\n    """Return the number of Unicode code points in s (not UTF-8 bytes)."""\n    return len(s.encode("utf-8"))\n',
        'gold': 'def codepoint_count(s):\n    """Return the number of Unicode code points in s (not UTF-8 bytes)."""\n    return len(s)\n',
    },
    'with_item': {
        'func_name': 'with_item',
        'signature': 'def with_item(items, x):',
        'doc': 'Return a new list equal to items plus x appended. Do not mutate items.',
        'buggy': 'def with_item(items, x):\n    """Return a new list equal to items plus x appended. Do not mutate items."""\n    items.append(x)\n    return items\n',
        'gold': 'def with_item(items, x):\n    """Return a new list equal to items plus x appended. Do not mutate items."""\n    return list(items) + [x]\n',
    },
    'closed_slice': {
        'func_name': 'closed_slice',
        'signature': 'def closed_slice(xs, lo, hi):',
        'doc': 'Return xs[lo] through xs[hi] inclusive (closed interval on both ends).',
        'buggy': 'def closed_slice(xs, lo, hi):\n    """Return xs[lo] through xs[hi] inclusive (closed interval on both ends)."""\n    return xs[lo:hi]\n',
        'gold': 'def closed_slice(xs, lo, hi):\n    """Return xs[lo] through xs[hi] inclusive (closed interval on both ends)."""\n    return xs[lo:hi + 1]\n',
    },
    'rotate_left': {
        'func_name': 'rotate_left',
        'signature': 'def rotate_left(items, n):',
        'doc': 'Rotate items left by n positions. Empty list returns empty list.',
        'buggy': 'def rotate_left(items, n):\n    """Rotate items left by n positions. Empty list returns empty list."""\n    n = n % len(items)\n    return items[n:] + items[:n]\n',
        'gold': 'def rotate_left(items, n):\n    """Rotate items left by n positions. Empty list returns empty list."""\n    if not items:\n        return list(items)\n    n = n % len(items)\n    return items[n:] + items[:n]\n',
    },
    'safe_ratio': {
        'func_name': 'safe_ratio',
        'signature': 'def safe_ratio(num, den):',
        'doc': 'Return num/den as a float, or None when den is zero.',
        'buggy': 'def safe_ratio(num, den):\n    """Return num/den as a float, or None when den is zero."""\n    return num / den\n',
        'gold': 'def safe_ratio(num, den):\n    """Return num/den as a float, or None when den is zero."""\n    if den == 0:\n        return None\n    return num / den\n',
    },
    'window_count': {
        'func_name': 'window_count',
        'signature': 'def window_count(n, k):',
        'doc': 'How many contiguous windows of length k fit in a sequence of length n.',
        'buggy': 'def window_count(n, k):\n    """How many contiguous windows of length k fit in a sequence of length n."""\n    return max(0, n - k)\n',
        'gold': 'def window_count(n, k):\n    """How many contiguous windows of length k fit in a sequence of length n."""\n    if k <= 0:\n        return 0\n    return max(0, n - k + 1)\n',
    },
    'same_letters': {
        'func_name': 'same_letters',
        'signature': 'def same_letters(a, b):',
        'doc': 'True iff a and b are equal ignoring Unicode case (use casefold, not lower).',
        'buggy': 'def same_letters(a, b):\n    """True iff a and b are equal ignoring Unicode case (use casefold, not lower)."""\n    return a.lower() == b.lower()\n',
        'gold': 'def same_letters(a, b):\n    """True iff a and b are equal ignoring Unicode case (use casefold, not lower)."""\n    return a.casefold() == b.casefold()\n',
    },
    'field_count': {
        'func_name': 'field_count',
        'signature': 'def field_count(s):',
        'doc': 'Count whitespace-separated fields; treat any Unicode whitespace as a separator.',
        'buggy': 'def field_count(s):\n    """Count whitespace-separated fields; treat any Unicode whitespace as a separator."""\n    if not s:\n        return 0\n    return len(s.split(" "))\n',
        'gold': 'def field_count(s):\n    """Count whitespace-separated fields; treat any Unicode whitespace as a separator."""\n    return len(s.split())\n',
    },
    'unique_keep_order': {
        'func_name': 'unique_keep_order',
        'signature': 'def unique_keep_order(xs):',
        'doc': 'Return unique elements of xs preserving first-seen order.',
        'buggy': 'def unique_keep_order(xs):\n    """Return unique elements of xs preserving first-seen order."""\n    return sorted(set(xs))\n',
        'gold': 'def unique_keep_order(xs):\n    """Return unique elements of xs preserving first-seen order."""\n    return list(dict.fromkeys(xs))\n',
    },
    'clamp': {
        'func_name': 'clamp',
        'signature': 'def clamp(x, lo, hi):',
        'doc': 'Clamp x into the closed interval [lo, hi].',
        'buggy': 'def clamp(x, lo, hi):\n    """Clamp x into the closed interval [lo, hi]."""\n    return max(lo, min(x, hi - 1))\n',
        'gold': 'def clamp(x, lo, hi):\n    """Clamp x into the closed interval [lo, hi]."""\n    return max(lo, min(x, hi))\n',
    },
    'chunks': {
        'func_name': 'chunks',
        'signature': 'def chunks(xs, n):',
        'doc': 'Split xs into contiguous chunks of length n; keep a short final remainder.',
        'buggy': 'def chunks(xs, n):\n    """Split xs into contiguous chunks of length n; keep a short final remainder."""\n    return [xs[i:i + n] for i in range(0, len(xs) - len(xs) % n, n)]\n',
        'gold': 'def chunks(xs, n):\n    """Split xs into contiguous chunks of length n; keep a short final remainder."""\n    return [xs[i:i + n] for i in range(0, len(xs), n)]\n',
    },
    'first_index': {
        'func_name': 'first_index',
        'signature': 'def first_index(xs, target):',
        'doc': 'Return the index of the first occurrence of target in xs, or -1.',
        'buggy': 'def first_index(xs, target):\n    """Return the index of the first occurrence of target in xs, or -1."""\n    found = -1\n    for i, v in enumerate(xs):\n        if v == target:\n            found = i\n    return found\n',
        'gold': 'def first_index(xs, target):\n    """Return the index of the first occurrence of target in xs, or -1."""\n    for i, v in enumerate(xs):\n        if v == target:\n            return i\n    return -1\n',
    },
}


# ---------------------------------------------------------------------------
# Sandbox - AST gate + capped builtins + step/wall limits (no subprocess)
# ---------------------------------------------------------------------------


class SandboxError(Exception):
    """Raised when submitted code violates sandbox policy or limits."""


_IO_NAMES = frozenset(
    {
        "open",
        "input",
        "print",
        "exec",
        "eval",
        "compile",
        "__import__",
        "breakpoint",
        "exit",
        "quit",
        "help",
        "memoryview",
        "bytearray",
    }
)


class _SandboxVisitor(ast.NodeVisitor):
    def visit_Import(self, node: ast.Import) -> None:
        raise SandboxError("imports are not allowed")

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        raise SandboxError("imports are not allowed")

    def visit_While(self, node: ast.While) -> None:
        test = node.test
        if isinstance(test, ast.Constant) and test.value is True:
            raise SandboxError("while True is not allowed")
        if isinstance(test, ast.Name) and test.id == "True":
            raise SandboxError("while True is not allowed")
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if "__" in node.id:
            raise SandboxError("dunder names are not allowed")
        if node.id in _IO_NAMES:
            raise SandboxError(f"builtin {node.id!r} is not allowed")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if "__" in node.attr:
            raise SandboxError("dunder attributes are not allowed")
        self.generic_visit(node)

    def visit_Global(self, node: ast.Global) -> None:
        raise SandboxError("global is not allowed")

    def visit_Nonlocal(self, node: ast.Nonlocal) -> None:
        raise SandboxError("nonlocal is not allowed")


def _validate_ast(source: str) -> ast.Module:
    try:
        tree = ast.parse(source, mode="exec")
    except SyntaxError as exc:
        raise SandboxError(f"syntax error: {exc}") from exc
    _SandboxVisitor().visit(tree)
    return tree


def _safe_range(*args: int) -> range:
    r = range(*args)
    try:
        length = len(r)
    except OverflowError as exc:
        raise SandboxError("range too large") from exc
    if length > MAX_RANGE_LEN:
        raise SandboxError("range too large")
    return r


def _safe_sum(iterable: Any, start: Any = 0) -> Any:
    if isinstance(iterable, range):
        if len(iterable) > MAX_SUM_LEN:
            raise SandboxError("sum too large")
        return sum(iterable, start)
    total = start
    count = 0
    for item in iterable:
        count += 1
        if count > MAX_SUM_LEN:
            raise SandboxError("sum too large")
        total = total + item
    return total


def _make_trace(deadline: float) -> Callable:
    steps = {"n": 0}

    def _trace(frame: Any, event: str, arg: Any) -> Callable:
        if event == "line":
            steps["n"] += 1
            if steps["n"] > MAX_STEPS:
                raise SandboxError("step limit exceeded")
            if time.perf_counter() > deadline:
                raise SandboxError("wall-clock limit exceeded")
        return _trace

    return _trace


def _sandbox_builtins() -> dict[str, Any]:
    return {
        "abs": abs,
        "all": all,
        "any": any,
        "bool": bool,
        "dict": dict,
        "enumerate": enumerate,
        "filter": filter,
        "float": float,
        "frozenset": frozenset,
        "int": int,
        "isinstance": isinstance,
        "issubclass": issubclass,
        "len": len,
        "list": list,
        "map": map,
        "max": max,
        "min": min,
        "next": next,
        "pow": pow,
        "range": _safe_range,
        "reversed": reversed,
        "round": round,
        "set": set,
        "slice": slice,
        "sorted": sorted,
        "str": str,
        "sum": _safe_sum,
        "tuple": tuple,
        "zip": zip,
        "True": True,
        "False": False,
        "None": None,
    }


def load_function(source: str, func_name: str) -> Callable[..., Any]:
    """Parse, sandbox-validate, and exec ``source``; return ``func_name``."""
    tree = _validate_ast(source)
    code = compile(tree, filename="<unit_test>", mode="exec")
    glb: dict[str, Any] = {"__builtins__": _sandbox_builtins()}
    deadline = time.perf_counter() + MAX_WALL_SECONDS
    trace = _make_trace(deadline)
    prev = sys.gettrace()
    try:
        sys.settrace(trace)
        exec(code, glb, glb)
    finally:
        sys.settrace(prev)
    fn = glb.get(func_name)
    if not callable(fn):
        raise SandboxError(f"function {func_name!r} not defined")
    return fn


def _values_equal(a: Any, b: Any) -> bool:
    if isinstance(a, float) or isinstance(b, float):
        try:
            return abs(float(a) - float(b)) < 1e-9
        except (TypeError, ValueError):
            return False
    return a == b


@dataclass
class TestResult:
    passed: int = 0
    failed: int = 0
    visible_passed: int = 0
    visible_total: int = 0
    hidden_passed: int = 0
    hidden_total: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def all_passed(self) -> bool:
        total = self.visible_total + self.hidden_total
        return total > 0 and self.passed == total and self.failed == 0

    @property
    def visible_all_passed(self) -> bool:
        return self.visible_total > 0 and self.visible_passed == self.visible_total


def _normalize_tests(tests: Any) -> dict[str, list[dict[str, Any]]]:
    if isinstance(tests, str):
        tests = json.loads(tests)
    if not isinstance(tests, dict):
        raise TypeError("tests must be a dict or JSON object string")
    return {
        "visible": list(tests.get("visible") or []),
        "hidden": list(tests.get("hidden") or []),
    }


def _run_one(fn: Callable[..., Any], case: dict[str, Any]) -> None:
    args = copy.deepcopy(case.get("args", []))
    kwargs = copy.deepcopy(case.get("kwargs", {}))
    snapshot = copy.deepcopy(args)
    expected = case["expected"]
    deadline = time.perf_counter() + MAX_WALL_SECONDS
    trace = _make_trace(deadline)
    prev = sys.gettrace()
    try:
        sys.settrace(trace)
        result = fn(*args, **kwargs)
    finally:
        sys.settrace(prev)
    if not _values_equal(result, expected):
        raise AssertionError(f"expected {expected!r}, got {result!r}")
    if case.get("preserve_args") and args != snapshot:
        raise AssertionError("function mutated its arguments")


def run_tests(code: str, func_name: str, tests: Any) -> TestResult:
    """Execute visible + hidden cases against ``code`` inside the sandbox."""
    bundle = _normalize_tests(tests)
    result = TestResult(
        visible_total=len(bundle["visible"]),
        hidden_total=len(bundle["hidden"]),
    )
    try:
        fn = load_function(code, func_name)
    except Exception as exc:  # noqa: BLE001
        result.failed = result.visible_total + result.hidden_total
        result.errors.append(f"load: {exc}")
        return result

    for label, cases in (("visible", bundle["visible"]), ("hidden", bundle["hidden"])):
        for idx, case in enumerate(cases):
            try:
                _run_one(fn, case)
                result.passed += 1
                if label == "visible":
                    result.visible_passed += 1
                else:
                    result.hidden_passed += 1
            except Exception as exc:  # noqa: BLE001
                result.failed += 1
                result.errors.append(f"{label}[{idx}]: {exc}")
    return result

# ---------------------------------------------------------------------------
# Test-case generators per family
# ---------------------------------------------------------------------------


def _gold_call(family: str, args: list[Any], *, preserve_args: bool = False) -> dict[str, Any]:
    impl = FAMILY_IMPL[family]
    fn = load_function(impl["gold"], impl["func_name"])
    expected = fn(*copy.deepcopy(args))
    case: dict[str, Any] = {"args": copy.deepcopy(args), "expected": expected}
    if preserve_args:
        case["preserve_args"] = True
    return case


def _gen_nth_item(rng: random.Random, difficulty: Difficulty) -> tuple[list, list]:
    n = {"easy": 4, "medium": 7, "hard": 10}[difficulty]
    items = [rng.randint(0, 50) for _ in range(n)]
    visible = [_gold_call("nth_item", [items, 1]), _gold_call("nth_item", [items, n])]
    mid = max(1, n // 2)
    other = list(reversed(items)) if items != list(reversed(items)) else [x + 1 for x in items]
    hidden = [
        _gold_call("nth_item", [items, mid]),
        _gold_call("nth_item", [other, 2 if n > 1 else 1]),
    ]
    return visible, hidden


def _gen_sum_through(rng: random.Random, difficulty: Difficulty) -> tuple[list, list]:
    span = {"easy": 5, "medium": 20, "hard": 40}[difficulty]
    lo = rng.randint(0, 20)
    hi = lo + span
    visible = [_gold_call("sum_through", [lo, hi])]
    hidden = [
        _gold_call("sum_through", [lo, lo]),
        _gold_call("sum_through", [lo + 1, hi - 1 if hi > lo + 1 else hi]),
    ]
    return visible, hidden


def _gen_mean_or_none(rng: random.Random, difficulty: Difficulty) -> tuple[list, list]:
    n = {"easy": 3, "medium": 6, "hard": 10}[difficulty]
    xs = [rng.randint(0, 20) for _ in range(n)]
    # Empty list must be visible so buggy ZeroDivision cannot farm partial.
    visible = [_gold_call("mean_or_none", [[]]), _gold_call("mean_or_none", [xs])]
    hidden = [_gold_call("mean_or_none", [xs[::-1]])]
    return visible, hidden

def _gen_codepoint_count(rng: random.Random, difficulty: Difficulty) -> tuple[list, list]:
    ascii_s = "".join(
        rng.choice("abcdef ") for _ in range({"easy": 4, "medium": 8, "hard": 12}[difficulty])
    )
    # Non-ASCII must be visible so byte-count buggy fails visibly.
    visible = [
        _gold_call("codepoint_count", ["café"]),
        _gold_call("codepoint_count", [ascii_s]),
    ]
    hidden = [
        _gold_call("codepoint_count", ["a🙂b"]),
        _gold_call("codepoint_count", ["日本語"]),
    ]
    return visible, hidden

def _gen_with_item(rng: random.Random, difficulty: Difficulty) -> tuple[list, list]:
    n = {"easy": 3, "medium": 5, "hard": 8}[difficulty]
    items = [rng.randint(0, 30) for _ in range(n)]
    x = rng.randint(0, 30)
    visible = [_gold_call("with_item", [items, x], preserve_args=True)]
    hidden = [
        _gold_call("with_item", [[], 7], preserve_args=True),
        _gold_call("with_item", [items[::-1], x + 1], preserve_args=True),
    ]
    return visible, hidden


def _gen_closed_slice(rng: random.Random, difficulty: Difficulty) -> tuple[list, list]:
    n = {"easy": 5, "medium": 8, "hard": 12}[difficulty]
    xs = list(range(n))
    lo, hi = 1, max(1, n - 2)
    visible = [_gold_call("closed_slice", [xs, lo, hi])]
    hidden = [
        _gold_call("closed_slice", [xs, 0, 0]),
        _gold_call("closed_slice", [xs, 0, n - 1]),
    ]
    return visible, hidden


def _gen_rotate_left(rng: random.Random, difficulty: Difficulty) -> tuple[list, list]:
    n = {"easy": 4, "medium": 6, "hard": 9}[difficulty]
    items = [rng.randint(0, 20) for _ in range(n)]
    k = rng.randint(1, max(1, n - 1))
    # Empty list must be visible so buggy modulo-by-zero fails visibly.
    visible = [
        _gold_call("rotate_left", [[], 3]),
        _gold_call("rotate_left", [items, k]),
    ]
    hidden = [
        _gold_call("rotate_left", [items, 0]),
        _gold_call("rotate_left", [items, n]),
    ]
    return visible, hidden

def _gen_safe_ratio(rng: random.Random, difficulty: Difficulty) -> tuple[list, list]:
    num = rng.randint(1, 50)
    den = rng.randint(1, 20)
    # Zero denominator must be visible so buggy ZeroDivision fails visibly.
    visible = [
        _gold_call("safe_ratio", [num, 0]),
        _gold_call("safe_ratio", [num, den]),
    ]
    hidden = [
        _gold_call("safe_ratio", [0, den]),
        _gold_call("safe_ratio", [-num, den]),
    ]
    return visible, hidden

def _gen_window_count(rng: random.Random, difficulty: Difficulty) -> tuple[list, list]:
    n = {"easy": 5, "medium": 10, "hard": 20}[difficulty]
    k = rng.randint(1, max(1, n // 2))
    visible = [_gold_call("window_count", [n, k])]
    hidden = [
        _gold_call("window_count", [n, 1]),
        _gold_call("window_count", [n, n]),
        _gold_call("window_count", [k, n + 1]),
    ]
    return visible, hidden


def _gen_same_letters(rng: random.Random, difficulty: Difficulty) -> tuple[list, list]:
    word = rng.choice(["Hello", "World", "Python", "Cafe"])
    # ß/ss must be visible so lower()-based buggy fails visibly.
    visible = [
        _gold_call("same_letters", ["straße", "STRASSE"]),
        _gold_call("same_letters", [word, word.lower()]),
    ]
    hidden = [
        _gold_call("same_letters", ["ß", "SS"]),
        _gold_call("same_letters", ["ABC", "abd"]),
    ]
    return visible, hidden

def _gen_field_count(rng: random.Random, difficulty: Difficulty) -> tuple[list, list]:
    words = [
        rng.choice(["alpha", "beta", "gamma", "delta"])
        for _ in range({"easy": 2, "medium": 3, "hard": 4}[difficulty])
    ]
    s = " ".join(words)
    # NBSP must be visible so split(" ") buggy fails visibly.
    visible = [
        _gold_call("field_count", ["a b"]),
        _gold_call("field_count", [s]),
    ]
    hidden = [
        _gold_call("field_count", ["  spaced  out  "]),
        _gold_call("field_count", ["a\tb\nc"]),
        _gold_call("field_count", [""]),
    ]
    return visible, hidden

def _gen_unique_keep_order(rng: random.Random, difficulty: Difficulty) -> tuple[list, list]:
    # Fixed discriminator where sorted(set) != first-seen order.
    visible = [_gold_call("unique_keep_order", [[3, 1, 2, 1, 3]])]
    n = {"easy": 5, "medium": 8, "hard": 12}[difficulty]
    xs = [rng.randint(0, 5) for _ in range(n)]
    hidden = [
        _gold_call("unique_keep_order", [xs]),
        _gold_call("unique_keep_order", [["b", "a", "b", "c"]]),
        _gold_call("unique_keep_order", [[]]),
    ]
    return visible, hidden

def _gen_clamp(rng: random.Random, difficulty: Difficulty) -> tuple[list, list]:
    lo = rng.randint(0, 10)
    hi = lo + {"easy": 5, "medium": 10, "hard": 20}[difficulty]
    x = rng.randint(lo, hi)
    # Upper bound must be visible so hi-1 buggy fails visibly.
    visible = [
        _gold_call("clamp", [hi, lo, hi]),
        _gold_call("clamp", [x, lo, hi]),
    ]
    hidden = [
        _gold_call("clamp", [hi + 5, lo, hi]),
        _gold_call("clamp", [lo - 5, lo, hi]),
    ]
    return visible, hidden

def _gen_chunks(rng: random.Random, difficulty: Difficulty) -> tuple[list, list]:
    n = {"easy": 2, "medium": 3, "hard": 4}[difficulty]
    length = n * rng.randint(2, 4) + rng.randint(1, max(1, n - 1))
    xs = list(range(length))
    visible = [_gold_call("chunks", [xs, n])]
    hidden = [
        _gold_call("chunks", [list(range(n)), n]),
        _gold_call("chunks", [list(range(n + 1)), n]),
        _gold_call("chunks", [[], n]),
    ]
    return visible, hidden


def _gen_first_index(rng: random.Random, difficulty: Difficulty) -> tuple[list, list]:
    n = {"easy": 5, "medium": 8, "hard": 12}[difficulty]
    xs = [rng.randint(0, 4) for _ in range(n)]
    target = xs[rng.randint(0, n - 1)]
    if xs.count(target) < 2 and n >= 2:
        xs[-1] = target
        xs[0] = target
    visible = [_gold_call("first_index", [xs, target])]
    hidden = [
        _gold_call("first_index", [[1, 2, 1, 2, 1], 1]),
        _gold_call("first_index", [[1, 2, 3], 9]),
        _gold_call("first_index", [[7, 7, 7], 7]),
    ]
    return visible, hidden


_GENERATORS: dict[str, Callable[[random.Random, Difficulty], tuple[list, list]]] = {
    "nth_item": _gen_nth_item,
    "sum_through": _gen_sum_through,
    "mean_or_none": _gen_mean_or_none,
    "codepoint_count": _gen_codepoint_count,
    "with_item": _gen_with_item,
    "closed_slice": _gen_closed_slice,
    "rotate_left": _gen_rotate_left,
    "safe_ratio": _gen_safe_ratio,
    "window_count": _gen_window_count,
    "same_letters": _gen_same_letters,
    "field_count": _gen_field_count,
    "unique_keep_order": _gen_unique_keep_order,
    "clamp": _gen_clamp,
    "chunks": _gen_chunks,
    "first_index": _gen_first_index,
}

# Curated eval edges - one per family.
EDGE_CASES: tuple[dict[str, Any], ...] = (
    {
        "family": "nth_item",
        "difficulty": "hard",
        "visible": [{"args": [[10, 20, 30], 1], "expected": 10}],
        "hidden": [
            {"args": [[10, 20, 30], 2], "expected": 20},
            {"args": [[10, 20, 30], 3], "expected": 30},
        ],
    },
    {
        "family": "sum_through",
        "difficulty": "hard",
        "visible": [{"args": [1, 3], "expected": 6}],
        "hidden": [
            {"args": [5, 5], "expected": 5},
            {"args": [0, 10], "expected": 55},
        ],
    },
    {
        "family": "mean_or_none",
        "difficulty": "hard",
        "visible": [{"args": [[]], "expected": None}],
        "hidden": [
            {"args": [[2, 4]], "expected": 3.0},
            {"args": [[1, 2, 3]], "expected": 2.0},
        ],
    },
    {
        "family": "codepoint_count",
        "difficulty": "hard",
        "visible": [{"args": ["é"], "expected": 1}],
        "hidden": [
            {"args": ["abc"], "expected": 3},
            {"args": ["🙂"], "expected": 1},
            {"args": ["a🙂é"], "expected": 3},
        ],
    },
    {
        "family": "with_item",
        "difficulty": "hard",
        "visible": [{"args": [[1, 2], 3], "expected": [1, 2, 3], "preserve_args": True}],
        "hidden": [
            {"args": [[], 9], "expected": [9], "preserve_args": True},
            {"args": [[1], 1], "expected": [1, 1], "preserve_args": True},
        ],
    },
    {
        "family": "closed_slice",
        "difficulty": "hard",
        "visible": [{"args": [[0, 1, 2, 3, 4], 1, 3], "expected": [1, 2, 3]}],
        "hidden": [
            {"args": [[0, 1, 2, 3, 4], 0, 0], "expected": [0]},
            {"args": [[0, 1, 2, 3, 4], 2, 4], "expected": [2, 3, 4]},
        ],
    },
    {
        "family": "rotate_left",
        "difficulty": "hard",
        "visible": [{"args": [[], 5], "expected": []}],
        "hidden": [
            {"args": [[1, 2, 3, 4], 1], "expected": [2, 3, 4, 1]},
            {"args": [[1, 2, 3], 3], "expected": [1, 2, 3]},
            {"args": [[1, 2, 3], 0], "expected": [1, 2, 3]},
        ],
    },
    {
        "family": "safe_ratio",
        "difficulty": "hard",
        "visible": [{"args": [10, 0], "expected": None}],
        "hidden": [
            {"args": [10, 2], "expected": 5.0},
            {"args": [0, 5], "expected": 0.0},
        ],
    },
    {
        "family": "window_count",
        "difficulty": "hard",
        "visible": [{"args": [5, 3], "expected": 3}],
        "hidden": [
            {"args": [5, 1], "expected": 5},
            {"args": [5, 5], "expected": 1},
            {"args": [3, 5], "expected": 0},
        ],
    },
    {
        "family": "same_letters",
        "difficulty": "hard",
        "visible": [{"args": ["straße", "STRASSE"], "expected": True}],
        "hidden": [
            {"args": ["Hello", "hello"], "expected": True},
            {"args": ["ß", "SS"], "expected": True},
            {"args": ["ab", "ac"], "expected": False},
        ],
    },
    {
        "family": "field_count",
        "difficulty": "hard",
        "visible": [{"args": ["one two"], "expected": 2}],
        "hidden": [
            {"args": ["one two"], "expected": 2},
            {"args": ["  a  b  "], "expected": 2},
            {"args": [""], "expected": 0},
        ],
    },
    {
        "family": "unique_keep_order",
        "difficulty": "hard",
        "visible": [{"args": [[3, 1, 2, 1, 3]], "expected": [3, 1, 2]}],
        "hidden": [
            {"args": [[1, 2, 1]], "expected": [1, 2]},
            {"args": [["b", "a", "b"]], "expected": ["b", "a"]},
        ],
    },
    {
        "family": "clamp",
        "difficulty": "hard",
        "visible": [{"args": [10, 0, 10], "expected": 10}],
        "hidden": [
            {"args": [5, 0, 10], "expected": 5},
            {"args": [11, 0, 10], "expected": 10},
            {"args": [-1, 0, 10], "expected": 0},
        ],
    },
    {
        "family": "chunks",
        "difficulty": "hard",
        "visible": [{"args": [[1, 2, 3, 4, 5], 2], "expected": [[1, 2], [3, 4], [5]]}],
        "hidden": [
            {"args": [[1, 2, 3], 2], "expected": [[1, 2], [3]]},
            {"args": [[1, 2], 2], "expected": [[1, 2]]},
            {"args": [[], 3], "expected": []},
        ],
    },
    {
        "family": "first_index",
        "difficulty": "hard",
        "visible": [{"args": [[1, 2, 1], 1], "expected": 0}],
        "hidden": [
            {"args": [[1, 2, 1, 2, 1], 2], "expected": 1},
            {"args": [[7, 7, 7], 7], "expected": 0},
            {"args": [[1, 2, 3], 9], "expected": -1},
        ],
    },
)

# ---------------------------------------------------------------------------
# Prompt / dataset rows
# ---------------------------------------------------------------------------


def _format_case(case: dict[str, Any]) -> str:
    args = case.get("args", [])
    kwargs = case.get("kwargs") or {}
    parts = [repr(a) for a in args]
    parts.extend(f"{k}={v!r}" for k, v in kwargs.items())
    return f"({', '.join(parts)}) -> {case['expected']!r}"


def _question_for(info: dict[str, Any]) -> str:
    impl = FAMILY_IMPL[info["family"]]
    tests = info["tests"]
    if isinstance(tests, str):
        tests = json.loads(tests)
    visible_lines = "\n".join(
        f"  - {impl['func_name']}{_format_case(c)}" for c in tests["visible"]
    )
    return (
        "Repair the following Python function so it matches its docstring "
        "and passes the unit tests.\n\n"
        f"```python\n{info['buggy_code'].rstrip()}\n```\n\n"
        f"Visible tests (must pass):\n{visible_lines}\n\n"
        "Return the complete fixed function inside <answer>...</answer>."
    )


def example_from_spec(spec: dict[str, Any]) -> dict[str, Any]:
    """Materialize a hand-written or generated spec into a dataset row.

    Executes gold (must pass every test) and buggy (must fail a visible test).
    """
    family: str = spec["family"]
    impl = FAMILY_IMPL[family]
    difficulty = spec.get("difficulty", "hard")
    if "visible" in spec and "hidden" in spec:
        tests = {"visible": spec["visible"], "hidden": spec["hidden"]}
    else:
        rng = spec.get("rng") or random.Random(0)
        tests_vis, tests_hid = _GENERATORS[family](rng, difficulty)
        tests = {"visible": tests_vis, "hidden": tests_hid}

    gold_code = impl["gold"]
    buggy_code = impl["buggy"]
    gold_result = run_tests(gold_code, impl["func_name"], tests)
    if not gold_result.all_passed:
        raise RuntimeError(f"gold failed for {family}: {gold_result.errors}")
    buggy_result = run_tests(buggy_code, impl["func_name"], tests)
    if buggy_result.visible_all_passed:
        raise RuntimeError(
            f"buggy unexpectedly passed all visible tests for {family}"
        )

    info = {
        "family": family,
        "difficulty": difficulty,
        "func_name": impl["func_name"],
        "buggy_code": buggy_code,
        "gold_code": gold_code,
        "tests": tests,
    }
    return {
        "question": _question_for({**info, "tests": tests}),
        "answer": gold_code,
        "info": info,
    }


def generate_example(
    rng: random.Random,
    *,
    family: FamilyName | None = None,
    difficulty: Difficulty | None = None,
) -> dict[str, Any]:
    difficulty = difficulty or rng.choices(
        population=list(DIFFICULTY_WEIGHTS),
        weights=list(DIFFICULTY_WEIGHTS.values()),
        k=1,
    )[0]
    family = family or rng.choice(list(FAMILIES))
    return example_from_spec({"family": family, "difficulty": difficulty, "rng": rng})


def build_rows(
    *,
    n: int,
    seed: int,
    include_edge_cases: bool = False,
    family: FamilyName | None = None,
    difficulty: Difficulty | None = None,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    if include_edge_cases:
        for spec in EDGE_CASES:
            if family is not None and spec["family"] != family:
                continue
            if difficulty is not None and spec.get("difficulty") != difficulty:
                continue
            rows.append(example_from_spec(spec))
    seen = {row["question"] for row in rows}
    guard = 0
    while len(rows) < n and guard < n * 40:
        guard += 1
        example = generate_example(rng, family=family, difficulty=difficulty)
        if example["question"] in seen:
            continue
        seen.add(example["question"])
        rows.append(example)
    if len(rows) < n:
        raise RuntimeError(f"could only generate {len(rows)} unique rows (wanted {n})")
    return rows[:n]


def rows_for_dataset(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """JSON-stringify nested tests; never emit a top-level/info ``task`` column."""
    out: list[dict[str, Any]] = []
    for row in rows:
        cloned = dict(row)
        info = dict(cloned["info"])
        tests = info.get("tests")
        if not isinstance(tests, str):
            info["tests"] = json.dumps(tests, sort_keys=True)
        info.pop("task", None)
        cloned.pop("task", None)
        cloned["info"] = info
        out.append(cloned)
    return out

# ---------------------------------------------------------------------------
# Parser + rubric (stdlib - usable without verifiers installed)
# ---------------------------------------------------------------------------


def extract_completion_text(completion: Any) -> str:
    if completion is None:
        return ""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        for message in reversed(completion):
            role, content = _message_role_content(message)
            if role not in (None, "assistant"):
                continue
            text = _content_to_text(content)
            if text:
                return text
        return ""
    if isinstance(completion, dict):
        return _content_to_text(completion.get("content", ""))
    role, content = _message_role_content(completion)
    if content is not None:
        return _content_to_text(content)
    return str(completion)


def _message_role_content(message: Any) -> tuple[Any, Any]:
    if isinstance(message, dict):
        return message.get("role"), message.get("content")
    role = getattr(message, "role", None)
    content = getattr(message, "content", None)
    if role is None and content is None and hasattr(message, "model_dump"):
        dumped = message.model_dump()
        if isinstance(dumped, dict):
            return dumped.get("role"), dumped.get("content")
    return role, content


def _content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
            elif isinstance(block, str):
                parts.append(block)
            else:
                text = getattr(block, "text", None)
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    return str(content)


def parse_answer(completion: Any) -> str | None:
    text = extract_completion_text(completion)
    match = ANSWER_RE.search(text)
    if not match:
        return None
    body = match.group(1).strip()
    if not body:
        return None
    body = FENCE_RE.sub("", body).strip()
    return body or None


def has_answer_tags(completion: Any) -> bool:
    return parse_answer(completion) is not None


def gold_completion(gold_code: str) -> str:
    return f"<answer>\n{gold_code.rstrip()}\n</answer>"


def naive_completion(info: dict[str, Any]) -> str:
    return f"<answer>\n{info['buggy_code'].rstrip()}\n</answer>"


def format_score(completion: Any) -> float:
    return 1.0 if has_answer_tags(completion) else 0.0


def exact_match_score(completion: Any, answer: str, info: dict[str, Any] | None = None) -> float:
    """Behavioral exact match: submitted code passes every test."""
    del answer
    info = info or {}
    parsed = parse_answer(completion)
    if parsed is None:
        return 0.0
    func_name = info.get("func_name")
    tests = info.get("tests")
    if not func_name or tests is None:
        return 0.0
    return 1.0 if run_tests(parsed, func_name, tests).all_passed else 0.0


def partial_credit_score(completion: Any, answer: str, info: dict[str, Any] | None = None) -> float:
    """0.5 when every visible test passes but a hidden test fails; 0 if exact."""
    del answer
    info = info or {}
    parsed = parse_answer(completion)
    if parsed is None:
        return 0.0
    func_name = info.get("func_name")
    tests = info.get("tests")
    if not func_name or tests is None:
        return 0.0
    result = run_tests(parsed, func_name, tests)
    if result.all_passed:
        return 0.0
    if result.visible_all_passed:
        return 0.5
    return 0.0


def grade(completion: Any, answer: str, info: dict[str, Any] | None = None) -> dict[str, float]:
    exact = exact_match_score(completion, answer, info)
    fmt = format_score(completion)
    partial = partial_credit_score(completion, answer, info)
    weighted = 1.0 * exact + 0.2 * fmt + 0.2 * partial
    return {
        "exact_match": exact,
        "format": fmt,
        "partial_credit": partial,
        "reward": weighted,
    }


# ---------------------------------------------------------------------------
# load_environment - Hub / verifiers v0 entrypoint
# ---------------------------------------------------------------------------


def load_environment(
    num_train_examples: int = 500,
    num_eval_examples: int = 100,
    seed: int = 42,
    family: FamilyName | None = None,
    difficulty: Difficulty | None = None,
    **kwargs: Any,
) -> Any:
    """Build a ``vf.SingleTurnEnv`` for sandboxed Python repair.

    Dataset rows never use a column named ``task``. Nested ``tests`` are
    JSON-stringified before ``Dataset.from_list``. Eval always prepends the
    15 curated edge cases (one per family), then fills with unique random rows.
    """
    import verifiers as vf
    from datasets import Dataset

    train_rows = build_rows(
        n=num_train_examples,
        seed=seed,
        family=family,
        difficulty=difficulty,
    )
    eval_rows = build_rows(
        n=num_eval_examples,
        seed=seed + 1,
        include_edge_cases=True,
        family=family,
        difficulty=difficulty,
    )
    train_ds = Dataset.from_list(rows_for_dataset(train_rows))
    eval_ds = Dataset.from_list(rows_for_dataset(eval_rows))

    parser = vf.XMLParser(["answer"], answer_field="answer")

    def exact_match_reward(completion, answer, info=None, **_kwargs) -> float:
        return exact_match_score(completion, answer, info)

    def format_reward(completion, **_kwargs) -> float:
        return format_score(completion)

    def partial_credit_reward(completion, answer, info=None, **_kwargs) -> float:
        return partial_credit_score(completion, answer, info)

    rubric = vf.Rubric(
        funcs=[exact_match_reward, format_reward, partial_credit_reward],
        weights=[1.0, 0.2, 0.2],
        parser=parser,
    )

    return vf.SingleTurnEnv(
        dataset=train_ds,
        eval_dataset=eval_ds,
        system_prompt=SYSTEM_PROMPT,
        parser=parser,
        rubric=rubric,
        **{k: v for k, v in kwargs.items() if k in ("max_concurrent",)},
    )
