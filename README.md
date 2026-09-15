# unit-fix

Single-turn **Python repair** for RLVR / evals on the [Prime Intellect Environments Hub](https://app.primeintellect.ai/dashboard/environments).

Hub: [devtechedge/unit-fix](https://app.primeintellect.ai/dashboard/environments/devtechedge/unit-fix) · Source: [github.com/devtechedge/unit-fix](https://github.com/devtechedge/unit-fix)

A third environment next to [calendar-math](https://github.com/devtechedge/calendar-math) (datetime gold) and [meeting-slot](https://github.com/devtechedge/meeting-slot) (multi-turn tools). This one is **code**: a small broken function plus a failing unit test. The model puts a patched function in `<answer>` tags. The grader execs the tests in a stdlib sandbox - no LLM-as-judge, no source-string match on the main reward.

| Family | Bug the eval edge exists to catch |
| --- | --- |
| `nth_item` | 1-based indexing, uses `items[n]` |
| `sum_through` | `range` exclusive of `hi` |
| `mean_or_none` | empty list divides by zero |
| `codepoint_count` | UTF-8 bytes, not code points |
| `with_item` | mutates the input list |
| `closed_slice` | Python slice drops the end index |
| `rotate_left` | `n % len([])` crashes |
| `safe_ratio` | no zero-denominator guard |
| `window_count` | fencepost: `n-k` instead of `n-k+1` |
| `same_letters` | `str.lower` misses `ß → ss` |
| `field_count` | `split(" ")` misses NBSP / unicode whitespace |
| `unique_keep_order` | `sorted(set)` loses first-seen order |
| `clamp` | closed interval treated as `hi-1` |
| `chunks` | drops the remainder chunk |
| `first_index` | returns last match, not first |

## Why this design

- **Verifiable.** Gold answers are in-repo repairs. Dataset construction **executes** gold (must pass) and the original function (must fail).
- **Behavioral exact, not AST match.** Any repair that passes the hidden tests scores 1.0 on the main term. Hardcoding the visible example does not.
- **Hard where it matters.** Eval is 15 curated edges: off-by-one, empty list, unicode, mutation vs copy, inclusive/exclusive slice.
- **Not gameable by format alone.** Format is a 0.2 bonus. Exact match is the 1.0 term (all tests pass).
- **Shaping, not noise.** Visible tests pass but a hidden test still fails → 0.5 partial credit. Naive echo of the original function never gets there.
- **Sandbox.** AST gate (no imports, no dunders, no I/O), capped `range`/`sum`, `sys.settrace` step + wall-clock limit. No subprocess, so Hub CI does not fork.
- **Configurable.** `num_train_examples`, `num_eval_examples`, `seed`, optional `family` / `difficulty` pin.

## Reward

```
reward = 1.0 * exact_match + 0.2 * format + 0.2 * partial_credit
```

| Term | 1.0 when | Notes |
| --- | --- | --- |
| `exact_match` | every test (visible + hidden) passes | behavioral; source need not match gold |
| `format` | `<answer>...</answer>` present with a body | extra prose outside the tags is ignored; markdown fences inside are stripped |
| `partial_credit` | every **visible** test passes, a hidden test fails | 0.5; 0 when exact already fired, so a perfect answer is **1.2** not 1.4 |

## Eval

15 curated edge cases (`num_eval_examples=15`, 1 rollout each).

| Policy | avg reward | exact | format | partial |
| --- | --- | --- | --- | --- |
| Gold repair (ceiling) | **1.200** | 1.000 | 1.000 | 0.000 |
| Naive (echo original function) | **0.200** | 0.000 | 1.000 | 0.000 |

The gold policy is a harness check: install, `load_environment`, the sandbox, and the rubric all fire 1.2. The naive policy is a discrimination check: returning the prompt's broken function does not rubber-stamp 1.2 - it fails a visible test on every eval edge, so it never collects partial credit either.

Model row pending a fresh OpenRouter key (`minimax/minimax-m2.7`, T=0, 2048 tok, `--max-concurrent 1`).

```bash
uv run vf-eval unit-fix -n 15 -r 1 -p openrouter \
  -m minimax/minimax-m2.7 --max-tokens 2048 \
  --temperature 0 --max-concurrent 1 --disable-tui --disable-env-server
```

## Installation

```bash
uv pip install -e .
python -m pytest tests/test_unit_fix.py -q
```

From the Hub:

```bash
prime env install devtechedge/unit-fix
```

```python
import verifiers as vf

env = vf.load_environment("unit-fix")
```

Requires `verifiers>=0.1.14,<0.2`.

## `load_environment` arguments

| Arg | Default | Meaning |
| --- | --- | --- |
| `num_train_examples` | `500` | train split size |
| `num_eval_examples` | `100` | eval split size (15 edges prepended) |
| `seed` | `42` | train RNG; eval uses `seed + 1` |
| `family` | `None` | pin to one bug family, or mixed |
| `difficulty` | `None` | `"easy"` \| `"medium"` \| `"hard"` \| mixed |

```bash
uv run vf-eval unit-fix -n 20
uv run vf-eval unit-fix -a '{"family": "with_item", "num_eval_examples": 40}'
```

Dataset rows never use a column named `task`. Verifiers ≥0.1 treats `info["task"]` as a nested rollout payload. Nested tests are JSON-stringified before `Dataset.from_list`.

## Gold solution

Dataset construction **is** the gold solver. Each family has a buggy body and a repair in `FAMILY_IMPL`. `example_from_spec` runs both through the sandbox: gold must pass every test; the original must fail a visible test (so naive cannot farm partial).

```python
from unit_fix import run_tests, gold_completion, naive_completion, grade

run_tests(info["gold_code"], info["func_name"], info["tests"]).all_passed  # True
run_tests(info["buggy_code"], info["func_name"], info["tests"]).all_passed  # False

grade(gold_completion(info["gold_code"]), info["gold_code"], info)["reward"]  # 1.2
grade(naive_completion(info), info["gold_code"], info)["reward"]              # 0.2
```

`tests/test_unit_fix.py` asserts the 15 edges, the sandbox rejects `import os` / dunder escapes / `while True`, and a function that hardcodes the visible expected value scores 0.3 not 1.2.

## Files

```
unit_fix.py                 # generator, sandbox, grader, load_environment
pyproject.toml
README.md
LICENSE
tests/test_unit_fix.py
```

## What this is not

- Not a wrap of HumanEval, MBPP, or any public coding dataset.
- Not LLM-judged.
- Not multi-turn / tool-using (that's meeting-slot).
- Not calendar arithmetic (that's calendar-math).
- Not source-diff matching. Hidden tests are the spec.

## License

MIT
