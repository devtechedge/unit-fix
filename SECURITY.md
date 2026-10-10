# Security Policy

## Reporting a vulnerability

Please do not open a public issue for security problems.

Report vulnerabilities privately using GitHub Security Advisories:
https://github.com/devtechedge/unit-fix/security/advisories/new

Please include:
- a description of the issue and its impact
- steps to reproduce
- the affected version or commit

## Response expectations

I aim to acknowledge reports within 7 days and provide a fix or a status
update within 30 days. You will be credited in the advisory if you would like.

## Scope

This repository is a personal project. Please report only issues that affect
this codebase, not third party services it may connect to.

## Accepted dependency risk

Two Dependabot alerts are open by choice and are not a maintenance oversight.
Both are pinned by the eval framework rather than by a version range we control:

- **`datasets` 4.6.1** (medium, path traversal in folder-based builders). `verifiers`
  requires `datasets>=3.0.0,<4.7.0`, so 5.x cannot be installed. The affected builder
  path is not used: this task ships a single local dataset, not folder-based loading.
- **`fsspec` 2026.2.0** (high, SSTI to RCE in `ReferenceFileSystem`). `datasets 4.6.1`
  caps it at `<=2026.2.0`. Reachable only when reading datasets over HTTP, which this
  task does not do. This is the one to revisit first if anything changes.

Re-evaluate both when `verifiers` ships a release that allows `datasets` 5.x, which
would unblock the pair.
