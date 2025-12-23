# High-Leverage Python & Algorithm Patterns brainstorming

This document captures **rare but extremely useful patterns** that repeatedly appear
in medium–hard LeetCode problems and system-style interviews.

The goal is **recognition + fast recall**, not theory.
---

## 1. `zip()` — Parallel Iteration & Transpose

### Core idea
`zip()` groups elements index-wise across iterables.

```python
zip(a, b)  →  (a[0], b[0]), (a[1], b[1]) ...
