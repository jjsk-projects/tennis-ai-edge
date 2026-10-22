"""Pipeline configuration loader for tennis-ai-edge.

Reads ``config.yaml`` into a nested dict that the orchestrator
(``tools/run_video.py``) consumes to parameterise every stage. This is the
single place config values enter the runtime — previously ``config.yaml``
was dead code (nothing imported it).

Design goals
------------
- **Zero hard dependency.** The loader uses PyYAML when it is installed,
  but falls back to a tiny built-in parser that handles the exact subset of
  YAML ``config.yaml`` uses (nested mappings, ``#`` comments, scalars).
  That keeps the Windows / edge prototype working with only the core deps
  (numpy, opencv, mediapipe) — no extra install required.
- **Graceful degradation.** A missing or unparseable file yields ``{}``;
  callers read values through :func:`get` with defaults, so the pipeline
  always runs even with no config file present.

The on-disk format (see ``config.yaml``) mirrors the real stage-constructor
signatures so each key maps 1:1 to a parameter.
"""

from __future__ import annotations

import os
from typing import Any

# --------------------------------------------------------------------------- #
# YAML parsing: PyYAML if present, else a minimal built-in fallback.
# --------------------------------------------------------------------------- #


def _coerce_scalar(value: str) -> Any:
    """Interpret a single scalar token as None/bool/int/float/str."""
    v = value.strip()
    if v == "" or v in ("~", "null", "None"):
        return None
    if v in ("true", "True"):
        return True
    if v in ("false", "False"):
        return False
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        pass
    if len(v) >= 2 and v[0] == v[-1] and v[0] in ("'", '"'):
        return v[1:-1]
    return v


def _strip_comment(line: str) -> str:
    """Drop a trailing or full-line ``#`` comment (quote-aware)."""
    quote = None
    for i, ch in enumerate(line):
        if quote:
            if ch == quote:
                quote = None
        elif ch in ("'", '"'):
            quote = ch
        elif ch == "#" and (i == 0 or line[i - 1] in " \t"):
            return line[:i]
    return line


def _parse_simple_yaml(text: str) -> dict:
    """Parse a *nested-mapping-only* YAML subset (no lists/anchors/flow).

    This is exactly the shape ``config.yaml`` uses. It is intentionally
    small and correct-for-that-subset rather than a full YAML implementation.
    """
    root: dict = {}
    # Stack of (indent, container). Root sits at indent -1 so that any
    # top-level key (indent >= 0) attaches to it.
    stack: list[tuple[int, dict]] = [(-1, root)]

    for raw in text.splitlines():
        line = _strip_comment(raw).rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        body = line.strip()
        if body.startswith("- "):
            # Lists are not part of the config subset; ignore defensively.
            continue
        if ":" not in body:
            continue
        key, _, val = body.partition(":")
        key = key.strip().strip("'\"")
        val = val.strip()

        # Pop any containers at depth >= this line's indent so we attach to
        # the nearest shallower mapping.
        while stack and stack[-1][0] >= indent:
            stack.pop()
        parent = stack[-1][1] if stack else root

        if val == "":
            child: dict = {}
            parent[key] = child
            stack.append((indent, child))
        else:
            parent[key] = _coerce_scalar(val)
    return root


def _load_yaml_file(path: str) -> dict:
    """Parse a YAML file via PyYAML, falling back to the built-in parser."""
    with open(path, "r", encoding="utf-8") as fh:
        text = fh.read()
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(text)
        return data if isinstance(data, dict) else {}
    except ImportError:
        return _parse_simple_yaml(text)


def load_config(path: str | None = None) -> dict:
    """Load a pipeline config file into a nested dict.

    Parameters
    ----------
    path
        Path to a YAML config. ``None`` resolves to ``config.yaml`` in the
        current working directory. A missing file returns ``{}`` (the
        pipeline then runs on in-code defaults via :func:`get`).

    Returns
    -------
    dict
        Nested mapping mirroring the file's structure. Never raises for a
        missing file; returns ``{}`` in that case.
    """
    if path is None:
        path = "config.yaml"
    if not os.path.exists(path):
        return {}
    try:
        return _load_yaml_file(path)
    except Exception:
        # A malformed config should not crash the pipeline; surface a warning
        # and fall back to defaults.
        import sys

        print(f"[warn] could not parse config {path!r}; using defaults",
              file=sys.stderr)
        return {}


def get(cfg: dict, dotted: str, default: Any = None) -> Any:
    """Fetch a value by dotted path (``"pipeline.pose.every_frames"``).

    Missing intermediate keys or a ``None`` leaf return ``default``. This is
    how callers pull individual knobs without defensive ``.get`` chains.
    """
    node: Any = cfg
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    if node is None:
        return default
    return node


def resolve_path(cfg: dict, dotted: str, root_dir: str,
                 default: str | None = None) -> str | None:
    """Like :func:`get`, but resolve a relative path against ``root_dir``.

    Useful for model/classifier paths in config that are relative to the
    project root (e.g. ``models/pose_landmarker.task``).
    """
    value = get(cfg, dotted, default)
    if value is None:
        return None
    value = str(value)
    if os.path.isabs(value):
        return value
    return os.path.join(root_dir, value)
