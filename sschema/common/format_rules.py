"""Deterministic formatting rules for fixed-schema JSONs.

Applied by `FBPipeline.apply_format()` as the s3 format pass
(runs after s2 verify, before s4 normalize). Pure Python, idempotent, no LLM calls.
"""

import re

# Common manufacturing acronyms that should stay all-uppercase in Title Case
PRESERVE_ACRONYMS = {
    "CNC", "EDM", "PLC", "CAD", "CAM", "NC",
    "MIG", "TIG", "EDC", "EDA", "DRO", "HSS",
    "ID", "OD", "RPM", "PPM", "PSI",
}


def _title_word(word: str) -> str:
    """Capitalize a single word; keep known acronyms uppercase."""
    if not word:
        return word
    if word.upper() in PRESERVE_ACRONYMS:
        return word.upper()
    return word[:1].upper() + word[1:].lower()


def _split_camel(text: str) -> str:
    """Insert a space between camelCase / PascalCase / ACRONYMWord boundaries."""
    # lowercase → UPPERCASE  (e.g. "cuttingSpeed" → "cutting Speed")
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
    # ACRONYM → Word  (e.g. "CNCMachine" → "CNC Machine")
    text = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", text)
    return text


def title_case(text: str) -> str:
    """Title Case, preserving punctuation, parens, and known acronyms.

    Examples:
        "shearing"               -> "Shearing"
        "shear (sheet metal)"    -> "Shear (Sheet Metal)"
        "wire edm machine"       -> "Wire EDM Machine"
        "WireEDM"                -> "Wire EDM"
    """
    if not text:
        return text
    text = _split_camel(str(text))
    return re.sub(r"[A-Za-z]+", lambda m: _title_word(m.group(0)), text)


def pascal_case(text: str) -> str:
    """PascalCase with no spaces; keep acronyms uppercase.

    Examples:
        "sheet metal"      -> "SheetMetal"
        "cut_sheet_metal"  -> "CutSheetMetal"
        "SheetMetal"       -> "SheetMetal"
        "cnc turning"      -> "CNCTurning"
    """
    if not text:
        return text
    text = _split_camel(str(text))
    parts = re.split(r"[\s_\-]+", text.strip())
    return "".join(_title_word(p) for p in parts if p)


def lowercase_words(text: str) -> str:
    """Lowercase with single spaces (no underscores, no camelCase).

    Examples:
        "cutting speed"  -> "cutting speed"
        "cuttingSpeed"   -> "cutting speed"
        "cutting_speed"  -> "cutting speed"
        "CuttingSpeed"   -> "cutting speed"
    """
    if not text:
        return text
    text = _split_camel(str(text))
    text = re.sub(r"[_\-]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


def format_step(step: dict) -> dict:
    """Apply formatting rules to a single step.

    Rules embodied here (these used to live in src/prompts/step2_format.txt
    before the 2-stage refactor; they are now declared in code only):
      operation              -> Title Case
      machine                -> Title Case
      duration               -> unchanged (can't fabricate units)
      pre/post.component_type -> PascalCase (no spaces)
      pre/post.component      -> Title Case
      pre/post.container      -> PascalCase or "" (empty preserved)
      parameters keys         -> lowercase with spaces
      parameters values       -> unchanged
    """
    if not isinstance(step, dict):
        return step

    out = dict(step)

    if "operation" in out:
        out["operation"] = title_case(out["operation"])
    if "machine" in out:
        out["machine"] = title_case(out["machine"])

    for field in ("precondition", "postcondition"):
        items = out.get(field)
        if not isinstance(items, list):
            continue
        new_items = []
        for item in items:
            if isinstance(item, dict):
                ni = dict(item)
                if "component_type" in ni:
                    ni["component_type"] = pascal_case(ni["component_type"])
                if "component" in ni:
                    ni["component"] = title_case(ni["component"])
                if "container" in ni:
                    c = str(ni["container"]).strip()
                    ni["container"] = pascal_case(c) if c else ""
                new_items.append(ni)
            else:
                new_items.append(item)
        out[field] = new_items

    params = out.get("parameters")
    if isinstance(params, dict):
        out["parameters"] = {lowercase_words(str(k)): v for k, v in params.items()}

    return out


def format_data(data: dict) -> dict:
    """Apply formatting to a full fixed-schema document. Preserves bad_case status."""
    if not isinstance(data, dict):
        return data
    if "bad_case" in data:
        return data
    new = dict(data)
    new["steps"] = [format_step(s) for s in data.get("steps", [])]
    return new
