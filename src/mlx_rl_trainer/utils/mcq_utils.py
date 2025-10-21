"""Utility functions specific for Multiple Choice Question (MCQ) processing."""
import re
from typing import List, Optional, Dict, Any
from mlx_rl_trainer.core.config import GenerationConfig
from mlx_rl_trainer.utils.text_utils import (
    extract_answer_region,
    _normalize_ans_for_match,
    LETTER_ALPH,
    _letters_to_canonical,
)


def _extract_predicted_letters(
    generated_text: str, options: Optional[List[str]], cfg: GenerationConfig
) -> str:
    """
    Extracts the predicted MCQ letter(s) from the answer region.

    Returns a canonical string: "A", "B,C", or "" if none found.
    """
    ans = extract_answer_region(generated_text or "", cfg)

    # 1. Look for explicit letter(s) immediately followed by punctuation/space
    letters = set()
    m_list = re.findall(r"\b([A-Z])(?:[\)\.\:\-]\s*|\s+)", ans.upper())
    for L in m_list:
        if L in LETTER_ALPH:
            letters.add(L)

    if letters:
        return _letters_to_canonical(",".join(letters))

    # 2. Look for the answer text matching an option
    if options:
        an = _normalize_ans_for_match(ans)
        for j, opt in enumerate(options):
            on = _normalize_ans_for_match(opt)
            if an == on:
                return LETTER_ALPH[j] if 0 <= j < len(LETTER_ALPH) else ""

    return ""
