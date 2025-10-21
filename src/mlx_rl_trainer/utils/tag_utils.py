"""
Tag Extraction and Processing Utilities

This module provides a comprehensive set of utilities for extracting and processing
tag-based content from generated text in the MLX RL Trainer framework. It centralizes
tag handling logic to ensure consistency across different reward functions and components.

The utilities handle various edge cases including:
- Missing or malformed tags
- Nested tags
- Empty content
- Whitespace normalization
- Consistent error handling

These utilities are designed to be used by reward functions, particularly those that
need to extract thinking and answer sections from generated text.
"""

import logging
import re
from typing import Dict, List, Optional, Tuple, Union, Any

from mlx_rl_trainer.core.config import GenerationConfig

logger = logging.getLogger(__name__)


def extract_think_region(
    text: str, 
    gen_config: GenerationConfig, 
    fallback_tags: Optional[Dict[str, str]] = None
) -> str:
    """
    Extracts the content between the FIRST think start tag and the FIRST think end tag.
    
    This function handles various edge cases including missing tags, malformed tags,
    and empty content. It uses the tags defined in the GenerationConfig, with optional
    fallback tags if the config doesn't provide them.
    
    Args:
        text: The text to extract from
        gen_config: Configuration object containing tag definitions
        fallback_tags: Optional dictionary with fallback tags if config doesn't provide them
        
    Returns:
        The extracted thinking region text, or empty string if not found
        
    Examples:
        >>> config = GenerationConfig(think_start_tag="<think>", think_end_tag="</think>")
        >>> extract_think_region("<think>This is thinking</think>Answer", config)
        'This is thinking'
        >>> extract_think_region("No tags here", config)
        ''
    """
    # Robust string conversion
    text_str = str(text) if text is not None else ""
    
    # Get tags from config with fallbacks
    fallback_tags = fallback_tags or {"think_start_tag": "<think>", "think_end_tag": "</think>"}
    start_tag = getattr(gen_config, 'think_start_tag', fallback_tags.get("think_start_tag"))
    end_tag = getattr(gen_config, 'think_end_tag', fallback_tags.get("think_end_tag"))
    
    # Handle edge cases
    if not text_str or not start_tag or not end_tag:
        return ""
        
    if text_str.endswith(start_tag) or text_str.startswith(end_tag):
        return ""
        
    # Extract the thinking region using a non-greedy pattern
    pattern = re.escape(start_tag) + r"(.*?)" + re.escape(end_tag)
    m = re.search(pattern, text_str, flags=re.IGNORECASE | re.DOTALL)
    
    # Return the content (group 1) if found, otherwise empty string
    # Limit to 8000 chars for safety
    return (m.group(1).strip() if m else "")[:8000]


def extract_answer_region(
    text: str, 
    gen_config: GenerationConfig, 
    fallback_tags: Optional[Dict[str, str]] = None
) -> str:
    """
    Extracts text that comes AFTER the LAST think end tag.
    
    This function handles various edge cases including missing tags, malformed tags,
    and empty content. It uses the tags defined in the GenerationConfig, with optional
    fallback tags if the config doesn't provide them.
    
    Args:
        text: The text to extract from
        gen_config: Configuration object containing tag definitions
        fallback_tags: Optional dictionary with fallback tags if config doesn't provide them
        
    Returns:
        The extracted answer region text, or the original text if no tags found
        
    Examples:
        >>> config = GenerationConfig(think_end_tag="</think>")
        >>> extract_answer_region("<think>This is thinking</think>This is the answer", config)
        'This is the answer'
        >>> extract_answer_region("No tags here", config)
        'No tags here'
    """
    # Robust string conversion
    text_str = str(text) if text is not None else ""
    
    # Get tags from config with fallbacks
    fallback_tags = fallback_tags or {"think_end_tag": "</think>"}
    end_tag = getattr(gen_config, 'think_end_tag', fallback_tags.get("think_end_tag"))
    start_tag = getattr(gen_config, 'think_start_tag', fallback_tags.get("think_start_tag", "<think>"))
    
    # Handle edge cases
    if not text_str:
        return ""
        
    if text_str.endswith(start_tag) or text_str.startswith(end_tag):
        return ""
        
    if not end_tag:
        return text_str.strip()[:2000]  # Limit to 2000 chars for safety
        
    # Find the last end tag and extract everything after it
    last_idx = text_str.lower().rfind(end_tag.lower())
    if last_idx != -1:
        answer_text = text_str[last_idx + len(end_tag):].strip()
        return answer_text.lstrip("\n").strip()[:2000]  # Limit to 2000 chars for safety
        
    # If no end tag found, return the original text (robust fallback)
    return text_str.strip()[:2000]  # Limit to 2000 chars for safety


def extract_think_answer_lengths(
    text: Union[str, Any], 
    gen_config: GenerationConfig,
    fallback_tags: Optional[Dict[str, str]] = None
) -> Tuple[int, int]:
    """
    Extracts character lengths of thinking and answer sections from text.
    
    This function uses the extract_think_region and extract_answer_region functions
    to extract the thinking and answer sections, then returns their lengths.
    
    Args:
        text: The text to extract from
        gen_config: Configuration object containing tag definitions
        fallback_tags: Optional dictionary with fallback tags if config doesn't provide them
        
    Returns:
        A tuple of (thinking_length, answer_length)
        
    Examples:
        >>> config = GenerationConfig(think_start_tag="<think>", think_end_tag="</think>")
        >>> extract_think_answer_lengths("<think>ABC</think>DEF", config)
        (3, 3)
    """
    # Robustly convert input 'text' to string to prevent AttributeError
    text_str = str(text) if text is not None else ""
    if not text_str:
        return 0, 0
    
    try:
        think_content = extract_think_region(text_str, gen_config, fallback_tags)
        answer_content = extract_answer_region(text_str, gen_config, fallback_tags)
        return len(think_content.strip()), len(answer_content.strip())
    except Exception as e:
        logger.debug(f"Failed to extract think/answer lengths: {e}")
        return 0, 0


def count_tag_occurrences(
    text: str, 
    gen_config: GenerationConfig,
    fallback_tags: Optional[Dict[str, str]] = None
) -> Dict[str, int]:
    """
    Counts occurrences of thinking tags in the text.
    
    This function counts how many times each tag appears in the text,
    which is useful for detecting malformed tag structures.
    
    Args:
        text: The text to analyze
        gen_config: Configuration object containing tag definitions
        fallback_tags: Optional dictionary with fallback tags if config doesn't provide them
        
    Returns:
        A dictionary with counts for each tag type
        
    Examples:
        >>> config = GenerationConfig(think_start_tag="<think>", think_end_tag="</think>")
        >>> count_tag_occurrences("<think>ABC</think><think>DEF</think>", config)
        {'think_start': 2, 'think_end': 2}
    """
    # Robust string conversion
    text_str = str(text) if text is not None else ""
    
    # Get tags from config with fallbacks
    fallback_tags = fallback_tags or {
        "think_start_tag": "<think>", 
        "think_end_tag": "</think>",
        "answer_start_tag": "",
        "answer_end_tag": ""
    }
    
    start_tag = getattr(gen_config, 'think_start_tag', fallback_tags.get("think_start_tag"))
    end_tag = getattr(gen_config, 'think_end_tag', fallback_tags.get("think_end_tag"))
    answer_start_tag = getattr(gen_config, 'answer_start_tag', fallback_tags.get("answer_start_tag"))
    answer_end_tag = getattr(gen_config, 'answer_end_tag', fallback_tags.get("answer_end_tag"))
    
    result = {
        "think_start": 0,
        "think_end": 0,
        "answer_start": 0,
        "answer_end": 0
    }
    
    # Count occurrences of each tag
    try:
        if start_tag:
            result["think_start"] = len(re.findall(re.escape(start_tag), text_str, flags=re.I))
        if end_tag:
            result["think_end"] = len(re.findall(re.escape(end_tag), text_str, flags=re.I))
        if answer_start_tag:
            result["answer_start"] = len(re.findall(re.escape(answer_start_tag), text_str, flags=re.I))
        if answer_end_tag:
            result["answer_end"] = len(re.findall(re.escape(answer_end_tag), text_str, flags=re.I))
    except Exception as e:
        logger.warning(f"Error counting tags: {e}")
    
    return result


def validate_tag_structure(
    text: str, 
    gen_config: GenerationConfig,
    fallback_tags: Optional[Dict[str, str]] = None
) -> Dict[str, Any]:
    """
    Validates the tag structure of the text and returns a detailed analysis.
    
    This function checks for various issues with tag structure:
    - Missing tags
    - Unbalanced tags (more start than end or vice versa)
    - Nested tags
    - Empty content
    
    Args:
        text: The text to validate
        gen_config: Configuration object containing tag definitions
        fallback_tags: Optional dictionary with fallback tags if config doesn't provide them
        
    Returns:
        A dictionary with validation results
        
    Examples:
        >>> config = GenerationConfig(think_start_tag="<think>", think_end_tag="</think>")
        >>> validate_tag_structure("<think>ABC</think>DEF", config)
        {'valid': True, 'has_think': True, 'has_answer': True, 'balanced_tags': True, 'issues': []}
    """
    # Robust string conversion
    text_str = str(text) if text is not None else ""
    
    # Initialize result
    result = {
        "valid": False,
        "has_think": False,
        "has_answer": False,
        "balanced_tags": False,
        "issues": []
    }
    
    # Count tag occurrences
    tag_counts = count_tag_occurrences(text_str, gen_config, fallback_tags)
    
    # Check if thinking section exists
    result["has_think"] = tag_counts["think_start"] > 0 and tag_counts["think_end"] > 0
    
    # Check if tags are balanced
    result["balanced_tags"] = tag_counts["think_start"] == tag_counts["think_end"]
    
    # Extract thinking and answer content
    think_content = extract_think_region(text_str, gen_config, fallback_tags)
    answer_content = extract_answer_region(text_str, gen_config, fallback_tags)
    
    # Check if answer section exists
    result["has_answer"] = len(answer_content.strip()) > 0
    
    # Collect issues
    if not result["has_think"]:
        result["issues"].append("missing_think_section")
    
    if not result["has_answer"]:
        result["issues"].append("missing_answer_section")
    
    if not result["balanced_tags"]:
        result["issues"].append("unbalanced_tags")
    
    if tag_counts["think_start"] > 1 or tag_counts["think_end"] > 1:
        result["issues"].append("multiple_think_tags")
    
    if not think_content and result["has_think"]:
        result["issues"].append("empty_think_content")
    
    if not answer_content and result["has_answer"]:
        result["issues"].append("empty_answer_content")
    
    # Set valid flag if no issues
    result["valid"] = len(result["issues"]) == 0
    
    return result


def clean_tag_structure(
    text: str, 
    gen_config: GenerationConfig,
    fallback_tags: Optional[Dict[str, str]] = None
) -> str:
    """
    Cleans and normalizes the tag structure of the text.
    
    This function attempts to fix common issues with tag structure:
    - Missing closing tags
    - Multiple or nested tags
    - Whitespace issues
    
    Args:
        text: The text to clean
        gen_config: Configuration object containing tag definitions
        fallback_tags: Optional dictionary with fallback tags if config doesn't provide them
        
    Returns:
        The cleaned text with normalized tag structure
        
    Examples:
        >>> config = GenerationConfig(think_start_tag="<think>", think_end_tag="</think>")
        >>> clean_tag_structure("<think>ABC<think>nested</think>DEF", config)
        '<think>ABCnested</think>DEF'
    """
    # Robust string conversion
    text_str = str(text) if text is not None else ""
    if not text_str:
        return ""
    
    # Get tags from config with fallbacks
    fallback_tags = fallback_tags or {
        "think_start_tag": "<think>", 
        "think_end_tag": "</think>"
    }
    
    start_tag = getattr(gen_config, 'think_start_tag', fallback_tags.get("think_start_tag"))
    end_tag = getattr(gen_config, 'think_end_tag', fallback_tags.get("think_end_tag"))
    
    if not start_tag or not end_tag:
        return text_str
    
    # Extract thinking and answer content
    think_content = extract_think_region(text_str, gen_config, fallback_tags)
    answer_content = extract_answer_region(text_str, gen_config, fallback_tags)
    
    # If no thinking section found but start tag exists
    if not think_content and start_tag in text_str:
        # Try to extract everything after the first start tag
        start_idx = text_str.find(start_tag)
        if start_idx != -1:
            content_after_start = text_str[start_idx + len(start_tag):]
            
            # Check if there's an end tag
            end_idx = content_after_start.find(end_tag)
            if end_idx != -1:
                think_content = content_after_start[:end_idx].strip()
                answer_content = content_after_start[end_idx + len(end_tag):].strip()
            else:
                # No end tag, use everything after start tag as thinking
                think_content = content_after_start.strip()
                answer_content = ""
    
    # Clean thinking content by removing any nested tags
    if think_content:
        think_content = re.sub(f"{re.escape(start_tag)}|{re.escape(end_tag)}", "", think_content)
    
    # Reconstruct the text with clean tags
    if think_content or answer_content:
        return f"{start_tag}{think_content}{end_tag}\n{answer_content}"
    
    return text_str


def calculate_verbosity_penalty(
    text_length: int,
    reference_length: int,
    penalty_strength: float = 0.01,
    min_penalty_factor: float = 0.5,
    threshold_ratio: float = 1.5
) -> float:
    """
    Calculates a standardized verbosity penalty factor.
    
    This function implements a consistent approach to penalizing verbosity
    across different reward functions. It penalizes text that is significantly
    longer than the reference.
    
    Args:
        text_length: Length of the text being evaluated
        reference_length: Length of the reference text
        penalty_strength: How strongly to penalize verbosity (higher = stronger penalty)
        min_penalty_factor: Minimum penalty factor (lower bound)
        threshold_ratio: Ratio above which penalty is applied
        
    Returns:
        A penalty factor between min_penalty_factor and 1.0
        
    Examples:
        >>> calculate_verbosity_penalty(150, 100, 0.01, 0.5, 1.5)
        1.0
        >>> calculate_verbosity_penalty(300, 100, 0.01, 0.5, 1.5)
        0.85
    """
    if text_length <= reference_length * threshold_ratio:
        return 1.0  # No penalty if not verbose enough
    
    # Calculate penalty factor
    penalty_factor = 1.0 - penalty_strength * (text_length / reference_length - threshold_ratio)
    
    # Ensure penalty factor is within bounds
    return max(min_penalty_factor, penalty_factor)