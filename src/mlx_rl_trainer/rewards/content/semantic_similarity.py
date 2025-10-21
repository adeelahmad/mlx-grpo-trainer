# file_path: mlx_rl_trainer/src/mlx_rl_trainer/rewards/content/semantic_similarity.py
# revision_no: 009
# goals_of_writing_code_block: Provide a robust, stable, and simplified semantic_similarity reward.
# type_of_code_response: change existing
"""
Semantic similarity-based content reward, simplified for stability and correctness.

This module provides a reward function that measures the semantic similarity between
generated and reference answers. It uses TF-IDF vectorization with robust fallback
mechanisms to handle edge cases and failures.

Key features:
- Centralized configuration management
- Robust TF-IDF vectorization with multiple fallback mechanisms
- Standardized verbosity penalty calculation
- Comprehensive error handling
"""

import logging
import re
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from mlx_rl_trainer.rewards.base_reward import BaseReward
from mlx_rl_trainer.rewards.context import RewardContext
from mlx_rl_trainer.rewards.registry import RewardRegistry
from mlx_rl_trainer.rewards.config_provider import RewardConfigProvider, RewardConfigurable
from mlx_rl_trainer.utils.tag_utils import extract_answer_region, calculate_verbosity_penalty

logger = logging.getLogger(__name__)


def _clean_text_for_tfidf(text: str) -> str:
    """A robust text cleaner for TF-IDF vectorization."""
    if not isinstance(text, str) or not text:
        return ""
    # Standardize and clean the text
    text = text.lower()
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"'re", " are", text)
    text = re.sub(r"'ve", " have", text)
    # Remove all non-alphanumeric characters
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    # Collapse multiple spaces into one
    text = re.sub(r"\s+", " ", text).strip()
    return text


@RewardRegistry.register("semantic_similarity")
class SemanticSimilarityReward(BaseReward, RewardConfigurable):
    """
    Rewards semantic similarity between generated and reference answers.

    This version is simplified for stability and uses a centralized configuration provider.
    It implements multiple fallback mechanisms for TF-IDF vectorization to ensure robustness
    in all scenarios, including very short texts and vectorization failures.

    Configuration:
        min_length (int): Min characters for text to be considered valid.
        apply_verbosity_penalty (bool): Penalize generated text that is much longer than the reference.
        verbosity_penalty_strength (float): Strength of the verbosity penalty.
        method (str): Similarity calculation method ('tfidf' or 'overlap').
        min_word_count (int): Minimum number of words required for TF-IDF vectorization.
        fallback_similarity_threshold (float): Threshold below which to try alternative similarity methods.
    """

    def __init__(self, config: Dict[str, Any]):
        BaseReward.__init__(self, config)
        RewardConfigurable.__init__(self)
        
        # Get reward-specific configuration with defaults
        reward_config = self.get_reward_config("semantic_similarity")
        
        # Initialize parameters from config with defaults
        self.min_length = int(config.get("min_length", reward_config.get("min_length", 10)))
        self.apply_verbosity_penalty = bool(config.get("apply_verbosity_penalty",
                                                      reward_config.get("apply_verbosity_penalty", True)))
        self.verbosity_penalty_strength = float(config.get("verbosity_penalty_strength",
                                                          reward_config.get("verbosity_penalty_strength", 0.01)))
        self.method = str(config.get("method", reward_config.get("method", "tfidf"))).lower()
        
        # Additional parameters for improved robustness
        self.min_word_count = int(config.get("min_word_count", reward_config.get("min_word_count", 3)))
        self.fallback_similarity_threshold = float(config.get("fallback_similarity_threshold",
                                                             reward_config.get("fallback_similarity_threshold", 0.1)))
        
        # Get generation config for tag extraction
        self.gen_config = self.get_generation_config()
        
        logger.info(f"SemanticSimilarityReward initialized: min_length={self.min_length}, "
                   f"method={self.method}, verbosity_penalty={self.apply_verbosity_penalty}")

    def compute(self, context: RewardContext) -> Dict[str, Any]:
        """
        Computes the semantic similarity score for a single context.
        
        This method extracts the answer sections from both the generated text and reference
        completion, then calculates their semantic similarity using TF-IDF vectorization
        with multiple fallback mechanisms for robustness.
        
        Args:
            context: The reward context containing generated text and reference completion
            
        Returns:
            Dictionary with reward score and log information
        """
        try:
            # Extract and validate text content using shared utilities
            generated_answer = extract_answer_region(context.generated_text or "", self.gen_config)
            reference_answer = extract_answer_region(context.reference_completion or "", self.gen_config)

            # If either text is too short or missing, the reward is zero.
            if len(generated_answer) < self.min_length or len(reference_answer) < self.min_length:
                return {"reward": 0.0, "log": "text_too_short_or_missing"}

            # Clean the texts to get a good representation for TF-IDF
            cleaned_gen = _clean_text_for_tfidf(generated_answer)
            cleaned_ref = _clean_text_for_tfidf(reference_answer)

            if not cleaned_gen or not cleaned_ref:
                return {"reward": 0.0, "log": "empty_after_cleaning"}
                
            # Get word counts for additional validation
            gen_words = cleaned_gen.split()
            ref_words = cleaned_ref.split()
            
            # Check if texts have enough words for meaningful TF-IDF
            if len(gen_words) < self.min_word_count or len(ref_words) < self.min_word_count:
                # For very short texts, use character-level n-grams or simple overlap
                logger.debug(f"Texts too short for standard TF-IDF: {len(gen_words)}/{len(ref_words)} words")
                raw_score = self._compute_similarity_for_short_texts(cleaned_gen, cleaned_ref)
                score = float(np.clip(raw_score, 0.0, 1.0))
                return {"reward": score, "log": f"short_text_similarity={raw_score:.4f}"}

            # Calculate similarity based on the selected method
            if self.method == "tfidf":
                raw_score = self._compute_tfidf_similarity(cleaned_gen, cleaned_ref)
            else:
                # Simple word overlap as fallback
                raw_score = self._compute_word_overlap(cleaned_gen, cleaned_ref)

            # Ensure score is in valid range
            score = float(np.clip(raw_score, 0.0, 1.0))
            
            # If similarity is suspiciously low, try alternative methods
            if score < self.fallback_similarity_threshold:
                logger.debug(f"Low similarity score ({score:.4f}), trying alternative methods")
                alt_score = self._compute_alternative_similarity(cleaned_gen, cleaned_ref)
                
                # Use the higher score
                if alt_score > score:
                    logger.debug(f"Using alternative similarity score: {alt_score:.4f} > {score:.4f}")
                    score = alt_score

            # Apply verbosity penalty if configured
            if self.apply_verbosity_penalty:
                len_gen, len_ref = len(generated_answer), len(reference_answer)
                if len_gen > len_ref * 1.5:  # Penalize if >50% longer
                    # Use standardized verbosity penalty calculation
                    penalty_factor = calculate_verbosity_penalty(
                        len_gen,
                        len_ref,
                        self.verbosity_penalty_strength
                    )
                    score *= penalty_factor
                    return {
                        "reward": score,
                        "log": f"raw_score={raw_score:.4f}, verbosity_penalty={penalty_factor:.4f}"
                    }

            return {"reward": score, "log": f"raw_score={raw_score:.4f}"}

        except Exception as e:
            logger.error(f"SemanticSimilarityReward failed: {e}", exc_info=True)
            return {"reward": 0.0, "log": f"exception_in_compute: {str(e)[:100]}"}
            
    def _compute_tfidf_similarity(self, text1: str, text2: str) -> float:
        """
        Compute TF-IDF based cosine similarity with comprehensive error handling.
        
        Args:
            text1: First text to compare
            text2: Second text to compare
            
        Returns:
            Similarity score between 0.0 and 1.0
        """
        try:
            # Create a new vectorizer for each comparison to avoid state issues
            vectorizer = TfidfVectorizer(
                stop_words="english",
                norm="l2",
                min_df=1,
                ngram_range=(1, 2)  # Use unigrams and bigrams for better coverage
            )
            vectors = vectorizer.fit_transform([text1, text2])
            
            # Verify vectors are not empty
            if vectors.shape[0] != 2 or vectors.getnnz() == 0:
                logger.warning("TF-IDF produced empty vectors, falling back to word overlap")
                return self._compute_word_overlap(text1, text2)
            
            # Calculate cosine similarity safely
            try:
                # Get the vectors as arrays (handles both sparse and dense matrices)
                vec1 = vectors[0].toarray().flatten() if hasattr(vectors[0], "toarray") else vectors[0].flatten()
                vec2 = vectors[1].toarray().flatten() if hasattr(vectors[1], "toarray") else vectors[1].flatten()
                
                # Calculate dot product and magnitudes manually for safety
                dot_product = np.dot(vec1, vec2)
                magnitude1 = np.sqrt(np.dot(vec1, vec1))
                magnitude2 = np.sqrt(np.dot(vec2, vec2))
                
                # Avoid division by zero
                if magnitude1 > 0 and magnitude2 > 0:
                    return float(dot_product / (magnitude1 * magnitude2))
                else:
                    logger.warning("Zero magnitude in TF-IDF vectors, falling back to word overlap")
                    return self._compute_word_overlap(text1, text2)
                    
            except Exception as e:
                logger.warning(f"Manual cosine similarity calculation failed: {e}. Using fallback.")
                return self._compute_word_overlap(text1, text2)
                
        except Exception as vec_error:
            logger.warning(f"TF-IDF vectorization failed: {vec_error}. Falling back to word overlap.")
            return self._compute_word_overlap(text1, text2)
            
    def _compute_word_overlap(self, text1: str, text2: str) -> float:
        """
        Compute simple word overlap (Jaccard similarity) between two texts.
        
        Args:
            text1: First text to compare
            text2: Second text to compare
            
        Returns:
            Similarity score between 0.0 and 1.0
        """
        try:
            # Split into words and create sets
            words1 = set(text1.split())
            words2 = set(text2.split())
            
            # Handle empty sets
            if not words1 or not words2:
                return 0.0
                
            # Calculate Jaccard similarity
            overlap = len(words1.intersection(words2))
            total = len(words1.union(words2))
            
            return float(overlap / total) if total > 0 else 0.0
            
        except Exception as e:
            logger.warning(f"Word overlap calculation failed: {e}. Returning 0.0")
            return 0.0
            
    def _compute_similarity_for_short_texts(self, text1: str, text2: str) -> float:
        """
        Compute similarity for very short texts using character n-grams.
        
        Args:
            text1: First text to compare
            text2: Second text to compare
            
        Returns:
            Similarity score between 0.0 and 1.0
        """
        try:
            # For very short texts, try character-level n-grams
            vectorizer = TfidfVectorizer(
                analyzer='char',
                ngram_range=(2, 4),  # Character bigrams, trigrams, and 4-grams
                min_df=1
            )
            
            try:
                vectors = vectorizer.fit_transform([text1, text2])
                similarity = cosine_similarity(vectors[0], vectors[1])[0][0]
                return float(similarity)
            except Exception:
                # If character n-grams fail, fall back to simple character overlap
                chars1 = set(text1)
                chars2 = set(text2)
                
                if not chars1 or not chars2:
                    return 0.0
                    
                overlap = len(chars1.intersection(chars2))
                total = len(chars1.union(chars2))
                
                return float(overlap / total) if total > 0 else 0.0
                
        except Exception as e:
            logger.warning(f"Short text similarity calculation failed: {e}. Returning 0.0")
            return 0.0
            
    def _compute_alternative_similarity(self, text1: str, text2: str) -> float:
        """
        Try multiple similarity methods and return the highest score.
        
        Args:
            text1: First text to compare
            text2: Second text to compare
            
        Returns:
            Best similarity score between 0.0 and 1.0
        """
        scores = []
        
        # Try word overlap
        scores.append(self._compute_word_overlap(text1, text2))
        
        # Try character n-grams
        scores.append(self._compute_similarity_for_short_texts(text1, text2))
        
        # Try TF-IDF with different parameters
        try:
            vectorizer = TfidfVectorizer(
                stop_words=None,  # Don't remove stop words
                ngram_range=(1, 3),  # Unigrams, bigrams, and trigrams
                min_df=1
            )
            vectors = vectorizer.fit_transform([text1, text2])
            similarity = cosine_similarity(vectors[0], vectors[1])[0][0]
            scores.append(float(similarity))
        except Exception:
            pass
            
        # Return the highest score
        return max(scores) if scores else 0.0

    def batch_compute(self, contexts: List[RewardContext]) -> List[Dict[str, Any]]:
        """
        Computes rewards for a batch of contexts by calling compute sequentially.
        This is more robust than a complex batch implementation.
        
        Args:
            contexts: List of RewardContext objects to evaluate
            
        Returns:
            List of dictionaries containing reward scores and logs
        """
        results = []
        for context in contexts:
            try:
                # Get the raw result from compute
                result_dict = self.compute(context)
                
                # Extract the reward value
                raw_score = result_dict.get('reward', 0.0)
                
                # Apply smoothing
                smoothed_score = self._smooth_reward(raw_score)
                
                # Create properly formatted output with total key
                output = {
                    self.name: smoothed_score,
                    "total": smoothed_score,  # Add the total key
                    "log": result_dict.get('log', {})
                }
                
                results.append(output)
            except Exception as e:
                logger.error(f"Exception in batch_compute for context: {e}")
                results.append({
                    self.name: 0.0,
                    "total": 0.0,  # Include total key in error case too
                    "log": {"error": f"batch_exception: {str(e)[:100]}"}
                })
        
        return results
