# Tokenizer Requirements and Thinking Length Constraints

This document provides detailed information about tokenizer requirements and thinking length constraints in the MLX RL Trainer reward shaping system.

## Table of Contents

- [Overview](#overview)
- [Tokenizer Requirements](#tokenizer-requirements)
  - [Character-based Fallbacks](#character-based-fallbacks)
  - [Tokenizer Integration](#tokenizer-integration)
- [Thinking Length Constraints](#thinking-length-constraints)
  - [Character-based Constraints](#character-based-constraints)
  - [Token-based Constraints](#token-based-constraints)
  - [Hardware-specific Considerations](#hardware-specific-considerations)
- [ThinkingConstrainedRewardWrapper](#thinkingconstrainedrewardwrapper)
  - [Usage Examples](#usage-examples)
  - [Configuration Options](#configuration-options)
- [Best Practices](#best-practices)
  - [Balancing Thinking Quality and Length](#balancing-thinking-quality-and-length)
  - [Model-specific Tuning](#model-specific-tuning)

## Overview

The reward shaping system includes mechanisms to evaluate and constrain the length of thinking sections in model outputs. These constraints serve multiple purposes:

1. **Efficiency**: Prevent excessive token usage in thinking sections
2. **Quality**: Encourage concise, focused reasoning
3. **Hardware Compatibility**: Adapt to specific hardware limitations

## Tokenizer Requirements

### Character-based Fallbacks

The reward functions are designed to work without requiring a tokenizer, using character-based approximations as fallbacks:

- `TagStructureReward` uses character counts for length calculations
- `calculate_verbosity_penalty()` in `tag_utils.py` works with character counts
- All reward functions have graceful degradation when tokenizers aren't available

Example of character-based length calculation:

```python
# Character-based length calculation (from TagStructureReward)
think_text = extract_think_region(generated, gen_config)
think_len = len(think_text.strip())
```

### Tokenizer Integration

While not strictly required, integrating a tokenizer provides more accurate length calculations, especially for non-Latin scripts and special tokens:

1. **Passing Tokenizer Information**:

```python
# Example: Passing tokenizer information through context metadata
context = RewardContext(
    generated_text=generated_text,
    prompt_text=prompt,
    reference_completion=reference,
    metadata={
        "tokenizer_name": "gpt2",  # Identifier for the tokenizer
        "max_thinking_tokens": 128  # Maximum thinking tokens allowed
    }
)
```

2. **Accessing Tokenizer in Reward Functions**:

```python
# Example: Using tokenizer information in a reward function
tokenizer_name = context.metadata.get("tokenizer_name")
if tokenizer_name:
    # Use appropriate tokenizer based on name
    pass
else:
    # Fall back to character-based approximation
    pass
```

## Thinking Length Constraints

### Character-based Constraints

`TagStructureReward` implements character-based thinking length constraints:

- `min_think_length`: Minimum acceptable thinking length (default: 20 characters)
- `think_target_min`: Target minimum thinking length (default: 100 characters)
- `think_target_max`: Target maximum thinking length (default: 250 characters)
- `length_penalty_strength`: Strength of length deviation penalty (default: 0.5)
- `verbosity_penalty_factor`: Multiplier for verbosity penalty (default: 2.0)

The reward function applies penalties for thinking sections that are too short or too long:

```python
# Configuration example for TagStructureReward
tag_structure_reward = TagStructureReward({
    "weight": 0.3,
    "min_think_length": 30,
    "think_length_target_min": 150,
    "think_length_target_max": 300,
    "length_penalty_strength": 0.7,
    "verbosity_penalty_factor": 1.5
})
```

### Token-based Constraints

`ThinkingQualityReward` implements token-based thinking length constraints:

- `target_length_min`: Minimum target token count (default: 30 tokens)
- `target_length_max`: Maximum target token count (default: 80 tokens)
- `optimal_length_min`: Optimal minimum token count (default: 40 tokens)
- `optimal_length_max`: Optimal maximum token count (default: 60 tokens)
- `excessive_length_threshold`: Threshold for excessive length (default: 90 tokens)
- `excessive_length_penalty`: Penalty for excessive length (default: 0.5)

The reward function can also integrate with trainer limits:

```python
# Configuration example for ThinkingQualityReward
thinking_quality_reward = ThinkingQualityReward({
    "weight": 0.4,
    "target_length_min": 40,
    "target_length_max": 100,
    "use_trainer_thinking_limits": True,  # Use limits from trainer config
    "excessive_length_threshold": 120,
    "excessive_length_penalty": 0.6
})
```

### Hardware-specific Considerations

The thinking length constraints can be adjusted based on hardware limitations:

- For memory-constrained environments, use stricter limits
- For high-performance environments, allow longer thinking sections

Example configuration for different hardware profiles:

```python
# Low-memory configuration (e.g., mobile devices)
low_memory_config = {
    "target_length_min": 20,
    "target_length_max": 50,
    "excessive_length_threshold": 60,
    "excessive_length_penalty": 0.8
}

# Standard configuration (e.g., laptops)
standard_config = {
    "target_length_min": 30,
    "target_length_max": 80,
    "excessive_length_threshold": 100,
    "excessive_length_penalty": 0.5
}

# High-performance configuration (e.g., servers)
high_performance_config = {
    "target_length_min": 50,
    "target_length_max": 150,
    "excessive_length_threshold": 200,
    "excessive_length_penalty": 0.3
}
```

## Integration with Reward System

### Using Thinking Length Constraints

To effectively use thinking length constraints in your reward system:

1. **Configure the reward functions with appropriate length constraints**:

```python
from mlx_rl_trainer.rewards.format.tag_structure import TagStructureReward
from mlx_rl_trainer.rewards.reasoning.thinking_quality import ThinkingQualityReward
from mlx_rl_trainer.rewards.base_reward import RewardComposer

# Create reward functions with length constraints
tag_structure = TagStructureReward({
    "weight": 0.3,
    "think_length_target_min": 100,
    "think_length_target_max": 250
})

thinking_quality = ThinkingQualityReward({
    "weight": 0.7,
    "target_length_min": 30,
    "target_length_max": 80,
    "use_trainer_thinking_limits": True
})

# Compose rewards
reward_composer = RewardComposer([
    (tag_structure, 0.3),
    (thinking_quality, 0.7)
], auto_normalize=True)
```

2. **Pass trainer limits through context metadata**:

```python
# Create context with trainer limits
context = RewardContext(
    generated_text=generated_text,
    prompt_text=prompt,
    reference_completion=reference,
    metadata={
        "max_thinking_tokens": 128  # Maximum thinking tokens allowed by trainer
    }
)

# Compute rewards
result = reward_composer.batch_compute([context])
```

### Accessing Thinking Length Information

You can access thinking length information from the reward results:

```python
# Get thinking length information from reward results
result = tag_structure.compute(context)
think_len = result["log"].get("think_len", 0)
print(f"Thinking length: {think_len} characters")

result = thinking_quality.compute(context)
think_len = result["log"].get("length", 0)
print(f"Thinking length: {think_len} tokens")
```

## Best Practices

### Balancing Thinking Quality and Length

When configuring thinking length constraints, consider the following trade-offs:

- **Too short**: May result in incomplete reasoning and lower quality outputs
- **Too long**: May waste computational resources and lead to verbosity
- **Just right**: Encourages concise, focused reasoning

Recommended approach:

1. Start with default constraints
2. Monitor thinking lengths in your training data
3. Adjust constraints based on observed quality and efficiency
4. Consider model-specific tuning

### Model-specific Tuning

Different models may require different thinking length constraints:

- **Smaller models** (1-3B parameters): May benefit from shorter thinking sections
- **Medium models** (7-13B parameters): Can handle moderate thinking lengths
- **Larger models** (>30B parameters): May produce higher quality with longer thinking sections

Example model-specific configurations:

```python
# Small model configuration
small_model_config = {
    "think_length_target_min": 50,
    "think_length_target_max": 150
}

# Medium model configuration
medium_model_config = {
    "think_length_target_min": 100,
    "think_length_target_max": 250
}

# Large model configuration
large_model_config = {
    "think_length_target_min": 150,
    "think_length_target_max": 400
}
```

## Conclusion

Proper configuration of tokenizer requirements and thinking length constraints is essential for effective reward shaping. By understanding and leveraging these features, you can train models that produce high-quality, efficient reasoning while respecting hardware limitations.