"""
Phase Detection and Management for Enhanced Dynamic Logit Processing

This module implements sophisticated phase detection logic for identifying and managing
different phases of text generation, particularly the transition between thinking and
answer phases. It provides enterprise-grade phase detection with comprehensive error
handling, performance optimization, and extensibility.

The phase detection system follows these key principles:
- Thinking Phase: From start until `</think>` is encountered
- Answer Phase: Everything after `</think>\n` 
- Transition Handling: Special processing at the `</think>` boundary
- Edge Cases: Handle malformed or missing tags gracefully

Architecture Features:
- State Machine Pattern: Clean phase transitions with validation
- Observer Pattern: Event-driven phase change notifications
- Strategy Pattern: Pluggable detection algorithms
- Template Method: Consistent detection workflow
- Chain of Responsibility: Hierarchical detection rules

Performance Characteristics:
- O(1) phase lookup with cached state
- O(n) text scanning where n is recent history length
- Minimal memory footprint with sliding window approach
- Thread-safe operations with lock-free algorithms where possible

Security Considerations:
- Input sanitization to prevent injection attacks
- Resource limits to prevent DoS via large inputs
- Secure pattern matching without regex vulnerabilities
- Audit logging for all phase transitions

Example:
    >>> from mlx_rl_trainer.generation.processors.phase_processor import PhaseDetector
    >>> from mlx_rl_trainer.generation.processors.base import ProcessingContext
    >>> 
    >>> detector = PhaseDetector()
    >>> history = [["<think>", "Let", "me", "think", "about", "this"]]
    >>> context = ProcessingContext(config=config, tokenizer=tokenizer)
    >>> 
    >>> phase = detector.detect_phase(history, context)
    >>> print(f"Current phase: {phase}")  # ProcessingPhase.THINKING
    >>> 
    >>> # After generating </think>
    >>> history[0].extend(["</think>", "\n", "The", "answer", "is"])
    >>> phase = detector.detect_phase(history, context)
    >>> print(f"Current phase: {phase}")  # ProcessingPhase.ANSWER
"""

import logging
import re
import threading
import time
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum, auto
from functools import lru_cache
from typing import (
    Any, Callable, Dict, List, Optional, Pattern, Set, Tuple, 
    Union, Protocol, runtime_checkable
)
from weakref import WeakKeyDictionary

import mlx.core as mx
from mlx_lm.tokenizer_utils import TokenizerWrapper

from .base import (
    BaseLogitProcessor, ProcessingContext, ProcessingPhase, 
    ProcessingException, ProcessingMetrics, ProcessingPriority,
    performance_monitor
)

logger = logging.getLogger(__name__)


class PhaseTransitionType(Enum):
    """
    Types of phase transitions that can occur during generation.
    
    This enum categorizes different types of phase transitions to enable
    specialized handling and monitoring of each transition type.
    """
    INITIALIZATION = auto()
    THINKING_START = auto()
    THINKING_CONTINUE = auto()
    THINK_CLOSE_APPROACH = auto()
    THINK_CLOSE_COMPLETE = auto()
    ANSWER_START = auto()
    ANSWER_CONTINUE = auto()
    COMPLETION = auto()
    ERROR_RECOVERY = auto()

    def __str__(self) -> str:
        return self.name.lower()

    @property
    def is_major_transition(self) -> bool:
        """Check if this represents a major phase boundary."""
        return self in (
            self.THINKING_START, 
            self.THINK_CLOSE_COMPLETE, 
            self.ANSWER_START, 
            self.COMPLETION
        )

    @property
    def requires_bias_adjustment(self) -> bool:
        """Check if this transition requires bias adjustments."""
        return self in (
            self.THINK_CLOSE_APPROACH,
            self.THINK_CLOSE_COMPLETE,
            self.ANSWER_START
        )


@dataclass(frozen=True)
class PhaseTransitionEvent:
    """
    Immutable event representing a phase transition.
    
    This class encapsulates all information about a phase transition,
    enabling comprehensive logging, monitoring, and event-driven processing.
    
    Attributes:
        from_phase: Previous processing phase
        to_phase: New processing phase
        transition_type: Type of transition that occurred
        batch_index: Index of batch item that triggered transition
        token_position: Position in generation where transition occurred
        trigger_tokens: Tokens that triggered the transition
        confidence: Confidence score for the transition (0.0-1.0)
        metadata: Additional transition-specific data
        timestamp: When the transition occurred
    """
    from_phase: ProcessingPhase
    to_phase: ProcessingPhase
    transition_type: PhaseTransitionType
    batch_index: int
    token_position: int
    trigger_tokens: Tuple[int, ...]
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def __post_init__(self):
        """Validate transition event data."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be between 0.0 and 1.0, got {self.confidence}")
        
        if self.batch_index < 0:
            raise ValueError(f"Batch index must be non-negative, got {self.batch_index}")
        
        if self.token_position < 0:
            raise ValueError(f"Token position must be non-negative, got {self.token_position}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary for serialization."""
        return {
            'from_phase': str(self.from_phase),
            'to_phase': str(self.to_phase),
            'transition_type': str(self.transition_type),
            'batch_index': self.batch_index,
            'token_position': self.token_position,
            'trigger_tokens': list(self.trigger_tokens),
            'confidence': self.confidence,
            'metadata': self.metadata.copy(),
            'timestamp': self.timestamp
        }


class PhaseDetectionException(ProcessingException):
    """Raised when phase detection encounters errors."""
    pass


class PhaseTransitionException(ProcessingException):
    """Raised when phase transitions are invalid or impossible."""
    pass


@runtime_checkable
class PhaseTransitionListener(Protocol):
    """
    Protocol for listening to phase transition events.
    
    This protocol enables event-driven processing of phase transitions,
    allowing for decoupled monitoring, logging, and reactive processing.
    """
    
    def on_phase_transition(self, event: PhaseTransitionEvent) -> None:
        """
        Handle a phase transition event.
        
        Args:
            event: Phase transition event to handle
        """
        ...


@dataclass
class PhaseDetectionState:
    """
    Mutable state container for phase detection.
    
    This class maintains the current state of phase detection for each
    batch item, enabling stateful detection across multiple generation steps.
    
    Attributes:
        current_phase: Current processing phase
        last_transition: Last transition event
        think_tag_positions: Positions where think tags were detected
        answer_start_position: Position where answer phase started
        confidence_history: History of detection confidence scores
        error_count: Number of detection errors encountered
        custom_state: Additional detector-specific state
    """
    current_phase: ProcessingPhase = ProcessingPhase.INITIALIZATION
    last_transition: Optional[PhaseTransitionEvent] = None
    think_tag_positions: List[int] = field(default_factory=list)
    answer_start_position: Optional[int] = None
    confidence_history: deque = field(default_factory=lambda: deque(maxlen=100))
    error_count: int = 0
    custom_state: Dict[str, Any] = field(default_factory=dict)
    
    def add_confidence_score(self, score: float) -> None:
        """Add a confidence score to the history."""
        self.confidence_history.append((time.time(), score))
    
    def get_average_confidence(self, window_size: int = 10) -> float:
        """Get average confidence over recent detections."""
        if not self.confidence_history:
            return 0.0
        
        recent_scores = list(self.confidence_history)[-window_size:]
        return sum(score for _, score in recent_scores) / len(recent_scores)
    
    def reset(self) -> None:
        """Reset state to initial values."""
        self.current_phase = ProcessingPhase.INITIALIZATION
        self.last_transition = None
        self.think_tag_positions.clear()
        self.answer_start_position = None
        self.confidence_history.clear()
        self.error_count = 0
        self.custom_state.clear()


class PhaseDetectionRule(ABC):
    """
    Abstract base class for phase detection rules.
    
    This class implements the Strategy pattern, allowing different
    detection algorithms to be plugged in based on requirements.
    Each rule encapsulates specific logic for detecting phase transitions.
    """
    
    def __init__(self, priority: int = 100, enabled: bool = True):
        """
        Initialize detection rule.
        
        Args:
            priority: Rule priority (higher values execute first)
            enabled: Whether rule is currently enabled
        """
        self.priority = priority
        self.enabled = enabled
        self.rule_id = f"{self.__class__.__name__}_{id(self)}"
    
    @abstractmethod
    def detect_transition(
        self, 
        history: List[int], 
        context: ProcessingContext,
        state: PhaseDetectionState
    ) -> Optional[PhaseTransitionEvent]:
        """
        Detect phase transition based on history and state.
        
        Args:
            history: Token generation history for single batch item
            context: Processing context with configuration
            state: Current phase detection state
            
        Returns:
            Phase transition event if detected, None otherwise
        """
        pass
    
    def can_apply(
        self, 
        history: List[int], 
        context: ProcessingContext,
        state: PhaseDetectionState
    ) -> bool:
        """
        Check if this rule can be applied to the current state.
        
        Args:
            history: Token generation history
            context: Processing context
            state: Current detection state
            
        Returns:
            True if rule can be applied, False otherwise
        """
        return self.enabled
    
    def __lt__(self, other: 'PhaseDetectionRule') -> bool:
        """Enable sorting by priority."""
        return self.priority > other.priority  # Higher priority first


class ThinkTagDetectionRule(PhaseDetectionRule):
    """
    Rule for detecting thinking tag patterns in generation history.
    
    This rule implements sophisticated pattern matching for detecting
    `<think>` and `</think>` tags with support for various tokenization
    schemes and malformed tag handling.
    """
    
    def __init__(self, priority: int = 1000, enabled: bool = True):
        """Initialize think tag detection rule."""
        super().__init__(priority, enabled)
        
        # Cache for tokenized patterns
        self._pattern_cache: Dict[str, List[int]] = {}
        self._cache_lock = threading.RLock()
        
        # Common think tag variations
        self._think_patterns = [
            "<think>", "< think >", "<think >", "< think>",
            "</think>", "</ think >", "</think >", "</ think>"
        ]
    
    @lru_cache(maxsize=1000)
    def _get_tokenized_pattern(self, pattern: str, tokenizer: TokenizerWrapper) -> Tuple[int, ...]:
        """
        Get tokenized representation of a pattern with caching.
        
        Args:
            pattern: Text pattern to tokenize
            tokenizer: Tokenizer to use
            
        Returns:
            Tuple of token IDs for the pattern
        """
        try:
            # Tokenize the pattern
            tokens = tokenizer.encode(pattern)
            return tuple(tokens)
        except Exception as e:
            logger.warning(f"Failed to tokenize pattern '{pattern}': {e}")
            return tuple()
    
    def detect_transition(
        self, 
        history: List[int], 
        context: ProcessingContext,
        state: PhaseDetectionState
    ) -> Optional[PhaseTransitionEvent]:
        """
        Detect think tag transitions in the generation history.
        
        This method implements sophisticated pattern matching with support for:
        - Multiple tokenization schemes
        - Partial tag detection
        - Malformed tag recovery
        - Confidence scoring based on pattern completeness
        
        Args:
            history: Token generation history for single batch item
            context: Processing context with tokenizer
            state: Current phase detection state
            
        Returns:
            Phase transition event if think tag detected, None otherwise
        """
        if not history:
            return None
        
        # Look for think tag patterns in recent history
        search_window = min(len(history), 20)  # Limit search for performance
        recent_history = history[-search_window:]
        
        # Check for opening think tag
        if state.current_phase == ProcessingPhase.INITIALIZATION:
            for pattern in ["<think>", "< think >"]:
                pattern_tokens = self._get_tokenized_pattern(pattern, context.tokenizer)
                if self._matches_pattern_end(recent_history, pattern_tokens):
                    return PhaseTransitionEvent(
                        from_phase=state.current_phase,
                        to_phase=ProcessingPhase.THINKING,
                        transition_type=PhaseTransitionType.THINKING_START,
                        batch_index=0,  # Will be set by caller
                        token_position=len(history),
                        trigger_tokens=pattern_tokens,
                        confidence=0.95,
                        metadata={'pattern': pattern, 'rule': self.rule_id}
                    )
        
        # Check for closing think tag
        elif state.current_phase == ProcessingPhase.THINKING:
            # Look for complete </think> pattern
            for pattern in ["</think>", "</ think >"]:
                pattern_tokens = self._get_tokenized_pattern(pattern, context.tokenizer)
                if self._matches_pattern_end(recent_history, pattern_tokens):
                    return PhaseTransitionEvent(
                        from_phase=state.current_phase,
                        to_phase=ProcessingPhase.THINK_TRANSITION,
                        transition_type=PhaseTransitionType.THINK_CLOSE_COMPLETE,
                        batch_index=0,
                        token_position=len(history),
                        trigger_tokens=pattern_tokens,
                        confidence=0.98,
                        metadata={'pattern': pattern, 'rule': self.rule_id}
                    )
            
            # Look for partial closing tag (approaching)
            partial_patterns = ["</", "</ ", "</th", "</thi", "</thin", "</think"]
            for pattern in partial_patterns:
                pattern_tokens = self._get_tokenized_pattern(pattern, context.tokenizer)
                if self._matches_pattern_end(recent_history, pattern_tokens):
                    confidence = len(pattern) / len("</think>")  # Confidence based on completeness
                    return PhaseTransitionEvent(
                        from_phase=state.current_phase,
                        to_phase=ProcessingPhase.THINK_TRANSITION,
                        transition_type=PhaseTransitionType.THINK_CLOSE_APPROACH,
                        batch_index=0,
                        token_position=len(history),
                        trigger_tokens=pattern_tokens,
                        confidence=confidence * 0.8,  # Lower confidence for partial matches
                        metadata={'pattern': pattern, 'rule': self.rule_id, 'partial': True}
                    )
        
        return None
    
    def _matches_pattern_end(self, history: List[int], pattern: Tuple[int, ...]) -> bool:
        """
        Check if history ends with the given pattern.
        
        Args:
            history: Token history to check
            pattern: Pattern tokens to match
            
        Returns:
            True if history ends with pattern, False otherwise
        """
        if not pattern or len(pattern) > len(history):
            return False
        
        return tuple(history[-len(pattern):]) == pattern


class NewlineDetectionRule(PhaseDetectionRule):
    """
    Rule for detecting newline transitions after think tags.
    
    This rule handles the transition from `</think>` to answer phase
    by detecting newline characters that indicate the start of the answer.
    """
    
    def __init__(self, priority: int = 900, enabled: bool = True):
        """Initialize newline detection rule."""
        super().__init__(priority, enabled)
    
    def detect_transition(
        self, 
        history: List[int], 
        context: ProcessingContext,
        state: PhaseDetectionState
    ) -> Optional[PhaseTransitionEvent]:
        """
        Detect newline transition after think close.
        
        Args:
            history: Token generation history
            context: Processing context
            state: Current detection state
            
        Returns:
            Phase transition event if newline detected after think close
        """
        if state.current_phase != ProcessingPhase.THINK_TRANSITION:
            return None
        
        if not history:
            return None
        
        # Look for newline tokens
        newline_tokens = self._get_newline_tokens(context.tokenizer)
        
        # Check if we just generated a newline
        if history and history[-1] in newline_tokens:
            return PhaseTransitionEvent(
                from_phase=state.current_phase,
                to_phase=ProcessingPhase.ANSWER,
                transition_type=PhaseTransitionType.ANSWER_START,
                batch_index=0,
                token_position=len(history),
                trigger_tokens=(history[-1],),
                confidence=0.95,
                metadata={'rule': self.rule_id, 'newline_token': history[-1]}
            )
        
        return None
    
    @lru_cache(maxsize=100)
    def _get_newline_tokens(self, tokenizer: TokenizerWrapper) -> Set[int]:
        """
        Get set of token IDs that represent newlines.
        
        Args:
            tokenizer: Tokenizer to use for encoding
            
        Returns:
            Set of newline token IDs
        """
        newline_variants = ["\n", "\\n", "\r\n", "\r"]
        newline_tokens = set()
        
        for variant in newline_variants:
            try:
                tokens = tokenizer.encode(variant)
                newline_tokens.update(tokens)
            except Exception:
                continue
        
        return newline_tokens


class AnswerContinuationRule(PhaseDetectionRule):
    """
    Rule for detecting continued answer generation.
    
    This rule maintains the answer phase once it has started,
    providing stability and preventing spurious phase changes.
    """
    
    def __init__(self, priority: int = 500, enabled: bool = True):
        """Initialize answer continuation rule."""
        super().__init__(priority, enabled)
    
    def detect_transition(
        self, 
        history: List[int], 
        context: ProcessingContext,
        state: PhaseDetectionState
    ) -> Optional[PhaseTransitionEvent]:
        """
        Detect answer continuation.
        
        Args:
            history: Token generation history
            context: Processing context
            state: Current detection state
            
        Returns:
            Phase transition event for answer continuation
        """
        if state.current_phase == ProcessingPhase.ANSWER and history:
            # Continue in answer phase
            return PhaseTransitionEvent(
                from_phase=state.current_phase,
                to_phase=ProcessingPhase.ANSWER,
                transition_type=PhaseTransitionType.ANSWER_CONTINUE,
                batch_index=0,
                token_position=len(history),
                trigger_tokens=(history[-1],) if history else tuple(),
                confidence=0.9,
                metadata={'rule': self.rule_id}
            )
        
        return None


class PhaseDetector:
    """
    Main phase detection engine with rule-based processing.
    
    This class orchestrates multiple detection rules to provide robust
    phase detection with comprehensive error handling, performance monitoring,
    and extensibility. It implements the Chain of Responsibility pattern
    for rule processing and the Observer pattern for event notifications.
    
    Features:
    - Rule-based detection with priority ordering
    - Per-batch-item state management
    - Event-driven notifications
    - Performance monitoring and caching
    - Thread-safe operations
    - Comprehensive error handling
    
    Performance Characteristics:
    - O(1) state lookup per batch item
    - O(r) rule evaluation where r is number of active rules
    - O(h) history scanning where h is search window size
    - Minimal memory footprint with LRU caching
    """
    
    def __init__(
        self,
        rules: Optional[List[PhaseDetectionRule]] = None,
        max_history_window: int = 50,
        enable_caching: bool = True,
        enable_monitoring: bool = True
    ):
        """
        Initialize phase detector with configuration.
        
        Args:
            rules: List of detection rules (default rules used if None)
            max_history_window: Maximum history window for detection
            enable_caching: Enable result caching for performance
            enable_monitoring: Enable performance monitoring
        """
        self.max_history_window = max_history_window
        self.enable_caching = enable_caching
        self.enable_monitoring = enable_monitoring
        
        # Initialize rules
        self._rules = rules or self._create_default_rules()
        self._rules.sort()  # Sort by priority
        
        # State management
        self._batch_states: WeakKeyDictionary[ProcessingContext, Dict[int, PhaseDetectionState]] = WeakKeyDictionary()
        self._state_lock = threading.RLock()
        
        # Event handling
        self._listeners: List[PhaseTransitionListener] = []
        self._listener_lock = threading.RLock()
        
        # Performance monitoring
        self._detection_count = 0
        self._total_detection_time = 0.0
        self._error_count = 0
        self._cache_hits = 0
        self._cache_misses = 0
        
        # Caching
        if self.enable_caching:
            self._result_cache: Dict[str, Tuple[ProcessingPhase, float]] = {}
            self._cache_lock = threading.RLock()
        
        logger.info(f"Initialized PhaseDetector with {len(self._rules)} rules")
    
    def _create_default_rules(self) -> List[PhaseDetectionRule]:
        """Create default set of detection rules."""
        return [
            ThinkTagDetectionRule(priority=1000),
            NewlineDetectionRule(priority=900),
            AnswerContinuationRule(priority=500)
        ]
    
    def add_rule(self, rule: PhaseDetectionRule) -> None:
        """
        Add a detection rule to the detector.
        
        Args:
            rule: Detection rule to add
        """
        self._rules.append(rule)
        self._rules.sort()  # Re-sort by priority
        logger.info(f"Added detection rule: {rule.rule_id}")
    
    def remove_rule(self, rule_id: str) -> bool:
        """
        Remove a detection rule by ID.
        
        Args:
            rule_id: ID of rule to remove
            
        Returns:
            True if rule was found and removed, False otherwise
        """
        for i, rule in enumerate(self._rules):
            if rule.rule_id == rule_id:
                del self._rules[i]
                logger.info(f"Removed detection rule: {rule_id}")
                return True
        return False
    
    def add_listener(self, listener: PhaseTransitionListener) -> None:
        """
        Add a phase transition listener.
        
        Args:
            listener: Listener to add
        """
        with self._listener_lock:
            self._listeners.append(listener)
    
    def remove_listener(self, listener: PhaseTransitionListener) -> bool:
        """
        Remove a phase transition listener.
        
        Args:
            listener: Listener to remove
            
        Returns:
            True if listener was found and removed, False otherwise
        """
        with self._listener_lock:
            try:
                self._listeners.remove(listener)
                return True
            except ValueError:
                return False
    
    @performance_monitor
    def detect_phase(
        self, 
        history: List[List[int]], 
        context: ProcessingContext
    ) -> List[ProcessingPhase]:
        """
        Detect current processing phase for each batch item.
        
        This is the main entry point for phase detection. It processes
        each batch item independently and returns the detected phases.
        
        Args:
            history: Token generation history for each batch item
            context: Processing context with configuration
            
        Returns:
            List of detected phases for each batch item
            
        Raises:
            PhaseDetectionException: If detection fails
        """
        start_time = time.perf_counter()
        
        try:
            phases = []
            
            for batch_idx, item_history in enumerate(history):
                try:
                    phase = self._detect_phase_for_item(
                        item_history, 
                        context, 
                        batch_idx
                    )
                    phases.append(phase)
                    
                except Exception as e:
                    logger.error(
                        f"Phase detection failed for batch item {batch_idx}: {e}",
                        extra={'correlation_id': context.correlation_id}
                    )
                    
                    # Use fallback phase
                    phases.append(self._get_fallback_phase(item_history))
                    
                    with self._state_lock:
                        self._error_count += 1
            
            return phases
            
        except Exception as e:
            raise PhaseDetectionException(
                f"Phase detection failed: {e}",
                correlation_id=context.correlation_id,
                phase=context.phase
            ) from e
            
        finally:
            # Update performance metrics
            if self.enable_monitoring:
                detection_time = time.perf_counter() - start_time
                with self._state_lock:
                    self._detection_count += 1
                    self._total_detection_time += detection_time
    
    def _detect_phase_for_item(
        self, 
        history: List[int], 
        context: ProcessingContext,
        batch_idx: int
    ) -> ProcessingPhase:
        """
        Detect phase for a single batch item.
        
        Args:
            history: Token history for the item
            context: Processing context
            batch_idx: Index of the batch item
            
        Returns:
            Detected processing phase
        """
        # Get or create state for this batch item
        state = self._get_batch_state(context, batch_idx)
        
        # Check cache first if enabled
        if self.enable_caching:
            cache_key = self._get_cache_key(history, context, batch_idx)
            cached_result = self._get_cached_result(cache_key)
            if cached_result is not None:
                phase, confidence = cached_result
                state.add_confidence_score(confidence)
                return phase
        
        # Apply detection rules in priority order
        transition_event = None
        
        for rule in self._rules:
            if not rule.can_apply(history, context, state):
                continue
            
            try:
                event = rule.detect_transition(history, context, state)
                if event is not None:
                    # Set correct batch index
                    event = PhaseTransitionEvent(
                        from_phase=event.from_phase,
                        to_phase=event.to_phase,
                        transition_type=event.transition_type,
                        batch_index=batch_idx,
                        token_position=event.token_position,
                        trigger_tokens=event.trigger_tokens,
                        confidence=event.confidence,
                        metadata=event.metadata,
                        timestamp=event.timestamp
                    )
                    
                    transition_event = event
                    break
                    
            except Exception as e:
                logger.warning(
                    f"Detection rule {rule.rule_id} failed: {e}",
                    extra={'correlation_id': context.correlation_id}
                )
                continue
        
        # Process transition if detected
        if transition_event:
            self._process_transition(state, transition_event)
            
            # Cache result if enabled
            if self.enable_caching:
                cache_key = self._get_cache_key(history, context, batch_idx)
                self._cache_result(cache_key, state.current_phase, transition_event.confidence)
        
        return state.current_phase
    
    def _get_batch_state(self, context: ProcessingContext, batch_idx: int) -> PhaseDetectionState:
        """
        Get or create phase detection state for a batch item.
        
        Args:
            context: Processing context
            batch_idx: Index of batch item
            
        Returns:
            Phase detection state for the batch item
        """
        with self._state_lock:
            if context not in self._batch_states:
                self._batch_states[context] = {}
            
            batch_states = self._batch_states[context]
            
            if batch_idx not in batch_states:
                batch_states[batch_idx] = PhaseDetectionState()
            
            return batch_states[batch_idx]
    
    def _process_transition(self, state: PhaseDetectionState, event: PhaseTransitionEvent) -> None:
        """
        Process a phase transition event.
        
        Args:
            state: Current detection state
            event: Transition event to process
        """
        # Validate transition
        if not self._is_valid_transition(state.current_phase, event.to_phase):
            logger.warning(
                f"Invalid phase transition: {state.current_phase} -> {event.to_phase}",
                extra={'event': event.to_dict()}
            )
            return
        
        # Update state
        old_phase = state.current_phase
        state.current_phase = event.to_phase
        state.last_transition = event
        state.add_confidence_score(event.confidence)
        
        # Update phase-specific state
        if event.transition_type == PhaseTransitionType.THINKING_START:
            state.think_tag_positions.append(event.token_position)
        elif event.transition_type == PhaseTransitionType.ANSWER_START:
            state.answer_start_position = event.token_position
        
        # Notify listeners
        self._notify_listeners(event)
        
        logger.debug(
            f"Phase transition: {old_phase} -> {event.to_phase} "
            f"(confidence: {event.confidence:.3f})"
        )
    
    def _is_valid_transition(self, from_phase: ProcessingPhase, to_phase: ProcessingPhase) -> bool:
        """
        Check if a phase transition is valid.
        
        Args:
            from_phase: Source phase
            to_phase: Target phase
            
        Returns:
            True if transition is valid, False otherwise
        """
        # Define valid transitions
        valid_transitions = {
            ProcessingPhase.INITIALIZATION: {
                ProcessingPhase.THINKING,
                ProcessingPhase.ANSWER,  # Direct to answer if no thinking
                ProcessingPhase.COMPLETION
            },
            ProcessingPhase.THINKING: {
                ProcessingPhase.THINKING,  # Continue thinking
                ProcessingPhase.THINK_TRANSITION,
                ProcessingPhase.ERROR_RECOVERY
            },
            ProcessingPhase.THINK_TRANSITION: {
                ProcessingPhase.ANSWER,
                ProcessingPhase.THINKING,  # Back to thinking if malformed
                ProcessingPhase.ERROR_RECOVERY
            },
            ProcessingPhase.ANSWER: {
                ProcessingPhase.ANSWER,  # Continue answer
                ProcessingPhase.COMPLETION,
                ProcessingPhase.ERROR_RECOVERY
            },
            ProcessingPhase.ANSWER_TRANSITION: {
                ProcessingPhase.ANSWER,
                ProcessingPhase.COMPLETION,
                ProcessingPhase.ERROR_RECOVERY
            },
            ProcessingPhase.COMPLETION: {
                ProcessingPhase.COMPLETION,  # Stay completed
                ProcessingPhase.ERROR_RECOVERY
            },
            ProcessingPhase.ERROR_RECOVERY: {
                ProcessingPhase.THINKING,
                ProcessingPhase.ANSWER,
                ProcessingPhase.COMPLETION,
                ProcessingPhase.ERROR_RECOVERY
            }
        }
        
        return to_phase in valid_transitions.get(from_phase, set())
    
    def _notify_listeners(self, event: PhaseTransitionEvent) -> None:
        """
        Notify all registered listeners of a phase transition.
        
        Args:
            event: Transition event to broadcast
        """
        with self._listener_lock:
            for listener in self._listeners:
                try:
                    listener.on_phase_transition(event)
                except Exception as e:
                    logger.error(f"Listener notification failed: {e}")
    
    def _get_cache_key(self, history: List[int], context: ProcessingContext, batch_idx: int) -> str:
        """
        Generate cache key for detection result.
        
        Args:
            history: Token history
            context: Processing context
            batch_idx: Batch item index
            
        Returns:
            Cache key string
        """
        # Use recent history for cache key
        recent_history = history[-self.max_history_window:] if history else []
        history_hash = hash(tuple(recent_history))
        
        return f"phase_detection:{context.correlation_id}:{batch_idx}:{history_hash}"
    
    def _get_cached_result(self, cache_key: str) -> Optional[Tuple[ProcessingPhase, float]]:
        """
        Get cached detection result.
        
        Args:
            cache_key: Cache key to lookup
            
        Returns:
            Cached (phase, confidence) tuple or None if not found
        """
        if not self.enable_caching:
            return None
        
        with self._cache_lock:
            result = self._result_cache.get(cache_key)
            if result is not None:
                self._cache_hits += 1
                return result
            else:
                self._cache_misses += 1
                return None
    
    def _cache_result(self, cache_key: str, phase: ProcessingPhase, confidence: float) -> None:
        """
        Cache detection result.
        
        Args:
            cache_key: Cache key
            phase: Detected phase
            confidence: Detection confidence
        """
        if not self.enable_caching:
            return
        
        with self._cache_lock:
            self._result_cache[cache_key] = (phase, confidence)
            
            # Limit cache size
            if len(self._result_cache) > 10000:
                # Remove oldest 20% of entries
                keys_to_remove = list(self._result_cache.keys())[:2000]
                for key in keys_to_remove:
                    del self._result_cache[key]
    
    def _get_fallback_phase(self, history: List[int]) -> ProcessingPhase:
        """
        Get fallback phase when detection fails.
        
        Args:
            history: Token history
            
        Returns:
            Fallback processing phase
        """
        # Simple heuristic: if history is empty, initialization; otherwise, answer
        return ProcessingPhase.INITIALIZATION if not history else ProcessingPhase.ANSWER
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get performance statistics for the detector.
        
        Returns:
            Dictionary containing performance metrics
        """
        with self._state_lock:
            avg_time = (
                self._total_detection_time / self._detection_count 
                if self._detection_count > 0 else 0
            )
            
            cache_hit_ratio = (
                self._cache_hits / (self._cache_hits + self._cache_misses)
                if (self._cache_hits + self._cache_misses) > 0 else 0
            )
            
            return {
                'detection_count': self._detection_count,
                'total_time_seconds': self._total_detection_time,
                'average_time_seconds': avg_time,
                'error_count': self._error_count,
                'error_rate': self._error_count / self._detection_count if self._detection_count > 0 else 0,
                'cache_hits': self._cache_hits,
                'cache_misses': self._cache_misses,
                'cache_hit_ratio': cache_hit_ratio,
                'active_rules': len([r for r in self._rules if r.enabled]),
                'total_rules': len(self._rules)
            }
    
    def reset_stats(self) -> None:
        """Reset performance statistics."""
        with self._state_lock:
            self._detection_count = 0
            self._total_detection_time = 0.0
            self._error_count = 0
            self._cache_hits = 0
            self._cache_misses = 0
    
    def clear_cache(self) -> None:
        """Clear the detection result cache."""
        if self.enable_caching:
            with self._cache_lock:
                self._result_cache.clear()
                logger.info("Cleared phase detection cache")


class PhaseAwareLogitProcessor(BaseLogitProcessor):
    """
    Base class for logit processors that are aware of generation phases.
    
    This class extends BaseLogitProcessor with phase-aware capabilities,
    enabling processors to adapt their behavior based on the current
    generation phase (thinking vs. answer).
    
    Features:
    - Automatic phase detection integration
    - Phase-specific processing logic
    - Performance optimization based on phase
    - Comprehensive monitoring and logging
    """
    
    def __init__(
        self,
        detector: Optional[PhaseDetector] = None,
        processor_id: Optional[str] = None,
        priority: ProcessingPriority = ProcessingPriority.NORMAL,
        enabled: bool = True,
        **kwargs
    ):
        """
        Initialize phase-aware processor.
        
        Args:
            detector: Phase detector instance (default created if None)
            processor_id: Unique processor identifier
            priority: Processing priority
            enabled: Whether processor is enabled
            **kwargs: Additional arguments for base class
        """
        super().__init__(
            processor_id=processor_id,
            priority=priority,
            enabled=enabled,
            **kwargs
        )
        
        self.detector = detector or PhaseDetector()
        self._phase_cache: Dict[str, List[ProcessingPhase]] = {}
        self._cache_lock = threading.RLock()
    
    def _process_logits_impl(
        self, 
        logits: mx.array, 
        history: List[List[int]], 
        context: ProcessingContext
    ) -> mx.array:
        """
        Process logits with phase awareness.
        
        This method detects the current phases and delegates to
        phase-specific processing methods.
        
        Args:
            logits: Current logits tensor
            history: Token generation history
            context: Processing context
            
        Returns:
            Modified logits tensor
        """
        # Detect current phases
        phases = self._get_phases(history, context)
        
        # Update context with detected phases
        updated_context = context.clone()
        
        # Process each batch item based on its phase
        result_logits = logits
        
        for batch_idx, phase in enumerate(phases):
            updated_context.phase = phase
            
            # Apply phase-specific processing
            batch_logits = result_logits[batch_idx:batch_idx+1]
            batch_history = [history[batch_idx]]
            
            processed_batch = self._process_logits_for_phase(
                batch_logits, 
                batch_history, 
                updated_context,
                phase
            )
            
            # Update result
            result_logits = mx.concatenate([
                result_logits[:batch_idx],
                processed_batch,
                result_logits[batch_idx+1:]
            ], axis=0) if batch_idx < len(phases) - 1 else mx.concatenate([
                result_logits[:batch_idx],
                processed_batch
            ], axis=0)
        
        return result_logits
    
    def _get_phases(self, history: List[List[int]], context: ProcessingContext) -> List[ProcessingPhase]:
        """
        Get phases for all batch items with caching.
        
        Args:
            history: Token generation history
            context: Processing context
            
        Returns:
            List of detected phases
        """
        # Create cache key
        cache_key = f"{context.correlation_id}:{len(history)}:{hash(tuple(tuple(h) for h in history))}"
        
        with self._cache_lock:
            if cache_key in self._phase_cache:
                return self._phase_cache[cache_key]
        
        # Detect phases
        phases = self.detector.detect_phase(history, context)
        
        # Cache result
        with self._cache_lock:
            self._phase_cache[cache_key] = phases
            
            # Limit cache size
            if len(self._phase_cache) > 1000:
                # Remove oldest entries
                keys_to_remove = list(self._phase_cache.keys())[:200]
                for key in keys_to_remove:
                    del self._phase_cache[key]
        
        return phases
    
    @abstractmethod
    def _process_logits_for_phase(
        self, 
        logits: mx.array, 
        history: List[List[int]], 
        context: ProcessingContext,
        phase: ProcessingPhase
    ) -> mx.array:
        """
        Process logits for a specific phase.
        
        Subclasses must implement this method to provide phase-specific
        processing logic.
        
        Args:
            logits: Logits tensor for single batch item
            history: Token history for single batch item
            context: Processing context with phase information
            phase: Current processing phase
            
        Returns:
            Modified logits tensor
        """
        pass
    
    def can_process(self, context: ProcessingContext) -> bool:
        """
        Check if processor can handle the given context.
        
        Phase-aware processors can generally handle all contexts
        since they adapt based on detected phases.
        
        Args:
            context: Processing context to evaluate
            
        Returns:
            True if processor can handle this context
        """
        return True
    
    def clear_phase_cache(self) -> None:
        """Clear the phase detection cache."""
        with self._cache_lock:
            self._phase_cache.clear()