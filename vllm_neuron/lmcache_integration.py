# SPDX-License-Identifier: Apache-2.0
"""LMCache integration for vLLM-Neuron plugin."""

import logging
import os
import yaml
import threading
import time
import torch
import json
import psutil
from typing import TYPE_CHECKING, Optional, Dict, Any, List, Tuple, Union
from pathlib import Path
from enum import Enum
from dataclasses import dataclass, field
from collections import defaultdict, deque

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.sequence import SequenceGroup

# Import LMCache components
try:
    from lmcache.config import LMCacheEngineMetadata
    from lmcache.utils import CacheEngineKey
    from lmcache.v1.config import LMCacheEngineConfig
    from lmcache.v1.cache_engine import LMCacheEngine
    LMCACHE_AVAILABLE = True
except ImportError:
    LMCACHE_AVAILABLE = False
    LMCacheEngine = None
    CacheEngineKey = Any
    LMCacheEngineConfig = Any
    LMCacheEngineMetadata = Any

logger = logging.getLogger(__name__)


class CacheStrategy(Enum):
    """Cache strategy based on vLLM-Neuron capabilities."""
    FULL_SEQUENCE_ONLY = "full_sequence_only"  # Chunked prefill disabled
    CHUNKED_PREFILL = "chunked_prefill"        # Chunked prefill enabled


class ErrorSeverity(Enum):
    """Error severity levels for error handling."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ComponentType(Enum):
    """Component types for error tracking."""
    LMCACHE_ENGINE = "lmcache_engine"
    STORAGE_BACKEND = "storage_backend"
    CACHE_HANDLER = "cache_handler"
    CONFIG_MANAGER = "config_manager"
    DEVICE_DETECTOR = "device_detector"
    INTEGRATION_LAYER = "integration_layer"


@dataclass
class ErrorEvent:
    """Represents an error event in the system."""
    component: ComponentType
    severity: ErrorSeverity
    error_type: str
    message: str
    timestamp: float
    context: Dict[str, Any]
    retry_count: int = 0
    resolved: bool = False


class ComponentFailureDetector:
    """Detects and tracks component failures across the system."""
    
    def __init__(self):
        self.error_history: List[ErrorEvent] = []
        self.component_health: Dict[ComponentType, bool] = {
            component: True for component in ComponentType
        }
        self.failure_thresholds = {
            ComponentType.LMCACHE_ENGINE: 5,
            ComponentType.STORAGE_BACKEND: 3,
            ComponentType.CACHE_HANDLER: 10,
            ComponentType.CONFIG_MANAGER: 2,
            ComponentType.DEVICE_DETECTOR: 2,
            ComponentType.INTEGRATION_LAYER: 5,
        }
        self.failure_counts: Dict[ComponentType, int] = {
            component: 0 for component in ComponentType
        }
        self.lock = threading.Lock()
        
        # Time window for failure counting (5 minutes)
        self.failure_window_seconds = 300
        
        logger.info("ComponentFailureDetector initialized")
    
    def record_error(
        self, 
        component: ComponentType, 
        severity: ErrorSeverity, 
        error_type: str, 
        message: str, 
        context: Optional[Dict[str, Any]] = None
    ) -> ErrorEvent:
        """Record an error event and update component health."""
        error_event = ErrorEvent(
            component=component,
            severity=severity,
            error_type=error_type,
            message=message,
            timestamp=time.time(),
            context=context or {}
        )
        
        with self.lock:
            self.error_history.append(error_event)
            
            # Update failure count for this component
            self.failure_counts[component] += 1
            
            # Check if component should be marked as unhealthy
            if self._should_mark_unhealthy(component):
                self.component_health[component] = False
                logger.error(f"Component {component.value} marked as unhealthy after {self.failure_counts[component]} failures")
            
            # Clean old errors outside the time window
            self._cleanup_old_errors()
        
        logger.warning(f"Error recorded: {component.value} - {severity.value} - {error_type}: {message}")
        return error_event
    
    def _should_mark_unhealthy(self, component: ComponentType) -> bool:
        """Check if a component should be marked as unhealthy."""
        threshold = self.failure_thresholds.get(component, 5)
        recent_failures = self._count_recent_failures(component)
        return recent_failures >= threshold
    
    def _count_recent_failures(self, component: ComponentType) -> int:
        """Count failures for a component within the time window."""
        current_time = time.time()
        cutoff_time = current_time - self.failure_window_seconds
        
        return sum(
            1 for error in self.error_history
            if (error.component == component and 
                error.timestamp >= cutoff_time and 
                not error.resolved)
        )
    
    def _cleanup_old_errors(self):
        """Remove errors outside the time window."""
        current_time = time.time()
        cutoff_time = current_time - self.failure_window_seconds
        
        self.error_history = [
            error for error in self.error_history
            if error.timestamp >= cutoff_time
        ]
    
    def is_component_healthy(self, component: ComponentType) -> bool:
        """Check if a component is healthy."""
        with self.lock:
            return self.component_health.get(component, False)
    
    def mark_component_recovered(self, component: ComponentType):
        """Mark a component as recovered."""
        with self.lock:
            self.component_health[component] = True
            self.failure_counts[component] = 0
            
            # Mark related errors as resolved
            for error in self.error_history:
                if error.component == component and not error.resolved:
                    error.resolved = True
        
        logger.info(f"Component {component.value} marked as recovered")
    
    def get_component_status(self) -> Dict[str, Any]:
        """Get status of all components."""
        with self.lock:
            return {
                "component_health": {comp.value: healthy for comp, healthy in self.component_health.items()},
                "failure_counts": {comp.value: count for comp, count in self.failure_counts.items()},
                "recent_errors": len([e for e in self.error_history if not e.resolved]),
                "total_errors": len(self.error_history)
            }
    
    def get_recent_errors(self, limit: int = 10) -> List[ErrorEvent]:
        """Get recent error events."""
        with self.lock:
            return sorted(self.error_history, key=lambda x: x.timestamp, reverse=True)[:limit]


class RetryManager:
    """Manages retry logic for transient failures."""
    
    def __init__(self):
        self.max_retries = {
            "storage_operation": 3,
            "cache_lookup": 2,
            "engine_initialization": 2,
            "config_loading": 1,
            "device_detection": 1,
        }
        self.retry_delays = {
            "storage_operation": [0.1, 0.5, 1.0],  # Exponential backoff
            "cache_lookup": [0.05, 0.2],
            "engine_initialization": [1.0, 3.0],
            "config_loading": [0.5],
            "device_detection": [0.1],
        }
        self.retry_counts: Dict[str, int] = {}
        self.lock = threading.Lock()
        
        logger.info("RetryManager initialized")
    
    def should_retry(self, operation_type: str, error: Exception) -> bool:
        """Determine if an operation should be retried."""
        # Don't retry certain types of errors
        non_retryable_errors = (
            ValueError,  # Configuration errors
            TypeError,   # Programming errors
            ImportError, # Missing dependencies
            AttributeError,  # API misuse
        )
        
        if isinstance(error, non_retryable_errors):
            return False
        
        with self.lock:
            current_retries = self.retry_counts.get(operation_type, 0)
            max_retries = self.max_retries.get(operation_type, 0)
            
            return current_retries < max_retries
    
    def get_retry_delay(self, operation_type: str) -> float:
        """Get the delay before the next retry."""
        with self.lock:
            current_retries = self.retry_counts.get(operation_type, 0)
            delays = self.retry_delays.get(operation_type, [0.1])
            
            if current_retries < len(delays):
                return delays[current_retries]
            else:
                # Use last delay for additional retries
                return delays[-1]
    
    def record_retry(self, operation_type: str):
        """Record a retry attempt."""
        with self.lock:
            self.retry_counts[operation_type] = self.retry_counts.get(operation_type, 0) + 1
    
    def reset_retries(self, operation_type: str):
        """Reset retry count for an operation type."""
        with self.lock:
            self.retry_counts[operation_type] = 0
    
    def execute_with_retry(self, operation_type: str, operation_func, *args, **kwargs):
        """Execute an operation with retry logic."""
        last_error = None
        
        while True:
            try:
                result = operation_func(*args, **kwargs)
                self.reset_retries(operation_type)
                return result
                
            except Exception as e:
                last_error = e
                
                if not self.should_retry(operation_type, e):
                    logger.error(f"Operation {operation_type} failed permanently: {e}")
                    raise e
                
                self.record_retry(operation_type)
                delay = self.get_retry_delay(operation_type)
                
                logger.warning(f"Operation {operation_type} failed, retrying in {delay}s: {e}")
                time.sleep(delay)
        
        # This should never be reached, but just in case
        raise last_error


class StorageBackendFailureHandler:
    """Handles storage backend failures and recovery."""
    
    def __init__(self, failure_detector: ComponentFailureDetector, retry_manager: RetryManager):
        self.failure_detector = failure_detector
        self.retry_manager = retry_manager
        self.fallback_storage = {}  # In-memory fallback storage
        self.storage_backend = None
        self.fallback_mode = False
        self.lock = threading.Lock()
        
        logger.info("StorageBackendFailureHandler initialized")
    
    def set_storage_backend(self, storage_backend):
        """Set the primary storage backend."""
        self.storage_backend = storage_backend
    
    def handle_storage_operation(self, operation_type: str, operation_func, *args, **kwargs):
        """Handle storage operations with failure recovery."""
        if self.fallback_mode:
            return self._handle_fallback_operation(operation_type, *args, **kwargs)
        
        try:
            return self.retry_manager.execute_with_retry(
                f"storage_{operation_type}", 
                operation_func, 
                *args, 
                **kwargs
            )
        except Exception as e:
            self.failure_detector.record_error(
                ComponentType.STORAGE_BACKEND,
                ErrorSeverity.HIGH,
                f"storage_{operation_type}_failure",
                str(e),
                {"operation_type": operation_type, "args": str(args)[:200]}
            )
            
            # Switch to fallback mode if storage backend is unhealthy
            if not self.failure_detector.is_component_healthy(ComponentType.STORAGE_BACKEND):
                logger.warning("Switching to fallback storage mode due to backend failures")
                self.fallback_mode = True
                return self._handle_fallback_operation(operation_type, *args, **kwargs)
            
            raise e
    
    def _handle_fallback_operation(self, operation_type: str, *args, **kwargs):
        """Handle operations in fallback mode using in-memory storage."""
        with self.lock:
            if operation_type == "get":
                key = args[0] if args else kwargs.get("key")
                return self.fallback_storage.get(str(key))
            
            elif operation_type == "put":
                key = args[0] if args else kwargs.get("key")
                value = args[1] if len(args) > 1 else kwargs.get("value")
                self.fallback_storage[str(key)] = value
                return True
            
            elif operation_type == "contains":
                key = args[0] if args else kwargs.get("key")
                return str(key) in self.fallback_storage
            
            elif operation_type == "remove":
                key = args[0] if args else kwargs.get("key")
                return self.fallback_storage.pop(str(key), None) is not None
            
            else:
                logger.warning(f"Unsupported fallback operation: {operation_type}")
                return None
    
    def attempt_recovery(self) -> bool:
        """Attempt to recover the storage backend."""
        if not self.storage_backend:
            return False
        
        try:
            # Test basic storage operations
            test_key = "health_check_key"
            test_value = {"test": "data", "timestamp": time.time()}
            
            # Try a simple put/get cycle
            if hasattr(self.storage_backend, 'put'):
                self.storage_backend.put(test_key, test_value)
                retrieved = self.storage_backend.get(test_key)
                
                if retrieved == test_value:
                    # Recovery successful
                    self.failure_detector.mark_component_recovered(ComponentType.STORAGE_BACKEND)
                    self.fallback_mode = False
                    logger.info("Storage backend recovery successful")
                    return True
            
        except Exception as e:
            logger.warning(f"Storage backend recovery failed: {e}")
        
        return False
    
    def get_fallback_stats(self) -> Dict[str, Any]:
        """Get statistics about fallback storage usage."""
        with self.lock:
            return {
                "fallback_mode": self.fallback_mode,
                "fallback_storage_size": len(self.fallback_storage),
                "fallback_storage_keys": list(self.fallback_storage.keys())[:10]  # First 10 keys
            }


logger = logging.getLogger(__name__)


class KVCacheOperation:
    """Represents a KV cache operation for interception."""
    
    def __init__(self, operation_type: str, sequence_group: "SequenceGroup", **kwargs):
        self.operation_type = operation_type  # "store", "retrieve", "evict"
        self.sequence_group = sequence_group
        self.kwargs = kwargs
        self.timestamp = time.time()
    
    def get_cache_key(self) -> Optional[CacheEngineKey]:
        """Generate cache key for this operation."""
        if not LMCACHE_AVAILABLE:
            return None
        
        try:
            # Extract prompt tokens from sequence group
            if hasattr(self.sequence_group, 'get_seqs'):
                seqs = self.sequence_group.get_seqs()
                if seqs:
                    prompt_tokens = seqs[0].get_token_ids()
                    # Use a global key manager for consistency
                    if not hasattr(self, '_key_manager'):
                        self._key_manager = TokenCacheKeyManager()
                    return self._key_manager.create_cache_key(prompt_tokens)
        except Exception as e:
            logger.warning(f"Failed to generate cache key: {e}")
        
        return None


class KVCacheResult:
    """Result of a KV cache operation."""
    
    def __init__(self, success: bool, cache_hit: bool = False, data: Any = None, error: str = None):
        self.success = success
        self.cache_hit = cache_hit
        self.data = data
        self.error = error
        self.timestamp = time.time()


class CacheHitMissHandler:
    """Handles cache hit/miss logic and partial cache hits."""
    
    def __init__(self, lmcache_engine: Optional["LMCacheEngine"] = None):
        self.lmcache_engine = lmcache_engine
        self.key_manager = TokenCacheKeyManager()
        self.stats = {
            "total_requests": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "partial_hits": 0,
            "storage_operations": 0,
            "errors": 0,
            "lookup_latency_ms": [],
            "storage_latency_ms": []
        }
        self.stats_lock = threading.Lock()
        
        # Cache lookup optimization settings
        self.enable_prefix_matching = True
        self.min_prefix_length = 32  # Minimum tokens for prefix matching
        self.max_lookup_attempts = 3  # Maximum lookup attempts for partial hits
        
        # Performance tracking
        self.recent_lookups = []  # Track recent lookup performance
        self.max_recent_lookups = 1000
    
    def handle_cache_lookup(self, cache_key: CacheEngineKey) -> Tuple[bool, Any, int]:
        """
        Handle cache lookup for a given key with advanced logic.
        
        Args:
            cache_key: Key to look up in cache
            
        Returns:
            Tuple of (is_hit, cached_data, hit_length)
        """
        start_time = time.time()
        
        with self.stats_lock:
            self.stats["total_requests"] += 1
        
        if not self.lmcache_engine or not cache_key:
            with self.stats_lock:
                self.stats["cache_misses"] += 1
            return False, None, 0
        
        try:
            # Try exact match first
            is_hit, cached_data, hit_length = self._try_exact_lookup(cache_key)
            
            if is_hit:
                self._record_lookup_performance(start_time, "exact_hit")
                return is_hit, cached_data, hit_length
            
            # Try prefix matching if enabled
            tokens = self.key_manager.get_tokens_from_key(cache_key)
            if self.enable_prefix_matching and tokens and len(tokens) >= self.min_prefix_length:
                is_hit, cached_data, hit_length = self._try_prefix_lookup(cache_key)
                
                if is_hit:
                    self._record_lookup_performance(start_time, "prefix_hit")
                    return is_hit, cached_data, hit_length
            
            # Complete cache miss
            with self.stats_lock:
                self.stats["cache_misses"] += 1
            
            self._record_lookup_performance(start_time, "miss")
            return False, None, 0
            
        except Exception as e:
            logger.warning(f"Cache lookup failed: {e}")
            with self.stats_lock:
                self.stats["errors"] += 1
            
            self._record_lookup_performance(start_time, "error")
            return False, None, 0
    
    def _try_exact_lookup(self, cache_key: CacheEngineKey) -> Tuple[bool, Any, int]:
        """Try exact cache key lookup."""
        try:
            cached_data = self.lmcache_engine.get(cache_key)
            
            if cached_data is not None:
                # Get tokens from key manager
                tokens = self.key_manager.get_tokens_from_key(cache_key)
                hit_length = len(tokens) if tokens else 0
                
                with self.stats_lock:
                    self.stats["cache_hits"] += 1
                logger.debug(f"Exact cache hit for {hit_length} tokens")
                return True, cached_data, hit_length
            
            return False, None, 0
            
        except Exception as e:
            logger.debug(f"Exact lookup failed: {e}")
            return False, None, 0
    
    def _try_prefix_lookup(self, cache_key: CacheEngineKey) -> Tuple[bool, Any, int]:
        """Try prefix-based cache lookup for partial hits."""
        tokens = self.key_manager.get_tokens_from_key(cache_key)
        if not tokens:
            return False, None, 0
        
        max_prefix_length = len(tokens)
        
        # Try progressively shorter prefixes
        for prefix_length in range(max_prefix_length, self.min_prefix_length - 1, -32):
            try:
                prefix_key = self.key_manager.create_prefix_key(cache_key, prefix_length)
                if not prefix_key:
                    continue
                
                cached_data = self.lmcache_engine.get(prefix_key)
                
                if cached_data is not None:
                    # Validate that this is actually a useful partial hit
                    if self._validate_partial_hit(cached_data, prefix_length):
                        with self.stats_lock:
                            self.stats["partial_hits"] += 1
                        
                        logger.debug(f"Partial cache hit: {prefix_length}/{len(tokens)} tokens")
                        return True, cached_data, prefix_length
                
            except Exception as e:
                logger.debug(f"Prefix lookup failed for length {prefix_length}: {e}")
                continue
        
        return False, None, 0
    
    def _validate_partial_hit(self, cached_data: Any, expected_length: int) -> bool:
        """Validate that partial hit data is usable."""
        try:
            # Basic validation - check if cached data has expected dimensions
            if hasattr(cached_data, 'shape') and len(cached_data.shape) > 0:
                # Assume sequence dimension is the second-to-last dimension
                seq_dim = cached_data.shape[-2] if len(cached_data.shape) > 1 else cached_data.shape[0]
                return seq_dim >= expected_length * 0.8  # Allow some tolerance
            
            # If we can't validate, assume it's valid
            return True
            
        except Exception:
            return False
    
    def handle_cache_storage(self, cache_key: CacheEngineKey, kv_data: Any) -> bool:
        """
        Handle storing KV cache data after request completion with optimization.
        
        Args:
            cache_key: Key to store data under
            kv_data: KV cache data to store
            
        Returns:
            True if storage succeeded, False otherwise
        """
        if not self.lmcache_engine or not cache_key or kv_data is None:
            return False
        
        start_time = time.time()
        
        try:
            with self.stats_lock:
                self.stats["storage_operations"] += 1
            
            # Validate data before storage
            if not self._validate_storage_data(kv_data):
                logger.warning("Invalid KV data for storage, skipping")
                return False
            
            # Check if we should store (avoid duplicate storage)
            if self._should_store_data(cache_key, kv_data):
                # Store in LMCache
                self.lmcache_engine.put(cache_key, kv_data)
                
                # Record storage performance
                storage_time_ms = (time.time() - start_time) * 1000
                with self.stats_lock:
                    self.stats["storage_latency_ms"].append(storage_time_ms)
                    # Keep only recent measurements
                    if len(self.stats["storage_latency_ms"]) > self.max_recent_lookups:
                        self.stats["storage_latency_ms"] = self.stats["storage_latency_ms"][-self.max_recent_lookups:]
                
                # Get tokens for logging
                tokens = self.key_manager.get_tokens_from_key(cache_key)
                token_count = len(tokens) if tokens else 0
                logger.debug(f"Stored KV cache for key with {token_count} tokens")
                return True
            else:
                logger.debug("Skipped storage (data already exists or invalid)")
                return True  # Return True since we didn't need to store
            
        except Exception as e:
            logger.warning(f"Cache storage failed: {e}")
            with self.stats_lock:
                self.stats["errors"] += 1
            return False
    
    def _validate_storage_data(self, kv_data: Any) -> bool:
        """Validate KV data before storage."""
        try:
            # Basic validation checks
            if kv_data is None:
                return False
            
            # Check if it's a tensor with reasonable shape
            if hasattr(kv_data, 'shape'):
                # Should have at least 2 dimensions (batch, sequence, ...)
                if len(kv_data.shape) < 2:
                    return False
                
                # Check for reasonable sequence length
                seq_len = kv_data.shape[-2] if len(kv_data.shape) > 1 else kv_data.shape[0]
                if seq_len <= 0 or seq_len > 100000:  # Reasonable bounds
                    return False
            
            return True
            
        except Exception:
            return False
    
    def _should_store_data(self, cache_key: CacheEngineKey, kv_data: Any) -> bool:
        """Determine if data should be stored (avoid duplicates)."""
        try:
            # Quick check if exact key already exists
            existing_data = self.lmcache_engine.get(cache_key)
            if existing_data is not None:
                # Data already exists, no need to store again
                return False
            
            # Check storage policy (e.g., minimum sequence length)
            tokens = self.key_manager.get_tokens_from_key(cache_key)
            if tokens and len(tokens) < self.min_prefix_length:
                return False
            
            return True
            
        except Exception:
            # If we can't check, err on the side of storing
            return True
    
    def _record_lookup_performance(self, start_time: float, lookup_type: str):
        """Record lookup performance metrics."""
        lookup_time_ms = (time.time() - start_time) * 1000
        
        with self.stats_lock:
            self.stats["lookup_latency_ms"].append(lookup_time_ms)
            # Keep only recent measurements
            if len(self.stats["lookup_latency_ms"]) > self.max_recent_lookups:
                self.stats["lookup_latency_ms"] = self.stats["lookup_latency_ms"][-self.max_recent_lookups:]
        
        # Track recent lookups for adaptive optimization
        self.recent_lookups.append({
            "timestamp": time.time(),
            "latency_ms": lookup_time_ms,
            "type": lookup_type
        })
        
        # Keep only recent lookups
        if len(self.recent_lookups) > self.max_recent_lookups:
            self.recent_lookups = self.recent_lookups[-self.max_recent_lookups:]
    
    def optimize_lookup_settings(self):
        """Adaptively optimize lookup settings based on performance."""
        if len(self.recent_lookups) < 100:
            return  # Need more data
        
        try:
            # Analyze recent performance
            prefix_hits = [l for l in self.recent_lookups if l["type"] == "prefix_hit"]
            exact_hits = [l for l in self.recent_lookups if l["type"] == "exact_hit"]
            misses = [l for l in self.recent_lookups if l["type"] == "miss"]
            
            total_lookups = len(self.recent_lookups)
            prefix_hit_rate = len(prefix_hits) / total_lookups
            
            # Adjust prefix matching based on effectiveness
            if prefix_hit_rate < 0.05:  # Less than 5% prefix hits
                # Prefix matching not very effective, increase minimum length
                self.min_prefix_length = min(64, self.min_prefix_length + 8)
                logger.debug(f"Increased min_prefix_length to {self.min_prefix_length}")
            elif prefix_hit_rate > 0.20:  # More than 20% prefix hits
                # Prefix matching very effective, decrease minimum length
                self.min_prefix_length = max(16, self.min_prefix_length - 8)
                logger.debug(f"Decreased min_prefix_length to {self.min_prefix_length}")
            
        except Exception as e:
            logger.debug(f"Lookup optimization failed: {e}")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get detailed performance metrics."""
        with self.stats_lock:
            lookup_latencies = self.stats["lookup_latency_ms"].copy()
            storage_latencies = self.stats["storage_latency_ms"].copy()
        
        metrics = {}
        
        # Lookup performance
        if lookup_latencies:
            metrics["lookup_latency"] = {
                "avg_ms": sum(lookup_latencies) / len(lookup_latencies),
                "min_ms": min(lookup_latencies),
                "max_ms": max(lookup_latencies),
                "p95_ms": sorted(lookup_latencies)[int(len(lookup_latencies) * 0.95)] if len(lookup_latencies) > 20 else max(lookup_latencies),
                "count": len(lookup_latencies)
            }
        
        # Storage performance
        if storage_latencies:
            metrics["storage_latency"] = {
                "avg_ms": sum(storage_latencies) / len(storage_latencies),
                "min_ms": min(storage_latencies),
                "max_ms": max(storage_latencies),
                "p95_ms": sorted(storage_latencies)[int(len(storage_latencies) * 0.95)] if len(storage_latencies) > 20 else max(storage_latencies),
                "count": len(storage_latencies)
            }
        
        # Recent lookup analysis
        if self.recent_lookups:
            recent_by_type = {}
            for lookup in self.recent_lookups[-100:]:  # Last 100 lookups
                lookup_type = lookup["type"]
                if lookup_type not in recent_by_type:
                    recent_by_type[lookup_type] = []
                recent_by_type[lookup_type].append(lookup["latency_ms"])
            
            metrics["recent_lookups_by_type"] = {}
            for lookup_type, latencies in recent_by_type.items():
                metrics["recent_lookups_by_type"][lookup_type] = {
                    "count": len(latencies),
                    "avg_latency_ms": sum(latencies) / len(latencies)
                }
        
        # Configuration
        metrics["config"] = {
            "enable_prefix_matching": self.enable_prefix_matching,
            "min_prefix_length": self.min_prefix_length,
            "max_lookup_attempts": self.max_lookup_attempts
        }
        
        return metrics
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache hit/miss statistics."""
        with self.stats_lock:
            stats = self.stats.copy()
            
            # Calculate derived metrics
            total_requests = stats["total_requests"]
            if total_requests > 0:
                stats["hit_rate"] = stats["cache_hits"] / total_requests
                stats["miss_rate"] = stats["cache_misses"] / total_requests
                stats["partial_hit_rate"] = stats["partial_hits"] / total_requests
                stats["error_rate"] = stats["errors"] / total_requests
                stats["effective_hit_rate"] = (stats["cache_hits"] + stats["partial_hits"]) / total_requests
            else:
                stats["hit_rate"] = 0.0
                stats["miss_rate"] = 0.0
                stats["partial_hit_rate"] = 0.0
                stats["error_rate"] = 0.0
                stats["effective_hit_rate"] = 0.0
            
            # Add performance metrics
            stats["performance"] = self.get_performance_metrics()
            
            return stats


class TokenCacheKeyManager:
    """Manages the relationship between tokens and cache keys."""
    
    def __init__(self):
        self.token_to_key_map: Dict[str, CacheEngineKey] = {}
        self.key_to_tokens_map: Dict[str, List[int]] = {}
    
    def create_cache_key(self, tokens: List[int], model_name: str = "default_model") -> CacheEngineKey:
        """Create a cache key from tokens."""
        if not LMCACHE_AVAILABLE:
            raise RuntimeError("LMCache not available")
        
        # Create hash from tokens
        token_str = str(tokens)
        chunk_hash = hash(token_str) % (2**31)
        
        cache_key = CacheEngineKey(
            fmt="kv_cache",
            model_name=model_name,
            world_size=1,
            worker_id=0,
            chunk_hash=chunk_hash,
            dtype=torch.float16
        )
        
        # Store the mapping
        key_str = str(cache_key)
        self.token_to_key_map[token_str] = cache_key
        self.key_to_tokens_map[key_str] = tokens
        
        return cache_key
    
    def get_tokens_from_key(self, cache_key: CacheEngineKey) -> Optional[List[int]]:
        """Get tokens from a cache key."""
        key_str = str(cache_key)
        return self.key_to_tokens_map.get(key_str)
    
    def create_prefix_key(self, cache_key: CacheEngineKey, prefix_length: int) -> Optional[CacheEngineKey]:
        """Create a cache key for a prefix of the original tokens."""
        tokens = self.get_tokens_from_key(cache_key)
        if not tokens or prefix_length >= len(tokens):
            return None
        
        prefix_tokens = tokens[:prefix_length]
        return self.create_cache_key(prefix_tokens, cache_key.model_name)


class NeuronDeviceDetector:
    """Utility class for detecting and validating Neuron device environment."""
    
    @staticmethod
    def is_neuron_available() -> bool:
        """Check if Neuron devices are available in the system."""
        try:
            from vllm_neuron.platform import NeuronPlatform
            platform = NeuronPlatform()
            devices = platform.get_device_capability()
            return devices is not None and devices.get('device_count', 0) > 0
        except Exception as e:
            logger.warning(f"Failed to detect Neuron devices via platform: {e}")
            # Fallback to original method
            try:
                from vllm_neuron import _is_neuron_dev
                return _is_neuron_dev()
            except Exception:
                return False
    
    @staticmethod
    def get_neuron_device_count() -> int:
        """Get the number of available Neuron devices."""
        try:
            from vllm_neuron.platform import NeuronPlatform
            platform = NeuronPlatform()
            devices = platform.get_device_capability()
            if devices:
                return devices.get('device_count', 0)
        except Exception as e:
            logger.warning(f"Failed to get device count via platform: {e}")
        
        # Fallback to counting device files
        import glob
        neuron_devices = glob.glob('/dev/neuron*')
        return len(neuron_devices)
    
    @staticmethod
    def get_neuron_device_info() -> Optional[Dict[str, Any]]:
        """Get detailed Neuron device information."""
        try:
            from vllm_neuron.platform import NeuronPlatform
            platform = NeuronPlatform()
            return platform.get_device_capability()
        except Exception as e:
            logger.warning(f"Failed to get device info: {e}")
            return None
    
    @staticmethod
    def get_neuron_runtime_info() -> Dict[str, Any]:
        """Get Neuron runtime information."""
        device_info = NeuronDeviceDetector.get_neuron_device_info()
        
        info = {
            "devices_available": device_info is not None,
            "device_count": device_info.get('device_count', 0) if device_info else 0,
            "total_cores": device_info.get('total_cores', 0) if device_info else 0,
            "visible_cores": os.environ.get("NEURON_RT_VISIBLE_CORES"),
            "neuron_cc_flags": os.environ.get("NEURON_CC_FLAGS"),
        }
        
        # Add device details if available
        if device_info and 'devices' in device_info:
            info['devices'] = device_info['devices']
        
        # Try to get more detailed runtime info if available
        try:
            import neuronx_distributed_inference
            info["neuronx_distributed_available"] = True
        except ImportError:
            info["neuronx_distributed_available"] = False
            
        return info


class LMCacheConfigManager:
    """Manages LMCache configuration for Neuron environments."""
    
    DEFAULT_CONFIG_FILENAME = "lmcache_neuron_config.yaml"
    
    def __init__(self):
        self._config_cache: Optional[Dict[str, Any]] = None
    
    def load_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """Load LMCache configuration from file or environment."""
        if self._config_cache is not None:
            return self._config_cache
            
        # Determine config path
        if config_path is None:
            config_path = self._find_config_file()
        
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f) or {}
                logger.info(f"Loaded LMCache config from: {config_path}")
            except Exception as e:
                logger.warning(f"Failed to load config from {config_path}: {e}")
                config = self._get_default_config()
        else:
            logger.info("No config file found, using default configuration")
            config = self._get_default_config()
        
        # Apply environment variable overrides
        config = self._apply_env_overrides(config)
        
        # Validate configuration
        self._validate_config(config)
        
        self._config_cache = config
        return config
    
    def _find_config_file(self) -> Optional[str]:
        """Find LMCache configuration file in standard locations."""
        # Check environment variable first
        env_path = os.environ.get("LMCACHE_CONFIG_PATH")
        if env_path:
            return env_path
        
        # Check current working directory
        cwd_path = os.path.join(os.getcwd(), self.DEFAULT_CONFIG_FILENAME)
        if os.path.exists(cwd_path):
            return cwd_path
        
        # Check relative to this module
        module_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        module_path = os.path.join(module_dir, self.DEFAULT_CONFIG_FILENAME)
        if os.path.exists(module_path):
            return module_path
        
        return None
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default LMCache configuration optimized for Neuron."""
        return {
            "chunk_size": 256,
            "local_cpu": True,
            "max_local_cpu_size": 4,  # 4GB
            "save_unfull_chunk": True,
            "save_decode_cache": True,
            "remote_url": "fs://localhost:0/tmp/lmcache_neuron",
            "storage_backend": "local_cpu",
            "extra_config": {
                "save_chunk_meta": True,
                "use_gpu_connector_v3": False,
                "enable_async_loading": True,
                "use_layerwise": False,
                "enable_blending": False,
                "priority_limit": None,
            }
        }
    
    def _apply_env_overrides(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply environment variable overrides to configuration."""
        env_overrides = {
            "LMCACHE_CHUNK_SIZE": ("chunk_size", int),
            "LMCACHE_MAX_CPU_SIZE": ("max_local_cpu_size", int),
            "LMCACHE_REMOTE_URL": ("remote_url", str),
            "LMCACHE_ENABLE_ASYNC": ("extra_config.enable_async_loading", bool),
        }
        
        for env_var, (config_key, value_type) in env_overrides.items():
            env_value = os.environ.get(env_var)
            if env_value is not None:
                try:
                    if value_type == bool:
                        parsed_value = env_value.lower() in ('true', '1', 'yes', 'on')
                    else:
                        parsed_value = value_type(env_value)
                    
                    # Handle nested config keys
                    if "." in config_key:
                        keys = config_key.split(".")
                        target = config
                        for key in keys[:-1]:
                            if key not in target:
                                target[key] = {}
                            target = target[key]
                        target[keys[-1]] = parsed_value
                    else:
                        config[config_key] = parsed_value
                    
                    logger.info(f"Applied environment override: {env_var}={parsed_value}")
                except (ValueError, TypeError) as e:
                    logger.warning(f"Invalid environment variable {env_var}={env_value}: {e}")
        
        return config
    
    def _validate_config(self, config: Dict[str, Any]) -> None:
        """Validate LMCache configuration for Neuron environment."""
        required_fields = ["chunk_size", "local_cpu", "max_local_cpu_size"]
        
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Missing required configuration field: {field}")
        
        # Validate chunk size
        chunk_size = config.get("chunk_size", 0)
        if not isinstance(chunk_size, int) or chunk_size <= 0:
            raise ValueError(f"Invalid chunk_size: {chunk_size}. Must be a positive integer.")
        
        # Validate CPU memory size
        max_cpu_size = config.get("max_local_cpu_size", 0)
        if not isinstance(max_cpu_size, (int, float)) or max_cpu_size <= 0:
            raise ValueError(f"Invalid max_local_cpu_size: {max_cpu_size}. Must be a positive number.")
        
        # Ensure CPU storage is enabled for Neuron
        if not config.get("local_cpu", False):
            logger.warning("local_cpu is disabled, but Neuron requires CPU storage. Enabling local_cpu.")
            config["local_cpu"] = True
        
        # Validate extra config if present
        extra_config = config.get("extra_config", {})
        if extra_config.get("use_gpu_connector_v3", False):
            logger.warning("use_gpu_connector_v3 is enabled, but Neuron uses CPU storage. Disabling GPU connector.")
            extra_config["use_gpu_connector_v3"] = False
        
        logger.info("LMCache configuration validation passed")


@dataclass
class PerformanceMetrics:
    """Performance metrics data structure."""
    cache_hit_rate: float = 0.0
    cache_miss_rate: float = 0.0
    partial_hit_rate: float = 0.0
    effective_hit_rate: float = 0.0
    average_lookup_latency_ms: float = 0.0
    average_storage_latency_ms: float = 0.0
    p95_lookup_latency_ms: float = 0.0
    p95_storage_latency_ms: float = 0.0
    memory_utilization_percent: float = 0.0
    cpu_utilization_percent: float = 0.0
    storage_utilization_gb: float = 0.0
    total_requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    partial_hits: int = 0
    storage_operations: int = 0
    errors: int = 0
    timestamp: float = field(default_factory=time.time)


@dataclass
class PrometheusMetric:
    """Prometheus metric definition."""
    name: str
    metric_type: str  # "counter", "gauge", "histogram"
    help_text: str
    value: Union[int, float]
    labels: Dict[str, str] = field(default_factory=dict)


class NeuronPerformanceMonitor:
    """
    Performance monitoring system for LMCache-Neuron integration.
    
    Tracks cache hit rates, memory utilization, and exports Prometheus metrics.
    Validates: Requirements 7.1, 7.2, 7.5
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Monitoring configuration
        self.metrics_collection_interval = self.config.get("metrics_interval", 30.0)  # seconds
        self.metrics_retention_period = self.config.get("retention_period", 3600.0)  # 1 hour
        self.enable_prometheus_export = self.config.get("enable_prometheus", True)
        self.prometheus_port = self.config.get("prometheus_port", 8080)
        
        # Metrics storage
        self.metrics_history: deque = deque(maxlen=int(self.metrics_retention_period / self.metrics_collection_interval))
        self.current_metrics = PerformanceMetrics()
        
        # Cache operation tracking
        self.cache_operations: Dict[str, List[float]] = defaultdict(list)
        self.memory_samples: List[Tuple[float, float]] = []  # (timestamp, memory_usage_gb)
        self.cpu_samples: List[Tuple[float, float]] = []  # (timestamp, cpu_percent)
        
        # Prometheus metrics registry
        self.prometheus_metrics: Dict[str, PrometheusMetric] = {}
        
        # Thread safety
        self.metrics_lock = threading.Lock()
        
        # Background monitoring thread
        self.monitoring_thread: Optional[threading.Thread] = None
        self.monitoring_stop_event = threading.Event()
        self.monitoring_enabled = self.config.get("enable_monitoring", True)
        
        # Performance baselines for comparison
        self.baseline_metrics: Optional[PerformanceMetrics] = None
        self.performance_targets = {
            "min_cache_hit_rate": 0.30,  # 30% minimum hit rate
            "max_lookup_latency_ms": 10.0,  # 10ms max lookup latency
            "max_storage_latency_ms": 50.0,  # 50ms max storage latency
            "max_memory_utilization": 0.80,  # 80% max memory utilization
            "max_cpu_utilization": 0.70,  # 70% max CPU utilization
        }
        
        # Initialize Prometheus metrics
        self._initialize_prometheus_metrics()
        
        # Start monitoring if enabled
        if self.monitoring_enabled:
            self.start_monitoring()
        
        logger.info(f"NeuronPerformanceMonitor initialized with config: {self.config}")
    
    def _initialize_prometheus_metrics(self):
        """Initialize Prometheus metrics definitions."""
        metrics_definitions = [
            # Cache performance metrics
            ("lmcache_cache_hit_rate", "gauge", "Cache hit rate (0.0-1.0)", 0.0),
            ("lmcache_cache_miss_rate", "gauge", "Cache miss rate (0.0-1.0)", 0.0),
            ("lmcache_partial_hit_rate", "gauge", "Partial cache hit rate (0.0-1.0)", 0.0),
            ("lmcache_effective_hit_rate", "gauge", "Effective hit rate including partial hits (0.0-1.0)", 0.0),
            
            # Latency metrics
            ("lmcache_lookup_latency_ms", "gauge", "Average cache lookup latency in milliseconds", 0.0),
            ("lmcache_storage_latency_ms", "gauge", "Average cache storage latency in milliseconds", 0.0),
            ("lmcache_lookup_latency_p95_ms", "gauge", "95th percentile cache lookup latency in milliseconds", 0.0),
            ("lmcache_storage_latency_p95_ms", "gauge", "95th percentile cache storage latency in milliseconds", 0.0),
            
            # Resource utilization metrics
            ("lmcache_memory_utilization_percent", "gauge", "Memory utilization percentage", 0.0),
            ("lmcache_cpu_utilization_percent", "gauge", "CPU utilization percentage", 0.0),
            ("lmcache_storage_utilization_gb", "gauge", "Storage utilization in gigabytes", 0.0),
            
            # Operation counters
            ("lmcache_total_requests", "counter", "Total number of cache requests", 0),
            ("lmcache_cache_hits", "counter", "Total number of cache hits", 0),
            ("lmcache_cache_misses", "counter", "Total number of cache misses", 0),
            ("lmcache_partial_hits", "counter", "Total number of partial cache hits", 0),
            ("lmcache_storage_operations", "counter", "Total number of storage operations", 0),
            ("lmcache_errors", "counter", "Total number of errors", 0),
            
            # Performance indicators
            ("lmcache_performance_score", "gauge", "Overall performance score (0.0-1.0)", 0.0),
            ("lmcache_baseline_comparison", "gauge", "Performance compared to baseline (ratio)", 1.0),
        ]
        
        for name, metric_type, help_text, initial_value in metrics_definitions:
            self.prometheus_metrics[name] = PrometheusMetric(
                name=name,
                metric_type=metric_type,
                help_text=help_text,
                value=initial_value,
                labels={"component": "lmcache_neuron"}
            )
    
    def track_cache_operation(self, operation_type: str, duration_ms: float, success: bool = True):
        """
        Track a cache operation for performance monitoring.
        
        Args:
            operation_type: Type of operation ("lookup", "storage", "hit", "miss", "partial_hit", "error")
            duration_ms: Duration of the operation in milliseconds
            success: Whether the operation was successful
        """
        with self.metrics_lock:
            timestamp = time.time()
            
            # Record operation timing
            self.cache_operations[operation_type].append(duration_ms)
            
            # Keep only recent operations (last hour)
            cutoff_time = timestamp - 3600  # 1 hour
            self.cache_operations[operation_type] = [
                duration for duration in self.cache_operations[operation_type]
                if len(self.cache_operations[operation_type]) <= 1000  # Keep max 1000 samples
            ]
            
            # Update current metrics based on operation type
            if operation_type == "lookup":
                self.current_metrics.total_requests += 1
                if success:
                    # Update lookup latency
                    if self.current_metrics.total_requests > 0:
                        # Running average
                        current_avg = self.current_metrics.average_lookup_latency_ms
                        n = self.current_metrics.total_requests
                        self.current_metrics.average_lookup_latency_ms = (
                            (current_avg * (n - 1) + duration_ms) / n
                        )
            
            elif operation_type == "storage":
                self.current_metrics.storage_operations += 1
                if success:
                    # Update storage latency
                    if self.current_metrics.storage_operations > 0:
                        current_avg = self.current_metrics.average_storage_latency_ms
                        n = self.current_metrics.storage_operations
                        self.current_metrics.average_storage_latency_ms = (
                            (current_avg * (n - 1) + duration_ms) / n
                        )
            
            elif operation_type == "hit":
                self.current_metrics.cache_hits += 1
            
            elif operation_type == "miss":
                self.current_metrics.cache_misses += 1
            
            elif operation_type == "partial_hit":
                self.current_metrics.partial_hits += 1
            
            elif operation_type == "error":
                self.current_metrics.errors += 1
            
            # Recalculate derived metrics
            self._update_derived_metrics()
    
    def record_memory_usage(self, memory_usage_gb: float):
        """
        Record current memory usage.
        
        Args:
            memory_usage_gb: Memory usage in gigabytes
        """
        with self.metrics_lock:
            timestamp = time.time()
            self.memory_samples.append((timestamp, memory_usage_gb))
            
            # Keep only recent samples (last hour)
            cutoff_time = timestamp - 3600
            self.memory_samples = [
                (ts, usage) for ts, usage in self.memory_samples
                if ts >= cutoff_time
            ]
            
            # Update current memory utilization
            if self.memory_samples:
                recent_usage = [usage for _, usage in self.memory_samples[-10:]]  # Last 10 samples
                self.current_metrics.memory_utilization_percent = (
                    sum(recent_usage) / len(recent_usage)
                ) * 100.0  # Convert to percentage
                
                self.current_metrics.storage_utilization_gb = memory_usage_gb
    
    def record_cpu_usage(self, cpu_percent: float):
        """
        Record current CPU usage.
        
        Args:
            cpu_percent: CPU usage percentage (0-100)
        """
        with self.metrics_lock:
            timestamp = time.time()
            self.cpu_samples.append((timestamp, cpu_percent))
            
            # Keep only recent samples (last hour)
            cutoff_time = timestamp - 3600
            self.cpu_samples = [
                (ts, usage) for ts, usage in self.cpu_samples
                if ts >= cutoff_time
            ]
            
            # Update current CPU utilization
            if self.cpu_samples:
                recent_usage = [usage for _, usage in self.cpu_samples[-10:]]  # Last 10 samples
                self.current_metrics.cpu_utilization_percent = sum(recent_usage) / len(recent_usage)
    
    def _update_derived_metrics(self):
        """Update derived metrics based on current counters."""
        total_requests = self.current_metrics.total_requests
        
        if total_requests > 0:
            self.current_metrics.cache_hit_rate = self.current_metrics.cache_hits / total_requests
            self.current_metrics.cache_miss_rate = self.current_metrics.cache_misses / total_requests
            self.current_metrics.partial_hit_rate = self.current_metrics.partial_hits / total_requests
            self.current_metrics.effective_hit_rate = (
                (self.current_metrics.cache_hits + self.current_metrics.partial_hits) / total_requests
            )
        
        # Calculate percentile latencies
        if "lookup" in self.cache_operations and self.cache_operations["lookup"]:
            lookup_latencies = sorted(self.cache_operations["lookup"])
            if len(lookup_latencies) >= 20:  # Need sufficient samples for percentiles
                p95_index = int(len(lookup_latencies) * 0.95)
                self.current_metrics.p95_lookup_latency_ms = lookup_latencies[p95_index]
        
        if "storage" in self.cache_operations and self.cache_operations["storage"]:
            storage_latencies = sorted(self.cache_operations["storage"])
            if len(storage_latencies) >= 20:
                p95_index = int(len(storage_latencies) * 0.95)
                self.current_metrics.p95_storage_latency_ms = storage_latencies[p95_index]
        
        # Update timestamp
        self.current_metrics.timestamp = time.time()
    
    def start_monitoring(self):
        """Start background monitoring thread."""
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            logger.warning("Monitoring thread already running")
            return
        
        def monitoring_worker():
            """Background monitoring worker function."""
            while not self.monitoring_stop_event.wait(timeout=self.metrics_collection_interval):
                try:
                    self._collect_system_metrics()
                    self._update_prometheus_metrics()
                    self._store_metrics_snapshot()
                except Exception as e:
                    logger.warning(f"Monitoring worker error: {e}")
        
        self.monitoring_thread = threading.Thread(
            target=monitoring_worker,
            name="NeuronPerformanceMonitor",
            daemon=True
        )
        self.monitoring_thread.start()
        logger.info("Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop background monitoring thread."""
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_stop_event.set()
            self.monitoring_thread.join(timeout=5.0)
            logger.info("Performance monitoring stopped")
    
    def _collect_system_metrics(self):
        """Collect system-level metrics (CPU, memory)."""
        try:
            # Get CPU usage
            cpu_percent = psutil.cpu_percent(interval=1.0)
            self.record_cpu_usage(cpu_percent)
            
            # Get memory usage
            memory_info = psutil.virtual_memory()
            memory_usage_gb = (memory_info.total - memory_info.available) / (1024**3)
            self.record_memory_usage(memory_usage_gb)
            
        except Exception as e:
            logger.debug(f"System metrics collection failed: {e}")
    
    def _update_prometheus_metrics(self):
        """Update Prometheus metrics with current values."""
        if not self.enable_prometheus_export:
            return
        
        with self.metrics_lock:
            # Update all Prometheus metrics with current values
            metric_mappings = {
                "lmcache_cache_hit_rate": self.current_metrics.cache_hit_rate,
                "lmcache_cache_miss_rate": self.current_metrics.cache_miss_rate,
                "lmcache_partial_hit_rate": self.current_metrics.partial_hit_rate,
                "lmcache_effective_hit_rate": self.current_metrics.effective_hit_rate,
                "lmcache_lookup_latency_ms": self.current_metrics.average_lookup_latency_ms,
                "lmcache_storage_latency_ms": self.current_metrics.average_storage_latency_ms,
                "lmcache_lookup_latency_p95_ms": self.current_metrics.p95_lookup_latency_ms,
                "lmcache_storage_latency_p95_ms": self.current_metrics.p95_storage_latency_ms,
                "lmcache_memory_utilization_percent": self.current_metrics.memory_utilization_percent,
                "lmcache_cpu_utilization_percent": self.current_metrics.cpu_utilization_percent,
                "lmcache_storage_utilization_gb": self.current_metrics.storage_utilization_gb,
                "lmcache_total_requests": self.current_metrics.total_requests,
                "lmcache_cache_hits": self.current_metrics.cache_hits,
                "lmcache_cache_misses": self.current_metrics.cache_misses,
                "lmcache_partial_hits": self.current_metrics.partial_hits,
                "lmcache_storage_operations": self.current_metrics.storage_operations,
                "lmcache_errors": self.current_metrics.errors,
            }
            
            for metric_name, value in metric_mappings.items():
                if metric_name in self.prometheus_metrics:
                    self.prometheus_metrics[metric_name].value = value
            
            # Calculate performance score
            performance_score = self._calculate_performance_score()
            self.prometheus_metrics["lmcache_performance_score"].value = performance_score
            
            # Calculate baseline comparison
            baseline_comparison = self._calculate_baseline_comparison()
            self.prometheus_metrics["lmcache_baseline_comparison"].value = baseline_comparison
    
    def _store_metrics_snapshot(self):
        """Store a snapshot of current metrics in history."""
        with self.metrics_lock:
            # Create a copy of current metrics
            snapshot = PerformanceMetrics(
                cache_hit_rate=self.current_metrics.cache_hit_rate,
                cache_miss_rate=self.current_metrics.cache_miss_rate,
                partial_hit_rate=self.current_metrics.partial_hit_rate,
                effective_hit_rate=self.current_metrics.effective_hit_rate,
                average_lookup_latency_ms=self.current_metrics.average_lookup_latency_ms,
                average_storage_latency_ms=self.current_metrics.average_storage_latency_ms,
                p95_lookup_latency_ms=self.current_metrics.p95_lookup_latency_ms,
                p95_storage_latency_ms=self.current_metrics.p95_storage_latency_ms,
                memory_utilization_percent=self.current_metrics.memory_utilization_percent,
                cpu_utilization_percent=self.current_metrics.cpu_utilization_percent,
                storage_utilization_gb=self.current_metrics.storage_utilization_gb,
                total_requests=self.current_metrics.total_requests,
                cache_hits=self.current_metrics.cache_hits,
                cache_misses=self.current_metrics.cache_misses,
                partial_hits=self.current_metrics.partial_hits,
                storage_operations=self.current_metrics.storage_operations,
                errors=self.current_metrics.errors,
                timestamp=time.time()
            )
            
            self.metrics_history.append(snapshot)
    
    def _calculate_performance_score(self) -> float:
        """
        Calculate overall performance score (0.0-1.0) based on multiple factors.
        
        Returns:
            Performance score between 0.0 (poor) and 1.0 (excellent)
        """
        score_components = []
        
        # Cache hit rate component (40% weight)
        hit_rate_score = min(1.0, self.current_metrics.effective_hit_rate / 0.8)  # Target 80% hit rate
        score_components.append((hit_rate_score, 0.4))
        
        # Latency component (30% weight)
        target_lookup_latency = self.performance_targets["max_lookup_latency_ms"]
        if self.current_metrics.average_lookup_latency_ms > 0:
            latency_score = max(0.0, 1.0 - (self.current_metrics.average_lookup_latency_ms / target_lookup_latency))
        else:
            latency_score = 1.0
        score_components.append((latency_score, 0.3))
        
        # Resource utilization component (20% weight)
        memory_score = max(0.0, 1.0 - (self.current_metrics.memory_utilization_percent / 100.0))
        cpu_score = max(0.0, 1.0 - (self.current_metrics.cpu_utilization_percent / 100.0))
        resource_score = (memory_score + cpu_score) / 2.0
        score_components.append((resource_score, 0.2))
        
        # Error rate component (10% weight)
        if self.current_metrics.total_requests > 0:
            error_rate = self.current_metrics.errors / self.current_metrics.total_requests
            error_score = max(0.0, 1.0 - (error_rate * 10))  # Penalize errors heavily
        else:
            error_score = 1.0
        score_components.append((error_score, 0.1))
        
        # Calculate weighted average
        total_score = sum(score * weight for score, weight in score_components)
        return max(0.0, min(1.0, total_score))
    
    def _calculate_baseline_comparison(self) -> float:
        """
        Calculate performance compared to baseline.
        
        Returns:
            Ratio compared to baseline (1.0 = same as baseline, >1.0 = better, <1.0 = worse)
        """
        if not self.baseline_metrics:
            return 1.0
        
        current_score = self._calculate_performance_score()
        
        # Calculate baseline score using same method
        baseline_hit_rate_score = min(1.0, self.baseline_metrics.effective_hit_rate / 0.8)
        baseline_latency_score = max(0.0, 1.0 - (self.baseline_metrics.average_lookup_latency_ms / 
                                                self.performance_targets["max_lookup_latency_ms"]))
        baseline_memory_score = max(0.0, 1.0 - (self.baseline_metrics.memory_utilization_percent / 100.0))
        baseline_cpu_score = max(0.0, 1.0 - (self.baseline_metrics.cpu_utilization_percent / 100.0))
        baseline_resource_score = (baseline_memory_score + baseline_cpu_score) / 2.0
        
        if self.baseline_metrics.total_requests > 0:
            baseline_error_rate = self.baseline_metrics.errors / self.baseline_metrics.total_requests
            baseline_error_score = max(0.0, 1.0 - (baseline_error_rate * 10))
        else:
            baseline_error_score = 1.0
        
        baseline_score = (
            baseline_hit_rate_score * 0.4 +
            baseline_latency_score * 0.3 +
            baseline_resource_score * 0.2 +
            baseline_error_score * 0.1
        )
        
        if baseline_score > 0:
            return current_score / baseline_score
        else:
            return 1.0
    
    def set_baseline_metrics(self, metrics: Optional[PerformanceMetrics] = None):
        """
        Set baseline metrics for performance comparison.
        
        Args:
            metrics: Baseline metrics to use. If None, uses current metrics as baseline.
        """
        if metrics is None:
            with self.metrics_lock:
                self.baseline_metrics = PerformanceMetrics(
                    cache_hit_rate=self.current_metrics.cache_hit_rate,
                    cache_miss_rate=self.current_metrics.cache_miss_rate,
                    partial_hit_rate=self.current_metrics.partial_hit_rate,
                    effective_hit_rate=self.current_metrics.effective_hit_rate,
                    average_lookup_latency_ms=self.current_metrics.average_lookup_latency_ms,
                    average_storage_latency_ms=self.current_metrics.average_storage_latency_ms,
                    p95_lookup_latency_ms=self.current_metrics.p95_lookup_latency_ms,
                    p95_storage_latency_ms=self.current_metrics.p95_storage_latency_ms,
                    memory_utilization_percent=self.current_metrics.memory_utilization_percent,
                    cpu_utilization_percent=self.current_metrics.cpu_utilization_percent,
                    storage_utilization_gb=self.current_metrics.storage_utilization_gb,
                    total_requests=self.current_metrics.total_requests,
                    cache_hits=self.current_metrics.cache_hits,
                    cache_misses=self.current_metrics.cache_misses,
                    partial_hits=self.current_metrics.partial_hits,
                    storage_operations=self.current_metrics.storage_operations,
                    errors=self.current_metrics.errors,
                    timestamp=time.time()
                )
        else:
            self.baseline_metrics = metrics
        
        logger.info("Baseline metrics set for performance comparison")
    
    def export_prometheus_metrics(self) -> str:
        """
        Export metrics in Prometheus format.
        
        Returns:
            Prometheus-formatted metrics string
        """
        if not self.enable_prometheus_export:
            return ""
        
        lines = []
        
        with self.metrics_lock:
            for metric_name, metric in self.prometheus_metrics.items():
                # Add help text
                lines.append(f"# HELP {metric_name} {metric.help_text}")
                lines.append(f"# TYPE {metric_name} {metric.metric_type}")
                
                # Add metric with labels
                if metric.labels:
                    label_str = ",".join([f'{k}="{v}"' for k, v in metric.labels.items()])
                    lines.append(f"{metric_name}{{{label_str}}} {metric.value}")
                else:
                    lines.append(f"{metric_name} {metric.value}")
                
                lines.append("")  # Empty line between metrics
        
        return "\n".join(lines)
    
    def get_current_metrics(self) -> PerformanceMetrics:
        """
        Get current performance metrics.
        
        Returns:
            Current performance metrics
        """
        with self.metrics_lock:
            return PerformanceMetrics(
                cache_hit_rate=self.current_metrics.cache_hit_rate,
                cache_miss_rate=self.current_metrics.cache_miss_rate,
                partial_hit_rate=self.current_metrics.partial_hit_rate,
                effective_hit_rate=self.current_metrics.effective_hit_rate,
                average_lookup_latency_ms=self.current_metrics.average_lookup_latency_ms,
                average_storage_latency_ms=self.current_metrics.average_storage_latency_ms,
                p95_lookup_latency_ms=self.current_metrics.p95_lookup_latency_ms,
                p95_storage_latency_ms=self.current_metrics.p95_storage_latency_ms,
                memory_utilization_percent=self.current_metrics.memory_utilization_percent,
                cpu_utilization_percent=self.current_metrics.cpu_utilization_percent,
                storage_utilization_gb=self.current_metrics.storage_utilization_gb,
                total_requests=self.current_metrics.total_requests,
                cache_hits=self.current_metrics.cache_hits,
                cache_misses=self.current_metrics.cache_misses,
                partial_hits=self.current_metrics.partial_hits,
                storage_operations=self.current_metrics.storage_operations,
                errors=self.current_metrics.errors,
                timestamp=self.current_metrics.timestamp
            )
    
    def get_metrics_history(self, duration_seconds: Optional[float] = None) -> List[PerformanceMetrics]:
        """
        Get historical metrics within specified duration.
        
        Args:
            duration_seconds: Duration to look back. If None, returns all available history.
            
        Returns:
            List of historical performance metrics
        """
        with self.metrics_lock:
            if duration_seconds is None:
                return list(self.metrics_history)
            
            cutoff_time = time.time() - duration_seconds
            return [
                metrics for metrics in self.metrics_history
                if metrics.timestamp >= cutoff_time
            ]
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive performance summary.
        
        Returns:
            Dictionary containing performance summary
        """
        current = self.get_current_metrics()
        performance_score = self._calculate_performance_score()
        baseline_comparison = self._calculate_baseline_comparison()
        
        # Check if performance targets are met
        targets_met = {
            "cache_hit_rate": current.effective_hit_rate >= self.performance_targets["min_cache_hit_rate"],
            "lookup_latency": current.average_lookup_latency_ms <= self.performance_targets["max_lookup_latency_ms"],
            "storage_latency": current.average_storage_latency_ms <= self.performance_targets["max_storage_latency_ms"],
            "memory_utilization": (current.memory_utilization_percent / 100.0) <= self.performance_targets["max_memory_utilization"],
            "cpu_utilization": (current.cpu_utilization_percent / 100.0) <= self.performance_targets["max_cpu_utilization"],
        }
        
        return {
            "current_metrics": {
                "cache_hit_rate": current.cache_hit_rate,
                "effective_hit_rate": current.effective_hit_rate,
                "average_lookup_latency_ms": current.average_lookup_latency_ms,
                "average_storage_latency_ms": current.average_storage_latency_ms,
                "p95_lookup_latency_ms": current.p95_lookup_latency_ms,
                "memory_utilization_percent": current.memory_utilization_percent,
                "cpu_utilization_percent": current.cpu_utilization_percent,
                "total_requests": current.total_requests,
                "errors": current.errors,
            },
            "performance_score": performance_score,
            "baseline_comparison": baseline_comparison,
            "targets_met": targets_met,
            "all_targets_met": all(targets_met.values()),
            "performance_targets": self.performance_targets,
            "monitoring_config": {
                "monitoring_enabled": self.monitoring_enabled,
                "metrics_interval": self.metrics_collection_interval,
                "prometheus_enabled": self.enable_prometheus_export,
                "prometheus_port": self.prometheus_port,
            },
            "timestamp": current.timestamp,
        }
    
    def reset_metrics(self):
        """Reset all metrics to initial state."""
        with self.metrics_lock:
            self.current_metrics = PerformanceMetrics()
            self.cache_operations.clear()
            self.memory_samples.clear()
            self.cpu_samples.clear()
            self.metrics_history.clear()
            
            # Reset Prometheus metrics
            for metric in self.prometheus_metrics.values():
                if metric.metric_type == "counter":
                    metric.value = 0
                else:
                    metric.value = 0.0
        
        logger.info("Performance metrics reset")
    
    def shutdown(self):
        """Shutdown the performance monitor and cleanup resources."""
        logger.info("Shutting down NeuronPerformanceMonitor")
        
        # Stop monitoring thread
        self.stop_monitoring()
        
        # Clear data structures
        with self.metrics_lock:
            self.cache_operations.clear()
            self.memory_samples.clear()
            self.cpu_samples.clear()
            self.metrics_history.clear()
            self.prometheus_metrics.clear()
        
        logger.info("NeuronPerformanceMonitor shutdown completed")


@dataclass
class HealthCheckResult:
    """Result of a health check operation."""
    component: str
    healthy: bool
    status: str
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    response_time_ms: float = 0.0


@dataclass
class AlertRule:
    """Definition of an alerting rule."""
    name: str
    condition: str  # Python expression to evaluate
    severity: str  # "low", "medium", "high", "critical"
    description: str
    enabled: bool = True
    cooldown_seconds: float = 300.0  # 5 minutes default cooldown
    last_triggered: Optional[float] = None


@dataclass
class Alert:
    """An active alert."""
    rule_name: str
    severity: str
    message: str
    details: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    resolved: bool = False
    resolved_timestamp: Optional[float] = None


class HealthCheckEndpoint:
    """
    Health check endpoint system for LMCache-Neuron integration.
    
    Provides HTTP endpoints for health status and implements performance anomaly detection.
    Validates: Requirements 7.3, 7.4
    """
    
    def __init__(self, 
                 performance_monitor: Optional[NeuronPerformanceMonitor] = None,
                 config: Optional[Dict[str, Any]] = None):
        self.performance_monitor = performance_monitor
        self.config = config or {}
        
        # Health check configuration
        self.health_check_interval = self.config.get("health_check_interval", 60.0)  # seconds
        self.health_check_timeout = self.config.get("health_check_timeout", 10.0)  # seconds
        self.enable_http_endpoint = self.config.get("enable_http_endpoint", True)
        self.http_port = self.config.get("health_check_port", 8081)
        
        # Component health checkers
        self.health_checkers: Dict[str, callable] = {}
        self.health_results: Dict[str, HealthCheckResult] = {}
        
        # Anomaly detection configuration
        self.anomaly_detection_enabled = self.config.get("enable_anomaly_detection", True)
        self.anomaly_thresholds = {
            "cache_hit_rate_drop": 0.20,  # 20% drop in hit rate
            "latency_increase": 2.0,      # 2x increase in latency
            "error_rate_spike": 0.10,     # 10% error rate
            "memory_usage_spike": 0.90,   # 90% memory usage
            "cpu_usage_spike": 0.85,      # 85% CPU usage
        }
        self.anomaly_detection_window = self.config.get("anomaly_window_seconds", 300.0)  # 5 minutes
        
        # Alerting system
        self.alerting_enabled = self.config.get("enable_alerting", True)
        self.alert_rules: Dict[str, AlertRule] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        self.max_alert_history = self.config.get("max_alert_history", 1000)
        
        # Thread safety
        self.health_lock = threading.Lock()
        self.alert_lock = threading.Lock()
        
        # Background health checking
        self.health_check_thread: Optional[threading.Thread] = None
        self.health_check_stop_event = threading.Event()
        
        # HTTP server for health endpoints
        self.http_server = None
        
        # Initialize default health checkers and alert rules
        self._initialize_default_health_checkers()
        self._initialize_default_alert_rules()
        
        # Start health checking if enabled
        if self.config.get("enable_health_checks", True):
            self.start_health_checking()
        
        logger.info(f"HealthCheckEndpoint initialized with config: {self.config}")
    
    def _initialize_default_health_checkers(self):
        """Initialize default health check functions."""
        
        def check_performance_monitor():
            """Check if performance monitor is healthy."""
            if not self.performance_monitor:
                return HealthCheckResult(
                    component="performance_monitor",
                    healthy=False,
                    status="not_configured",
                    details={"message": "Performance monitor not configured"}
                )
            
            try:
                start_time = time.time()
                metrics = self.performance_monitor.get_current_metrics()
                response_time = (time.time() - start_time) * 1000
                
                # Check if metrics are reasonable
                if metrics.timestamp > 0 and (time.time() - metrics.timestamp) < 300:  # Within 5 minutes
                    return HealthCheckResult(
                        component="performance_monitor",
                        healthy=True,
                        status="healthy",
                        details={
                            "last_update": metrics.timestamp,
                            "total_requests": metrics.total_requests,
                            "cache_hit_rate": metrics.cache_hit_rate
                        },
                        response_time_ms=response_time
                    )
                else:
                    return HealthCheckResult(
                        component="performance_monitor",
                        healthy=False,
                        status="stale_metrics",
                        details={
                            "last_update": metrics.timestamp,
                            "age_seconds": time.time() - metrics.timestamp
                        },
                        response_time_ms=response_time
                    )
            except Exception as e:
                return HealthCheckResult(
                    component="performance_monitor",
                    healthy=False,
                    status="error",
                    details={"error": str(e)}
                )
        
        def check_system_resources():
            """Check system resource health."""
            try:
                start_time = time.time()
                
                # Check memory usage
                memory_info = psutil.virtual_memory()
                memory_percent = memory_info.percent
                
                # Check CPU usage
                cpu_percent = psutil.cpu_percent(interval=1.0)
                
                # Check disk usage
                disk_info = psutil.disk_usage('/')
                disk_percent = (disk_info.used / disk_info.total) * 100
                
                response_time = (time.time() - start_time) * 1000
                
                # Determine health based on resource usage
                healthy = (
                    memory_percent < 90 and
                    cpu_percent < 90 and
                    disk_percent < 95
                )
                
                status = "healthy" if healthy else "resource_pressure"
                
                return HealthCheckResult(
                    component="system_resources",
                    healthy=healthy,
                    status=status,
                    details={
                        "memory_percent": memory_percent,
                        "cpu_percent": cpu_percent,
                        "disk_percent": disk_percent
                    },
                    response_time_ms=response_time
                )
            except Exception as e:
                return HealthCheckResult(
                    component="system_resources",
                    healthy=False,
                    status="error",
                    details={"error": str(e)}
                )
        
        def check_neuron_devices():
            """Check Neuron device availability."""
            try:
                start_time = time.time()
                
                from vllm_neuron import _is_neuron_dev
                neuron_available = _is_neuron_dev()
                
                import glob
                neuron_devices = glob.glob('/dev/neuron*')
                device_count = len(neuron_devices)
                
                response_time = (time.time() - start_time) * 1000
                
                return HealthCheckResult(
                    component="neuron_devices",
                    healthy=neuron_available and device_count > 0,
                    status="healthy" if neuron_available else "no_devices",
                    details={
                        "neuron_available": neuron_available,
                        "device_count": device_count,
                        "devices": neuron_devices
                    },
                    response_time_ms=response_time
                )
            except Exception as e:
                return HealthCheckResult(
                    component="neuron_devices",
                    healthy=False,
                    status="error",
                    details={"error": str(e)}
                )
        
        # Register default health checkers
        self.health_checkers["performance_monitor"] = check_performance_monitor
        self.health_checkers["system_resources"] = check_system_resources
        self.health_checkers["neuron_devices"] = check_neuron_devices
    
    def _initialize_default_alert_rules(self):
        """Initialize default alerting rules."""
        default_rules = [
            AlertRule(
                name="high_error_rate",
                condition="metrics.errors / max(metrics.total_requests, 1) > 0.10",
                severity="high",
                description="Error rate exceeds 10%",
                cooldown_seconds=300.0
            ),
            AlertRule(
                name="low_cache_hit_rate",
                condition="metrics.effective_hit_rate < 0.20 and metrics.total_requests > 100",
                severity="medium",
                description="Cache hit rate below 20% with significant traffic",
                cooldown_seconds=600.0
            ),
            AlertRule(
                name="high_lookup_latency",
                condition="metrics.average_lookup_latency_ms > 50.0 and metrics.total_requests > 50",
                severity="medium",
                description="Average lookup latency exceeds 50ms",
                cooldown_seconds=300.0
            ),
            AlertRule(
                name="high_memory_usage",
                condition="metrics.memory_utilization_percent > 90.0",
                severity="high",
                description="Memory utilization exceeds 90%",
                cooldown_seconds=180.0
            ),
            AlertRule(
                name="high_cpu_usage",
                condition="metrics.cpu_utilization_percent > 85.0",
                severity="medium",
                description="CPU utilization exceeds 85%",
                cooldown_seconds=300.0
            ),
            AlertRule(
                name="performance_degradation",
                condition="performance_score < 0.5",
                severity="high",
                description="Overall performance score below 50%",
                cooldown_seconds=600.0
            ),
            AlertRule(
                name="component_unhealthy",
                condition="any(not result.healthy for result in health_results.values())",
                severity="high",
                description="One or more components are unhealthy",
                cooldown_seconds=300.0
            ),
        ]
        
        for rule in default_rules:
            self.alert_rules[rule.name] = rule
    
    def register_health_checker(self, component: str, checker_func: callable):
        """
        Register a custom health checker function.
        
        Args:
            component: Name of the component to check
            checker_func: Function that returns HealthCheckResult
        """
        self.health_checkers[component] = checker_func
        logger.info(f"Registered health checker for component: {component}")
    
    def add_alert_rule(self, rule: AlertRule):
        """
        Add a custom alert rule.
        
        Args:
            rule: AlertRule to add
        """
        with self.alert_lock:
            self.alert_rules[rule.name] = rule
        logger.info(f"Added alert rule: {rule.name}")
    
    def remove_alert_rule(self, rule_name: str):
        """
        Remove an alert rule.
        
        Args:
            rule_name: Name of the rule to remove
        """
        with self.alert_lock:
            if rule_name in self.alert_rules:
                del self.alert_rules[rule_name]
                logger.info(f"Removed alert rule: {rule_name}")
    
    def start_health_checking(self):
        """Start background health checking thread."""
        if self.health_check_thread and self.health_check_thread.is_alive():
            logger.warning("Health checking thread already running")
            return
        
        def health_check_worker():
            """Background health checking worker function."""
            while not self.health_check_stop_event.wait(timeout=self.health_check_interval):
                try:
                    self._perform_health_checks()
                    if self.anomaly_detection_enabled:
                        self._detect_performance_anomalies()
                    if self.alerting_enabled:
                        self._evaluate_alert_rules()
                except Exception as e:
                    logger.warning(f"Health check worker error: {e}")
        
        self.health_check_thread = threading.Thread(
            target=health_check_worker,
            name="HealthCheckWorker",
            daemon=True
        )
        self.health_check_thread.start()
        logger.info("Health checking started")
    
    def stop_health_checking(self):
        """Stop background health checking thread."""
        if self.health_check_thread and self.health_check_thread.is_alive():
            self.health_check_stop_event.set()
            self.health_check_thread.join(timeout=5.0)
            logger.info("Health checking stopped")
    
    def _perform_health_checks(self):
        """Perform all registered health checks."""
        with self.health_lock:
            for component, checker_func in self.health_checkers.items():
                try:
                    result = checker_func()
                    self.health_results[component] = result
                    
                    if not result.healthy:
                        logger.warning(f"Health check failed for {component}: {result.status}")
                    
                except Exception as e:
                    logger.error(f"Health checker error for {component}: {e}")
                    self.health_results[component] = HealthCheckResult(
                        component=component,
                        healthy=False,
                        status="check_error",
                        details={"error": str(e)}
                    )
    
    def _detect_performance_anomalies(self):
        """Detect performance anomalies based on historical data."""
        if not self.performance_monitor:
            return
        
        try:
            # Get recent metrics history
            recent_metrics = self.performance_monitor.get_metrics_history(
                duration_seconds=self.anomaly_detection_window
            )
            
            if len(recent_metrics) < 2:
                return  # Need at least 2 data points
            
            current_metrics = self.performance_monitor.get_current_metrics()
            
            # Calculate baseline from recent history (excluding current)
            baseline_metrics = recent_metrics[:-1] if len(recent_metrics) > 1 else recent_metrics
            
            if not baseline_metrics:
                return
            
            # Calculate baseline averages
            baseline_hit_rate = sum(m.effective_hit_rate for m in baseline_metrics) / len(baseline_metrics)
            baseline_latency = sum(m.average_lookup_latency_ms for m in baseline_metrics) / len(baseline_metrics)
            baseline_error_rate = sum(m.errors / max(m.total_requests, 1) for m in baseline_metrics) / len(baseline_metrics)
            
            # Detect anomalies
            anomalies = []
            
            # Cache hit rate drop
            if (baseline_hit_rate > 0 and 
                current_metrics.effective_hit_rate < baseline_hit_rate * (1 - self.anomaly_thresholds["cache_hit_rate_drop"])):
                anomalies.append({
                    "type": "cache_hit_rate_drop",
                    "severity": "medium",
                    "message": f"Cache hit rate dropped from {baseline_hit_rate:.2%} to {current_metrics.effective_hit_rate:.2%}",
                    "baseline": baseline_hit_rate,
                    "current": current_metrics.effective_hit_rate
                })
            
            # Latency increase
            if (baseline_latency > 0 and 
                current_metrics.average_lookup_latency_ms > baseline_latency * self.anomaly_thresholds["latency_increase"]):
                anomalies.append({
                    "type": "latency_increase",
                    "severity": "medium",
                    "message": f"Lookup latency increased from {baseline_latency:.1f}ms to {current_metrics.average_lookup_latency_ms:.1f}ms",
                    "baseline": baseline_latency,
                    "current": current_metrics.average_lookup_latency_ms
                })
            
            # Error rate spike
            current_error_rate = current_metrics.errors / max(current_metrics.total_requests, 1)
            if current_error_rate > self.anomaly_thresholds["error_rate_spike"]:
                anomalies.append({
                    "type": "error_rate_spike",
                    "severity": "high",
                    "message": f"Error rate spiked to {current_error_rate:.2%}",
                    "baseline": baseline_error_rate,
                    "current": current_error_rate
                })
            
            # Memory usage spike
            if current_metrics.memory_utilization_percent > self.anomaly_thresholds["memory_usage_spike"] * 100:
                anomalies.append({
                    "type": "memory_usage_spike",
                    "severity": "high",
                    "message": f"Memory usage spiked to {current_metrics.memory_utilization_percent:.1f}%",
                    "current": current_metrics.memory_utilization_percent
                })
            
            # CPU usage spike
            if current_metrics.cpu_utilization_percent > self.anomaly_thresholds["cpu_usage_spike"] * 100:
                anomalies.append({
                    "type": "cpu_usage_spike",
                    "severity": "medium",
                    "message": f"CPU usage spiked to {current_metrics.cpu_utilization_percent:.1f}%",
                    "current": current_metrics.cpu_utilization_percent
                })
            
            # Log detected anomalies
            for anomaly in anomalies:
                logger.warning(f"Performance anomaly detected: {anomaly['message']}")
                
                # Create alert for anomaly
                self._create_alert(
                    rule_name=f"anomaly_{anomaly['type']}",
                    severity=anomaly["severity"],
                    message=anomaly["message"],
                    details=anomaly
                )
        
        except Exception as e:
            logger.warning(f"Anomaly detection failed: {e}")
    
    def _evaluate_alert_rules(self):
        """Evaluate all alert rules and trigger alerts as needed."""
        if not self.performance_monitor:
            return
        
        try:
            # Get current metrics and health results
            metrics = self.performance_monitor.get_current_metrics()
            performance_score = self.performance_monitor._calculate_performance_score()
            
            with self.health_lock:
                health_results = self.health_results.copy()
            
            # Evaluate each alert rule
            with self.alert_lock:
                for rule_name, rule in self.alert_rules.items():
                    if not rule.enabled:
                        continue
                    
                    # Check cooldown period
                    if (rule.last_triggered and 
                        time.time() - rule.last_triggered < rule.cooldown_seconds):
                        continue
                    
                    try:
                        # Create evaluation context
                        eval_context = {
                            "metrics": metrics,
                            "performance_score": performance_score,
                            "health_results": health_results,
                            "time": time.time(),
                            "any": any,
                            "all": all,
                            "max": max,
                            "min": min,
                            "sum": sum,
                            "len": len,
                        }
                        
                        # Evaluate rule condition
                        condition_result = eval(rule.condition, {"__builtins__": {}}, eval_context)
                        
                        if condition_result:
                            # Trigger alert
                            self._create_alert(
                                rule_name=rule_name,
                                severity=rule.severity,
                                message=rule.description,
                                details={
                                    "condition": rule.condition,
                                    "metrics_snapshot": {
                                        "cache_hit_rate": metrics.cache_hit_rate,
                                        "effective_hit_rate": metrics.effective_hit_rate,
                                        "average_lookup_latency_ms": metrics.average_lookup_latency_ms,
                                        "memory_utilization_percent": metrics.memory_utilization_percent,
                                        "cpu_utilization_percent": metrics.cpu_utilization_percent,
                                        "total_requests": metrics.total_requests,
                                        "errors": metrics.errors,
                                    },
                                    "performance_score": performance_score,
                                }
                            )
                            
                            # Update last triggered time
                            rule.last_triggered = time.time()
                    
                    except Exception as e:
                        logger.warning(f"Alert rule evaluation failed for {rule_name}: {e}")
        
        except Exception as e:
            logger.warning(f"Alert rule evaluation failed: {e}")
    
    def _create_alert(self, rule_name: str, severity: str, message: str, details: Dict[str, Any]):
        """Create and register a new alert."""
        alert = Alert(
            rule_name=rule_name,
            severity=severity,
            message=message,
            details=details
        )
        
        with self.alert_lock:
            # Add to active alerts
            self.active_alerts[rule_name] = alert
            
            # Add to history
            self.alert_history.append(alert)
            
            # Trim history if needed
            if len(self.alert_history) > self.max_alert_history:
                self.alert_history = self.alert_history[-self.max_alert_history:]
        
        logger.error(f"ALERT [{severity.upper()}] {rule_name}: {message}")
    
    def resolve_alert(self, rule_name: str):
        """
        Resolve an active alert.
        
        Args:
            rule_name: Name of the alert rule to resolve
        """
        with self.alert_lock:
            if rule_name in self.active_alerts:
                alert = self.active_alerts[rule_name]
                alert.resolved = True
                alert.resolved_timestamp = time.time()
                
                # Remove from active alerts
                del self.active_alerts[rule_name]
                
                logger.info(f"Alert resolved: {rule_name}")
    
    def get_health_status(self) -> Dict[str, Any]:
        """
        Get comprehensive health status.
        
        Returns:
            Dictionary containing health status information
        """
        with self.health_lock:
            health_results = {name: result for name, result in self.health_results.items()}
        
        with self.alert_lock:
            active_alerts = list(self.active_alerts.values())
            recent_alerts = [alert for alert in self.alert_history[-10:]]  # Last 10 alerts
        
        # Calculate overall health
        all_healthy = all(result.healthy for result in health_results.values())
        
        # Get component status summary
        component_status = {}
        for name, result in health_results.items():
            component_status[name] = {
                "healthy": result.healthy,
                "status": result.status,
                "response_time_ms": result.response_time_ms,
                "last_check": result.timestamp,
            }
        
        return {
            "overall_healthy": all_healthy,
            "component_status": component_status,
            "active_alerts": [
                {
                    "rule_name": alert.rule_name,
                    "severity": alert.severity,
                    "message": alert.message,
                    "timestamp": alert.timestamp,
                }
                for alert in active_alerts
            ],
            "recent_alerts": [
                {
                    "rule_name": alert.rule_name,
                    "severity": alert.severity,
                    "message": alert.message,
                    "timestamp": alert.timestamp,
                    "resolved": alert.resolved,
                }
                for alert in recent_alerts
            ],
            "health_check_config": {
                "interval_seconds": self.health_check_interval,
                "timeout_seconds": self.health_check_timeout,
                "anomaly_detection_enabled": self.anomaly_detection_enabled,
                "alerting_enabled": self.alerting_enabled,
            },
            "timestamp": time.time(),
        }
    
    def get_health_endpoint_response(self) -> Tuple[Dict[str, Any], int]:
        """
        Get health endpoint response suitable for HTTP endpoints.
        
        Returns:
            Tuple of (response_dict, http_status_code)
        """
        health_status = self.get_health_status()
        
        # Determine HTTP status code
        if health_status["overall_healthy"]:
            status_code = 200  # OK
        else:
            # Check severity of issues
            has_critical_alerts = any(
                alert["severity"] == "critical" 
                for alert in health_status["active_alerts"]
            )
            
            if has_critical_alerts:
                status_code = 503  # Service Unavailable
            else:
                status_code = 200  # OK but with warnings
        
        # Create simplified response
        response = {
            "status": "healthy" if health_status["overall_healthy"] else "unhealthy",
            "timestamp": health_status["timestamp"],
            "components": {
                name: status["healthy"] 
                for name, status in health_status["component_status"].items()
            },
            "active_alerts_count": len(health_status["active_alerts"]),
            "details": health_status if self.config.get("include_detailed_health", False) else None
        }
        
        return response, status_code
    
    def get_metrics_endpoint_response(self) -> str:
        """
        Get Prometheus metrics endpoint response.
        
        Returns:
            Prometheus-formatted metrics string
        """
        if self.performance_monitor:
            return self.performance_monitor.export_prometheus_metrics()
        else:
            return "# No performance monitor configured\n"
    
    def shutdown(self):
        """Shutdown health checking and cleanup resources."""
        logger.info("Shutting down HealthCheckEndpoint")
        
        # Stop health checking thread
        self.stop_health_checking()
        
        # Clear data structures
        with self.health_lock:
            self.health_checkers.clear()
            self.health_results.clear()
        
        with self.alert_lock:
            self.alert_rules.clear()
            self.active_alerts.clear()
            self.alert_history.clear()
        
        logger.info("HealthCheckEndpoint shutdown completed")


class LMCacheNeuronIntegration:
    """Integration layer for LMCache with vLLM-Neuron."""
    
    def __init__(self):
        self.device_detector = NeuronDeviceDetector()
        self.config_manager = LMCacheConfigManager()
        self._initialized = False
        self._lmcache_engine: Optional["LMCacheEngine"] = None
        self._cache_handler: Optional[CacheHitMissHandler] = None
        self._graceful_degradation_enabled = True
        self._operation_stats = {
            "intercepted_operations": 0,
            "successful_operations": 0,
            "failed_operations": 0,
            "degraded_operations": 0
        }
        self._stats_lock = threading.Lock()
        
        # Chunked prefill detection and cache strategy
        self.chunked_prefill_enabled = self._detect_chunked_prefill_status()
        self.cache_strategy = self._determine_cache_strategy()
        self._warn_about_chunked_prefill_limitations()
        
        # Enhanced error handling components
        self.failure_detector = ComponentFailureDetector()
        self.retry_manager = RetryManager()
        self.storage_failure_handler = StorageBackendFailureHandler(
            self.failure_detector, self.retry_manager
        )
        
        # Performance monitoring and health checking
        self.performance_monitor: Optional[NeuronPerformanceMonitor] = None
        self.health_endpoint: Optional[HealthCheckEndpoint] = None
        
        # Recovery settings
        self.auto_recovery_enabled = True
        self.recovery_check_interval = 30.0  # seconds
        self.recovery_thread: Optional[threading.Thread] = None
        self.recovery_stop_event = threading.Event()
        
        # Start recovery thread if auto-recovery is enabled
        if self.auto_recovery_enabled:
            self._start_recovery_thread()
    
    def _detect_chunked_prefill_status(self) -> bool:
        """Detect if chunked prefill is enabled in the current environment."""
        # Check environment variable that enables chunked prefill
        disable_custom_scheduler = os.environ.get("DISABLE_NEURON_CUSTOM_SCHEDULER", "0")
        chunked_prefill_enabled = disable_custom_scheduler == "1"
        
        logger.info(f"Chunked prefill status: {'enabled' if chunked_prefill_enabled else 'disabled'}")
        if not chunked_prefill_enabled:
            logger.info("To enable chunked prefill: set DISABLE_NEURON_CUSTOM_SCHEDULER=1")
        
        return chunked_prefill_enabled
    
    def _determine_cache_strategy(self) -> CacheStrategy:
        """Determine the appropriate cache strategy based on chunked prefill availability."""
        if self.chunked_prefill_enabled:
            return CacheStrategy.CHUNKED_PREFILL
        else:
            return CacheStrategy.FULL_SEQUENCE_ONLY
    
    def _warn_about_chunked_prefill_limitations(self):
        """Warn users about limitations when chunked prefill is disabled."""
        if not self.chunked_prefill_enabled:
            warning_message = """
⚠️  CHUNKED PREFILL DISABLED - LIMITED LMCACHE EFFECTIVENESS

Current Status: Chunked prefill is disabled (vLLM-Neuron default)
Impact: LMCache can only cache complete sequences, not partial matches

To Enable Full LMCache Functionality:
1. Set environment variable: DISABLE_NEURON_CUSTOM_SCHEDULER=1
2. Configure num_gpu_blocks_override parameter appropriately
3. Restart your vLLM service

Benefits of Enabling:
- ✅ Partial cache hits for long sequences
- ✅ Better cache utilization  
- ✅ Improved TTFT for similar prompts

Current Limited Benefits:
- ✅ Full sequence matches still cached
- ✅ Decode phase caching active
- ✅ Identical prompts across requests cached

For more information, see: CHUNKED_PREFILL_LIMITATIONS.md
            """.strip()
            
            logger.warning(warning_message)
    
    def get_performance_expectations(self) -> Dict[str, str]:
        """Get realistic performance expectations based on current configuration."""
        if self.chunked_prefill_enabled:
            return {
                "ttft_improvement": "30-50% for partial cache hits",
                "cache_hit_rate": "High for similar prompts",
                "memory_efficiency": "Optimal with partial caching",
                "cache_strategy": "Full chunked prefill caching"
            }
        else:
            return {
                "ttft_improvement": "30-50% for identical prompts only",
                "cache_hit_rate": "Lower (full sequence matches only)",
                "memory_efficiency": "Reduced (no partial caching)",
                "cache_strategy": "Full sequence caching only",
                "recommendation": "Enable chunked prefill for full benefits"
            }
    
    def _start_recovery_thread(self):
        """Start the automatic recovery thread."""
        def recovery_worker():
            while not self.recovery_stop_event.wait(timeout=self.recovery_check_interval):
                try:
                    self._perform_health_checks()
                    self._attempt_component_recovery()
                except Exception as e:
                    logger.warning(f"Recovery thread error: {e}")
        
        self.recovery_thread = threading.Thread(
            target=recovery_worker,
            name="LMCache-Recovery",
            daemon=True
        )
        self.recovery_thread.start()
        logger.info("Started automatic recovery thread")
    
    def _perform_health_checks(self):
        """Perform periodic health checks on components."""
        try:
            # Check LMCache engine health
            if self._lmcache_engine and hasattr(self._lmcache_engine, 'get_health'):
                health = self._lmcache_engine.get_health()
                if not health.get('healthy', True):
                    self.failure_detector.record_error(
                        ComponentType.LMCACHE_ENGINE,
                        ErrorSeverity.MEDIUM,
                        "health_check_failure",
                        "LMCache engine health check failed",
                        {"health_status": health}
                    )
            
            # Check cache handler health
            if self._cache_handler:
                stats = self._cache_handler.get_stats()
                error_rate = stats.get("error_rate", 0.0)
                if error_rate > 0.1:  # More than 10% error rate
                    self.failure_detector.record_error(
                        ComponentType.CACHE_HANDLER,
                        ErrorSeverity.MEDIUM,
                        "high_error_rate",
                        f"Cache handler error rate: {error_rate:.2%}",
                        {"stats": stats}
                    )
            
        except Exception as e:
            logger.debug(f"Health check error: {e}")
    
    def _attempt_component_recovery(self):
        """Attempt to recover failed components."""
        try:
            # Attempt storage backend recovery
            if not self.failure_detector.is_component_healthy(ComponentType.STORAGE_BACKEND):
                if self.storage_failure_handler.attempt_recovery():
                    logger.info("Storage backend recovered successfully")
            
            # Attempt LMCache engine recovery
            if (not self.failure_detector.is_component_healthy(ComponentType.LMCACHE_ENGINE) and
                self._lmcache_engine is None):
                try:
                    config = self.config_manager.load_config()
                    if self._initialize_lmcache_engine(config):
                        self.failure_detector.mark_component_recovered(ComponentType.LMCACHE_ENGINE)
                        logger.info("LMCache engine recovered successfully")
                except Exception as e:
                    logger.debug(f"LMCache engine recovery failed: {e}")
            
        except Exception as e:
            logger.debug(f"Component recovery error: {e}")
    
    @classmethod
    def is_lmcache_available(cls) -> bool:
        """Check if LMCache is available in the environment."""
        return LMCACHE_AVAILABLE
    
    def initialize(self, vllm_config: "VllmConfig") -> bool:
        """Initialize LMCache integration with vLLM-Neuron."""
        if self._initialized:
            logger.info("LMCache integration already initialized")
            return True
        
        try:
            # Check prerequisites with error handling
            if not self._check_prerequisites():
                return self._handle_initialization_failure("Prerequisites not met")
            
            # Load and validate configuration with retry
            config = self._load_configuration_with_retry()
            if not config:
                return self._handle_initialization_failure("Configuration loading failed")
            
            # Initialize LMCache engine with retry
            if not self._initialize_lmcache_engine_with_retry(config):
                return self._handle_initialization_failure("LMCache engine initialization failed")
            
            # Set up cache hit/miss handler
            self._cache_handler = CacheHitMissHandler(self._lmcache_engine)
            
            # Set up performance monitoring
            monitoring_config = config.get("monitoring", {})
            self.performance_monitor = NeuronPerformanceMonitor(monitoring_config)
            
            # Set up health checking and alerting
            health_config = config.get("health_checks", {})
            self.health_endpoint = HealthCheckEndpoint(
                performance_monitor=self.performance_monitor,
                config=health_config
            )
            
            # Register integration-specific health checker
            self.health_endpoint.register_health_checker(
                "lmcache_integration", 
                self._check_integration_health
            )
            
            # Set up LMCache environment
            self._setup_lmcache_environment(config)
            
            # Configure vLLM for LMCache integration
            self._configure_vllm_integration(vllm_config, config)
            
            # Apply Neuron-specific optimizations
            self._apply_neuron_optimizations(vllm_config, config)
            
            # Set up storage backend failure handler
            if hasattr(self._lmcache_engine, 'storage_backend'):
                self.storage_failure_handler.set_storage_backend(self._lmcache_engine.storage_backend)
            
            self._initialized = True
            logger.info("LMCache integration with vLLM-Neuron initialized successfully")
            return True
            
        except Exception as e:
            return self._handle_initialization_failure(f"Unexpected error: {e}")
    
    def _check_prerequisites(self) -> bool:
        """Check prerequisites with error handling."""
        try:
            if not self.is_lmcache_available():
                self.failure_detector.record_error(
                    ComponentType.INTEGRATION_LAYER,
                    ErrorSeverity.CRITICAL,
                    "lmcache_unavailable",
                    "LMCache not available in environment"
                )
                logger.warning("LMCache not available, skipping integration")
                return False
            
            if not self.device_detector.is_neuron_available():
                self.failure_detector.record_error(
                    ComponentType.DEVICE_DETECTOR,
                    ErrorSeverity.HIGH,
                    "neuron_unavailable",
                    "No Neuron devices detected"
                )
                logger.warning("No Neuron devices detected, skipping LMCache integration")
                return False
            
            return True
            
        except Exception as e:
            self.failure_detector.record_error(
                ComponentType.INTEGRATION_LAYER,
                ErrorSeverity.HIGH,
                "prerequisite_check_error",
                str(e)
            )
            return False
    
    def _load_configuration_with_retry(self) -> Optional[Dict[str, Any]]:
        """Load configuration with retry logic."""
        try:
            return self.retry_manager.execute_with_retry(
                "config_loading",
                self.config_manager.load_config
            )
        except Exception as e:
            self.failure_detector.record_error(
                ComponentType.CONFIG_MANAGER,
                ErrorSeverity.HIGH,
                "config_loading_failure",
                str(e)
            )
            return None
    
    def _initialize_lmcache_engine_with_retry(self, config: Dict[str, Any]) -> bool:
        """Initialize LMCache engine with retry logic."""
        try:
            return self.retry_manager.execute_with_retry(
                "engine_initialization",
                self._initialize_lmcache_engine,
                config
            )
        except Exception as e:
            self.failure_detector.record_error(
                ComponentType.LMCACHE_ENGINE,
                ErrorSeverity.CRITICAL,
                "engine_initialization_failure",
                str(e),
                {"config": str(config)[:200]}
            )
            return False
    
    def _handle_initialization_failure(self, reason: str) -> bool:
        """Handle initialization failure with graceful degradation."""
        logger.error(f"LMCache integration initialization failed: {reason}")
        
        if self._graceful_degradation_enabled:
            logger.info("Graceful degradation enabled, continuing without LMCache")
            return True
        else:
            return False
    
    def _initialize_lmcache_engine(self, config: Dict[str, Any]) -> bool:
        """Initialize the LMCache engine."""
        try:
            if not LMCACHE_AVAILABLE:
                return False
            
            # Create LMCache configuration
            lmcache_config = LMCacheEngineConfig(
                chunk_size=config.get("chunk_size", 256),
                local_cpu=config.get("local_cpu", True),
                max_local_cpu_size=config.get("max_local_cpu_size", 4),
                save_unfull_chunk=config.get("save_unfull_chunk", True),
                save_decode_cache=config.get("save_decode_cache", True),
                remote_url=config.get("remote_url", "fs://localhost:0/tmp/lmcache_neuron"),
                **config.get("extra_config", {})
            )
            
            # Initialize LMCache engine
            self._lmcache_engine = LMCacheEngine(lmcache_config)
            logger.info("LMCache engine initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize LMCache engine: {e}")
            return False
    
    def _check_integration_health(self) -> HealthCheckResult:
        """Health checker for the LMCache integration itself."""
        try:
            start_time = time.time()
            
            # Check if integration is initialized
            if not self._initialized:
                return HealthCheckResult(
                    component="lmcache_integration",
                    healthy=False,
                    status="not_initialized",
                    details={"message": "Integration not initialized"}
                )
            
            # Check if LMCache engine is available
            if not self._lmcache_engine:
                return HealthCheckResult(
                    component="lmcache_integration",
                    healthy=False,
                    status="no_engine",
                    details={"message": "LMCache engine not available"}
                )
            
            # Check if cache handler is available
            if not self._cache_handler:
                return HealthCheckResult(
                    component="lmcache_integration",
                    healthy=False,
                    status="no_cache_handler",
                    details={"message": "Cache handler not available"}
                )
            
            # Check operation statistics for health indicators
            with self._stats_lock:
                total_ops = self._operation_stats["intercepted_operations"]
                failed_ops = self._operation_stats["failed_operations"]
                degraded_ops = self._operation_stats["degraded_operations"]
            
            error_rate = failed_ops / max(total_ops, 1)
            degraded_rate = degraded_ops / max(total_ops, 1)
            
            # Determine health based on error rates
            healthy = error_rate < 0.1 and degraded_rate < 0.2  # Less than 10% errors, 20% degraded
            status = "healthy" if healthy else "high_error_rate"
            
            response_time = (time.time() - start_time) * 1000
            
            return HealthCheckResult(
                component="lmcache_integration",
                healthy=healthy,
                status=status,
                details={
                    "total_operations": total_ops,
                    "failed_operations": failed_ops,
                    "degraded_operations": degraded_ops,
                    "error_rate": error_rate,
                    "degraded_rate": degraded_rate,
                    "graceful_degradation_enabled": self._graceful_degradation_enabled,
                },
                response_time_ms=response_time
            )
            
        except Exception as e:
            return HealthCheckResult(
                component="lmcache_integration",
                healthy=False,
                status="check_error",
                details={"error": str(e)}
            )
    
    def intercept_kv_operations(self, operation: KVCacheOperation) -> KVCacheResult:
        """
        Intercept KV cache operations from vLLM with enhanced error handling and performance tracking.
        
        Args:
            operation: KV cache operation to intercept
            
        Returns:
            Result of the cache operation
        """
        start_time = time.time()
        
        with self._stats_lock:
            self._operation_stats["intercepted_operations"] += 1
        
        # Track operation start for performance monitoring
        if self.performance_monitor:
            self.performance_monitor.track_cache_operation("lookup", 0.0, True)  # Will be updated with actual timing
        
        # Check if system is healthy enough to handle operations
        if not self._is_system_healthy():
            with self._stats_lock:
                self._operation_stats["degraded_operations"] += 1
            
            # Track degraded operation
            if self.performance_monitor:
                duration_ms = (time.time() - start_time) * 1000
                self.performance_monitor.track_cache_operation("error", duration_ms, False)
            
            return KVCacheResult(success=True, cache_hit=False, error="System degraded")
        
        if not self._initialized or not self._cache_handler:
            # Graceful degradation - return success but no caching
            with self._stats_lock:
                self._operation_stats["degraded_operations"] += 1
            
            # Track degraded operation
            if self.performance_monitor:
                duration_ms = (time.time() - start_time) * 1000
                self.performance_monitor.track_cache_operation("miss", duration_ms, True)
            
            return KVCacheResult(success=True, cache_hit=False)
        
        try:
            # Use storage failure handler for cache operations
            if operation.operation_type == "retrieve":
                result = self._handle_retrieve_operation_with_retry(operation)
                
                # Track performance metrics
                if self.performance_monitor:
                    duration_ms = (time.time() - start_time) * 1000
                    if result.cache_hit:
                        # Check if it's a partial hit
                        is_partial = result.data and result.data.get("is_partial_hit", False)
                        operation_type = "partial_hit" if is_partial else "hit"
                        self.performance_monitor.track_cache_operation(operation_type, duration_ms, True)
                    else:
                        self.performance_monitor.track_cache_operation("miss", duration_ms, True)
                
                return result
                
            elif operation.operation_type == "store":
                result = self._handle_store_operation_with_retry(operation)
                
                # Track storage performance
                if self.performance_monitor:
                    duration_ms = (time.time() - start_time) * 1000
                    self.performance_monitor.track_cache_operation("storage", duration_ms, result.success)
                
                return result
                
            else:
                self.failure_detector.record_error(
                    ComponentType.INTEGRATION_LAYER,
                    ErrorSeverity.LOW,
                    "unknown_operation_type",
                    f"Unknown operation type: {operation.operation_type}"
                )
                
                # Track error
                if self.performance_monitor:
                    duration_ms = (time.time() - start_time) * 1000
                    self.performance_monitor.track_cache_operation("error", duration_ms, False)
                
                return KVCacheResult(success=False, error=f"Unknown operation type: {operation.operation_type}")
                
        except Exception as e:
            self.failure_detector.record_error(
                ComponentType.INTEGRATION_LAYER,
                ErrorSeverity.MEDIUM,
                "kv_operation_failure",
                str(e),
                {"operation_type": operation.operation_type}
            )
            
            logger.warning(f"KV cache operation failed: {e}")
            with self._stats_lock:
                self._operation_stats["failed_operations"] += 1
            
            # Track error
            if self.performance_monitor:
                duration_ms = (time.time() - start_time) * 1000
                self.performance_monitor.track_cache_operation("error", duration_ms, False)
            
            if self._graceful_degradation_enabled:
                return KVCacheResult(success=True, cache_hit=False, error=str(e))
            else:
                return KVCacheResult(success=False, error=str(e))
    
    def _is_system_healthy(self) -> bool:
        """Check if the system is healthy enough to handle operations."""
        # Check critical components
        critical_components = [
            ComponentType.LMCACHE_ENGINE,
            ComponentType.INTEGRATION_LAYER
        ]
        
        for component in critical_components:
            if not self.failure_detector.is_component_healthy(component):
                return False
        
        return True
    
    def _handle_retrieve_operation_with_retry(self, operation: KVCacheOperation) -> KVCacheResult:
        """Handle cache retrieval operation with retry logic."""
        try:
            cache_key = operation.get_cache_key()
            if not cache_key:
                return KVCacheResult(success=False, error="Failed to generate cache key")
            
            # Use storage failure handler for retrieval
            def retrieve_operation():
                return self._cache_handler.handle_cache_lookup(cache_key)
            
            is_hit, cached_data, hit_length = self.storage_failure_handler.handle_storage_operation(
                "get", retrieve_operation
            )
            
            with self._stats_lock:
                self._operation_stats["successful_operations"] += 1
            
            return KVCacheResult(
                success=True,
                cache_hit=is_hit,
                data={
                    "cached_data": cached_data,
                    "hit_length": hit_length,
                    "is_partial_hit": is_hit and hit_length < len(self._cache_handler.key_manager.get_tokens_from_key(cache_key) or [])
                }
            )
            
        except Exception as e:
            self.failure_detector.record_error(
                ComponentType.CACHE_HANDLER,
                ErrorSeverity.MEDIUM,
                "retrieve_operation_failure",
                str(e)
            )
            raise e
    
    def _handle_store_operation_with_retry(self, operation: KVCacheOperation) -> KVCacheResult:
        """Handle cache storage operation with retry logic."""
        try:
            cache_key = operation.get_cache_key()
            if not cache_key:
                return KVCacheResult(success=False, error="Failed to generate cache key")
            
            kv_data = operation.kwargs.get("kv_data")
            if kv_data is None:
                return KVCacheResult(success=False, error="No KV data provided for storage")
            
            # Use storage failure handler for storage
            def store_operation():
                return self._cache_handler.handle_cache_storage(cache_key, kv_data)
            
            success = self.storage_failure_handler.handle_storage_operation(
                "put", store_operation
            )
            
            if success:
                with self._stats_lock:
                    self._operation_stats["successful_operations"] += 1
            else:
                with self._stats_lock:
                    self._operation_stats["failed_operations"] += 1
            
            return KVCacheResult(
                success=success,
                cache_hit=False,  # Storage operations don't have cache hits
                error=None if success else "Storage operation failed"
            )
            
        except Exception as e:
            self.failure_detector.record_error(
                ComponentType.CACHE_HANDLER,
                ErrorSeverity.MEDIUM,
                "store_operation_failure",
                str(e)
            )
            raise e
    
    def _handle_retrieve_operation(self, cache_key: CacheEngineKey) -> KVCacheResult:
        """Handle cache retrieval operation."""
        is_hit, cached_data, hit_length = self._cache_handler.handle_cache_lookup(cache_key)
        
        with self._stats_lock:
            self._operation_stats["successful_operations"] += 1
        
        return KVCacheResult(
            success=True,
            cache_hit=is_hit,
            data={
                "cached_data": cached_data,
                "hit_length": hit_length,
                "is_partial_hit": is_hit and hit_length < len(self._cache_handler.key_manager.get_tokens_from_key(cache_key) or [])
            }
        )
    
    def _handle_store_operation(self, cache_key: CacheEngineKey, kv_data: Any) -> KVCacheResult:
        """Handle cache storage operation."""
        success = self._cache_handler.handle_cache_storage(cache_key, kv_data)
        
        if success:
            with self._stats_lock:
                self._operation_stats["successful_operations"] += 1
        else:
            with self._stats_lock:
                self._operation_stats["failed_operations"] += 1
        
        return KVCacheResult(
            success=success,
            cache_hit=False,  # Storage operations don't have cache hits
            error=None if success else "Storage operation failed"
        )
    
    def handle_cache_miss(self, tokens: List[int]) -> Optional[Any]:
        """
        Handle cache miss scenario.
        
        Args:
            tokens: Token sequence that missed in cache
            
        Returns:
            None (cache miss means no cached data available)
        """
        if not self._initialized or not LMCACHE_AVAILABLE:
            return None
        
        try:
            cache_key = self._cache_handler.key_manager.create_cache_key(tokens)
            is_hit, cached_data, hit_length = self._cache_handler.handle_cache_lookup(cache_key)
            
            if is_hit and hit_length > 0:
                # Partial hit - return available cached data
                logger.debug(f"Partial cache hit: {hit_length}/{len(tokens)} tokens")
                return cached_data
            
            # Complete miss
            logger.debug(f"Cache miss for {len(tokens)} tokens")
            return None
            
        except Exception as e:
            logger.warning(f"Cache miss handling failed: {e}")
            return None
    
    def store_kv_cache(self, tokens: List[int], kv_data: Any) -> bool:
        """
        Store KV cache data after request completion.
        
        Args:
            tokens: Token sequence for the cache
            kv_data: KV cache data to store
            
        Returns:
            True if storage succeeded, False otherwise
        """
        if not self._initialized or not LMCACHE_AVAILABLE:
            return False
        
        try:
            cache_key = self._cache_handler.key_manager.create_cache_key(tokens)
            return self._cache_handler.handle_cache_storage(cache_key, kv_data)
            
        except Exception as e:
            logger.warning(f"KV cache storage failed: {e}")
            return False
    
    def enable_graceful_degradation(self, enabled: bool = True):
        """Enable or disable graceful degradation mode."""
        self._graceful_degradation_enabled = enabled
        logger.info(f"Graceful degradation {'enabled' if enabled else 'disabled'}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        stats = {}
        
        # Integration-level stats
        with self._stats_lock:
            stats["integration"] = self._operation_stats.copy()
        
        # Cache handler stats
        if self._cache_handler:
            stats["cache_handler"] = self._cache_handler.get_stats()
        
        # LMCache engine stats (if available)
        if self._lmcache_engine and hasattr(self._lmcache_engine, 'get_stats'):
            try:
                stats["lmcache_engine"] = self._lmcache_engine.get_stats()
            except Exception as e:
                logger.warning(f"Failed to get LMCache engine stats: {e}")
        
        return stats
    
    def optimize_cache_performance(self):
        """Optimize cache performance based on recent usage patterns."""
        if not self._cache_handler:
            return
        
        try:
            # Run cache handler optimization
            self._cache_handler.optimize_lookup_settings()
            
            # Log optimization results
            performance_metrics = self._cache_handler.get_performance_metrics()
            if "config" in performance_metrics:
                config = performance_metrics["config"]
                logger.info(f"Cache optimization completed: "
                          f"min_prefix_length={config['min_prefix_length']}, "
                          f"prefix_matching={config['enable_prefix_matching']}")
            
        except Exception as e:
            logger.warning(f"Cache optimization failed: {e}")
    
    def configure_cache_behavior(self, **kwargs):
        """Configure cache behavior parameters."""
        if not self._cache_handler:
            logger.warning("Cache handler not initialized")
            return
        
        # Update cache handler settings
        if "enable_prefix_matching" in kwargs:
            self._cache_handler.enable_prefix_matching = kwargs["enable_prefix_matching"]
            logger.info(f"Prefix matching {'enabled' if kwargs['enable_prefix_matching'] else 'disabled'}")
        
        if "min_prefix_length" in kwargs:
            self._cache_handler.min_prefix_length = max(16, kwargs["min_prefix_length"])
            logger.info(f"Min prefix length set to {self._cache_handler.min_prefix_length}")
        
        if "max_lookup_attempts" in kwargs:
            self._cache_handler.max_lookup_attempts = max(1, kwargs["max_lookup_attempts"])
            logger.info(f"Max lookup attempts set to {self._cache_handler.max_lookup_attempts}")
    
    def force_cache_cleanup(self) -> Dict[str, Any]:
        """Force cleanup of cache resources and return cleanup stats."""
        cleanup_stats = {
            "cache_handler_reset": False,
            "lmcache_engine_cleanup": False,
            "stats_reset": False
        }
        
        try:
            # Reset cache handler stats
            if self._cache_handler:
                with self._cache_handler.stats_lock:
                    self._cache_handler.stats = {
                        "total_requests": 0,
                        "cache_hits": 0,
                        "cache_misses": 0,
                        "partial_hits": 0,
                        "storage_operations": 0,
                        "errors": 0,
                        "lookup_latency_ms": [],
                        "storage_latency_ms": []
                    }
                    self._cache_handler.recent_lookups = []
                cleanup_stats["cache_handler_reset"] = True
            
            # Cleanup LMCache engine if possible
            if self._lmcache_engine and hasattr(self._lmcache_engine, 'cleanup'):
                self._lmcache_engine.cleanup()
                cleanup_stats["lmcache_engine_cleanup"] = True
            
            # Reset integration stats
            with self._stats_lock:
                self._operation_stats = {
                    "intercepted_operations": 0,
                    "successful_operations": 0,
                    "failed_operations": 0,
                    "degraded_operations": 0
                }
            cleanup_stats["stats_reset"] = True
            
            logger.info("Cache cleanup completed successfully")
            
        except Exception as e:
            logger.warning(f"Cache cleanup failed: {e}")
            cleanup_stats["error"] = str(e)
        
        return cleanup_stats
    
    def _setup_lmcache_environment(self, config: Dict[str, Any]) -> None:
        """Set up LMCache environment variables and configuration."""
        # Set config path if not already set
        if not os.environ.get("LMCACHE_CONFIG_PATH"):
            config_path = self.config_manager._find_config_file()
            if config_path:
                os.environ["LMCACHE_CONFIG_PATH"] = config_path
                logger.info(f"Set LMCACHE_CONFIG_PATH to: {config_path}")
        
        # Log Neuron runtime information
        runtime_info = self.device_detector.get_neuron_runtime_info()
        logger.info(f"Neuron runtime info: {runtime_info}")
    
    def _configure_vllm_integration(self, vllm_config: "VllmConfig", config: Dict[str, Any]) -> None:
        """Configure vLLM KV transfer settings for LMCache integration."""
        # Enable KV transfer if not already configured
        if vllm_config.kv_transfer_config is None:
            logger.info("Enabling KV transfer for LMCache integration")
            # Import here to avoid circular imports
            from vllm.config import KVTransferConfig
            
            # Build connector extra config from LMCache config
            connector_extra_config = {
                "lmcache.chunk_size": config.get("chunk_size", 256),
                "lmcache.save_decode_cache": config.get("save_decode_cache", True),
                "lmcache.local_cpu": config.get("local_cpu", True),
                "discard_partial_chunks": not config.get("save_unfull_chunk", True),
            }
            
            # Add extra config items
            extra_config = config.get("extra_config", {})
            for key, value in extra_config.items():
                connector_extra_config[f"lmcache.{key}"] = value
            
            # Configure KV transfer for LMCache
            vllm_config.kv_transfer_config = KVTransferConfig(
                kv_connector="lmcache",
                kv_role="kv_both",  # Can both produce and consume KV cache
                kv_rank=0,
                kv_parallel_size=1,
                kv_buffer_size=1024 * 1024 * 1024,  # 1GB buffer
                kv_connector_extra_config=connector_extra_config
            )
            logger.info("KV transfer configured for LMCache")
        else:
            logger.info("KV transfer already configured")
    
    def _apply_neuron_optimizations(self, vllm_config: "VllmConfig", config: Dict[str, Any]) -> None:
        """Apply Neuron-specific optimizations for LMCache."""
        if vllm_config.kv_transfer_config and vllm_config.kv_transfer_config.kv_connector_extra_config:
            extra_config = vllm_config.kv_transfer_config.kv_connector_extra_config
            
            # Force CPU storage for Neuron
            extra_config["lmcache.local_cpu"] = True
            extra_config["lmcache.use_gpu_connector_v3"] = False
            
            # Optimize for Neuron memory patterns
            max_cpu_size = config.get("max_local_cpu_size", 4)
            extra_config["lmcache.max_local_cpu_size"] = max_cpu_size
            extra_config["lmcache.enable_async_loading"] = config.get("extra_config", {}).get("enable_async_loading", True)
            
            logger.info("Applied Neuron-specific LMCache optimizations")
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status of the LMCache integration."""
        # Get base health status
        health_status = {
            "initialized": self._initialized,
            "lmcache_available": self.is_lmcache_available(),
            "neuron_devices": self.device_detector.get_neuron_runtime_info(),
            "config_loaded": self.config_manager._config_cache is not None,
            "graceful_degradation_enabled": self._graceful_degradation_enabled,
            "auto_recovery_enabled": self.auto_recovery_enabled,
            "recovery_thread_running": (
                self.recovery_thread is not None and 
                self.recovery_thread.is_alive()
            ),
        }
        
        # Add component health status
        health_status["component_health"] = self.failure_detector.get_component_status()
        
        # Add recent errors
        recent_errors = self.failure_detector.get_recent_errors(5)
        health_status["recent_errors"] = [
            {
                "component": error.component.value,
                "severity": error.severity.value,
                "error_type": error.error_type,
                "message": error.message,
                "timestamp": error.timestamp,
                "resolved": error.resolved
            }
            for error in recent_errors
        ]
        
        # Add LMCache engine status
        if self._lmcache_engine:
            health_status["lmcache_engine_initialized"] = True
            try:
                # Try to get engine health if available
                if hasattr(self._lmcache_engine, 'get_health'):
                    health_status["lmcache_engine_health"] = self._lmcache_engine.get_health()
            except Exception as e:
                health_status["lmcache_engine_error"] = str(e)
        else:
            health_status["lmcache_engine_initialized"] = False
        
        # Add cache handler status
        if self._cache_handler:
            health_status["cache_handler_initialized"] = True
            health_status["cache_stats"] = self._cache_handler.get_stats()
        else:
            health_status["cache_handler_initialized"] = False
        
        # Add storage backend status
        health_status["storage_backend_status"] = self.storage_failure_handler.get_fallback_stats()
        
        # Add operation statistics
        with self._stats_lock:
            health_status["operation_stats"] = self._operation_stats.copy()
        
        # Add performance monitoring status
        if self.performance_monitor:
            health_status["performance_monitor"] = {
                "initialized": True,
                "monitoring_enabled": self.performance_monitor.monitoring_enabled,
                "current_metrics": self.performance_monitor.get_current_metrics().__dict__,
                "performance_summary": self.performance_monitor.get_performance_summary(),
            }
        else:
            health_status["performance_monitor"] = {"initialized": False}
        
        # Add health endpoint status
        if self.health_endpoint:
            endpoint_health = self.health_endpoint.get_health_status()
            health_status["health_endpoint"] = {
                "initialized": True,
                "overall_healthy": endpoint_health["overall_healthy"],
                "active_alerts_count": len(endpoint_health["active_alerts"]),
                "component_status": endpoint_health["component_status"],
                "recent_alerts": endpoint_health["recent_alerts"][-3:],  # Last 3 alerts
            }
        else:
            health_status["health_endpoint"] = {"initialized": False}
        
        # Calculate overall health score
        health_status["overall_healthy"] = self._calculate_overall_health()
        
        return health_status
    
    def _calculate_overall_health(self) -> bool:
        """Calculate overall system health based on component status."""
        # Check critical components
        critical_components = [
            ComponentType.LMCACHE_ENGINE,
            ComponentType.INTEGRATION_LAYER
        ]
        
        for component in critical_components:
            if not self.failure_detector.is_component_healthy(component):
                return False
        
        # Check error rates
        with self._stats_lock:
            total_ops = self._operation_stats["intercepted_operations"]
            failed_ops = self._operation_stats["failed_operations"]
            
            if total_ops > 0:
                error_rate = failed_ops / total_ops
                if error_rate > 0.1:  # More than 10% error rate
                    return False
        
        return True
    
    def shutdown(self):
        """Shutdown the integration and cleanup resources."""
        logger.info("Shutting down LMCache integration")
        
        try:
            # Stop recovery thread
            if self.recovery_thread and self.recovery_thread.is_alive():
                self.recovery_stop_event.set()
                self.recovery_thread.join(timeout=5.0)
                logger.info("Recovery thread stopped")
            
            # Shutdown health endpoint
            if self.health_endpoint:
                self.health_endpoint.shutdown()
                logger.info("Health endpoint shutdown")
            
            # Shutdown performance monitor
            if self.performance_monitor:
                self.performance_monitor.shutdown()
                logger.info("Performance monitor shutdown")
            
            # Cleanup cache handler
            if self._cache_handler:
                # Reset cache handler if it has cleanup methods
                if hasattr(self._cache_handler, 'cleanup'):
                    self._cache_handler.cleanup()
            
            # Cleanup LMCache engine
            if self._lmcache_engine:
                if hasattr(self._lmcache_engine, 'close'):
                    self._lmcache_engine.close()
                elif hasattr(self._lmcache_engine, 'shutdown'):
                    self._lmcache_engine.shutdown()
            
            # Reset state
            self._initialized = False
            self._lmcache_engine = None
            self._cache_handler = None
            self.performance_monitor = None
            self.health_endpoint = None
            
            logger.info("LMCache integration shutdown completed")
            
        except Exception as e:
            logger.warning(f"Error during shutdown: {e}")
    
    def force_component_recovery(self, component: ComponentType) -> bool:
        """Force recovery of a specific component."""
        try:
            if component == ComponentType.LMCACHE_ENGINE:
                config = self.config_manager.load_config()
                if self._initialize_lmcache_engine(config):
                    self.failure_detector.mark_component_recovered(component)
                    return True
            
            elif component == ComponentType.STORAGE_BACKEND:
                return self.storage_failure_handler.attempt_recovery()
            
            elif component == ComponentType.CACHE_HANDLER:
                if self._lmcache_engine:
                    self._cache_handler = CacheHitMissHandler(self._lmcache_engine)
                    self.failure_detector.mark_component_recovered(component)
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to recover component {component.value}: {e}")
            return False
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get a summary of recent errors and system status."""
        return {
            "component_status": self.failure_detector.get_component_status(),
            "recent_errors": [
                {
                    "component": error.component.value,
                    "severity": error.severity.value,
                    "error_type": error.error_type,
                    "message": error.message,
                    "timestamp": error.timestamp
                }
                for error in self.failure_detector.get_recent_errors(10)
            ],
            "storage_fallback": self.storage_failure_handler.get_fallback_stats(),
            "overall_healthy": self._calculate_overall_health()
        }


def integrate_lmcache_with_neuron(vllm_config: "VllmConfig") -> bool:
    """Main integration function to setup LMCache with vLLM-Neuron."""
    logger.info("Starting LMCache integration with vLLM-Neuron")
    
    integration = LMCacheNeuronIntegration()
    success = integration.initialize(vllm_config)
    
    if success:
        logger.info("LMCache integration with vLLM-Neuron completed successfully")
    else:
        logger.warning("LMCache integration with vLLM-Neuron failed or was skipped")
    
    return success