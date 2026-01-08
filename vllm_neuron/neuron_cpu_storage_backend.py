# SPDX-License-Identifier: Apache-2.0
"""Neuron-optimized CPU storage backend for LMCache integration."""

import logging
import os
import threading
import time
import multiprocessing
import psutil
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Union, Tuple

import torch

# Import LMCache components
try:
    from lmcache.config import LMCacheEngineMetadata
    from lmcache.utils import CacheEngineKey
    from lmcache.v1.config import LMCacheEngineConfig
    from lmcache.v1.memory_management import (
        MemoryAllocatorInterface,
        MemoryFormat,
        MemoryObj,
        MixedMemoryAllocator,
    )
    from lmcache.v1.storage_backend.abstract_backend import AllocatorBackendInterface
    from lmcache.v1.system_detection import NUMADetector
    LMCACHE_AVAILABLE = True
except ImportError:
    # Fallback types when LMCache is not available
    LMCACHE_AVAILABLE = False
    CacheEngineKey = Any
    MemoryObj = Any
    LMCacheEngineConfig = Any
    LMCacheEngineMetadata = Any
    MemoryAllocatorInterface = Any
    AllocatorBackendInterface = object
    MemoryFormat = Any

if TYPE_CHECKING and LMCACHE_AVAILABLE:
    pass

logger = logging.getLogger(__name__)


class ChunkMetadata:
    """Metadata for cached chunks to support advanced eviction policies."""
    
    def __init__(self, key: CacheEngineKey, memory_obj: MemoryObj, size_bytes: int):
        self.key = key
        self.memory_obj = memory_obj
        self.size_bytes = size_bytes
        self.access_count = 0
        self.last_access_time = time.time()
        self.creation_time = time.time()
        self.is_pinned = False
    
    def update_access(self):
        """Update access statistics."""
        self.access_count += 1
        self.last_access_time = time.time()
    
    def get_age_seconds(self) -> float:
        """Get age of chunk in seconds."""
        return time.time() - self.creation_time
    
    def get_idle_time_seconds(self) -> float:
        """Get time since last access in seconds."""
        return time.time() - self.last_access_time
    
    def can_evict(self) -> bool:
        """Check if chunk can be evicted."""
        return not self.is_pinned and self.memory_obj.can_evict


class MemoryPressureMonitor:
    """Monitor memory pressure and trigger eviction policies."""
    
    def __init__(self, max_memory_bytes: int):
        self.max_memory_bytes = max_memory_bytes
        self.current_memory_bytes = 0
        self.lock = threading.Lock()
        
        # Pressure thresholds
        self.low_pressure_threshold = 0.70   # 70% - start gentle eviction
        self.medium_pressure_threshold = 0.85  # 85% - aggressive eviction
        self.high_pressure_threshold = 0.95   # 95% - emergency eviction
        self.critical_pressure_threshold = 0.98  # 98% - disable caching temporarily
        
        # Eviction batch sizes for different pressure levels
        self.low_pressure_batch_size = 5
        self.medium_pressure_batch_size = 15
        self.high_pressure_batch_size = 30
        self.critical_pressure_batch_size = 50
        
        # Memory pressure handling settings
        self.cache_disabled = False
        self.disable_cache_duration = 60.0  # seconds to keep cache disabled
        self.cache_disabled_until = 0.0
        self.pressure_history = []  # Track pressure over time
        self.max_history_size = 100
        
        # Adaptive thresholds
        self.adaptive_thresholds = True
        self.threshold_adjustment_factor = 0.05
        
        logger.info(f"MemoryPressureMonitor initialized with {max_memory_bytes} bytes limit")
    
    def update_memory_usage(self, delta_bytes: int):
        """Update current memory usage."""
        with self.lock:
            self.current_memory_bytes += delta_bytes
            self.current_memory_bytes = max(0, self.current_memory_bytes)
            
            # Record pressure history
            current_ratio = self.get_memory_usage_ratio()
            self.pressure_history.append({
                "timestamp": time.time(),
                "ratio": current_ratio,
                "delta": delta_bytes
            })
            
            # Limit history size
            if len(self.pressure_history) > self.max_history_size:
                self.pressure_history = self.pressure_history[-self.max_history_size:]
            
            # Check if we need to disable caching temporarily
            if current_ratio >= self.critical_pressure_threshold and not self.cache_disabled:
                self._disable_cache_temporarily()
            elif current_ratio < self.low_pressure_threshold and self.cache_disabled:
                self._check_cache_re_enable()
    
    def _disable_cache_temporarily(self):
        """Temporarily disable caching due to critical memory pressure."""
        self.cache_disabled = True
        self.cache_disabled_until = time.time() + self.disable_cache_duration
        logger.warning(f"Caching temporarily disabled due to critical memory pressure "
                      f"({self.get_memory_usage_ratio():.1%} usage)")
    
    def _check_cache_re_enable(self):
        """Check if caching can be re-enabled."""
        current_time = time.time()
        if (self.cache_disabled and 
            current_time >= self.cache_disabled_until and 
            self.get_memory_usage_ratio() < self.low_pressure_threshold):
            
            self.cache_disabled = False
            logger.info("Caching re-enabled after memory pressure relief")
    
    def is_caching_disabled(self) -> bool:
        """Check if caching is temporarily disabled."""
        if self.cache_disabled:
            self._check_cache_re_enable()
        return self.cache_disabled
    
    def get_memory_usage_ratio(self) -> float:
        """Get current memory usage ratio."""
        # Don't acquire lock here since this is called from update_memory_usage which already holds the lock
        if self.max_memory_bytes <= 0:
            return 0.0
        return self.current_memory_bytes / self.max_memory_bytes
    
    def get_pressure_level(self) -> str:
        """Get current memory pressure level."""
        # Use internal method to avoid lock issues
        ratio = self.current_memory_bytes / self.max_memory_bytes if self.max_memory_bytes > 0 else 0.0
        if ratio >= self.critical_pressure_threshold:
            return "critical"
        elif ratio >= self.high_pressure_threshold:
            return "high"
        elif ratio >= self.medium_pressure_threshold:
            return "medium"
        elif ratio >= self.low_pressure_threshold:
            return "low"
        else:
            return "none"
    
    def get_eviction_batch_size(self) -> int:
        """Get recommended eviction batch size based on pressure."""
        pressure = self.get_pressure_level()
        if pressure == "critical":
            return self.critical_pressure_batch_size
        elif pressure == "high":
            return self.high_pressure_batch_size
        elif pressure == "medium":
            return self.medium_pressure_batch_size
        elif pressure == "low":
            return self.low_pressure_batch_size
        else:
            return 0
    
    def should_evict(self) -> bool:
        """Check if eviction should be triggered."""
        return self.get_pressure_level() != "none"
    
    def should_reject_new_allocations(self) -> bool:
        """Check if new allocations should be rejected."""
        return self.get_pressure_level() in ["critical", "high"]
    
    def get_pressure_trend(self) -> str:
        """Get memory pressure trend over recent history."""
        if len(self.pressure_history) < 10:
            return "stable"
        
        recent_ratios = [entry["ratio"] for entry in self.pressure_history[-10:]]
        
        # Calculate trend
        if len(recent_ratios) >= 2:
            trend = recent_ratios[-1] - recent_ratios[0]
            if trend > 0.05:  # 5% increase
                return "increasing"
            elif trend < -0.05:  # 5% decrease
                return "decreasing"
        
        return "stable"
    
    def adapt_thresholds_based_on_workload(self):
        """Adapt pressure thresholds based on workload patterns."""
        if not self.adaptive_thresholds or len(self.pressure_history) < 50:
            return
        
        try:
            # Analyze recent pressure patterns
            recent_entries = self.pressure_history[-50:]
            avg_ratio = sum(entry["ratio"] for entry in recent_entries) / len(recent_entries)
            
            # If we're consistently running at high memory usage, adjust thresholds
            if avg_ratio > 0.8:  # Consistently high usage
                # Increase thresholds slightly to be more aggressive
                self.low_pressure_threshold = min(0.75, self.low_pressure_threshold + self.threshold_adjustment_factor)
                self.medium_pressure_threshold = min(0.88, self.medium_pressure_threshold + self.threshold_adjustment_factor)
                self.high_pressure_threshold = min(0.97, self.high_pressure_threshold + self.threshold_adjustment_factor)
                
                logger.debug(f"Adapted thresholds for high-usage workload: "
                           f"low={self.low_pressure_threshold:.2f}, "
                           f"medium={self.medium_pressure_threshold:.2f}, "
                           f"high={self.high_pressure_threshold:.2f}")
            
            elif avg_ratio < 0.5:  # Consistently low usage
                # Decrease thresholds to be less aggressive
                self.low_pressure_threshold = max(0.65, self.low_pressure_threshold - self.threshold_adjustment_factor)
                self.medium_pressure_threshold = max(0.80, self.medium_pressure_threshold - self.threshold_adjustment_factor)
                self.high_pressure_threshold = max(0.90, self.high_pressure_threshold - self.threshold_adjustment_factor)
                
                logger.debug(f"Adapted thresholds for low-usage workload: "
                           f"low={self.low_pressure_threshold:.2f}, "
                           f"medium={self.medium_pressure_threshold:.2f}, "
                           f"high={self.high_pressure_threshold:.2f}")
        
        except Exception as e:
            logger.warning(f"Failed to adapt thresholds: {e}")
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics."""
        with self.lock:
            pressure_level = self.get_pressure_level()
            trend = self.get_pressure_trend()
            
            stats = {
                "current_bytes": self.current_memory_bytes,
                "max_bytes": self.max_memory_bytes,
                "usage_ratio": self.get_memory_usage_ratio(),
                "pressure_level": pressure_level,
                "pressure_trend": trend,
                "cache_disabled": self.cache_disabled,
                "cache_disabled_until": self.cache_disabled_until,
                "thresholds": {
                    "low": self.low_pressure_threshold,
                    "medium": self.medium_pressure_threshold,
                    "high": self.high_pressure_threshold,
                    "critical": self.critical_pressure_threshold
                },
                "batch_sizes": {
                    "low": self.low_pressure_batch_size,
                    "medium": self.medium_pressure_batch_size,
                    "high": self.high_pressure_batch_size,
                    "critical": self.critical_pressure_batch_size
                },
                "adaptive_thresholds": self.adaptive_thresholds
            }
            
            # Add recent pressure history
            if self.pressure_history:
                recent_history = self.pressure_history[-10:]
                stats["recent_history"] = [
                    {
                        "timestamp": entry["timestamp"],
                        "ratio": entry["ratio"],
                        "delta": entry["delta"]
                    }
                    for entry in recent_history
                ]
            
            return stats
    
    def force_cache_disable(self, duration_seconds: float = None):
        """Force disable caching for a specified duration."""
        if duration_seconds is None:
            duration_seconds = self.disable_cache_duration
        
        self.cache_disabled = True
        self.cache_disabled_until = time.time() + duration_seconds
        logger.warning(f"Caching forcibly disabled for {duration_seconds} seconds")
    
    def force_cache_enable(self):
        """Force enable caching regardless of memory pressure."""
        self.cache_disabled = False
        self.cache_disabled_until = 0.0
        logger.info("Caching forcibly enabled")
    
    def reset_adaptive_thresholds(self):
        """Reset thresholds to default values."""
        self.low_pressure_threshold = 0.70
        self.medium_pressure_threshold = 0.85
        self.high_pressure_threshold = 0.95
        self.critical_pressure_threshold = 0.98
        logger.info("Memory pressure thresholds reset to defaults")


class ResourceCleanupManager:
    """Manages resource cleanup during memory pressure."""
    
    def __init__(self, storage_backend):
        self.storage_backend = storage_backend
        self.cleanup_strategies = [
            self._cleanup_expired_chunks,
            self._cleanup_large_unused_chunks,
            self._cleanup_duplicate_chunks,
            self._cleanup_old_chunks,
            self._emergency_cleanup
        ]
        self.cleanup_stats = {
            "total_cleanups": 0,
            "bytes_freed": 0,
            "chunks_removed": 0,
            "last_cleanup": 0.0
        }
        self.lock = threading.Lock()
        
        # Cleanup thresholds
        self.min_cleanup_interval = 5.0  # seconds
        self.chunk_expiry_time = 3600.0  # 1 hour
        self.large_chunk_threshold = 10 * 1024 * 1024  # 10MB
        self.duplicate_detection_enabled = True
        
        logger.info("ResourceCleanupManager initialized")
    
    def perform_cleanup(self, pressure_level: str) -> Dict[str, Any]:
        """Perform resource cleanup based on pressure level."""
        with self.lock:
            current_time = time.time()
            
            # Check minimum cleanup interval
            if current_time - self.cleanup_stats["last_cleanup"] < self.min_cleanup_interval:
                return {"skipped": True, "reason": "too_soon"}
            
            cleanup_result = {
                "pressure_level": pressure_level,
                "strategies_used": [],
                "bytes_freed": 0,
                "chunks_removed": 0,
                "success": False
            }
            
            try:
                # Apply cleanup strategies based on pressure level
                if pressure_level in ["low", "medium"]:
                    # Gentle cleanup
                    strategies = self.cleanup_strategies[:2]
                elif pressure_level == "high":
                    # Aggressive cleanup
                    strategies = self.cleanup_strategies[:4]
                else:  # critical
                    # Emergency cleanup
                    strategies = self.cleanup_strategies
                
                for strategy in strategies:
                    try:
                        result = strategy()
                        if result["chunks_removed"] > 0:
                            cleanup_result["strategies_used"].append(result["strategy"])
                            cleanup_result["bytes_freed"] += result["bytes_freed"]
                            cleanup_result["chunks_removed"] += result["chunks_removed"]
                    except Exception as e:
                        logger.warning(f"Cleanup strategy failed: {e}")
                
                # Update stats
                self.cleanup_stats["total_cleanups"] += 1
                self.cleanup_stats["bytes_freed"] += cleanup_result["bytes_freed"]
                self.cleanup_stats["chunks_removed"] += cleanup_result["chunks_removed"]
                self.cleanup_stats["last_cleanup"] = current_time
                
                cleanup_result["success"] = cleanup_result["chunks_removed"] > 0
                
                if cleanup_result["success"]:
                    logger.info(f"Cleanup completed: freed {cleanup_result['bytes_freed']} bytes, "
                              f"removed {cleanup_result['chunks_removed']} chunks")
                
                return cleanup_result
                
            except Exception as e:
                logger.error(f"Cleanup failed: {e}")
                cleanup_result["error"] = str(e)
                return cleanup_result
    
    def _cleanup_expired_chunks(self) -> Dict[str, Any]:
        """Remove chunks that have expired based on age."""
        result = {"strategy": "expired_chunks", "chunks_removed": 0, "bytes_freed": 0}
        
        current_time = time.time()
        expired_keys = []
        
        with self.storage_backend.storage_lock:
            for key, chunk_meta in self.storage_backend.chunk_metadata.items():
                if (chunk_meta.can_evict() and 
                    chunk_meta.get_age_seconds() > self.chunk_expiry_time):
                    expired_keys.append(key)
        
        # Remove expired chunks
        for key in expired_keys:
            if self.storage_backend._remove_chunk_internal(key):
                result["chunks_removed"] += 1
                # Estimate bytes freed (we don't have exact size here)
                result["bytes_freed"] += 1024 * 1024  # Estimate 1MB per chunk
        
        return result
    
    def _cleanup_large_unused_chunks(self) -> Dict[str, Any]:
        """Remove large chunks that haven't been accessed recently."""
        result = {"strategy": "large_unused_chunks", "chunks_removed": 0, "bytes_freed": 0}
        
        large_unused_keys = []
        
        with self.storage_backend.storage_lock:
            for key, chunk_meta in self.storage_backend.chunk_metadata.items():
                if (chunk_meta.can_evict() and 
                    chunk_meta.size_bytes > self.large_chunk_threshold and
                    chunk_meta.get_idle_time_seconds() > 300):  # 5 minutes idle
                    large_unused_keys.append((key, chunk_meta.size_bytes))
        
        # Sort by size (largest first)
        large_unused_keys.sort(key=lambda x: x[1], reverse=True)
        
        # Remove up to 5 large unused chunks
        for key, size in large_unused_keys[:5]:
            if self.storage_backend._remove_chunk_internal(key):
                result["chunks_removed"] += 1
                result["bytes_freed"] += size
        
        return result
    
    def _cleanup_duplicate_chunks(self) -> Dict[str, Any]:
        """Remove duplicate or similar chunks."""
        result = {"strategy": "duplicate_chunks", "chunks_removed": 0, "bytes_freed": 0}
        
        if not self.duplicate_detection_enabled:
            return result
        
        # Simple duplicate detection based on size and access patterns
        size_groups = {}
        
        with self.storage_backend.storage_lock:
            for key, chunk_meta in self.storage_backend.chunk_metadata.items():
                if chunk_meta.can_evict():
                    size = chunk_meta.size_bytes
                    if size not in size_groups:
                        size_groups[size] = []
                    size_groups[size].append((key, chunk_meta))
        
        # Look for groups with multiple chunks of the same size
        for size, chunks in size_groups.items():
            if len(chunks) > 1:
                # Sort by access count (keep most accessed)
                chunks.sort(key=lambda x: x[1].access_count, reverse=True)
                
                # Remove all but the most accessed chunk
                for key, chunk_meta in chunks[1:]:
                    if self.storage_backend._remove_chunk_internal(key):
                        result["chunks_removed"] += 1
                        result["bytes_freed"] += chunk_meta.size_bytes
        
        return result
    
    def _cleanup_old_chunks(self) -> Dict[str, Any]:
        """Remove old chunks based on LRU policy."""
        result = {"strategy": "old_chunks", "chunks_removed": 0, "bytes_freed": 0}
        
        # Use existing eviction policy
        eviction_candidates = self.storage_backend.eviction_policy.select_eviction_candidates(
            self.storage_backend.chunk_metadata, 10  # Remove up to 10 old chunks
        )
        
        for key in eviction_candidates:
            chunk_meta = self.storage_backend.chunk_metadata.get(key)
            if chunk_meta and self.storage_backend._remove_chunk_internal(key):
                result["chunks_removed"] += 1
                result["bytes_freed"] += chunk_meta.size_bytes
        
        return result
    
    def _emergency_cleanup(self) -> Dict[str, Any]:
        """Emergency cleanup - remove a significant portion of cache."""
        result = {"strategy": "emergency_cleanup", "chunks_removed": 0, "bytes_freed": 0}
        
        # Remove 50% of all evictable chunks
        evictable_keys = []
        
        with self.storage_backend.storage_lock:
            for key, chunk_meta in self.storage_backend.chunk_metadata.items():
                if chunk_meta.can_evict():
                    evictable_keys.append((key, chunk_meta.last_access_time, chunk_meta.size_bytes))
        
        # Sort by last access time (oldest first)
        evictable_keys.sort(key=lambda x: x[1])
        
        # Remove half of the evictable chunks
        target_count = len(evictable_keys) // 2
        
        for key, _, size in evictable_keys[:target_count]:
            if self.storage_backend._remove_chunk_internal(key):
                result["chunks_removed"] += 1
                result["bytes_freed"] += size
        
        return result
    
    def get_cleanup_stats(self) -> Dict[str, Any]:
        """Get cleanup statistics."""
        with self.lock:
            return self.cleanup_stats.copy()
    
    def configure_cleanup(self, **kwargs):
        """Configure cleanup parameters."""
        if "chunk_expiry_time" in kwargs:
            self.chunk_expiry_time = max(300, kwargs["chunk_expiry_time"])  # Min 5 minutes
        
        if "large_chunk_threshold" in kwargs:
            self.large_chunk_threshold = max(1024*1024, kwargs["large_chunk_threshold"])  # Min 1MB
        
        if "duplicate_detection_enabled" in kwargs:
            self.duplicate_detection_enabled = kwargs["duplicate_detection_enabled"]
        
        if "min_cleanup_interval" in kwargs:
            self.min_cleanup_interval = max(1.0, kwargs["min_cleanup_interval"])  # Min 1 second
        
        logger.info(f"Cleanup configuration updated: "
                   f"expiry_time={self.chunk_expiry_time}, "
                   f"large_threshold={self.large_chunk_threshold}, "
                   f"duplicate_detection={self.duplicate_detection_enabled}")


class LRUEvictionPolicy:
    """Advanced LRU eviction policy with multiple strategies."""
    
    def __init__(self):
        self.strategy = "lru"  # Can be "lru", "lfu", "aging", or "hybrid"
    
    def select_eviction_candidates(
        self, 
        chunks: Dict[CacheEngineKey, ChunkMetadata], 
        target_count: int
    ) -> List[CacheEngineKey]:
        """Select chunks for eviction based on the current strategy."""
        evictable_chunks = [
            (key, metadata) for key, metadata in chunks.items() 
            if metadata.can_evict()
        ]
        
        if not evictable_chunks:
            return []
        
        if self.strategy == "lru":
            # Sort by last access time (oldest first)
            evictable_chunks.sort(key=lambda x: x[1].last_access_time)
        elif self.strategy == "lfu":
            # Sort by access count (least frequently used first)
            evictable_chunks.sort(key=lambda x: x[1].access_count)
        elif self.strategy == "aging":
            # Sort by age (oldest first)
            evictable_chunks.sort(key=lambda x: x[1].creation_time)
        elif self.strategy == "hybrid":
            # Hybrid strategy: consider both recency and frequency
            def hybrid_score(metadata):
                # Lower score = higher eviction priority
                idle_time = metadata.get_idle_time_seconds()
                access_frequency = metadata.access_count / max(1, metadata.get_age_seconds())
                return access_frequency / max(1, idle_time)
            
            evictable_chunks.sort(key=lambda x: hybrid_score(x[1]))
        
        # Return the keys of selected candidates
        selected_count = min(target_count, len(evictable_chunks))
        return [key for key, _ in evictable_chunks[:selected_count]]


class NeuronCoreDetector:
    """Detects and manages available Neuron cores for cache distribution."""
    
    def __init__(self):
        self.available_cores = self._detect_available_cores()
        self.core_count = len(self.available_cores)
        self.cpu_cores = multiprocessing.cpu_count()
        
        logger.info(f"Detected {self.core_count} Neuron cores: {self.available_cores}")
        logger.info(f"System has {self.cpu_cores} CPU cores")
    
    def _detect_available_cores(self) -> List[int]:
        """Detect available Neuron cores from environment variables."""
        cores = []
        
        # Check NEURON_RT_VISIBLE_CORES environment variable
        visible_cores = os.environ.get("NEURON_RT_VISIBLE_CORES")
        if visible_cores:
            try:
                if "-" in visible_cores:
                    # Range format: "0-7"
                    start, end = map(int, visible_cores.split("-"))
                    cores = list(range(start, end + 1))
                elif "," in visible_cores:
                    # Comma-separated format: "0,1,2,3"
                    cores = [int(x.strip()) for x in visible_cores.split(",")]
                else:
                    # Single core: "0"
                    cores = [int(visible_cores)]
            except ValueError as e:
                logger.warning(f"Failed to parse NEURON_RT_VISIBLE_CORES '{visible_cores}': {e}")
        
        # Fallback: assume single core if no environment variable
        if not cores:
            cores = [0]
            logger.info("No NEURON_RT_VISIBLE_CORES found, assuming single core")
        
        return cores
    
    def get_core_for_operation(self, operation_id: str) -> int:
        """Get the optimal Neuron core for a cache operation."""
        if not self.available_cores:
            return 0
        
        # Use hash-based distribution for consistent core assignment
        core_index = hash(operation_id) % len(self.available_cores)
        return self.available_cores[core_index]
    
    def get_all_cores(self) -> List[int]:
        """Get all available Neuron cores."""
        return self.available_cores.copy()
    
    def is_multi_core(self) -> bool:
        """Check if multiple Neuron cores are available."""
        return len(self.available_cores) > 1


class CoreAffinityManager:
    """Manages CPU core affinity for cache operations to optimize NUMA locality."""
    
    def __init__(self, neuron_cores: List[int]):
        self.neuron_cores = neuron_cores
        self.cpu_cores = multiprocessing.cpu_count()
        self.affinity_mapping = self._create_affinity_mapping()
        self.current_process_affinity = None
        
        # Try to get current process affinity
        try:
            current_process = psutil.Process()
            self.current_process_affinity = current_process.cpu_affinity()
            logger.debug(f"Current process CPU affinity: {self.current_process_affinity}")
        except (AttributeError, psutil.AccessDenied):
            logger.warning("Cannot access CPU affinity information")
    
    def _create_affinity_mapping(self) -> Dict[int, List[int]]:
        """Create mapping from Neuron cores to optimal CPU cores."""
        mapping = {}
        
        # Simple strategy: distribute CPU cores evenly across Neuron cores
        cpu_cores_per_neuron = max(1, self.cpu_cores // len(self.neuron_cores))
        
        for i, neuron_core in enumerate(self.neuron_cores):
            start_cpu = i * cpu_cores_per_neuron
            end_cpu = min(start_cpu + cpu_cores_per_neuron, self.cpu_cores)
            mapping[neuron_core] = list(range(start_cpu, end_cpu))
        
        logger.debug(f"Created CPU affinity mapping: {mapping}")
        return mapping
    
    def set_affinity_for_neuron_core(self, neuron_core: int) -> bool:
        """Set CPU affinity for operations on a specific Neuron core."""
        if neuron_core not in self.affinity_mapping:
            return False
        
        try:
            cpu_cores = self.affinity_mapping[neuron_core]
            current_process = psutil.Process()
            current_process.cpu_affinity(cpu_cores)
            logger.debug(f"Set CPU affinity to {cpu_cores} for Neuron core {neuron_core}")
            return True
        except (AttributeError, psutil.AccessDenied) as e:
            logger.warning(f"Failed to set CPU affinity: {e}")
            return False
    
    def restore_original_affinity(self) -> bool:
        """Restore original CPU affinity."""
        if self.current_process_affinity is None:
            return False
        
        try:
            current_process = psutil.Process()
            current_process.cpu_affinity(self.current_process_affinity)
            logger.debug(f"Restored original CPU affinity: {self.current_process_affinity}")
            return True
        except (AttributeError, psutil.AccessDenied) as e:
            logger.warning(f"Failed to restore CPU affinity: {e}")
            return False


class BlockAlignedMemoryManager:
    """Manages memory allocation aligned with Neuron's block-based memory model."""
    
    def __init__(self, block_size: int = 4096):
        self.block_size = block_size  # Neuron's typical block size
        self.alignment_stats = {
            "aligned_allocations": 0,
            "unaligned_allocations": 0,
            "total_padding_bytes": 0,
        }
        self.stats_lock = threading.Lock()
    
    def calculate_aligned_size(self, requested_size: int) -> int:
        """Calculate block-aligned size for a memory allocation."""
        if requested_size <= 0:
            return self.block_size
        
        # Round up to nearest block boundary
        aligned_size = ((requested_size + self.block_size - 1) // self.block_size) * self.block_size
        
        with self.stats_lock:
            if aligned_size == requested_size:
                self.alignment_stats["aligned_allocations"] += 1
            else:
                self.alignment_stats["unaligned_allocations"] += 1
                self.alignment_stats["total_padding_bytes"] += (aligned_size - requested_size)
        
        return aligned_size
    
    def is_aligned(self, size: int) -> bool:
        """Check if a size is block-aligned."""
        return size % self.block_size == 0
    
    def get_alignment_stats(self) -> Dict[str, int]:
        """Get memory alignment statistics."""
        with self.stats_lock:
            return self.alignment_stats.copy()
    
    def optimize_tensor_layout(self, tensor: torch.Tensor) -> torch.Tensor:
        """Optimize tensor memory layout for Neuron's block-based access patterns."""
        if not tensor.is_contiguous():
            # Make tensor contiguous for better block access
            tensor = tensor.contiguous()
        
        # For KV cache tensors, ensure optimal memory layout
        if tensor.dim() >= 2:
            # Transpose if it improves memory access patterns
            # This is a heuristic - in practice, you'd want to profile different layouts
            if tensor.shape[-1] % 64 == 0:  # Common Neuron optimization
                return tensor
        
        return tensor


class MultiCoreDistributor:
    """Distributes cache operations across multiple Neuron cores."""
    
    def __init__(self, max_workers: Optional[int] = None):
        self.core_detector = NeuronCoreDetector()
        self.affinity_manager = CoreAffinityManager(self.core_detector.get_all_cores())
        self.memory_manager = BlockAlignedMemoryManager()
        
        # Thread pool for parallel operations
        self.max_workers = max_workers or min(4, self.core_detector.core_count * 2)
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers, thread_name_prefix="NeuronCache")
        
        # Per-core operation queues for load balancing
        self.core_queues = {core: [] for core in self.core_detector.get_all_cores()}
        self.core_queue_locks = {core: threading.Lock() for core in self.core_detector.get_all_cores()}
        
        # Performance tracking
        self.operation_stats = {
            "operations_per_core": {core: 0 for core in self.core_detector.get_all_cores()},
            "average_latency_per_core": {core: 0.0 for core in self.core_detector.get_all_cores()},
            "total_operations": 0,
            "failed_operations": 0,
        }
        self.stats_lock = threading.Lock()
        
        logger.info(f"Initialized MultiCoreDistributor with {self.max_workers} workers "
                   f"for {self.core_detector.core_count} Neuron cores")
    
    def distribute_cache_operation(
        self, 
        operation_func: callable, 
        operation_id: str, 
        *args, 
        **kwargs
    ) -> Future:
        """
        Distribute a cache operation across available cores.
        
        Args:
            operation_func: Function to execute
            operation_id: Unique identifier for the operation
            *args, **kwargs: Arguments for the operation function
            
        Returns:
            Future object for the operation result
        """
        # Select optimal core for this operation
        target_core = self.core_detector.get_core_for_operation(operation_id)
        
        # Wrap the operation with core affinity management
        def wrapped_operation():
            start_time = time.time()
            
            try:
                # Set CPU affinity for this thread
                self.affinity_manager.set_affinity_for_neuron_core(target_core)
                
                # Execute the operation
                result = operation_func(*args, **kwargs)
                
                # Update statistics
                latency = time.time() - start_time
                self._update_operation_stats(target_core, latency, success=True)
                
                return result
                
            except Exception as e:
                # Update error statistics
                latency = time.time() - start_time
                self._update_operation_stats(target_core, latency, success=False)
                raise e
            
            finally:
                # Restore original affinity (optional, may impact performance)
                # self.affinity_manager.restore_original_affinity()
                pass
        
        # Submit to thread pool
        future = self.executor.submit(wrapped_operation)
        return future
    
    def distribute_batch_operation(
        self, 
        operation_func: callable, 
        operation_items: List[Tuple[str, Any]], 
        **kwargs
    ) -> List[Future]:
        """
        Distribute a batch of cache operations across cores.
        
        Args:
            operation_func: Function to execute for each item
            operation_items: List of (operation_id, item_data) tuples
            **kwargs: Common arguments for all operations
            
        Returns:
            List of Future objects for the operation results
        """
        futures = []
        
        for operation_id, item_data in operation_items:
            future = self.distribute_cache_operation(
                operation_func, 
                operation_id, 
                item_data, 
                **kwargs
            )
            futures.append(future)
        
        return futures
    
    def _update_operation_stats(self, core: int, latency: float, success: bool):
        """Update operation statistics for a core."""
        with self.stats_lock:
            if success:
                self.operation_stats["operations_per_core"][core] += 1
                
                # Update rolling average latency
                current_avg = self.operation_stats["average_latency_per_core"][core]
                current_count = self.operation_stats["operations_per_core"][core]
                new_avg = ((current_avg * (current_count - 1)) + latency) / current_count
                self.operation_stats["average_latency_per_core"][core] = new_avg
                
                self.operation_stats["total_operations"] += 1
            else:
                self.operation_stats["failed_operations"] += 1
    
    def get_load_balance_stats(self) -> Dict[str, Any]:
        """Get load balancing statistics across cores."""
        with self.stats_lock:
            stats = self.operation_stats.copy()
        
        # Add additional metrics
        total_ops = sum(stats["operations_per_core"].values())
        if total_ops > 0:
            stats["load_balance_ratio"] = {
                core: ops / total_ops 
                for core, ops in stats["operations_per_core"].items()
            }
        else:
            stats["load_balance_ratio"] = {core: 0.0 for core in stats["operations_per_core"]}
        
        stats["core_count"] = self.core_detector.core_count
        stats["worker_count"] = self.max_workers
        stats["alignment_stats"] = self.memory_manager.get_alignment_stats()
        
        return stats
    
    def optimize_memory_for_neuron(self, tensor: torch.Tensor) -> torch.Tensor:
        """Optimize tensor memory layout for Neuron cores."""
        return self.memory_manager.optimize_tensor_layout(tensor)
    
    def calculate_optimal_chunk_size(self, base_chunk_size: int) -> int:
        """Calculate optimal chunk size aligned with Neuron's block model."""
        return self.memory_manager.calculate_aligned_size(base_chunk_size)
    
    def shutdown(self):
        """Shutdown the multi-core distributor."""
        logger.info("Shutting down MultiCoreDistributor")
        self.executor.shutdown(wait=True)
        
        # Log final statistics
        stats = self.get_load_balance_stats()
        logger.info(f"Final load balance stats: {stats}")


class NeuronMemoryStats:
    """Enhanced statistics tracking for Neuron CPU memory usage."""
    
    def __init__(self):
        self.total_allocated_bytes = 0
        self.peak_allocated_bytes = 0
        self.current_cached_chunks = 0
        self.total_cache_hits = 0
        self.total_cache_misses = 0
        self.total_evictions = 0
        self.eviction_by_pressure = {"low": 0, "medium": 0, "high": 0}
        self.memory_pressure_events = 0
        self.failed_allocations = 0
        self.lock = threading.Lock()
    
    def update_allocation(self, bytes_allocated: int):
        with self.lock:
            self.total_allocated_bytes += bytes_allocated
            self.peak_allocated_bytes = max(self.peak_allocated_bytes, self.total_allocated_bytes)
    
    def update_deallocation(self, bytes_deallocated: int):
        with self.lock:
            self.total_allocated_bytes -= bytes_deallocated
    
    def record_cache_hit(self):
        with self.lock:
            self.total_cache_hits += 1
    
    def record_cache_miss(self):
        with self.lock:
            self.total_cache_misses += 1
    
    def record_eviction(self, pressure_level: str = "unknown"):
        with self.lock:
            self.total_evictions += 1
            if pressure_level in self.eviction_by_pressure:
                self.eviction_by_pressure[pressure_level] += 1
    
    def record_memory_pressure_event(self):
        with self.lock:
            self.memory_pressure_events += 1
    
    def record_failed_allocation(self):
        with self.lock:
            self.failed_allocations += 1
    
    def get_stats(self) -> Dict[str, Any]:
        with self.lock:
            hit_rate = 0.0
            total_requests = self.total_cache_hits + self.total_cache_misses
            if total_requests > 0:
                hit_rate = self.total_cache_hits / total_requests
            
            return {
                "total_allocated_bytes": self.total_allocated_bytes,
                "peak_allocated_bytes": self.peak_allocated_bytes,
                "current_cached_chunks": self.current_cached_chunks,
                "cache_hit_rate": hit_rate,
                "total_cache_hits": self.total_cache_hits,
                "total_cache_misses": self.total_cache_misses,
                "total_evictions": self.total_evictions,
                "eviction_by_pressure": self.eviction_by_pressure.copy(),
                "memory_pressure_events": self.memory_pressure_events,
                "failed_allocations": self.failed_allocations,
            }


class NeuronCPUStorageBackend(AllocatorBackendInterface if LMCACHE_AVAILABLE else object):
    """
    Neuron-optimized CPU storage backend for LMCache.
    
    This backend is specifically designed for AWS Neuron devices and provides:
    - CPU-based memory allocation and management
    - NUMA-aware memory placement
    - LRU eviction policy optimized for Neuron workloads
    - Chunk-based storage with configurable sizes
    - Memory pressure handling
    """
    
    def __init__(
        self,
        config: Optional["LMCacheEngineConfig"] = None,
        metadata: Optional["LMCacheEngineMetadata"] = None,
        dst_device: str = "cpu",
        max_memory_gb: float = 4.0,
        chunk_size: int = 256,
        numa_node: Optional[int] = None,
        enable_multi_core: bool = True,
        max_workers: Optional[int] = None,
    ):
        """
        Initialize the Neuron CPU storage backend.
        
        Args:
            config: LMCache engine configuration
            metadata: LMCache engine metadata
            dst_device: Target device for tensor operations (should be "cpu" for Neuron)
            max_memory_gb: Maximum CPU memory to use in GB
            chunk_size: Size of KV cache chunks in tokens
            numa_node: Specific NUMA node to bind memory to (None for auto-detection)
            enable_multi_core: Enable multi-core cache distribution
            max_workers: Maximum worker threads for multi-core operations
        """
        if not LMCACHE_AVAILABLE:
            raise RuntimeError("LMCache is not available. Please install LMCache to use this backend.")
        
        super().__init__(dst_device)
        
        # Ensure we're using CPU for Neuron
        if dst_device != "cpu":
            logger.warning(f"Neuron backend requires CPU device, got {dst_device}. Forcing to CPU.")
            self.dst_device = "cpu"
        
        # Configuration
        self.config = config
        self.metadata = metadata
        self.max_memory_bytes = int(max_memory_gb * 1024 * 1024 * 1024)
        self.chunk_size = chunk_size
        
        # Multi-core distribution setup
        self.enable_multi_core = enable_multi_core
        self.multi_core_distributor = None
        if self.enable_multi_core:
            try:
                self.multi_core_distributor = MultiCoreDistributor(max_workers=max_workers)
                # Optimize chunk size for Neuron's block model
                self.chunk_size = self.multi_core_distributor.calculate_optimal_chunk_size(chunk_size)
                logger.info(f"Multi-core distribution enabled with {self.multi_core_distributor.core_detector.core_count} cores")
            except Exception as e:
                logger.warning(f"Failed to initialize multi-core distribution: {e}")
                self.enable_multi_core = False
        
        # Enhanced storage with metadata tracking
        self.chunk_metadata: Dict[CacheEngineKey, ChunkMetadata] = {}
        self.storage_lock = threading.RLock()
        
        # Memory allocator
        self.memory_allocator: Optional[MemoryAllocatorInterface] = None
        if config and metadata:
            self.memory_allocator = self.initialize_allocator(config, metadata)
        
        # NUMA configuration
        self.numa_node = numa_node
        if self.numa_node is None:
            self.numa_node = self._detect_optimal_numa_node()
        
        # Enhanced memory management components
        self.memory_monitor = MemoryPressureMonitor(self.max_memory_bytes)
        self.eviction_policy = LRUEvictionPolicy()
        self.stats = NeuronMemoryStats()
        self.cleanup_manager = ResourceCleanupManager(self)
        
        # Configurable chunk sizes for different workloads
        self.min_chunk_size = max(64, chunk_size // 4)   # Minimum 64 tokens
        self.max_chunk_size = min(1024, chunk_size * 4)  # Maximum 1024 tokens
        self.adaptive_chunking = True  # Enable adaptive chunk sizing
        
        # Background eviction settings
        self.background_eviction_enabled = True
        self.eviction_thread: Optional[threading.Thread] = None
        self.eviction_stop_event = threading.Event()
        
        # Memory pressure response settings
        self.auto_cleanup_enabled = True
        self.emergency_mode = False
        self.last_pressure_check = 0.0
        self.pressure_check_interval = 1.0  # Check every second
        
        if self.background_eviction_enabled:
            self._start_background_eviction()
        
        logger.info(f"Initialized NeuronCPUStorageBackend with {max_memory_gb}GB memory limit, "
                   f"chunk_size={chunk_size}, NUMA node={self.numa_node}, "
                   f"adaptive_chunking={self.adaptive_chunking}, "
                   f"multi_core_enabled={self.enable_multi_core}")
    
    def _start_background_eviction(self):
        """Start enhanced background eviction and memory pressure handling thread."""
        def background_eviction_worker():
            while not self.eviction_stop_event.wait(timeout=self.pressure_check_interval):
                try:
                    current_time = time.time()
                    
                    # Check if caching is temporarily disabled
                    if self.memory_monitor.is_caching_disabled():
                        continue
                    
                    # Perform memory pressure checks
                    if current_time - self.last_pressure_check >= self.pressure_check_interval:
                        self._handle_memory_pressure()
                        self.last_pressure_check = current_time
                    
                    # Adapt thresholds based on workload patterns
                    if current_time % 60 < 1:  # Every minute
                        self.memory_monitor.adapt_thresholds_based_on_workload()
                    
                except Exception as e:
                    logger.warning(f"Background eviction error: {e}")
        
        self.eviction_thread = threading.Thread(
            target=background_eviction_worker,
            name="NeuronCPU-MemoryManager",
            daemon=True
        )
        self.eviction_thread.start()
        logger.debug("Started enhanced background memory management thread")
    
    def _handle_memory_pressure(self):
        """Handle memory pressure with comprehensive response strategies."""
        pressure_level = self.memory_monitor.get_pressure_level()
        
        if pressure_level == "none":
            # Reset emergency mode if pressure is relieved
            if self.emergency_mode:
                self.emergency_mode = False
                logger.info("Exiting emergency mode - memory pressure relieved")
            return
        
        self.stats.record_memory_pressure_event()
        
        # Handle different pressure levels
        if pressure_level == "low":
            self._handle_low_pressure()
        elif pressure_level == "medium":
            self._handle_medium_pressure()
        elif pressure_level == "high":
            self._handle_high_pressure()
        elif pressure_level == "critical":
            self._handle_critical_pressure()
    
    def _handle_low_pressure(self):
        """Handle low memory pressure with gentle eviction."""
        batch_size = self.memory_monitor.get_eviction_batch_size()
        evicted = self._evict_lru_chunks(target_count=batch_size)
        
        if evicted > 0:
            logger.debug(f"Low pressure eviction: removed {evicted} chunks")
    
    def _handle_medium_pressure(self):
        """Handle medium memory pressure with more aggressive eviction."""
        batch_size = self.memory_monitor.get_eviction_batch_size()
        
        # First try normal eviction
        evicted = self._evict_lru_chunks(target_count=batch_size)
        
        # If not enough freed, try cleanup
        if evicted < batch_size // 2 and self.auto_cleanup_enabled:
            cleanup_result = self.cleanup_manager.perform_cleanup("medium")
            if cleanup_result.get("success"):
                logger.debug(f"Medium pressure cleanup: freed {cleanup_result['bytes_freed']} bytes")
        
        if evicted > 0:
            logger.debug(f"Medium pressure eviction: removed {evicted} chunks")
    
    def _handle_high_pressure(self):
        """Handle high memory pressure with aggressive cleanup."""
        batch_size = self.memory_monitor.get_eviction_batch_size()
        
        # Aggressive eviction
        evicted = self._evict_lru_chunks(target_count=batch_size)
        
        # Perform cleanup
        if self.auto_cleanup_enabled:
            cleanup_result = self.cleanup_manager.perform_cleanup("high")
            if cleanup_result.get("success"):
                logger.info(f"High pressure cleanup: freed {cleanup_result['bytes_freed']} bytes, "
                          f"removed {cleanup_result['chunks_removed']} chunks")
        
        # Reject new allocations temporarily
        if not self.emergency_mode:
            logger.warning("Entering high memory pressure mode - rejecting new allocations")
        
        logger.warning(f"High pressure response: evicted {evicted} chunks")
    
    def _handle_critical_pressure(self):
        """Handle critical memory pressure with emergency measures."""
        if not self.emergency_mode:
            self.emergency_mode = True
            logger.error("Entering emergency mode due to critical memory pressure")
        
        # Emergency eviction
        batch_size = self.memory_monitor.get_eviction_batch_size()
        evicted = self._evict_lru_chunks(target_count=batch_size)
        
        # Emergency cleanup
        if self.auto_cleanup_enabled:
            cleanup_result = self.cleanup_manager.perform_cleanup("critical")
            if cleanup_result.get("success"):
                logger.error(f"Emergency cleanup: freed {cleanup_result['bytes_freed']} bytes, "
                           f"removed {cleanup_result['chunks_removed']} chunks")
        
        # Temporarily disable caching
        if not self.memory_monitor.is_caching_disabled():
            self.memory_monitor.force_cache_disable(60.0)  # Disable for 1 minute
        
        logger.error(f"Critical pressure response: evicted {evicted} chunks, "
                    f"caching disabled temporarily")
    
    def _detect_optimal_numa_node(self) -> Optional[int]:
        """Detect the optimal NUMA node for memory allocation."""
        try:
            if LMCACHE_AVAILABLE:
                numa_mapping = NUMADetector.get_numa_mapping(self.config or {})
                if numa_mapping and hasattr(numa_mapping, 'cpu_to_numa'):
                    # Use the first available NUMA node
                    return next(iter(numa_mapping.cpu_to_numa.values()), None)
        except Exception as e:
            logger.warning(f"Failed to detect NUMA configuration: {e}")
        
        return None
    
    def _get_memory_usage_ratio(self) -> float:
        """Get current memory usage as a ratio of maximum memory."""
        return self.memory_monitor.get_memory_usage_ratio()
    
    def _is_memory_pressure(self) -> bool:
        """Check if we're under memory pressure."""
        return self.memory_monitor.should_evict()
    
    def _evict_lru_chunks(self, target_count: int = None) -> int:
        """
        Evict LRU chunks to free memory using advanced eviction policy.
        
        Args:
            target_count: Number of chunks to evict (None for pressure-based)
            
        Returns:
            Number of chunks actually evicted
        """
        if target_count is None:
            target_count = self.memory_monitor.get_eviction_batch_size()
        
        if target_count <= 0:
            return 0
        
        evicted_count = 0
        pressure_level = self.memory_monitor.get_pressure_level()
        
        with self.storage_lock:
            # Use eviction policy to select candidates
            eviction_candidates = self.eviction_policy.select_eviction_candidates(
                self.chunk_metadata, target_count
            )
            
            # Remove selected candidates
            for key in eviction_candidates:
                if self._remove_chunk_internal(key):
                    evicted_count += 1
                    self.stats.record_eviction(pressure_level)
        
        if evicted_count > 0:
            logger.debug(f"Evicted {evicted_count} chunks using {self.eviction_policy.strategy} "
                        f"policy under {pressure_level} pressure")
        
        return evicted_count
    
    def _remove_chunk_internal(self, key: CacheEngineKey) -> bool:
        """
        Internal method to remove a chunk (must be called with lock held).
        
        Args:
            key: Key of chunk to remove
            
        Returns:
            True if chunk was removed, False otherwise
        """
        if key not in self.chunk_metadata:
            return False
        
        chunk_meta = self.chunk_metadata.pop(key)
        memory_obj = chunk_meta.memory_obj
        
        # Update memory tracking
        self.memory_monitor.update_memory_usage(-chunk_meta.size_bytes)
        self.stats.update_deallocation(chunk_meta.size_bytes)
        self.stats.current_cached_chunks -= 1
        
        # Release memory object
        memory_obj.ref_count_down()
        
        return True
    
    def _ensure_memory_available(self, required_bytes: int) -> bool:
        """
        Ensure sufficient memory is available, evicting if necessary.
        
        Args:
            required_bytes: Bytes of memory required
            
        Returns:
            True if memory is available, False otherwise
        """
        max_attempts = 10
        attempt = 0
        
        while attempt < max_attempts:
            current_usage = self.memory_monitor.current_memory_bytes
            available_bytes = self.max_memory_bytes - current_usage
            
            if available_bytes >= required_bytes:
                return True
            
            # Calculate how many chunks we need to evict
            bytes_to_free = required_bytes - available_bytes
            
            # Estimate chunks needed (assume average chunk size)
            avg_chunk_size = self._get_average_chunk_size()
            estimated_chunks_to_evict = max(1, int(bytes_to_free / avg_chunk_size) + 1)
            
            # Try to evict chunks
            evicted = self._evict_lru_chunks(target_count=estimated_chunks_to_evict)
            if evicted == 0:
                # No more chunks to evict
                self.stats.record_failed_allocation()
                break
            
            attempt += 1
        
        return False
    
    def _get_average_chunk_size(self) -> int:
        """Get average chunk size in bytes."""
        with self.storage_lock:
            if not self.chunk_metadata:
                return 1024 * 1024  # 1MB default
            
            total_size = sum(meta.size_bytes for meta in self.chunk_metadata.values())
            return total_size // len(self.chunk_metadata)
    
    def _adapt_chunk_size(self, workload_pattern: str) -> int:
        """
        Adapt chunk size based on workload patterns.
        
        Args:
            workload_pattern: Type of workload ("conversation", "batch", "mixed")
            
        Returns:
            Adapted chunk size
        """
        if not self.adaptive_chunking:
            return self.chunk_size
        
        if workload_pattern == "conversation":
            # Smaller chunks for interactive workloads
            return max(self.min_chunk_size, self.chunk_size // 2)
        elif workload_pattern == "batch":
            # Larger chunks for batch processing
            return min(self.max_chunk_size, self.chunk_size * 2)
        else:
            # Default chunk size for mixed workloads
            return self.chunk_size
    
    # Implementation of StorageBackendInterface methods
    
    def contains(self, key: CacheEngineKey, pin: bool = False) -> bool:
        """Check whether key is in the storage backend."""
        with self.storage_lock:
            exists = key in self.chunk_metadata
            if exists:
                self.stats.record_cache_hit()
                chunk_meta = self.chunk_metadata[key]
                chunk_meta.update_access()
                
                if pin:
                    chunk_meta.is_pinned = True
                    chunk_meta.memory_obj.pin()
            else:
                self.stats.record_cache_miss()
            return exists
    
    def exists_in_put_tasks(self, key: CacheEngineKey) -> bool:
        """Check whether key is in the ongoing put tasks."""
        # For synchronous CPU backend, no ongoing put tasks
        return False
    
    def batched_submit_put_task(
        self,
        keys: Sequence[CacheEngineKey],
        objs: List[MemoryObj],
        transfer_spec: Any = None,
    ) -> Union[List[Future], None]:
        """Synchronously put the MemoryObjs into the storage backend with multi-core distribution."""
        if len(keys) != len(objs):
            raise ValueError(f"Keys and objects length mismatch: {len(keys)} != {len(objs)}")
        
        if self.enable_multi_core and self.multi_core_distributor:
            # Use multi-core distribution for batch operations
            return self._batched_put_with_distribution(keys, objs, transfer_spec)
        else:
            # Fallback to single-threaded operation
            return self._batched_put_single_threaded(keys, objs, transfer_spec)
    
    def _batched_put_with_distribution(
        self,
        keys: Sequence[CacheEngineKey],
        objs: List[MemoryObj],
        transfer_spec: Any = None,
    ) -> List[Future]:
        """Put operations using multi-core distribution."""
        def put_single_item(key_obj_pair):
            key, memory_obj = key_obj_pair
            return self._put_single_item(key, memory_obj)
        
        # Create operation items for distribution
        operation_items = [(str(key), (key, obj)) for key, obj in zip(keys, objs)]
        
        # Distribute across cores
        futures = self.multi_core_distributor.distribute_batch_operation(
            put_single_item,
            operation_items
        )
        
        return futures
    
    def _batched_put_single_threaded(
        self,
        keys: Sequence[CacheEngineKey],
        objs: List[MemoryObj],
        transfer_spec: Any = None,
    ) -> None:
        """Single-threaded put operations (original implementation)."""
        with self.storage_lock:
            for key, memory_obj in zip(keys, objs):
                self._put_single_item(key, memory_obj)
        
        return None  # Synchronous operation
    
    def _put_single_item(self, key: CacheEngineKey, memory_obj: MemoryObj) -> bool:
        """Put a single item into storage with memory pressure handling."""
        with self.storage_lock:
            if key in self.chunk_metadata:
                return True  # Already exists
            
            # Check if caching is temporarily disabled
            if self.memory_monitor.is_caching_disabled():
                logger.debug("Caching temporarily disabled due to memory pressure")
                return False
            
            # Check if we should reject new allocations due to high pressure
            if self.memory_monitor.should_reject_new_allocations():
                logger.debug("Rejecting new allocation due to high memory pressure")
                return False
            
            # Estimate memory usage for this object
            estimated_bytes = self._estimate_memory_obj_size(memory_obj)
            
            # Ensure memory is available with enhanced pressure handling
            if not self._ensure_memory_available_with_pressure_handling(estimated_bytes):
                logger.warning(f"Failed to allocate memory for key {key}, skipping")
                return False
            
            # Optimize memory layout for Neuron if multi-core is enabled
            if self.enable_multi_core and self.multi_core_distributor:
                if hasattr(memory_obj, 'data') and isinstance(memory_obj.data, torch.Tensor):
                    memory_obj.data = self.multi_core_distributor.optimize_memory_for_neuron(memory_obj.data)
            
            # Create chunk metadata
            chunk_meta = ChunkMetadata(key, memory_obj, estimated_bytes)
            
            # Store the object
            memory_obj.ref_count_up()
            self.chunk_metadata[key] = chunk_meta
            
            # Update memory tracking
            self.memory_monitor.update_memory_usage(estimated_bytes)
            self.stats.update_allocation(estimated_bytes)
            self.stats.current_cached_chunks += 1
            
            return True
    
    def _ensure_memory_available_with_pressure_handling(self, required_bytes: int) -> bool:
        """Ensure sufficient memory is available with enhanced pressure handling."""
        max_attempts = 15  # Increased attempts for better recovery
        attempt = 0
        
        while attempt < max_attempts:
            current_usage = self.memory_monitor.current_memory_bytes
            available_bytes = self.max_memory_bytes - current_usage
            
            if available_bytes >= required_bytes:
                return True
            
            pressure_level = self.memory_monitor.get_pressure_level()
            
            # Try different strategies based on pressure level
            if pressure_level in ["low", "medium"]:
                # Standard eviction
                bytes_to_free = required_bytes - available_bytes
                avg_chunk_size = self._get_average_chunk_size()
                estimated_chunks_to_evict = max(1, int(bytes_to_free / avg_chunk_size) + 1)
                
                evicted = self._evict_lru_chunks(target_count=estimated_chunks_to_evict)
                if evicted == 0:
                    break
                
            elif pressure_level == "high":
                # Aggressive eviction + cleanup
                evicted = self._evict_lru_chunks(target_count=10)
                
                if evicted < 5 and self.auto_cleanup_enabled:
                    cleanup_result = self.cleanup_manager.perform_cleanup("high")
                    if not cleanup_result.get("success"):
                        break
                
            else:  # critical pressure
                # Emergency measures
                if self.auto_cleanup_enabled:
                    cleanup_result = self.cleanup_manager.perform_cleanup("critical")
                    if not cleanup_result.get("success"):
                        break
                else:
                    break
            
            attempt += 1
        
        # If we couldn't free enough memory, record the failure
        if attempt >= max_attempts:
            self.stats.record_failed_allocation()
        
        return False
    
    async def async_batched_submit_put_task(
        self,
        keys: Sequence[CacheEngineKey],
        objs: List[MemoryObj],
        transfer_spec: Any = None,
    ) -> None:
        """Async version of batched_submit_put_task."""
        self.batched_submit_put_task(keys, objs, transfer_spec)
    
    def get_blocking(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        """Get the KV cache from the storage backend (blocking)."""
        with self.storage_lock:
            if key not in self.chunk_metadata:
                self.stats.record_cache_miss()
                return None
            
            self.stats.record_cache_hit()
            chunk_meta = self.chunk_metadata[key]
            chunk_meta.update_access()
            
            # Increment ref count for caller
            memory_obj = chunk_meta.memory_obj
            memory_obj.ref_count_up()
            return memory_obj
    
    def get_non_blocking(
        self,
        key: CacheEngineKey,
        location: Optional[str] = None,
    ) -> Optional[Future]:
        """Non-blocking get (not implemented for CPU backend)."""
        # CPU backend is synchronous
        return None
    
    async def batched_async_contains(
        self,
        lookup_id: str,
        keys: List[CacheEngineKey],
        pin: bool = False,
    ) -> int:
        """Check whether keys are in the storage backend (async)."""
        hit_count = 0
        with self.storage_lock:
            for key in keys:
                if key not in self.chunk_metadata:
                    break  # Stop at first miss for prefix matching
                
                hit_count += 1
                chunk_meta = self.chunk_metadata[key]
                chunk_meta.update_access()
                
                if pin:
                    chunk_meta.is_pinned = True
                    chunk_meta.memory_obj.pin()
        
        return hit_count
    
    async def batched_get_non_blocking(
        self,
        lookup_id: str,
        keys: list[CacheEngineKey],
        transfer_spec: Any = None,
    ) -> list[MemoryObj]:
        """Non-blocking batched get with multi-core distribution."""
        if self.enable_multi_core and self.multi_core_distributor and len(keys) > 1:
            # Use multi-core distribution for batch operations
            return await self._batched_get_with_distribution(lookup_id, keys, transfer_spec)
        else:
            # Fallback to single-threaded operation
            return self._batched_get_single_threaded(lookup_id, keys, transfer_spec)
    
    async def _batched_get_with_distribution(
        self,
        lookup_id: str,
        keys: list[CacheEngineKey],
        transfer_spec: Any = None,
    ) -> list[MemoryObj]:
        """Batched get using multi-core distribution."""
        def get_single_item(key_data):
            key = key_data
            return self._get_single_item(key)
        
        # Create operation items for distribution
        operation_items = [(str(key), key) for key in keys]
        
        # Distribute across cores
        futures = self.multi_core_distributor.distribute_batch_operation(
            get_single_item,
            operation_items
        )
        
        # Collect results in order
        mem_objs = []
        for future in futures:
            try:
                result = future.result(timeout=1.0)  # 1 second timeout
                if result is not None:
                    mem_objs.append(result)
                else:
                    break  # Stop at first miss for prefix matching
            except Exception as e:
                logger.warning(f"Multi-core get operation failed: {e}")
                break
        
        return mem_objs
    
    def _batched_get_single_threaded(
        self,
        lookup_id: str,
        keys: list[CacheEngineKey],
        transfer_spec: Any = None,
    ) -> list[MemoryObj]:
        """Single-threaded batched get (original implementation)."""
        mem_objs = []
        with self.storage_lock:
            for key in keys:
                result = self._get_single_item(key)
                if result is not None:
                    mem_objs.append(result)
                else:
                    break  # Stop at first miss
        
        return mem_objs
    
    def _get_single_item(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        """Get a single item from storage."""
        with self.storage_lock:
            if key not in self.chunk_metadata:
                return None
            
            chunk_meta = self.chunk_metadata[key]
            chunk_meta.update_access()
            
            # Increment ref count for caller
            memory_obj = chunk_meta.memory_obj
            memory_obj.ref_count_up()
            return memory_obj
    
    def pin(self, key: CacheEngineKey) -> bool:
        """Pin a memory object so it will not be evicted."""
        with self.storage_lock:
            if key not in self.chunk_metadata:
                return False
            
            chunk_meta = self.chunk_metadata[key]
            chunk_meta.is_pinned = True
            chunk_meta.memory_obj.pin()
            return True
    
    def unpin(self, key: CacheEngineKey) -> bool:
        """Unpin a memory object so it can be evicted."""
        with self.storage_lock:
            if key not in self.chunk_metadata:
                return False
            
            chunk_meta = self.chunk_metadata[key]
            chunk_meta.is_pinned = False
            chunk_meta.memory_obj.unpin()
            return True
    
    def remove(self, key: CacheEngineKey, force: bool = True) -> bool:
        """Remove a memory object."""
        with self.storage_lock:
            return self._remove_chunk_internal(key)
    
    def close(self) -> None:
        """Close the storage backend."""
        # Stop background eviction
        if self.background_eviction_enabled and self.eviction_thread:
            self.eviction_stop_event.set()
            self.eviction_thread.join(timeout=5.0)
        
        # Shutdown multi-core distributor
        if self.enable_multi_core and self.multi_core_distributor:
            self.multi_core_distributor.shutdown()
        
        with self.storage_lock:
            # Clear all cached objects
            for chunk_meta in self.chunk_metadata.values():
                chunk_meta.memory_obj.ref_count_down()
            
            self.chunk_metadata.clear()
            self.stats.current_cached_chunks = 0
            
            if self.memory_allocator:
                self.memory_allocator.close()
        
        logger.info("NeuronCPUStorageBackend closed")
    
    # Implementation of AllocatorBackendInterface methods
    
    def initialize_allocator(
        self, 
        config: LMCacheEngineConfig, 
        metadata: LMCacheEngineMetadata
    ) -> MemoryAllocatorInterface:
        """Initialize the memory allocator for this backend."""
        if not LMCACHE_AVAILABLE:
            raise RuntimeError("LMCache is not available")
        
        # Detect NUMA mapping for optimal memory placement
        numa_mapping = None
        try:
            numa_mapping = NUMADetector.get_numa_mapping(config)
            logger.info(f"Detected NUMA mapping: {numa_mapping}")
        except Exception as e:
            logger.warning(f"Failed to detect NUMA mapping: {e}")
        
        # Create mixed memory allocator optimized for CPU
        allocator = MixedMemoryAllocator(
            self.max_memory_bytes,
            numa_mapping=numa_mapping,
        )
        
        logger.info(f"Initialized memory allocator with {self.max_memory_bytes} bytes")
        return allocator
    
    def get_memory_allocator(self) -> MemoryAllocatorInterface:
        """Get the underlying memory allocator."""
        if self.memory_allocator is None:
            raise RuntimeError("Memory allocator not initialized")
        return self.memory_allocator
    
    def get_allocator_backend(self) -> "AllocatorBackendInterface":
        """Get the allocator backend (self)."""
        return self
    
    def allocate(
        self,
        shapes: Union[torch.Size, list[torch.Size]],
        dtypes: Union[torch.dtype, list[torch.dtype]],
        fmt: MemoryFormat = MemoryFormat.KV_2LTD,
        eviction: bool = True,
        busy_loop: bool = True,
    ) -> Optional[MemoryObj]:
        """Allocate a memory object with eviction if necessary."""
        if self.memory_allocator is None:
            logger.error("Memory allocator not initialized")
            return None
        
        # Try direct allocation first
        memory_obj = self.memory_allocator.allocate(shapes, dtypes, fmt)
        if memory_obj is not None:
            return memory_obj
        
        if not eviction:
            return None
        
        # Try eviction and allocation in a loop
        max_attempts = 10
        for attempt in range(max_attempts):
            # Evict some chunks
            evicted = self._evict_lru_chunks()
            if evicted == 0:
                logger.warning("No chunks available for eviction")
                break
            
            # Try allocation again
            memory_obj = self.memory_allocator.allocate(shapes, dtypes, fmt)
            if memory_obj is not None:
                return memory_obj
            
            if busy_loop and attempt < max_attempts - 1:
                time.sleep(0.01)  # Brief pause before retry
        
        logger.warning("Failed to allocate memory after eviction attempts")
        return None
    
    def batched_allocate(
        self,
        shapes: Union[torch.Size, list[torch.Size]],
        dtypes: Union[torch.dtype, list[torch.dtype]],
        batch_size: int,
        fmt: MemoryFormat = MemoryFormat.KV_2LTD,
        eviction: bool = True,
        busy_loop: bool = True,
    ) -> Optional[list[MemoryObj]]:
        """Batched allocate memory objects with eviction if necessary."""
        if self.memory_allocator is None:
            logger.error("Memory allocator not initialized")
            return None
        
        # Try direct allocation first
        memory_objs = self.memory_allocator.batched_allocate(shapes, dtypes, batch_size, fmt)
        if memory_objs is not None:
            return memory_objs
        
        if not eviction:
            return None
        
        # Try eviction and allocation in a loop
        max_attempts = 10
        for attempt in range(max_attempts):
            # Evict some chunks
            evicted = self._evict_lru_chunks(target_count=batch_size)
            if evicted == 0:
                logger.warning("No chunks available for eviction")
                break
            
            # Try allocation again
            memory_objs = self.memory_allocator.batched_allocate(shapes, dtypes, batch_size, fmt)
            if memory_objs is not None:
                return memory_objs
            
            if busy_loop and attempt < max_attempts - 1:
                time.sleep(0.01)  # Brief pause before retry
        
        logger.warning("Failed to batched allocate memory after eviction attempts")
        return None
    
    def calculate_chunk_budget(self) -> int:
        """Calculate the chunk budget for the allocator backend."""
        if not self.metadata:
            return 0
        
        # Estimate chunk size based on metadata
        chunk_tokens = self.chunk_size
        kv_shape = self.metadata.kv_shape  # [num_layers, kv_size, chunk_size, num_heads, head_size]
        num_layers = kv_shape[0]
        kv_size = kv_shape[1]
        num_heads = kv_shape[3]
        head_size = kv_shape[4]
        hidden_dim = num_heads * head_size
        dtype_size = self.metadata.kv_dtype.itemsize
        
        # Calculate bytes per chunk
        chunk_bytes = kv_size * num_layers * chunk_tokens * hidden_dim * dtype_size
        
        # Calculate budget with safety margin (80% of available memory)
        usable_memory = int(self.max_memory_bytes * 0.8)
        max_chunks = usable_memory // chunk_bytes
        
        logger.debug(f"Calculated chunk budget: {max_chunks} chunks "
                    f"(chunk_bytes={chunk_bytes}, usable_memory={usable_memory})")
        
        return max_chunks
    
    # Helper methods
    
    def _estimate_memory_obj_size(self, memory_obj: MemoryObj) -> int:
        """Estimate the memory size of a MemoryObj."""
        try:
            if hasattr(memory_obj, 'get_memory_size'):
                return memory_obj.get_memory_size()
            elif hasattr(memory_obj, 'data') and isinstance(memory_obj.data, torch.Tensor):
                return memory_obj.data.numel() * memory_obj.data.element_size()
            else:
                # Fallback estimation
                return 1024 * 1024  # 1MB default
        except Exception as e:
            logger.warning(f"Failed to estimate memory object size: {e}")
            return 1024 * 1024  # 1MB default
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive storage backend statistics including memory pressure info."""
        base_stats = self.stats.get_stats()
        base_stats.update({
            "max_memory_bytes": self.max_memory_bytes,
            "memory_usage_ratio": self._get_memory_usage_ratio(),
            "memory_pressure_level": self.memory_monitor.get_pressure_level(),
            "numa_node": self.numa_node,
            "chunk_size": self.chunk_size,
            "adaptive_chunking": self.adaptive_chunking,
            "eviction_policy": self.eviction_policy.strategy,
            "background_eviction_enabled": self.background_eviction_enabled,
            "average_chunk_size_bytes": self._get_average_chunk_size(),
            "multi_core_enabled": self.enable_multi_core,
            "emergency_mode": self.emergency_mode,
            "auto_cleanup_enabled": self.auto_cleanup_enabled,
        })
        
        # Add detailed memory pressure statistics
        base_stats["memory_pressure"] = self.memory_monitor.get_memory_stats()
        
        # Add cleanup statistics
        base_stats["cleanup_stats"] = self.cleanup_manager.get_cleanup_stats()
        
        # Add multi-core statistics if enabled
        if self.enable_multi_core and self.multi_core_distributor:
            multi_core_stats = self.multi_core_distributor.get_load_balance_stats()
            base_stats["multi_core_stats"] = multi_core_stats
        
        return base_stats
    
    def get_memory_pressure_status(self) -> Dict[str, Any]:
        """Get detailed memory pressure status."""
        return {
            "pressure_level": self.memory_monitor.get_pressure_level(),
            "pressure_trend": self.memory_monitor.get_pressure_trend(),
            "usage_ratio": self.memory_monitor.get_memory_usage_ratio(),
            "cache_disabled": self.memory_monitor.is_caching_disabled(),
            "emergency_mode": self.emergency_mode,
            "should_reject_allocations": self.memory_monitor.should_reject_new_allocations(),
            "memory_stats": self.memory_monitor.get_memory_stats(),
            "cleanup_stats": self.cleanup_manager.get_cleanup_stats()
        }
    
    def configure_memory_pressure_handling(self, **kwargs):
        """Configure memory pressure handling parameters."""
        if "auto_cleanup_enabled" in kwargs:
            self.auto_cleanup_enabled = kwargs["auto_cleanup_enabled"]
            logger.info(f"Auto cleanup {'enabled' if self.auto_cleanup_enabled else 'disabled'}")
        
        if "pressure_check_interval" in kwargs:
            self.pressure_check_interval = max(0.1, kwargs["pressure_check_interval"])
            logger.info(f"Pressure check interval set to {self.pressure_check_interval}s")
        
        # Configure memory monitor thresholds
        monitor_config = {}
        for key in ["low_pressure_threshold", "medium_pressure_threshold", 
                   "high_pressure_threshold", "critical_pressure_threshold"]:
            if key in kwargs:
                setattr(self.memory_monitor, key, kwargs[key])
                monitor_config[key] = kwargs[key]
        
        if monitor_config:
            logger.info(f"Memory pressure thresholds updated: {monitor_config}")
        
        # Configure cleanup manager
        cleanup_config = {}
        for key in ["chunk_expiry_time", "large_chunk_threshold", 
                   "duplicate_detection_enabled", "min_cleanup_interval"]:
            if key in kwargs:
                cleanup_config[key] = kwargs[key]
        
        if cleanup_config:
            self.cleanup_manager.configure_cleanup(**cleanup_config)
    
    def force_memory_cleanup(self, pressure_level: str = "high") -> Dict[str, Any]:
        """Force memory cleanup with specified pressure level."""
        return self.cleanup_manager.perform_cleanup(pressure_level)
    
    def disable_caching_temporarily(self, duration_seconds: float = 60.0):
        """Temporarily disable caching due to memory pressure."""
        self.memory_monitor.force_cache_disable(duration_seconds)
        logger.warning(f"Caching disabled for {duration_seconds} seconds")
    
    def enable_caching(self):
        """Re-enable caching."""
        self.memory_monitor.force_cache_enable()
        logger.info("Caching re-enabled")
    
    def reset_memory_pressure_thresholds(self):
        """Reset memory pressure thresholds to defaults."""
        self.memory_monitor.reset_adaptive_thresholds()
        logger.info("Memory pressure thresholds reset to defaults")
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of the storage backend."""
        memory_ratio = self._get_memory_usage_ratio()
        pressure_level = self.memory_monitor.get_pressure_level()
        
        # Determine health based on memory pressure and error rates
        is_healthy = (
            memory_ratio < 0.95 and  # Not critically full
            pressure_level != "high" and  # Not under high pressure
            self.stats.failed_allocations < 10  # Low failure rate
        )
        
        return {
            "backend_type": "NeuronCPUStorageBackend",
            "is_healthy": is_healthy,
            "memory_allocator_initialized": self.memory_allocator is not None,
            "background_eviction_running": (
                self.background_eviction_enabled and 
                self.eviction_thread and 
                self.eviction_thread.is_alive()
            ),
            "stats": self.get_stats(),
        }
    
    def set_eviction_strategy(self, strategy: str) -> bool:
        """
        Set the eviction strategy.
        
        Args:
            strategy: One of "lru", "lfu", "aging", "hybrid"
            
        Returns:
            True if strategy was set successfully
        """
        valid_strategies = ["lru", "lfu", "aging", "hybrid"]
        if strategy not in valid_strategies:
            logger.warning(f"Invalid eviction strategy: {strategy}. "
                          f"Valid options: {valid_strategies}")
            return False
        
        self.eviction_policy.strategy = strategy
        logger.info(f"Eviction strategy changed to: {strategy}")
        return True
    
    def force_eviction(self, count: int = None) -> int:
        """
        Force eviction of chunks regardless of memory pressure.
        
        Args:
            count: Number of chunks to evict (None for default batch size)
            
        Returns:
            Number of chunks evicted
        """
        if count is None:
            count = self.memory_monitor.medium_pressure_batch_size
        
        return self._evict_lru_chunks(target_count=count)
    
    def get_chunk_info(self, key: CacheEngineKey) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a specific chunk.
        
        Args:
            key: Key of the chunk
            
        Returns:
            Dictionary with chunk information or None if not found
        """
        with self.storage_lock:
            if key not in self.chunk_metadata:
                return None
            
            chunk_meta = self.chunk_metadata[key]
            return {
                "key": str(key),
                "size_bytes": chunk_meta.size_bytes,
                "access_count": chunk_meta.access_count,
                "age_seconds": chunk_meta.get_age_seconds(),
                "idle_time_seconds": chunk_meta.get_idle_time_seconds(),
                "is_pinned": chunk_meta.is_pinned,
                "can_evict": chunk_meta.can_evict(),
                "creation_time": chunk_meta.creation_time,
                "last_access_time": chunk_meta.last_access_time,
            }
    
    def get_all_chunk_keys(self) -> List[CacheEngineKey]:
        """Get all cached chunk keys."""
        with self.storage_lock:
            return list(self.chunk_metadata.keys())
    
    def clear_cache(self, force: bool = False) -> int:
        """
        Clear all cached chunks.
        
        Args:
            force: If True, clear even pinned chunks
            
        Returns:
            Number of chunks cleared
        """
        cleared_count = 0
        
        with self.storage_lock:
            keys_to_remove = []
            
            for key, chunk_meta in self.chunk_metadata.items():
                if force or chunk_meta.can_evict():
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                if self._remove_chunk_internal(key):
                    cleared_count += 1
        
        logger.info(f"Cleared {cleared_count} chunks from cache")
        return cleared_count
    
    def get_multi_core_info(self) -> Dict[str, Any]:
        """Get information about multi-core configuration."""
        if not self.enable_multi_core or not self.multi_core_distributor:
            return {"enabled": False}
        
        return {
            "enabled": True,
            "neuron_cores": self.multi_core_distributor.core_detector.get_all_cores(),
            "core_count": self.multi_core_distributor.core_detector.core_count,
            "worker_count": self.multi_core_distributor.max_workers,
            "load_balance_stats": self.multi_core_distributor.get_load_balance_stats(),
        }
    
    def set_multi_core_enabled(self, enabled: bool) -> bool:
        """Enable or disable multi-core distribution."""
        if enabled and not self.multi_core_distributor:
            try:
                self.multi_core_distributor = MultiCoreDistributor()
                self.enable_multi_core = True
                logger.info("Multi-core distribution enabled")
                return True
            except Exception as e:
                logger.error(f"Failed to enable multi-core distribution: {e}")
                return False
        elif not enabled and self.multi_core_distributor:
            self.multi_core_distributor.shutdown()
            self.multi_core_distributor = None
            self.enable_multi_core = False
            logger.info("Multi-core distribution disabled")
            return True
        
        return self.enable_multi_core == enabled
    
    def optimize_for_workload(self, workload_type: str) -> bool:
        """
        Optimize cache configuration for specific workload types.
        
        Args:
            workload_type: "conversation", "batch", "mixed", or "inference"
            
        Returns:
            True if optimization was applied successfully
        """
        try:
            if workload_type == "conversation":
                # Optimize for interactive workloads
                self.eviction_policy.strategy = "lru"  # Favor recent interactions
                self.adaptive_chunking = True
                if self.multi_core_distributor:
                    # Use fewer workers to reduce context switching
                    self.multi_core_distributor.max_workers = min(2, self.multi_core_distributor.core_detector.core_count)
                
            elif workload_type == "batch":
                # Optimize for batch processing
                self.eviction_policy.strategy = "hybrid"  # Balance frequency and recency
                self.adaptive_chunking = True
                if self.multi_core_distributor:
                    # Use more workers for parallel processing
                    self.multi_core_distributor.max_workers = self.multi_core_distributor.core_detector.core_count * 2
                
            elif workload_type == "inference":
                # Optimize for inference workloads
                self.eviction_policy.strategy = "lfu"  # Favor frequently used patterns
                self.adaptive_chunking = False  # Use consistent chunk sizes
                if self.multi_core_distributor:
                    # Balanced worker count
                    self.multi_core_distributor.max_workers = self.multi_core_distributor.core_detector.core_count
                
            else:  # "mixed" or default
                # Balanced configuration
                self.eviction_policy.strategy = "hybrid"
                self.adaptive_chunking = True
                if self.multi_core_distributor:
                    self.multi_core_distributor.max_workers = min(4, self.multi_core_distributor.core_detector.core_count * 2)
            
            logger.info(f"Optimized cache configuration for {workload_type} workload")
            return True
            
        except Exception as e:
            logger.error(f"Failed to optimize for workload {workload_type}: {e}")
            return False