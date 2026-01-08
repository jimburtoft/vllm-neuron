#!/usr/bin/env python3
"""
Basic functionality tests for NeuronCPUStorageBackend core components.

This test suite validates the core functionality without full LMCache integration.
"""

import pytest
import threading
import time
from unittest.mock import Mock, patch

# Test imports - handle LMCache availability gracefully
try:
    from vllm_neuron.neuron_cpu_storage_backend import (
        ChunkMetadata,
        MemoryPressureMonitor,
        LRUEvictionPolicy,
        NeuronMemoryStats,
        LMCACHE_AVAILABLE
    )
    
    if LMCACHE_AVAILABLE:
        from vllm_neuron.neuron_cpu_storage_backend import NeuronCPUStorageBackend
        
except ImportError as e:
    pytest.skip(f"Storage backend not available: {e}", allow_module_level=True)


class MockMemoryObj:
    """Mock MemoryObj for testing."""
    
    def __init__(self, size_bytes: int = 1024):
        self.size_bytes = size_bytes
        self.ref_count = 0
        self.pinned = False
        self.evictable = True
        
    def ref_count_up(self):
        self.ref_count += 1
        
    def ref_count_down(self):
        self.ref_count = max(0, self.ref_count - 1)
        
    def pin(self):
        self.pinned = True
        
    def unpin(self):
        self.pinned = False
        
    @property
    def can_evict(self) -> bool:
        return self.evictable and not self.pinned
        
    def get_memory_size(self) -> int:
        return self.size_bytes


class TestStorageComponents:
    """Test individual storage backend components."""
    
    def test_chunk_metadata_basic_functionality(self):
        """Test ChunkMetadata basic functionality."""
        key = "test_key"
        memory_obj = MockMemoryObj(1024)
        size_bytes = 1024
        
        chunk_meta = ChunkMetadata(key, memory_obj, size_bytes)
        
        # Test initialization
        assert chunk_meta.key == key
        assert chunk_meta.memory_obj == memory_obj
        assert chunk_meta.size_bytes == size_bytes
        assert chunk_meta.access_count == 0
        assert not chunk_meta.is_pinned
        assert chunk_meta.can_evict()
        
        # Test access tracking
        initial_time = chunk_meta.last_access_time
        time.sleep(0.01)
        chunk_meta.update_access()
        
        assert chunk_meta.access_count == 1
        assert chunk_meta.last_access_time > initial_time
        
        # Test age calculations
        age = chunk_meta.get_age_seconds()
        idle_time = chunk_meta.get_idle_time_seconds()
        assert age > 0
        assert idle_time > 0
    
    def test_memory_pressure_monitor_functionality(self):
        """Test MemoryPressureMonitor functionality."""
        max_memory = 1000
        monitor = MemoryPressureMonitor(max_memory)
        
        # Test initialization
        assert monitor.max_memory_bytes == max_memory
        assert monitor.current_memory_bytes == 0
        assert monitor.get_memory_usage_ratio() == 0.0
        assert monitor.get_pressure_level() == "none"
        assert not monitor.should_evict()
        
        # Test memory usage tracking
        monitor.update_memory_usage(500)
        assert monitor.current_memory_bytes == 500
        assert monitor.get_memory_usage_ratio() == 0.5
        assert monitor.get_pressure_level() == "none"
        
        # Test pressure levels
        monitor.update_memory_usage(200)  # 70% - low pressure
        assert monitor.get_pressure_level() == "low"
        assert monitor.should_evict()
        
        monitor.update_memory_usage(150)  # 85% - medium pressure
        assert monitor.get_pressure_level() == "medium"
        
        monitor.update_memory_usage(100)  # 95% - high pressure
        assert monitor.get_pressure_level() == "high"
        
        # Test eviction batch sizes
        assert monitor.get_eviction_batch_size() == monitor.high_pressure_batch_size
    
    def test_lru_eviction_policy_functionality(self):
        """Test LRUEvictionPolicy functionality."""
        policy = LRUEvictionPolicy()
        
        # Create test chunks with different access patterns
        chunks = {}
        for i in range(5):
            key = f"key_{i}"
            memory_obj = MockMemoryObj()
            chunk_meta = ChunkMetadata(key, memory_obj, 1024)
            chunk_meta.last_access_time = time.time() - (5 - i)  # Older chunks first
            chunk_meta.access_count = i  # Different access counts
            chunks[key] = chunk_meta
        
        # Test LRU strategy
        policy.strategy = "lru"
        candidates = policy.select_eviction_candidates(chunks, 3)
        assert len(candidates) == 3
        assert candidates[0] == "key_0"  # Oldest access time
        
        # Test LFU strategy
        policy.strategy = "lfu"
        candidates = policy.select_eviction_candidates(chunks, 3)
        assert len(candidates) == 3
        assert candidates[0] == "key_0"  # Least access count
        
        # Test with pinned chunks
        chunks["key_1"].is_pinned = True
        chunks["key_1"].memory_obj.pin()
        
        candidates = policy.select_eviction_candidates(chunks, 5)
        assert "key_1" not in candidates  # Pinned chunk should be excluded
    
    def test_memory_stats_functionality(self):
        """Test NeuronMemoryStats functionality."""
        stats = NeuronMemoryStats()
        
        # Test initialization
        assert stats.total_allocated_bytes == 0
        assert stats.peak_allocated_bytes == 0
        
        # Test allocation tracking
        stats.update_allocation(1000)
        assert stats.total_allocated_bytes == 1000
        assert stats.peak_allocated_bytes == 1000
        
        stats.update_allocation(500)
        assert stats.total_allocated_bytes == 1500
        assert stats.peak_allocated_bytes == 1500
        
        stats.update_deallocation(200)
        assert stats.total_allocated_bytes == 1300
        assert stats.peak_allocated_bytes == 1500  # Peak unchanged
        
        # Test cache statistics
        stats.record_cache_hit()
        stats.record_cache_hit()
        stats.record_cache_miss()
        
        stats_dict = stats.get_stats()
        assert stats_dict["cache_hit_rate"] == 2.0 / 3.0
        assert stats_dict["total_cache_hits"] == 2
        assert stats_dict["total_cache_misses"] == 1
        
        # Test eviction tracking
        stats.record_eviction("low")
        stats.record_eviction("medium")
        
        stats_dict = stats.get_stats()
        assert stats_dict["total_evictions"] == 2
        assert stats_dict["eviction_by_pressure"]["low"] == 1
        assert stats_dict["eviction_by_pressure"]["medium"] == 1


@pytest.mark.skipif(not LMCACHE_AVAILABLE, reason="LMCache not available")
class TestStorageBackendCore:
    """Test core storage backend functionality without full memory allocator."""
    
    def test_storage_backend_basic_operations(self):
        """Test basic storage backend operations without memory allocator."""
        # Create backend without config/metadata to avoid memory allocator issues
        with patch('vllm_neuron.neuron_cpu_storage_backend.NUMADetector'):
            backend = NeuronCPUStorageBackend(
                config=None,
                metadata=None,
                max_memory_gb=0.1,
                chunk_size=256
            )
        
        try:
            # Test basic properties
            assert backend.dst_device == "cpu"
            assert backend.chunk_size == 256
            assert backend.memory_allocator is None  # Should be None without config
            
            # Test contains functionality with manual chunk addition
            key = "test_key"
            assert not backend.contains(key)
            
            # Manually add a chunk for testing
            memory_obj = MockMemoryObj(1024)
            chunk_meta = ChunkMetadata(key, memory_obj, 1024)
            backend.chunk_metadata[key] = chunk_meta
            
            # Test contains now returns True
            assert backend.contains(key)
            assert chunk_meta.access_count == 1  # Should be incremented
            
            # Test get_blocking
            result = backend.get_blocking(key)
            assert result == memory_obj
            assert memory_obj.ref_count == 1
            
            # Test pin/unpin
            assert backend.pin(key)
            assert chunk_meta.is_pinned
            assert memory_obj.pinned
            
            assert backend.unpin(key)
            assert not chunk_meta.is_pinned
            assert not memory_obj.pinned
            
            # Test remove
            assert backend.remove(key)
            assert key not in backend.chunk_metadata
            
            # Test with non-existent key
            assert not backend.contains("non_existent")
            assert backend.get_blocking("non_existent") is None
            assert not backend.pin("non_existent")
            assert not backend.remove("non_existent")
            
        finally:
            backend.close()
    
    def test_storage_backend_memory_management(self):
        """Test memory management functionality."""
        with patch('vllm_neuron.neuron_cpu_storage_backend.NUMADetector'):
            backend = NeuronCPUStorageBackend(
                config=None,
                metadata=None,
                max_memory_gb=0.001,  # Very small for testing
                chunk_size=256
            )
        
        try:
            # Add chunks to test eviction
            chunks_added = 0
            for i in range(10):
                key = f"key_{i}"
                memory_obj = MockMemoryObj(1024)
                chunk_meta = ChunkMetadata(key, memory_obj, 1024)
                chunk_meta.last_access_time = time.time() - (10 - i)  # Different access times
                
                with backend.storage_lock:
                    backend.chunk_metadata[key] = chunk_meta
                    backend.memory_monitor.update_memory_usage(1024)
                    chunks_added += 1
            
            assert chunks_added == 10
            assert len(backend.chunk_metadata) == 10
            
            # Test forced eviction
            evicted = backend.force_eviction(5)
            assert evicted == 5
            assert len(backend.chunk_metadata) == 5
            
            # Test clear cache
            cleared = backend.clear_cache()
            assert cleared == 5
            assert len(backend.chunk_metadata) == 0
            
        finally:
            backend.close()
    
    def test_storage_backend_statistics(self):
        """Test statistics and health monitoring."""
        with patch('vllm_neuron.neuron_cpu_storage_backend.NUMADetector'):
            backend = NeuronCPUStorageBackend(
                config=None,
                metadata=None,
                max_memory_gb=0.1,
                chunk_size=256
            )
        
        try:
            # Test get_stats
            stats = backend.get_stats()
            assert isinstance(stats, dict)
            assert "max_memory_bytes" in stats
            assert "memory_usage_ratio" in stats
            assert "chunk_size" in stats
            assert stats["chunk_size"] == 256
            
            # Test get_health_status
            health = backend.get_health_status()
            assert isinstance(health, dict)
            assert "backend_type" in health
            assert "is_healthy" in health
            assert health["backend_type"] == "NeuronCPUStorageBackend"
            
            # Test eviction strategy setting
            assert backend.set_eviction_strategy("lfu")
            assert backend.eviction_policy.strategy == "lfu"
            
            assert not backend.set_eviction_strategy("invalid")
            assert backend.eviction_policy.strategy == "lfu"  # Should remain unchanged
            
        finally:
            backend.close()
    
    def test_storage_backend_chunk_management(self):
        """Test chunk information and management."""
        with patch('vllm_neuron.neuron_cpu_storage_backend.NUMADetector'):
            backend = NeuronCPUStorageBackend(
                config=None,
                metadata=None,
                max_memory_gb=0.1,
                chunk_size=256
            )
        
        try:
            # Initially no chunks
            assert backend.get_all_chunk_keys() == []
            
            # Add some chunks
            keys = ["key_1", "key_2", "key_3"]
            for key in keys:
                memory_obj = MockMemoryObj(1024)
                chunk_meta = ChunkMetadata(key, memory_obj, 1024)
                backend.chunk_metadata[key] = chunk_meta
            
            # Test get_all_chunk_keys
            all_keys = backend.get_all_chunk_keys()
            assert len(all_keys) == 3
            assert set(all_keys) == set(keys)
            
            # Test get_chunk_info
            info = backend.get_chunk_info("key_1")
            assert info is not None
            assert info["key"] == "key_1"
            assert info["size_bytes"] == 1024
            assert info["access_count"] == 0
            assert not info["is_pinned"]
            
            # Test with non-existent key
            assert backend.get_chunk_info("non_existent") is None
            
        finally:
            backend.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])