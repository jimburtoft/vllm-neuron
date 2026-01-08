#!/usr/bin/env python3
"""
Unit tests for NeuronCPUStorageBackend.

This test suite validates the core functionality of the Neuron-optimized
CPU storage backend for LMCache integration.
"""

import pytest
import threading
import time
import os
import psutil
from unittest.mock import Mock, patch
from typing import Dict, Any, Optional

import torch

# Test imports - handle LMCache availability gracefully
try:
    from vllm_neuron.neuron_cpu_storage_backend import (
        NeuronCPUStorageBackend,
        ChunkMetadata,
        MemoryPressureMonitor,
        LRUEvictionPolicy,
        NeuronMemoryStats,
        NeuronCoreDetector,
        CoreAffinityManager,
        BlockAlignedMemoryManager,
        MultiCoreDistributor,
        LMCACHE_AVAILABLE
    )
    
    if LMCACHE_AVAILABLE:
        from lmcache.config import LMCacheEngineMetadata
        from lmcache.v1.config import LMCacheEngineConfig
        from lmcache.v1.memory_management import MemoryObj, MemoryFormat
        from lmcache.utils import CacheEngineKey
    else:
        # Mock types for testing when LMCache is not available
        LMCacheEngineMetadata = Mock
        LMCacheEngineConfig = Mock
        MemoryObj = Mock
        MemoryFormat = Mock
        CacheEngineKey = str
        
except ImportError as e:
    pytest.skip(f"LMCache integration not available: {e}", allow_module_level=True)


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
        
    def can_evict(self) -> bool:
        return self.evictable and not self.pinned
        
    def get_memory_size(self) -> int:
        return self.size_bytes


class TestChunkMetadata:
    """Test ChunkMetadata functionality."""
    
    def test_chunk_metadata_initialization(self):
        """Test ChunkMetadata initialization."""
        key = "test_key"
        memory_obj = MockMemoryObj(1024)
        size_bytes = 1024
        
        chunk_meta = ChunkMetadata(key, memory_obj, size_bytes)
        
        assert chunk_meta.key == key
        assert chunk_meta.memory_obj == memory_obj
        assert chunk_meta.size_bytes == size_bytes
        assert chunk_meta.access_count == 0
        assert not chunk_meta.is_pinned
        assert chunk_meta.can_evict()
    
    def test_chunk_metadata_access_tracking(self):
        """Test access tracking functionality."""
        chunk_meta = ChunkMetadata("test", MockMemoryObj(), 1024)
        initial_time = chunk_meta.last_access_time
        
        time.sleep(0.01)  # Small delay
        chunk_meta.update_access()
        
        assert chunk_meta.access_count == 1
        assert chunk_meta.last_access_time > initial_time
    
    def test_chunk_metadata_age_calculation(self):
        """Test age and idle time calculations."""
        chunk_meta = ChunkMetadata("test", MockMemoryObj(), 1024)
        
        time.sleep(0.01)  # Small delay
        
        age = chunk_meta.get_age_seconds()
        idle_time = chunk_meta.get_idle_time_seconds()
        
        assert age > 0
        assert idle_time > 0
    
    def test_chunk_metadata_pinning(self):
        """Test chunk pinning functionality."""
        memory_obj = MockMemoryObj()
        chunk_meta = ChunkMetadata("test", memory_obj, 1024)
        
        assert chunk_meta.can_evict()
        
        chunk_meta.is_pinned = True
        memory_obj.pin()
        
        assert not chunk_meta.can_evict()


class TestMemoryPressureMonitor:
    """Test MemoryPressureMonitor functionality."""
    
    def test_memory_pressure_monitor_initialization(self):
        """Test MemoryPressureMonitor initialization."""
        max_memory = 1024 * 1024 * 1024  # 1GB
        monitor = MemoryPressureMonitor(max_memory)
        
        assert monitor.max_memory_bytes == max_memory
        assert monitor.current_memory_bytes == 0
        assert monitor.get_memory_usage_ratio() == 0.0
        assert monitor.get_pressure_level() == "none"
    
    def test_memory_usage_tracking(self):
        """Test memory usage tracking."""
        monitor = MemoryPressureMonitor(1000)
        
        monitor.update_memory_usage(500)
        assert monitor.current_memory_bytes == 500
        assert monitor.get_memory_usage_ratio() == 0.5
        
        monitor.update_memory_usage(-200)
        assert monitor.current_memory_bytes == 300
        assert monitor.get_memory_usage_ratio() == 0.3
        
        # Test negative overflow protection
        monitor.update_memory_usage(-500)
        assert monitor.current_memory_bytes == 0
    
    def test_pressure_level_detection(self):
        """Test pressure level detection."""
        monitor = MemoryPressureMonitor(1000)
        
        # No pressure
        monitor.update_memory_usage(600)  # 60%
        assert monitor.get_pressure_level() == "none"
        
        # Low pressure
        monitor.update_memory_usage(100)  # 70%
        assert monitor.get_pressure_level() == "low"
        
        # Medium pressure
        monitor.update_memory_usage(150)  # 85%
        assert monitor.get_pressure_level() == "medium"
        
        # High pressure
        monitor.update_memory_usage(100)  # 95%
        assert monitor.get_pressure_level() == "high"
    
    def test_eviction_batch_size(self):
        """Test eviction batch size calculation."""
        monitor = MemoryPressureMonitor(1000)
        
        # No pressure
        monitor.update_memory_usage(600)
        assert monitor.get_eviction_batch_size() == 0
        
        # Low pressure
        monitor.update_memory_usage(100)
        assert monitor.get_eviction_batch_size() == monitor.low_pressure_batch_size
        
        # Medium pressure
        monitor.update_memory_usage(150)
        assert monitor.get_eviction_batch_size() == monitor.medium_pressure_batch_size
        
        # High pressure
        monitor.update_memory_usage(100)
        assert monitor.get_eviction_batch_size() == monitor.high_pressure_batch_size


class TestLRUEvictionPolicy:
    """Test LRUEvictionPolicy functionality."""
    
    def test_lru_eviction_strategy(self):
        """Test LRU eviction strategy."""
        policy = LRUEvictionPolicy()
        policy.strategy = "lru"
        
        # Create test chunks with different access times
        chunks = {}
        for i in range(5):
            key = f"key_{i}"
            memory_obj = MockMemoryObj()
            chunk_meta = ChunkMetadata(key, memory_obj, 1024)
            chunk_meta.last_access_time = time.time() - (5 - i)  # Older chunks first
            chunks[key] = chunk_meta
        
        candidates = policy.select_eviction_candidates(chunks, 3)
        
        assert len(candidates) == 3
        assert candidates[0] == "key_0"  # Oldest
        assert candidates[1] == "key_1"
        assert candidates[2] == "key_2"
    
    def test_lfu_eviction_strategy(self):
        """Test LFU eviction strategy."""
        policy = LRUEvictionPolicy()
        policy.strategy = "lfu"
        
        # Create test chunks with different access counts
        chunks = {}
        for i in range(5):
            key = f"key_{i}"
            memory_obj = MockMemoryObj()
            chunk_meta = ChunkMetadata(key, memory_obj, 1024)
            chunk_meta.access_count = i  # Different access counts
            chunks[key] = chunk_meta
        
        candidates = policy.select_eviction_candidates(chunks, 3)
        
        assert len(candidates) == 3
        assert candidates[0] == "key_0"  # Least accessed
        assert candidates[1] == "key_1"
        assert candidates[2] == "key_2"
    
    def test_eviction_respects_pinned_chunks(self):
        """Test that eviction respects pinned chunks."""
        policy = LRUEvictionPolicy()
        
        chunks = {}
        for i in range(3):
            key = f"key_{i}"
            memory_obj = MockMemoryObj()
            chunk_meta = ChunkMetadata(key, memory_obj, 1024)
            if i == 1:  # Pin middle chunk
                chunk_meta.is_pinned = True
                memory_obj.pin()
            chunks[key] = chunk_meta
        
        candidates = policy.select_eviction_candidates(chunks, 3)
        
        assert len(candidates) == 2  # Only unpinned chunks
        assert "key_1" not in candidates  # Pinned chunk excluded


class TestNeuronMemoryStats:
    """Test NeuronMemoryStats functionality."""
    
    def test_memory_stats_initialization(self):
        """Test NeuronMemoryStats initialization."""
        stats = NeuronMemoryStats()
        
        assert stats.total_allocated_bytes == 0
        assert stats.peak_allocated_bytes == 0
        assert stats.current_cached_chunks == 0
        assert stats.total_cache_hits == 0
        assert stats.total_cache_misses == 0
    
    def test_allocation_tracking(self):
        """Test allocation and deallocation tracking."""
        stats = NeuronMemoryStats()
        
        stats.update_allocation(1000)
        assert stats.total_allocated_bytes == 1000
        assert stats.peak_allocated_bytes == 1000
        
        stats.update_allocation(500)
        assert stats.total_allocated_bytes == 1500
        assert stats.peak_allocated_bytes == 1500
        
        stats.update_deallocation(200)
        assert stats.total_allocated_bytes == 1300
        assert stats.peak_allocated_bytes == 1500  # Peak unchanged
    
    def test_cache_hit_rate_calculation(self):
        """Test cache hit rate calculation."""
        stats = NeuronMemoryStats()
        
        # No requests yet
        stats_dict = stats.get_stats()
        assert stats_dict["cache_hit_rate"] == 0.0
        
        # Add some hits and misses
        stats.record_cache_hit()
        stats.record_cache_hit()
        stats.record_cache_miss()
        
        stats_dict = stats.get_stats()
        assert stats_dict["cache_hit_rate"] == 2.0 / 3.0  # 2 hits out of 3 total
    
    def test_eviction_tracking(self):
        """Test eviction tracking by pressure level."""
        stats = NeuronMemoryStats()
        
        stats.record_eviction("low")
        stats.record_eviction("medium")
        stats.record_eviction("high")
        stats.record_eviction("low")
        
        stats_dict = stats.get_stats()
        assert stats_dict["total_evictions"] == 4
        assert stats_dict["eviction_by_pressure"]["low"] == 2
        assert stats_dict["eviction_by_pressure"]["medium"] == 1
        assert stats_dict["eviction_by_pressure"]["high"] == 1


@pytest.mark.skipif(not LMCACHE_AVAILABLE, reason="LMCache not available")
class TestNeuronCPUStorageBackend:
    """Test NeuronCPUStorageBackend functionality."""
    
    @pytest.fixture
    def mock_config(self):
        """Create mock LMCache configuration."""
        config = Mock(spec=LMCacheEngineConfig)
        return config
    
    @pytest.fixture
    def mock_metadata(self):
        """Create mock LMCache metadata."""
        metadata = Mock(spec=LMCacheEngineMetadata)
        metadata.kv_shape = [32, 2, 256, 32, 128]  # [layers, kv_size, chunk_size, heads, head_size]
        metadata.kv_dtype = torch.float16
        return metadata
    
    @pytest.fixture
    def storage_backend(self, mock_config, mock_metadata):
        """Create NeuronCPUStorageBackend instance for testing."""
        with patch('vllm_neuron.neuron_cpu_storage_backend.NUMADetector') as mock_numa:
            # Make NUMA detector return None to avoid GPU mapping issues
            mock_numa.get_numa_mapping.return_value = None
            backend = NeuronCPUStorageBackend(
                config=mock_config,
                metadata=mock_metadata,
                max_memory_gb=0.1,  # Small memory for testing
                chunk_size=256,
                enable_multi_core=False  # Disable multi-core for testing to avoid NUMA issues
            )
            yield backend
            backend.close()
    
    def test_backend_initialization(self, storage_backend):
        """Test storage backend initialization."""
        assert storage_backend.dst_device == "cpu"
        assert storage_backend.chunk_size == 256
        assert storage_backend.max_memory_bytes == int(0.1 * 1024 * 1024 * 1024)
        assert isinstance(storage_backend.memory_monitor, MemoryPressureMonitor)
        assert isinstance(storage_backend.eviction_policy, LRUEvictionPolicy)
        assert isinstance(storage_backend.stats, NeuronMemoryStats)
    
    def test_contains_functionality(self, storage_backend):
        """Test contains functionality."""
        key = "test_key"
        
        # Key should not exist initially
        assert not storage_backend.contains(key)
        
        # Add a mock chunk
        memory_obj = MockMemoryObj(1024)
        chunk_meta = ChunkMetadata(key, memory_obj, 1024)
        storage_backend.chunk_metadata[key] = chunk_meta
        
        # Key should exist now
        assert storage_backend.contains(key)
        
        # Check that access was tracked
        assert chunk_meta.access_count == 1
    
    def test_contains_with_pinning(self, storage_backend):
        """Test contains functionality with pinning."""
        key = "test_key"
        memory_obj = MockMemoryObj(1024)
        chunk_meta = ChunkMetadata(key, memory_obj, 1024)
        storage_backend.chunk_metadata[key] = chunk_meta
        
        # Test pinning
        assert storage_backend.contains(key, pin=True)
        assert chunk_meta.is_pinned
        assert memory_obj.pinned
    
    def test_get_blocking_functionality(self, storage_backend):
        """Test get_blocking functionality."""
        key = "test_key"
        
        # Should return None for non-existent key
        result = storage_backend.get_blocking(key)
        assert result is None
        
        # Add a mock chunk
        memory_obj = MockMemoryObj(1024)
        chunk_meta = ChunkMetadata(key, memory_obj, 1024)
        storage_backend.chunk_metadata[key] = chunk_meta
        
        # Should return the memory object
        result = storage_backend.get_blocking(key)
        assert result == memory_obj
        assert memory_obj.ref_count == 1  # Ref count should be incremented
    
    def test_pin_unpin_functionality(self, storage_backend):
        """Test pin and unpin functionality."""
        key = "test_key"
        memory_obj = MockMemoryObj(1024)
        chunk_meta = ChunkMetadata(key, memory_obj, 1024)
        storage_backend.chunk_metadata[key] = chunk_meta
        
        # Test pinning
        assert storage_backend.pin(key)
        assert chunk_meta.is_pinned
        assert memory_obj.pinned
        
        # Test unpinning
        assert storage_backend.unpin(key)
        assert not chunk_meta.is_pinned
        assert not memory_obj.pinned
        
        # Test with non-existent key
        assert not storage_backend.pin("non_existent")
        assert not storage_backend.unpin("non_existent")
    
    def test_remove_functionality(self, storage_backend):
        """Test remove functionality."""
        key = "test_key"
        memory_obj = MockMemoryObj(1024)
        chunk_meta = ChunkMetadata(key, memory_obj, 1024)
        storage_backend.chunk_metadata[key] = chunk_meta
        storage_backend.memory_monitor.update_memory_usage(1024)
        storage_backend.stats.current_cached_chunks = 1
        
        # Test removal
        assert storage_backend.remove(key)
        assert key not in storage_backend.chunk_metadata
        assert storage_backend.memory_monitor.current_memory_bytes == 0
        assert storage_backend.stats.current_cached_chunks == 0
        
        # Test removing non-existent key
        assert not storage_backend.remove("non_existent")
    
    def test_memory_pressure_eviction(self, storage_backend):
        """Test eviction under memory pressure."""
        # Add multiple chunks to trigger eviction
        for i in range(10):
            key = f"key_{i}"
            memory_obj = MockMemoryObj(1024 * 1024)  # 1MB each
            chunk_meta = ChunkMetadata(key, memory_obj, 1024 * 1024)
            chunk_meta.last_access_time = time.time() - (10 - i)  # Different access times
            storage_backend.chunk_metadata[key] = chunk_meta
            storage_backend.memory_monitor.update_memory_usage(1024 * 1024)
        
        initial_count = len(storage_backend.chunk_metadata)
        
        # Force eviction
        evicted = storage_backend.force_eviction(5)
        
        assert evicted == 5
        assert len(storage_backend.chunk_metadata) == initial_count - 5
        
        # Check that oldest chunks were evicted (LRU)
        remaining_keys = list(storage_backend.chunk_metadata.keys())
        assert "key_0" not in remaining_keys  # Oldest should be evicted
        assert "key_9" in remaining_keys  # Newest should remain
    
    def test_clear_cache_functionality(self, storage_backend):
        """Test cache clearing functionality."""
        # Add some chunks
        for i in range(5):
            key = f"key_{i}"
            memory_obj = MockMemoryObj(1024)
            chunk_meta = ChunkMetadata(key, memory_obj, 1024)
            if i == 2:  # Pin one chunk
                chunk_meta.is_pinned = True
                memory_obj.pin()
            storage_backend.chunk_metadata[key] = chunk_meta
        
        # Clear without force (should skip pinned)
        cleared = storage_backend.clear_cache(force=False)
        assert cleared == 4  # 4 unpinned chunks
        assert len(storage_backend.chunk_metadata) == 1  # 1 pinned chunk remains
        
        # Clear with force (should clear all)
        cleared = storage_backend.clear_cache(force=True)
        assert cleared == 1  # 1 remaining chunk
        assert len(storage_backend.chunk_metadata) == 0
    
    def test_get_stats_functionality(self, storage_backend):
        """Test statistics retrieval."""
        stats = storage_backend.get_stats()
        
        assert isinstance(stats, dict)
        assert "max_memory_bytes" in stats
        assert "memory_usage_ratio" in stats
        assert "memory_pressure_level" in stats
        assert "numa_node" in stats
        assert "chunk_size" in stats
        assert "cache_hit_rate" in stats
    
    def test_get_health_status_functionality(self, storage_backend):
        """Test health status retrieval."""
        health = storage_backend.get_health_status()
        
        assert isinstance(health, dict)
        assert "backend_type" in health
        assert "is_healthy" in health
        assert "memory_allocator_initialized" in health
        assert "background_eviction_running" in health
        assert "stats" in health
        
        assert health["backend_type"] == "NeuronCPUStorageBackend"
    
    def test_eviction_strategy_setting(self, storage_backend):
        """Test eviction strategy configuration."""
        # Test valid strategies
        assert storage_backend.set_eviction_strategy("lru")
        assert storage_backend.eviction_policy.strategy == "lru"
        
        assert storage_backend.set_eviction_strategy("lfu")
        assert storage_backend.eviction_policy.strategy == "lfu"
        
        assert storage_backend.set_eviction_strategy("aging")
        assert storage_backend.eviction_policy.strategy == "aging"
        
        assert storage_backend.set_eviction_strategy("hybrid")
        assert storage_backend.eviction_policy.strategy == "hybrid"
        
        # Test invalid strategy
        assert not storage_backend.set_eviction_strategy("invalid")
        assert storage_backend.eviction_policy.strategy == "hybrid"  # Should remain unchanged
    
    def test_chunk_info_retrieval(self, storage_backend):
        """Test chunk information retrieval."""
        key = "test_key"
        memory_obj = MockMemoryObj(1024)
        chunk_meta = ChunkMetadata(key, memory_obj, 1024)
        storage_backend.chunk_metadata[key] = chunk_meta
        
        info = storage_backend.get_chunk_info(key)
        
        assert info is not None
        assert info["key"] == str(key)
        assert info["size_bytes"] == 1024
        assert info["access_count"] == 0
        assert "age_seconds" in info
        assert "idle_time_seconds" in info
        assert not info["is_pinned"]
        assert info["can_evict"]
        
        # Test non-existent key
        assert storage_backend.get_chunk_info("non_existent") is None
    
    def test_get_all_chunk_keys(self, storage_backend):
        """Test retrieving all chunk keys."""
        # Initially empty
        assert storage_backend.get_all_chunk_keys() == []
        
        # Add some chunks
        keys = ["key_1", "key_2", "key_3"]
        for key in keys:
            memory_obj = MockMemoryObj(1024)
            chunk_meta = ChunkMetadata(key, memory_obj, 1024)
            storage_backend.chunk_metadata[key] = chunk_meta
        
        all_keys = storage_backend.get_all_chunk_keys()
        assert len(all_keys) == 3
        assert set(all_keys) == set(keys)
    
    def test_background_eviction_thread(self, storage_backend):
        """Test background eviction thread functionality."""
        # Background eviction should be enabled by default
        assert storage_backend.background_eviction_enabled
        assert storage_backend.eviction_thread is not None
        assert storage_backend.eviction_thread.is_alive()
        
        # Test that thread stops on close
        storage_backend.close()
        time.sleep(0.1)  # Give thread time to stop
        assert not storage_backend.eviction_thread.is_alive()


class TestMultiCoreOptimization:
    """Test multi-core optimization functionality."""
    
    def test_neuron_core_detector(self):
        """Test NeuronCoreDetector functionality."""
        # Test with no environment variable
        with patch.dict(os.environ, {}, clear=True):
            detector = NeuronCoreDetector()
            assert detector.get_all_cores() == [0]
            assert detector.core_count == 1
            assert not detector.is_multi_core()
        
        # Test with range format
        with patch.dict(os.environ, {"NEURON_RT_VISIBLE_CORES": "0-3"}):
            detector = NeuronCoreDetector()
            assert detector.get_all_cores() == [0, 1, 2, 3]
            assert detector.core_count == 4
            assert detector.is_multi_core()
        
        # Test with comma-separated format
        with patch.dict(os.environ, {"NEURON_RT_VISIBLE_CORES": "0,2,4"}):
            detector = NeuronCoreDetector()
            assert detector.get_all_cores() == [0, 2, 4]
            assert detector.core_count == 3
            assert detector.is_multi_core()
        
        # Test core assignment consistency
        detector = NeuronCoreDetector()
        core1 = detector.get_core_for_operation("test_op")
        core2 = detector.get_core_for_operation("test_op")
        assert core1 == core2  # Same operation should get same core
    
    def test_core_affinity_manager(self):
        """Test CoreAffinityManager functionality."""
        neuron_cores = [0, 1, 2, 3]
        manager = CoreAffinityManager(neuron_cores)
        
        # Test affinity mapping creation
        assert len(manager.affinity_mapping) == 4
        for core in neuron_cores:
            assert core in manager.affinity_mapping
            assert isinstance(manager.affinity_mapping[core], list)
            assert len(manager.affinity_mapping[core]) > 0
        
        # Test affinity setting (may fail due to permissions, but shouldn't crash)
        try:
            result = manager.set_affinity_for_neuron_core(0)
            assert isinstance(result, bool)
        except (AttributeError, psutil.AccessDenied):
            pass  # Expected on systems without affinity support
    
    def test_block_aligned_memory_manager(self):
        """Test BlockAlignedMemoryManager functionality."""
        manager = BlockAlignedMemoryManager(block_size=4096)
        
        # Test alignment calculations
        assert manager.calculate_aligned_size(100) == 4096
        assert manager.calculate_aligned_size(4096) == 4096
        assert manager.calculate_aligned_size(4097) == 8192
        assert manager.calculate_aligned_size(0) == 4096
        
        # Test alignment checking
        assert manager.is_aligned(4096)
        assert manager.is_aligned(8192)
        assert not manager.is_aligned(4097)
        assert not manager.is_aligned(100)
        
        # Test statistics tracking
        stats = manager.get_alignment_stats()
        assert "aligned_allocations" in stats
        assert "unaligned_allocations" in stats
        assert "total_padding_bytes" in stats
    
    def test_multi_core_distributor_initialization(self):
        """Test MultiCoreDistributor initialization."""
        distributor = MultiCoreDistributor(max_workers=4)
        
        assert distributor.core_detector is not None
        assert distributor.affinity_manager is not None
        assert distributor.memory_manager is not None
        assert distributor.executor is not None
        assert distributor.max_workers == 4
        
        # Test load balance stats
        stats = distributor.get_load_balance_stats()
        assert "operations_per_core" in stats
        assert "core_count" in stats
        assert "worker_count" in stats
        
        distributor.shutdown()
    
    def test_multi_core_operation_distribution(self):
        """Test cache operation distribution across cores."""
        distributor = MultiCoreDistributor(max_workers=2)
        
        def test_operation(data):
            return f"processed_{data}"
        
        # Test single operation
        future = distributor.distribute_cache_operation(
            test_operation, "test_op", "test_data"
        )
        result = future.result(timeout=1.0)
        assert result == "processed_test_data"
        
        # Test batch operations
        operation_items = [("op1", "data1"), ("op2", "data2")]
        futures = distributor.distribute_batch_operation(
            test_operation, operation_items
        )
        
        results = [f.result(timeout=1.0) for f in futures]
        assert results == ["processed_data1", "processed_data2"]
        
        distributor.shutdown()
    
    def test_multi_core_error_handling(self):
        """Test error handling in multi-core operations."""
        distributor = MultiCoreDistributor(max_workers=2)
        
        def failing_operation(data):
            raise ValueError(f"Test error for {data}")
        
        # Test single operation error
        future = distributor.distribute_cache_operation(
            failing_operation, "test_op", "test_data"
        )
        
        with pytest.raises(ValueError, match="Test error for test_data"):
            future.result(timeout=1.0)
        
        # Check error statistics
        stats = distributor.get_load_balance_stats()
        assert stats["failed_operations"] > 0
        
        distributor.shutdown()
    
    def test_storage_backend_with_multi_core(self):
        """Test storage backend with multi-core enabled."""
        with patch('vllm_neuron.neuron_cpu_storage_backend.NUMADetector') as mock_numa:
            mock_numa.get_numa_mapping.return_value = None
            
            backend = NeuronCPUStorageBackend(
                max_memory_gb=0.1,
                chunk_size=256,
                enable_multi_core=True,
                max_workers=2
            )
            
            # Test multi-core info
            info = backend.get_multi_core_info()
            assert info["enabled"] is True
            assert "neuron_cores" in info
            assert "core_count" in info
            assert "worker_count" in info
            
            # Test workload optimization
            assert backend.optimize_for_workload("conversation")
            assert backend.optimize_for_workload("batch")
            assert backend.optimize_for_workload("inference")
            assert backend.optimize_for_workload("mixed")
            
            # Test multi-core enable/disable
            assert backend.set_multi_core_enabled(False)
            assert not backend.enable_multi_core
            
            backend.close()
    
    def test_storage_backend_multi_core_operations(self):
        """Test storage operations with multi-core distribution."""
        with patch('vllm_neuron.neuron_cpu_storage_backend.NUMADetector') as mock_numa:
            mock_numa.get_numa_mapping.return_value = None
            
            backend = NeuronCPUStorageBackend(
                max_memory_gb=0.1,
                chunk_size=256,
                enable_multi_core=True,
                max_workers=2
            )
            
            # Create mock memory objects
            mock_keys = [f"test_key_{i}" for i in range(3)]
            mock_objs = []
            
            for i in range(3):
                mock_obj = Mock()
                mock_obj.ref_count_up = Mock()
                mock_obj.ref_count_down = Mock()
                mock_obj.can_evict = True
                mock_obj.data = torch.randn(10, 10)  # Mock tensor data
                mock_objs.append(mock_obj)
            
            # Test batched put with multi-core
            with patch.object(backend, '_estimate_memory_obj_size', return_value=1024):
                futures = backend.batched_submit_put_task(mock_keys, mock_objs)
                
                if futures:  # Multi-core returns futures
                    for future in futures:
                        result = future.result(timeout=1.0)
                        assert isinstance(result, bool)
            
            backend.close()


class TestIntegrationScenarios:
    """Test integration scenarios and edge cases."""
    
    @pytest.mark.skipif(not LMCACHE_AVAILABLE, reason="LMCache not available")
    def test_concurrent_access(self):
        """Test concurrent access to storage backend."""
        with patch('vllm_neuron.neuron_cpu_storage_backend.NUMADetector'):
            backend = NeuronCPUStorageBackend(max_memory_gb=0.1)
        
        try:
            # Add initial chunks
            for i in range(10):
                key = f"key_{i}"
                memory_obj = MockMemoryObj(1024)
                chunk_meta = ChunkMetadata(key, memory_obj, 1024)
                backend.chunk_metadata[key] = chunk_meta
            
            # Define concurrent operations
            def reader_thread():
                for i in range(10):
                    key = f"key_{i}"
                    backend.contains(key)
                    backend.get_blocking(key)
            
            def writer_thread():
                for i in range(10, 20):
                    key = f"key_{i}"
                    memory_obj = MockMemoryObj(1024)
                    chunk_meta = ChunkMetadata(key, memory_obj, 1024)
                    with backend.storage_lock:
                        backend.chunk_metadata[key] = chunk_meta
            
            def evictor_thread():
                backend.force_eviction(5)
            
            # Run concurrent operations
            threads = [
                threading.Thread(target=reader_thread),
                threading.Thread(target=writer_thread),
                threading.Thread(target=evictor_thread),
            ]
            
            for thread in threads:
                thread.start()
            
            for thread in threads:
                thread.join()
            
            # Verify no corruption occurred
            stats = backend.get_stats()
            assert isinstance(stats, dict)
            
        finally:
            backend.close()
    
    @pytest.mark.skipif(not LMCACHE_AVAILABLE, reason="LMCache not available")
    def test_memory_exhaustion_scenario(self):
        """Test behavior under memory exhaustion."""
        with patch('vllm_neuron.neuron_cpu_storage_backend.NUMADetector'):
            backend = NeuronCPUStorageBackend(max_memory_gb=0.001)  # Very small memory
        
        try:
            # Try to add more chunks than memory allows
            added_chunks = 0
            for i in range(100):
                key = f"key_{i}"
                memory_obj = MockMemoryObj(1024 * 1024)  # 1MB each
                chunk_meta = ChunkMetadata(key, memory_obj, 1024 * 1024)
                
                with backend.storage_lock:
                    if backend._ensure_memory_available(1024 * 1024):
                        backend.chunk_metadata[key] = chunk_meta
                        backend.memory_monitor.update_memory_usage(1024 * 1024)
                        added_chunks += 1
                    else:
                        break
            
            # Should have added some chunks but not all due to memory limits
            assert 0 < added_chunks < 100
            assert backend.memory_monitor.get_pressure_level() in ["medium", "high"]
            
        finally:
            backend.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])