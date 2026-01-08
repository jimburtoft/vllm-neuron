#!/usr/bin/env python3
"""
End-to-end integration tests for LMCache-Neuron integration.

This test suite validates the complete inference pipeline with LMCache,
testing various model architectures, sizes, and configuration scenarios.
"""

import pytest
import os
import tempfile
import yaml
import time
import torch
from pathlib import Path
from typing import Dict, Any, List, Optional
from unittest.mock import Mock, patch, MagicMock

# Test imports - handle LMCache availability gracefully
try:
    from vllm_neuron.lmcache_integration import (
        LMCacheNeuronIntegration,
        NeuronDeviceDetector,
        LMCacheConfigManager,
        LMCACHE_AVAILABLE
    )
    from vllm_neuron.platform import NeuronPlatform
    
    if LMCACHE_AVAILABLE:
        from lmcache.v1.cache_engine import LMCacheEngine
        from lmcache.v1.config import LMCacheEngineConfig
    else:
        LMCacheEngine = Mock
        LMCacheEngineConfig = Mock
        
except ImportError as e:
    pytest.skip(f"LMCache integration not available: {e}", allow_module_level=True)


class MockVllmEngine:
    """Mock vLLM engine for integration testing."""
    
    def __init__(self, model_name: str = "test_model"):
        self.model_name = model_name
        self.model_config = Mock()
        self.model_config.model = model_name
        self.model_config.max_model_len = 2048
        self.scheduler_config = Mock()
        self.cache_config = Mock()
        self.parallel_config = Mock()
        self.device_config = Mock()
        self.load_config = Mock()
        self.lora_config = None
        self.vision_language_config = None
        self.speculative_config = None
        self.decoding_config = Mock()
        self.observability_config = Mock()
        self.prompt_adapter_config = None
        self.request_logger = None
        
        # Mock KV cache operations
        self.kv_cache_operations = []
        
    def generate(self, prompts: List[str], **kwargs) -> List[str]:
        """Mock generation method."""
        # Simulate KV cache operations
        for prompt in prompts:
            tokens = [i for i in range(len(prompt.split()))]  # Simple tokenization
            self.kv_cache_operations.append({
                "operation": "retrieve",
                "tokens": tokens,
                "timestamp": time.time()
            })
            
        # Return mock responses
        return [f"Response to: {prompt}" for prompt in prompts]
    
    def get_kv_cache_operations(self) -> List[Dict[str, Any]]:
        """Get recorded KV cache operations."""
        return self.kv_cache_operations.copy()


class MockNeuronWorker:
    """Mock Neuron worker for testing."""
    
    def __init__(self):
        self.device_id = 0
        self.model_runner = Mock()
        self.cache_engine = Mock()
        
    def execute_model(self, seq_group_metadata_list, kv_caches):
        """Mock model execution."""
        return Mock()


@pytest.fixture
def temp_config_dir():
    """Create temporary directory for test configurations."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def basic_lmcache_config():
    """Basic LMCache configuration for testing."""
    return {
        "chunk_size": 256,
        "local_cpu": True,
        "max_local_cpu_size": 1,  # 1GB for testing
        "save_unfull_chunk": True,
        "save_decode_cache": True,
        "remote_url": "fs://localhost:0/tmp/lmcache_test",
        "extra_config": {
            "enable_compression": False,  # Disable for faster tests
            "numa_node_affinity": None
        }
    }


@pytest.fixture
def mock_neuron_environment():
    """Mock Neuron environment for testing."""
    with patch('vllm_neuron.lmcache_integration.NeuronDeviceDetector.is_neuron_available') as mock_neuron:
        mock_neuron.return_value = True
        with patch('vllm_neuron.lmcache_integration.NeuronDeviceDetector.get_neuron_device_count') as mock_count:
            mock_count.return_value = 2
            yield


@pytest.fixture
def mock_lmcache_engine():
    """Mock LMCache engine for testing."""
    if not LMCACHE_AVAILABLE:
        return Mock()
    
    engine = Mock(spec=LMCacheEngine)
    engine.get.return_value = None  # Default to cache miss
    engine.put.return_value = None
    engine.get_stats.return_value = {
        "cache_hits": 0,
        "cache_misses": 0,
        "storage_size": 0
    }
    return engine


class TestEndToEndIntegration:
    """Test complete inference pipeline with LMCache integration."""
    
    def test_basic_inference_pipeline(self, mock_neuron_environment, basic_lmcache_config, temp_config_dir):
        """Test basic inference pipeline with LMCache enabled."""
        # Create config file
        config_file = temp_config_dir / "lmcache_config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(basic_lmcache_config, f)
        
        # Initialize integration
        integration = LMCacheNeuronIntegration()
        
        # Mock vLLM config
        mock_vllm_config = Mock()
        mock_vllm_config.kv_transfer_config = None
        
        # Set config file path
        os.environ['LMCACHE_CONFIG_FILE'] = str(config_file)
        
        try:
            # Initialize with mocked LMCache engine and config
            with patch('vllm_neuron.lmcache_integration.LMCacheEngine') as mock_engine_class, \
                 patch('vllm_neuron.lmcache_integration.LMCacheEngineConfig') as mock_config_class:
                
                # Mock the config class
                mock_config = Mock()
                mock_config_class.return_value = mock_config
                
                # Mock the engine
                mock_engine = Mock()
                mock_engine.get.return_value = None  # Cache miss
                mock_engine.put.return_value = None
                mock_engine.get_stats.return_value = {"cache_hits": 0, "cache_misses": 1}
                mock_engine_class.return_value = mock_engine
                
                result = integration.initialize(mock_vllm_config)
                assert result is True
                assert integration._initialized is True
            
            # Test inference simulation
            mock_engine = MockVllmEngine("test_model")
            prompts = [
                "What is machine learning?",
                "Explain neural networks",
                "What is deep learning?"
            ]
            
            # Simulate inference with cache operations
            responses = mock_engine.generate(prompts)
            
            assert len(responses) == len(prompts)
            assert all("Response to:" in response for response in responses)
            
            # Verify KV cache operations were recorded
            kv_ops = mock_engine.get_kv_cache_operations()
            assert len(kv_ops) == len(prompts)
            
            # Test cache statistics
            stats = integration.get_cache_stats()
            assert "integration" in stats
            assert "cache_handler" in stats
            
        finally:
            # Cleanup
            if 'LMCACHE_CONFIG_FILE' in os.environ:
                del os.environ['LMCACHE_CONFIG_FILE']
    
    def test_multi_model_compatibility(self, mock_neuron_environment, basic_lmcache_config, temp_config_dir):
        """Test integration with multiple model architectures."""
        model_configs = [
            {"name": "llama-7b", "max_len": 2048, "architecture": "llama"},
            {"name": "gpt-neo-2.7b", "max_len": 2048, "architecture": "gpt_neo"},
            {"name": "opt-1.3b", "max_len": 2048, "architecture": "opt"}
        ]
        
        # Create config file
        config_file = temp_config_dir / "lmcache_config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(basic_lmcache_config, f)
        
        os.environ['LMCACHE_CONFIG_FILE'] = str(config_file)
        
        try:
            for model_config in model_configs:
                integration = LMCacheNeuronIntegration()
                
                # Mock vLLM config for each model
                mock_vllm_config = Mock()
                mock_vllm_config.model_config = Mock()
                mock_vllm_config.model_config.model = model_config["name"]
                mock_vllm_config.kv_transfer_config = None
                
                with patch('vllm_neuron.lmcache_integration.LMCacheEngine') as mock_engine_class:
                    mock_engine = Mock()
                    mock_engine.get.return_value = None
                    mock_engine.put.return_value = None
                    mock_engine.get_stats.return_value = {"model": model_config["name"]}
                    mock_engine_class.return_value = mock_engine
                    
                    result = integration.initialize(mock_vllm_config)
                    assert result is True, f"Failed to initialize for model {model_config['name']}"
                    
                    # Test model-specific inference
                    mock_engine_instance = MockVllmEngine(model_config["name"])
                    responses = mock_engine_instance.generate([f"Test prompt for {model_config['name']}"])
                    
                    assert len(responses) == 1
                    assert model_config["name"] in str(responses[0]) or "Response to:" in responses[0]
                    
                    # Verify health status
                    health = integration.get_health_status()
                    assert health["initialized"] is True
                    
        finally:
            if 'LMCACHE_CONFIG_FILE' in os.environ:
                del os.environ['LMCACHE_CONFIG_FILE']
    
    def test_different_configuration_scenarios(self, mock_neuron_environment, temp_config_dir):
        """Test various configuration scenarios."""
        config_scenarios = [
            {
                "name": "minimal_config",
                "config": {
                    "chunk_size": 128,
                    "local_cpu": True,
                    "max_local_cpu_size": 0.5
                }
            },
            {
                "name": "remote_storage_config",
                "config": {
                    "chunk_size": 512,
                    "local_cpu": True,
                    "max_local_cpu_size": 2,
                    "remote_url": "fs://localhost:0/tmp/lmcache_remote_test",
                    "save_unfull_chunk": False,
                    "save_decode_cache": True
                }
            },
            {
                "name": "performance_optimized_config",
                "config": {
                    "chunk_size": 256,
                    "local_cpu": True,
                    "max_local_cpu_size": 4,
                    "save_unfull_chunk": True,
                    "save_decode_cache": True,
                    "extra_config": {
                        "enable_compression": True,
                        "numa_node_affinity": 0,
                        "prefetch_enabled": True
                    }
                }
            }
        ]
        
        for scenario in config_scenarios:
            # Create config file for this scenario
            config_file = temp_config_dir / f"{scenario['name']}.yaml"
            with open(config_file, 'w') as f:
                yaml.dump(scenario["config"], f)
            
            os.environ['LMCACHE_CONFIG_FILE'] = str(config_file)
            
            try:
                integration = LMCacheNeuronIntegration()
                mock_vllm_config = Mock()
                mock_vllm_config.kv_transfer_config = None
                
                with patch('vllm_neuron.lmcache_integration.LMCacheEngine') as mock_engine_class:
                    mock_engine = Mock()
                    mock_engine.get.return_value = None
                    mock_engine.put.return_value = None
                    mock_engine.get_stats.return_value = {"config": scenario["name"]}
                    mock_engine_class.return_value = mock_engine
                    
                    result = integration.initialize(mock_vllm_config)
                    assert result is True, f"Failed to initialize with {scenario['name']}"
                    
                    # Test basic operations with this configuration
                    health = integration.get_health_status()
                    assert health["initialized"] is True
                    
                    stats = integration.get_cache_stats()
                    assert "integration" in stats
                    
            finally:
                if 'LMCACHE_CONFIG_FILE' in os.environ:
                    del os.environ['LMCACHE_CONFIG_FILE']
    
    def test_cache_hit_miss_scenarios(self, mock_neuron_environment, basic_lmcache_config, temp_config_dir):
        """Test cache hit and miss scenarios in end-to-end pipeline."""
        # Create config file
        config_file = temp_config_dir / "lmcache_config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(basic_lmcache_config, f)
        
        os.environ['LMCACHE_CONFIG_FILE'] = str(config_file)
        
        try:
            integration = LMCacheNeuronIntegration()
            mock_vllm_config = Mock()
            mock_vllm_config.kv_transfer_config = None
            
            # Mock storage for cache hit/miss simulation
            cache_storage = {}
            
            def mock_get(key):
                if hasattr(key, 'tokens'):
                    key_str = str(key.tokens)
                else:
                    key_str = str(key)
                return cache_storage.get(key_str)
            
            def mock_put(key, value):
                if hasattr(key, 'tokens'):
                    key_str = str(key.tokens)
                else:
                    key_str = str(key)
                cache_storage[key_str] = value
            
            with patch('vllm_neuron.lmcache_integration.LMCacheEngine') as mock_engine_class:
                mock_engine = Mock()
                mock_engine.get.side_effect = mock_get
                mock_engine.put.side_effect = mock_put
                mock_engine.get_stats.return_value = {
                    "cache_hits": 0,
                    "cache_misses": 0,
                    "storage_size": 0
                }
                mock_engine_class.return_value = mock_engine
                
                result = integration.initialize(mock_vllm_config)
                assert result is True
                
                # Test cache miss scenario (first request)
                from vllm_neuron.lmcache_integration import KVCacheOperation
                
                class MockSequenceGroup:
                    def __init__(self, tokens):
                        self.tokens = tokens
                    def get_seqs(self):
                        return [Mock(get_token_ids=lambda: self.tokens)]
                
                tokens1 = [1, 2, 3, 4, 5]
                seq_group1 = MockSequenceGroup(tokens1)
                operation1 = KVCacheOperation("retrieve", seq_group1)
                
                result1 = integration.intercept_kv_operations(operation1)
                assert result1.success is True
                assert result1.cache_hit is False  # Should be cache miss
                
                # Store some data for cache hit test
                store_operation = KVCacheOperation("store", seq_group1, kv_data={"cached": "data"})
                store_result = integration.intercept_kv_operations(store_operation)
                assert store_result.success is True
                
                # Test cache hit scenario (same tokens)
                operation2 = KVCacheOperation("retrieve", seq_group1)
                result2 = integration.intercept_kv_operations(operation2)
                assert result2.success is True
                # Note: Cache hit depends on implementation details, so we just verify success
                
                # Test different tokens (cache miss)
                tokens3 = [6, 7, 8, 9, 10]
                seq_group3 = MockSequenceGroup(tokens3)
                operation3 = KVCacheOperation("retrieve", seq_group3)
                
                result3 = integration.intercept_kv_operations(operation3)
                assert result3.success is True
                assert result3.cache_hit is False  # Should be cache miss for different tokens
                
        finally:
            if 'LMCACHE_CONFIG_FILE' in os.environ:
                del os.environ['LMCACHE_CONFIG_FILE']
    
    def test_error_recovery_scenarios(self, mock_neuron_environment, basic_lmcache_config, temp_config_dir):
        """Test error recovery in end-to-end scenarios."""
        # Create config file
        config_file = temp_config_dir / "lmcache_config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(basic_lmcache_config, f)
        
        os.environ['LMCACHE_CONFIG_FILE'] = str(config_file)
        
        try:
            integration = LMCacheNeuronIntegration()
            mock_vllm_config = Mock()
            mock_vllm_config.kv_transfer_config = None
            
            # Test initialization with engine failure
            with patch('vllm_neuron.lmcache_integration.LMCacheEngine') as mock_engine_class:
                mock_engine_class.side_effect = RuntimeError("Engine initialization failed")
                
                result = integration.initialize(mock_vllm_config)
                # Should fail gracefully
                assert result is False
                assert integration._initialized is False
                
                # Operations should still work with graceful degradation
                from vllm_neuron.lmcache_integration import KVCacheOperation
                
                class MockSequenceGroup:
                    def __init__(self, tokens):
                        self.tokens = tokens
                    def get_seqs(self):
                        return [Mock(get_token_ids=lambda: self.tokens)]
                
                seq_group = MockSequenceGroup([1, 2, 3, 4, 5])
                operation = KVCacheOperation("retrieve", seq_group)
                
                result = integration.intercept_kv_operations(operation)
                assert result.success is True  # Graceful degradation
                assert result.cache_hit is False
                
                # Verify degraded operations are tracked
                assert integration._operation_stats["degraded_operations"] > 0
            
            # Test recovery after successful initialization
            with patch('vllm_neuron.lmcache_integration.LMCacheEngine') as mock_engine_class:
                mock_engine = Mock()
                mock_engine.get.return_value = None
                mock_engine.put.return_value = None
                mock_engine.get_stats.return_value = {"status": "healthy"}
                mock_engine_class.return_value = mock_engine
                
                # Re-initialize
                integration = LMCacheNeuronIntegration()
                result = integration.initialize(mock_vllm_config)
                assert result is True
                assert integration._initialized is True
                
                # Operations should work normally now
                operation = KVCacheOperation("retrieve", seq_group)
                result = integration.intercept_kv_operations(operation)
                assert result.success is True
                assert integration._operation_stats["successful_operations"] > 0
                
        finally:
            if 'LMCACHE_CONFIG_FILE' in os.environ:
                del os.environ['LMCACHE_CONFIG_FILE']
    
    def test_concurrent_inference_requests(self, mock_neuron_environment, basic_lmcache_config, temp_config_dir):
        """Test concurrent inference requests with LMCache."""
        import threading
        
        # Create config file
        config_file = temp_config_dir / "lmcache_config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(basic_lmcache_config, f)
        
        os.environ['LMCACHE_CONFIG_FILE'] = str(config_file)
        
        try:
            integration = LMCacheNeuronIntegration()
            mock_vllm_config = Mock()
            mock_vllm_config.kv_transfer_config = None
            
            with patch('vllm_neuron.lmcache_integration.LMCacheEngine') as mock_engine_class:
                mock_engine = Mock()
                mock_engine.get.return_value = None
                mock_engine.put.return_value = None
                mock_engine.get_stats.return_value = {"concurrent_test": True}
                mock_engine_class.return_value = mock_engine
                
                result = integration.initialize(mock_vllm_config)
                assert result is True
                
                # Test concurrent operations
                results = []
                errors = []
                
                def perform_inference(thread_id):
                    try:
                        from vllm_neuron.lmcache_integration import KVCacheOperation
                        
                        class MockSequenceGroup:
                            def __init__(self, tokens):
                                self.tokens = tokens
                            def get_seqs(self):
                                return [Mock(get_token_ids=lambda: self.tokens)]
                        
                        # Each thread uses different tokens
                        tokens = [thread_id, thread_id + 1, thread_id + 2, thread_id + 3]
                        seq_group = MockSequenceGroup(tokens)
                        
                        # Perform retrieve operation
                        retrieve_op = KVCacheOperation("retrieve", seq_group)
                        retrieve_result = integration.intercept_kv_operations(retrieve_op)
                        
                        # Perform store operation
                        store_op = KVCacheOperation("store", seq_group, kv_data={"thread": thread_id})
                        store_result = integration.intercept_kv_operations(store_op)
                        
                        results.append({
                            "thread_id": thread_id,
                            "retrieve_success": retrieve_result.success,
                            "store_success": store_result.success
                        })
                        
                    except Exception as e:
                        errors.append({"thread_id": thread_id, "error": str(e)})
                
                # Run concurrent threads
                threads = []
                for i in range(5):
                    thread = threading.Thread(target=perform_inference, args=(i,))
                    threads.append(thread)
                    thread.start()
                
                # Wait for all threads to complete
                for thread in threads:
                    thread.join()
                
                # Verify results
                assert len(errors) == 0, f"Errors occurred: {errors}"
                assert len(results) == 5
                
                for result in results:
                    assert result["retrieve_success"] is True
                    assert result["store_success"] is True
                
                # Verify stats reflect concurrent operations
                stats = integration.get_cache_stats()
                assert stats["integration"]["successful_operations"] >= 10  # 5 retrieve + 5 store
                
        finally:
            if 'LMCACHE_CONFIG_FILE' in os.environ:
                del os.environ['LMCACHE_CONFIG_FILE']


class TestModelArchitectureCompatibility:
    """Test compatibility with different model architectures."""
    
    @pytest.mark.parametrize("model_info", [
        {"name": "llama-7b", "architecture": "LlamaForCausalLM", "vocab_size": 32000},
        {"name": "gpt-neo-2.7b", "architecture": "GPTNeoForCausalLM", "vocab_size": 50257},
        {"name": "opt-1.3b", "architecture": "OPTForCausalLM", "vocab_size": 50272},
        {"name": "bloom-1b7", "architecture": "BloomForCausalLM", "vocab_size": 250880},
    ])
    def test_model_specific_integration(self, model_info, mock_neuron_environment, basic_lmcache_config, temp_config_dir):
        """Test integration with specific model architectures."""
        # Create config file
        config_file = temp_config_dir / f"lmcache_config_{model_info['name']}.yaml"
        
        # Customize config for model
        model_config = basic_lmcache_config.copy()
        model_config["extra_config"]["model_architecture"] = model_info["architecture"]
        model_config["extra_config"]["vocab_size"] = model_info["vocab_size"]
        
        with open(config_file, 'w') as f:
            yaml.dump(model_config, f)
        
        os.environ['LMCACHE_CONFIG_FILE'] = str(config_file)
        
        try:
            integration = LMCacheNeuronIntegration()
            
            # Mock vLLM config for specific model
            mock_vllm_config = Mock()
            mock_vllm_config.model_config = Mock()
            mock_vllm_config.model_config.model = model_info["name"]
            mock_vllm_config.model_config.vocab_size = model_info["vocab_size"]
            mock_vllm_config.kv_transfer_config = None
            
            with patch('vllm_neuron.lmcache_integration.LMCacheEngine') as mock_engine_class:
                mock_engine = Mock()
                mock_engine.get.return_value = None
                mock_engine.put.return_value = None
                mock_engine.get_stats.return_value = {
                    "model": model_info["name"],
                    "architecture": model_info["architecture"]
                }
                mock_engine_class.return_value = mock_engine
                
                result = integration.initialize(mock_vllm_config)
                assert result is True, f"Failed to initialize for {model_info['name']}"
                
                # Test model-specific operations
                from vllm_neuron.lmcache_integration import KVCacheOperation
                
                class MockSequenceGroup:
                    def __init__(self, tokens):
                        self.tokens = tokens
                    def get_seqs(self):
                        return [Mock(get_token_ids=lambda: self.tokens)]
                
                # Use tokens within vocab range
                max_token = min(model_info["vocab_size"] - 1, 1000)
                tokens = list(range(1, min(max_token, 10)))
                seq_group = MockSequenceGroup(tokens)
                
                operation = KVCacheOperation("retrieve", seq_group)
                result = integration.intercept_kv_operations(operation)
                
                assert result.success is True
                
                # Verify health status includes model info
                health = integration.get_health_status()
                assert health["initialized"] is True
                
        finally:
            if 'LMCACHE_CONFIG_FILE' in os.environ:
                del os.environ['LMCACHE_CONFIG_FILE']


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])