#!/usr/bin/env python3
"""
Simple performance validation for LMCache-Neuron integration.
"""

import time
import logging
import statistics
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_storage_backend_performance():
    """Test storage backend performance."""
    try:
        from vllm_neuron.neuron_cpu_storage_backend import NeuronCPUStorageBackend
        import torch
        
        logger.info("Testing storage backend performance...")
        
        # Create storage backend
        config = {
            'memory_limit': 100 * 1024 * 1024,  # 100MB
            'chunk_size': 256,
            'numa_node': None,
            'adaptive_chunking': True,
            'multi_core_enabled': False
        }
        
        # Measure backend initialization time
        init_start = time.time()
        backend = NeuronCPUStorageBackend(config)
        init_end = time.time()
        
        init_time = (init_end - init_start) * 1000  # Convert to ms
        logger.info(f"Backend initialization time: {init_time:.2f}ms")
        
        # Test basic operations timing
        operation_times = []
        
        # Test configuration access
        for i in range(10):
            op_start = time.time()
            stats = backend.get_stats()
            health = backend.get_health_status()
            memory_status = backend.get_memory_pressure_status()
            op_end = time.time()
            
            operation_times.append((op_end - op_start) * 1000)  # Convert to ms
        
        backend.close()
        
        # Calculate statistics
        if operation_times:
            avg_operation_time = statistics.mean(operation_times)
            max_operation_time = max(operation_times)
            logger.info(f"Average operation time: {avg_operation_time:.2f}ms")
            logger.info(f"Max operation time: {max_operation_time:.2f}ms")
        else:
            avg_operation_time = 0
            max_operation_time = 0
        
        # Validate performance targets
        init_target = 1000.0  # ms - initialization should be fast
        operation_target = 10.0  # ms - operations should be fast
        
        init_passed = init_time < init_target
        operation_passed = avg_operation_time < operation_target if operation_times else True
        
        logger.info(f"Initialization performance target (<{init_target}ms): {'PASS' if init_passed else 'FAIL'}")
        logger.info(f"Operation performance target (<{operation_target}ms): {'PASS' if operation_passed else 'FAIL'}")
        
        return init_passed and operation_passed
        
    except Exception as e:
        logger.error(f"Storage backend performance test failed: {e}")
        return False

def test_integration_initialization_performance():
    """Test integration initialization performance."""
    try:
        from vllm_neuron.lmcache_integration import LMCacheNeuronIntegration
        
        logger.info("Testing integration initialization performance...")
        
        # Measure initialization time
        start_time = time.time()
        integration = LMCacheNeuronIntegration()
        end_time = time.time()
        
        init_time = (end_time - start_time) * 1000  # Convert to ms
        logger.info(f"Integration initialization time: {init_time:.2f}ms")
        
        # Performance target: initialization should be fast (< 1000ms)
        performance_target = 1000.0  # ms
        passed = init_time < performance_target
        
        logger.info(f"Initialization performance target (<{performance_target}ms): {'PASS' if passed else 'FAIL'}")
        
        return passed
        
    except Exception as e:
        logger.error(f"Integration initialization performance test failed: {e}")
        return False

def test_memory_usage():
    """Test memory usage is within acceptable limits."""
    try:
        import psutil
        import os
        
        logger.info("Testing memory usage...")
        
        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create integration and storage backend
        from vllm_neuron.lmcache_integration import LMCacheNeuronIntegration
        from vllm_neuron.neuron_cpu_storage_backend import NeuronCPUStorageBackend
        
        integration = LMCacheNeuronIntegration()
        
        config = {
            'memory_limit': 50 * 1024 * 1024,  # 50MB
            'chunk_size': 256,
            'numa_node': None,
            'adaptive_chunking': True,
            'multi_core_enabled': False
        }
        backend = NeuronCPUStorageBackend(config)
        
        # Get memory usage after initialization
        current_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_overhead = current_memory - initial_memory
        
        backend.close()
        
        logger.info(f"Initial memory usage: {initial_memory:.1f}MB")
        logger.info(f"Memory usage after initialization: {current_memory:.1f}MB")
        logger.info(f"Memory overhead: {memory_overhead:.1f}MB")
        
        # Performance target: memory overhead should be reasonable (< 100MB)
        memory_target = 100.0  # MB
        passed = memory_overhead < memory_target
        
        logger.info(f"Memory overhead target (<{memory_target}MB): {'PASS' if passed else 'FAIL'}")
        
        return passed
        
    except Exception as e:
        logger.error(f"Memory usage test failed: {e}")
        return False

def main():
    """Run simple performance validation tests."""
    logger.info("Starting LMCache-Neuron Performance Validation")
    logger.info("=" * 60)
    
    tests = [
        ("Storage Backend Performance", test_storage_backend_performance),
        ("Integration Initialization Performance", test_integration_initialization_performance),
        ("Memory Usage", test_memory_usage),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\nRunning: {test_name}")
        logger.info("-" * 40)
        
        try:
            result = test_func()
            results[test_name] = result
            logger.info(f"Result: {'PASS' if result else 'FAIL'}")
        except Exception as e:
            logger.error(f"Test failed with exception: {e}")
            results[test_name] = False
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("PERFORMANCE VALIDATION SUMMARY")
    logger.info("=" * 60)
    
    passed_tests = sum(1 for result in results.values() if result)
    total_tests = len(results)
    
    for test_name, result in results.items():
        status = "PASS" if result else "FAIL"
        logger.info(f"{test_name}: {status}")
    
    logger.info(f"\nOverall: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        logger.info("All performance targets met!")
        return 0
    else:
        logger.warning("Some performance targets not met")
        return 1

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)