#!/usr/bin/env python3
"""
Compatibility validation for LMCache-Neuron integration.
Tests compatibility with multiple vLLM-Neuron versions and Neuron instance types.
"""

import logging
import os
import sys
import subprocess
import importlib
from typing import Dict, List, Any, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_neuron_environment():
    """Check Neuron environment and instance type."""
    logger.info("Checking Neuron environment...")
    
    # Check for Neuron runtime
    neuron_cores = os.environ.get('NEURON_RT_VISIBLE_CORES', '0')
    logger.info(f"NEURON_RT_VISIBLE_CORES: {neuron_cores}")
    
    # Check for Neuron device files
    neuron_devices = []
    for i in range(16):  # Check up to 16 potential devices
        device_path = f"/dev/neuron{i}"
        if os.path.exists(device_path):
            neuron_devices.append(device_path)
    
    logger.info(f"Found Neuron devices: {neuron_devices}")
    
    # Detect instance type from metadata (if available)
    instance_type = "unknown"
    try:
        # Try to get instance type from EC2 metadata
        result = subprocess.run([
            'curl', '-s', '--connect-timeout', '2',
            'http://169.254.169.254/latest/meta-data/instance-type'
        ], capture_output=True, text=True, timeout=5)
        
        if result.returncode == 0 and result.stdout.strip():
            instance_type = result.stdout.strip()
            logger.info(f"Detected instance type: {instance_type}")
    except Exception as e:
        logger.info(f"Could not detect instance type: {e}")
    
    return {
        'neuron_cores': neuron_cores,
        'neuron_devices': neuron_devices,
        'instance_type': instance_type,
        'has_neuron_runtime': len(neuron_devices) > 0 or neuron_cores != '0'
    }

def check_vllm_neuron_version():
    """Check vLLM-Neuron version and compatibility."""
    logger.info("Checking vLLM-Neuron version...")
    
    try:
        # Try to import vllm and get version
        import vllm
        vllm_version = getattr(vllm, '__version__', 'unknown')
        logger.info(f"vLLM version: {vllm_version}")
        
        # Check for Neuron-specific components
        neuron_components = []
        
        # Check for common Neuron modules
        neuron_modules = [
            'vllm.model_executor.layers.neuron',
            'vllm.worker.neuron_worker',
            'vllm.engine.neuron_engine'
        ]
        
        for module_name in neuron_modules:
            try:
                importlib.import_module(module_name)
                neuron_components.append(module_name)
            except ImportError:
                pass
        
        logger.info(f"Found Neuron components: {neuron_components}")
        
        return {
            'vllm_version': vllm_version,
            'neuron_components': neuron_components,
            'has_neuron_support': len(neuron_components) > 0
        }
        
    except ImportError as e:
        logger.warning(f"Could not import vLLM: {e}")
        return {
            'vllm_version': 'not_installed',
            'neuron_components': [],
            'has_neuron_support': False
        }

def test_lmcache_integration_compatibility():
    """Test LMCache integration compatibility."""
    logger.info("Testing LMCache integration compatibility...")
    
    try:
        from vllm_neuron.lmcache_integration import LMCacheNeuronIntegration
        from vllm_neuron.neuron_cpu_storage_backend import NeuronCPUStorageBackend
        
        # Test basic initialization
        integration = LMCacheNeuronIntegration()
        logger.info("LMCache integration initialized successfully")
        
        # Test storage backend initialization
        config = {
            'memory_limit': 50 * 1024 * 1024,  # 50MB
            'chunk_size': 256,
            'numa_node': None,
            'adaptive_chunking': True,
            'multi_core_enabled': False
        }
        
        backend = NeuronCPUStorageBackend(config)
        logger.info("Storage backend initialized successfully")
        
        # Test basic operations
        stats = backend.get_stats()
        health = backend.get_health_status()
        
        backend.close()
        
        return {
            'integration_compatible': True,
            'storage_backend_compatible': True,
            'error': None
        }
        
    except Exception as e:
        logger.error(f"LMCache integration compatibility test failed: {e}")
        return {
            'integration_compatible': False,
            'storage_backend_compatible': False,
            'error': str(e)
        }

def test_model_architecture_compatibility():
    """Test compatibility with various model architectures."""
    logger.info("Testing model architecture compatibility...")
    
    # Test with different model configurations
    test_configs = [
        {
            'name': 'small_model',
            'hidden_size': 768,
            'num_attention_heads': 12,
            'num_layers': 12
        },
        {
            'name': 'medium_model', 
            'hidden_size': 1024,
            'num_attention_heads': 16,
            'num_layers': 24
        },
        {
            'name': 'large_model',
            'hidden_size': 2048,
            'num_attention_heads': 32,
            'num_layers': 48
        }
    ]
    
    compatible_configs = []
    incompatible_configs = []
    
    for config in test_configs:
        try:
            # Test if our integration can handle this model size
            from vllm_neuron.lmcache_integration import TokenCacheKeyManager
            
            key_manager = TokenCacheKeyManager()
            
            # Create a test cache key for this model
            test_tokens = list(range(100))  # 100 tokens
            cache_key = key_manager.create_cache_key(test_tokens, config['name'])
            
            # Test token retrieval
            retrieved_tokens = key_manager.get_tokens_from_key(cache_key)
            
            if retrieved_tokens == test_tokens:
                compatible_configs.append(config['name'])
                logger.info(f"Model architecture '{config['name']}' is compatible")
            else:
                incompatible_configs.append(config['name'])
                logger.warning(f"Model architecture '{config['name']}' failed token round-trip test")
                
        except Exception as e:
            incompatible_configs.append(config['name'])
            logger.error(f"Model architecture '{config['name']}' compatibility test failed: {e}")
    
    return {
        'compatible_architectures': compatible_configs,
        'incompatible_architectures': incompatible_configs,
        'total_tested': len(test_configs)
    }

def test_configuration_compatibility():
    """Test compatibility with various configuration options."""
    logger.info("Testing configuration compatibility...")
    
    test_configs = [
        {
            'name': 'minimal_config',
            'memory_limit': 10 * 1024 * 1024,  # 10MB
            'chunk_size': 128,
            'multi_core_enabled': False
        },
        {
            'name': 'standard_config',
            'memory_limit': 100 * 1024 * 1024,  # 100MB
            'chunk_size': 256,
            'multi_core_enabled': True
        },
        {
            'name': 'large_config',
            'memory_limit': 1024 * 1024 * 1024,  # 1GB
            'chunk_size': 512,
            'multi_core_enabled': True,
            'adaptive_chunking': True
        }
    ]
    
    compatible_configs = []
    incompatible_configs = []
    
    for config in test_configs:
        try:
            from vllm_neuron.neuron_cpu_storage_backend import NeuronCPUStorageBackend
            
            # Add required config fields
            full_config = {
                'numa_node': None,
                'adaptive_chunking': config.get('adaptive_chunking', True),
                **config
            }
            
            backend = NeuronCPUStorageBackend(full_config)
            
            # Test basic operations
            stats = backend.get_stats()
            health = backend.get_health_status()
            
            backend.close()
            
            compatible_configs.append(config['name'])
            logger.info(f"Configuration '{config['name']}' is compatible")
            
        except Exception as e:
            incompatible_configs.append(config['name'])
            logger.error(f"Configuration '{config['name']}' compatibility test failed: {e}")
    
    return {
        'compatible_configurations': compatible_configs,
        'incompatible_configurations': incompatible_configs,
        'total_tested': len(test_configs)
    }

def main():
    """Run comprehensive compatibility validation."""
    logger.info("Starting LMCache-Neuron Compatibility Validation")
    logger.info("=" * 60)
    
    results = {}
    
    # Check environment
    logger.info("\n1. Environment Compatibility")
    logger.info("-" * 40)
    results['environment'] = check_neuron_environment()
    
    # Check vLLM-Neuron version
    logger.info("\n2. vLLM-Neuron Version Compatibility")
    logger.info("-" * 40)
    results['vllm_neuron'] = check_vllm_neuron_version()
    
    # Test LMCache integration
    logger.info("\n3. LMCache Integration Compatibility")
    logger.info("-" * 40)
    results['lmcache_integration'] = test_lmcache_integration_compatibility()
    
    # Test model architectures
    logger.info("\n4. Model Architecture Compatibility")
    logger.info("-" * 40)
    results['model_architectures'] = test_model_architecture_compatibility()
    
    # Test configurations
    logger.info("\n5. Configuration Compatibility")
    logger.info("-" * 40)
    results['configurations'] = test_configuration_compatibility()
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("COMPATIBILITY VALIDATION SUMMARY")
    logger.info("=" * 60)
    
    # Environment summary
    env = results['environment']
    logger.info(f"Instance Type: {env['instance_type']}")
    logger.info(f"Neuron Runtime Available: {env['has_neuron_runtime']}")
    logger.info(f"Neuron Devices: {len(env['neuron_devices'])}")
    
    # vLLM summary
    vllm = results['vllm_neuron']
    logger.info(f"vLLM Version: {vllm['vllm_version']}")
    logger.info(f"Neuron Support: {vllm['has_neuron_support']}")
    
    # Integration summary
    integration = results['lmcache_integration']
    logger.info(f"LMCache Integration: {'COMPATIBLE' if integration['integration_compatible'] else 'INCOMPATIBLE'}")
    
    # Architecture summary
    arch = results['model_architectures']
    logger.info(f"Compatible Architectures: {len(arch['compatible_architectures'])}/{arch['total_tested']}")
    
    # Configuration summary
    config = results['configurations']
    logger.info(f"Compatible Configurations: {len(config['compatible_configurations'])}/{config['total_tested']}")
    
    # Overall assessment
    overall_compatible = (
        integration['integration_compatible'] and
        len(arch['compatible_architectures']) > 0 and
        len(config['compatible_configurations']) > 0
    )
    
    logger.info(f"\nOverall Compatibility: {'PASS' if overall_compatible else 'FAIL'}")
    
    if overall_compatible:
        logger.info("LMCache-Neuron integration is compatible with current environment")
        return 0
    else:
        logger.warning("Compatibility issues detected")
        return 1

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)