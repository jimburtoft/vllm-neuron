#!/usr/bin/env python3
"""
Simple Offline Inference Example with LMCache-Neuron Integration

This example demonstrates basic offline inference using vLLM-Neuron with LMCache
for KV cache reuse. It shows how to:
- Initialize vLLM with Neuron backend and LMCache integration
- Process single requests with cache reuse
- Monitor cache performance

Requirements: 1.1, 1.4
"""

import os
import sys
import time
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add the vllm_neuron package to the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_environment():
    """Setup environment variables for Neuron and LMCache."""
    # Ensure Neuron environment is configured
    if not os.environ.get("NEURON_RT_VISIBLE_CORES"):
        os.environ["NEURON_RT_VISIBLE_CORES"] = "0,1"
        logger.info("Set NEURON_RT_VISIBLE_CORES to 0,1")
    
    # Set LMCache configuration path
    config_path = Path(__file__).parent.parent / "configs" / "basic_neuron_config.yaml"
    os.environ["LMCACHE_CONFIG_PATH"] = str(config_path)
    logger.info(f"Set LMCACHE_CONFIG_PATH to {config_path}")
    
    # Enable LMCache integration
    os.environ["ENABLE_LMCACHE"] = "true"
    logger.info("Enabled LMCache integration")

def check_neuron_availability():
    """Check if Neuron devices are available."""
    try:
        from vllm_neuron.lmcache_integration import NeuronDeviceDetector
        
        if not NeuronDeviceDetector.is_neuron_available():
            logger.error("No Neuron devices detected. Please ensure you're running on a Neuron instance.")
            return False
        
        device_count = NeuronDeviceDetector.get_neuron_device_count()
        device_info = NeuronDeviceDetector.get_neuron_device_info()
        
        logger.info(f"✓ Detected {device_count} Neuron device(s)")
        if device_info:
            logger.info(f"✓ Total cores: {device_info.get('total_cores', 0)}")
            for device in device_info.get('devices', []):
                logger.info(f"  Device {device['device_id']}: {device['cores']} cores, {device['memory']} memory")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to check Neuron availability: {e}")
        return False

def initialize_lmcache_integration():
    """Initialize LMCache integration for Neuron."""
    try:
        from vllm_neuron.lmcache_integration import LMCacheNeuronIntegration
        
        # Create integration instance
        integration = LMCacheNeuronIntegration()
        
        # Initialize LMCache
        if integration.initialize(None):  # Pass None for vllm_config in dry-run mode
            logger.info("LMCache integration initialized successfully")
            return integration
        else:
            logger.warning("LMCache initialization failed, continuing without cache")
            return None
            
    except ImportError:
        logger.error("LMCache integration not available")
        return None
    except Exception as e:
        logger.error(f"Failed to initialize LMCache integration: {e}")
        return None

def create_vllm_engine(model_path: str, integration: Optional[Any] = None):
    """Create vLLM engine with Neuron backend and optional LMCache integration."""
    try:
        # Import vLLM components
        from vllm import LLM, SamplingParams
        from vllm.config import VllmConfig
        
        # Configure vLLM for Neuron
        engine_config = {
            "model": model_path,
            "tensor_parallel_size": 2,  # Use 2 Neuron cores
            "max_model_len": 2048,      # Reasonable context length
            "trust_remote_code": True,  # Allow custom model code
            "block_size": 16,           # Required for prefix caching
        }
        
        # Add LMCache integration if available
        # Note: LMCache integration is handled separately, not via vLLM engine config
        
        logger.info(f"Creating vLLM engine with config: {engine_config}")
        
        # Create the engine
        llm = LLM(**engine_config)
        
        logger.info("vLLM engine created successfully")
        return llm
        
    except Exception as e:
        logger.error(f"Failed to create vLLM engine: {e}")
        raise

def run_inference_examples(llm, integration: Optional[Any] = None):
    """Run inference examples to demonstrate cache behavior."""
    from vllm import SamplingParams
    
    # Configure sampling parameters
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=256,
        stop=["</s>", "<|endoftext|>"]
    )
    
    # Example prompts that demonstrate cache reuse
    prompts = [
        # First request - will be a cache miss
        "Explain the concept of machine learning in simple terms.",
        
        # Second request with same prefix - should hit cache
        "Explain the concept of machine learning in simple terms. What are the main types?",
        
        # Third request with different content - cache miss
        "What are the benefits of using AWS Neuron for machine learning inference?",
        
        # Fourth request - repeat of first - should hit cache
        "Explain the concept of machine learning in simple terms.",
        
        # Fifth request with common prefix - partial cache hit
        "Explain the concept of machine learning and its applications in healthcare.",
    ]
    
    results = []
    
    for i, prompt in enumerate(prompts, 1):
        logger.info(f"\n--- Request {i} ---")
        logger.info(f"Prompt: {prompt[:60]}...")
        
        # Record start time
        start_time = time.time()
        
        # Run inference
        try:
            outputs = llm.generate([prompt], sampling_params)
            inference_time = time.time() - start_time
            
            # Extract generated text
            generated_text = outputs[0].outputs[0].text
            
            logger.info(f"Generated text: {generated_text[:100]}...")
            logger.info(f"Inference time: {inference_time:.2f}s")
            
            # Get cache statistics if integration is available
            cache_stats = None
            if integration:
                try:
                    cache_stats = integration.get_cache_stats()
                    logger.info(f"Cache hit rate: {cache_stats.get('hit_rate', 0):.2%}")
                    logger.info(f"Total requests: {cache_stats.get('total_requests', 0)}")
                except Exception as e:
                    logger.warning(f"Failed to get cache stats: {e}")
            
            results.append({
                "request_id": i,
                "prompt": prompt,
                "generated_text": generated_text,
                "inference_time": inference_time,
                "cache_stats": cache_stats
            })
            
        except Exception as e:
            logger.error(f"Inference failed for request {i}: {e}")
            results.append({
                "request_id": i,
                "prompt": prompt,
                "error": str(e),
                "inference_time": time.time() - start_time
            })
        
        # Small delay between requests
        time.sleep(1)
    
    return results

def print_performance_summary(results: List[Dict[str, Any]], integration: Optional[Any] = None):
    """Print performance summary and cache effectiveness."""
    logger.info("\n" + "="*60)
    logger.info("PERFORMANCE SUMMARY")
    logger.info("="*60)
    
    # Calculate timing statistics
    successful_results = [r for r in results if "error" not in r]
    if successful_results:
        times = [r["inference_time"] for r in successful_results]
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        
        logger.info(f"Successful requests: {len(successful_results)}/{len(results)}")
        logger.info(f"Average inference time: {avg_time:.2f}s")
        logger.info(f"Min inference time: {min_time:.2f}s")
        logger.info(f"Max inference time: {max_time:.2f}s")
        
        # Show time improvement from caching
        if len(times) > 1:
            first_time = times[0]  # First request (cache miss)
            subsequent_times = times[1:]
            avg_subsequent = sum(subsequent_times) / len(subsequent_times)
            improvement = ((first_time - avg_subsequent) / first_time) * 100
            logger.info(f"Average speedup from caching: {improvement:.1f}%")
    
    # Print cache statistics if available
    if integration:
        try:
            final_stats = integration.get_cache_stats()
            logger.info("\nCACHE STATISTICS:")
            logger.info(f"  Total requests: {final_stats.get('total_requests', 0)}")
            logger.info(f"  Cache hits: {final_stats.get('cache_hits', 0)}")
            logger.info(f"  Cache misses: {final_stats.get('cache_misses', 0)}")
            logger.info(f"  Partial hits: {final_stats.get('partial_hits', 0)}")
            logger.info(f"  Hit rate: {final_stats.get('hit_rate', 0):.2%}")
            logger.info(f"  Effective hit rate: {final_stats.get('effective_hit_rate', 0):.2%}")
            
            # Performance metrics
            perf_metrics = final_stats.get('performance', {})
            if perf_metrics:
                lookup_latency = perf_metrics.get('lookup_latency', {})
                if lookup_latency:
                    logger.info(f"  Average lookup latency: {lookup_latency.get('avg_ms', 0):.1f}ms")
                    logger.info(f"  P95 lookup latency: {lookup_latency.get('p95_ms', 0):.1f}ms")
        except Exception as e:
            logger.warning(f"Failed to get final cache statistics: {e}")
    
    logger.info("="*60)

def cleanup_resources(integration: Optional[Any] = None):
    """Cleanup resources and print final statistics."""
    if integration:
        try:
            # Get final health status
            health_status = integration.get_health_status()
            logger.info(f"Final health status: {health_status}")
            
            # Cleanup integration if method exists
            if hasattr(integration, 'cleanup'):
                integration.cleanup()
                logger.info("LMCache integration cleaned up")
            else:
                logger.info("No cleanup method available, skipping cleanup")
        except Exception as e:
            logger.warning(f"Cleanup failed: {e}")

def main():
    """Main function to run the simple offline inference example."""
    parser = argparse.ArgumentParser(description="Simple Offline Inference Example with LMCache-Neuron")
    parser.add_argument("--dry-run", action="store_true", help="Run device detection and setup only, skip actual inference")
    args = parser.parse_args()
    
    logger.info("Starting Simple Offline Inference Example with LMCache-Neuron")
    
    try:
        # Setup environment
        setup_environment()
        
        # Check Neuron availability
        if not check_neuron_availability():
            logger.error("Neuron not available, exiting")
            return 1
        
        # Initialize LMCache integration
        integration = initialize_lmcache_integration()
        if integration:
            logger.info("Running with LMCache integration enabled")
        else:
            logger.info("Running without LMCache integration")
        
        # Model path - adjust as needed for your setup
        model_path = os.environ.get(
            "MODEL_PATH", 
            "/opt/ml/model"  # Default path for SageMaker
        )
        
        # Alternative model paths to try
        alternative_paths = [
            "./local-models/TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "../local-models/Qwen/Qwen3-8B",
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0",  # HuggingFace model
        ]
        
        # Find available model
        if not os.path.exists(model_path):
            for alt_path in alternative_paths:
                if os.path.exists(alt_path):
                    model_path = alt_path
                    break
            else:
                logger.warning(f"Model not found at {model_path}, using HuggingFace model")
                model_path = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        
        logger.info(f"Using model: {model_path}")
        
        # In dry-run mode, skip vLLM engine creation and inference
        if args.dry_run:
            logger.info("✓ Dry-run mode: Device detection and LMCache integration successful")
            logger.info("✓ Skipping vLLM engine creation and inference in dry-run mode")
            cleanup_resources(integration)
            logger.info("Simple offline inference dry-run completed successfully")
            return 0
        
        # Create vLLM engine
        llm = create_vllm_engine(model_path, integration)
        
        # Run inference examples
        results = run_inference_examples(llm, integration)
        
        # Print performance summary
        print_performance_summary(results, integration)
        
        # Cleanup
        cleanup_resources(integration)
        
        logger.info("Simple offline inference example completed successfully")
        return 0
        
    except KeyboardInterrupt:
        logger.info("Example interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Example failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)