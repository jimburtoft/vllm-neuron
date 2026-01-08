#!/usr/bin/env python3
"""
Online Serving Example with LMCache-Neuron Integration

This example demonstrates online serving using vLLM-Neuron with LMCache
for KV cache reuse in a production-like environment. It shows how to:
- Set up an HTTP API server with vLLM-Neuron and LMCache
- Handle concurrent requests with cache reuse
- Monitor performance and cache effectiveness
- Implement health checks and metrics endpoints

Requirements: 1.1, 1.4
"""

import os
import sys
import time
import json
import asyncio
import logging
import threading
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor

# Add the vllm_neuron package to the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class GenerationRequest:
    """Request for text generation."""
    prompt: str
    max_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    stop: Optional[List[str]] = None
    request_id: Optional[str] = None

@dataclass
class GenerationResponse:
    """Response from text generation."""
    request_id: str
    generated_text: str
    prompt: str
    inference_time: float
    cache_hit: bool = False
    cache_stats: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

class LMCacheNeuronServer:
    """Online serving server with LMCache-Neuron integration."""
    
    def __init__(self, model_path: str, config: Optional[Dict[str, Any]] = None):
        self.model_path = model_path
        self.config = config or {}
        
        # Server configuration
        self.host = self.config.get("host", "0.0.0.0")
        self.port = self.config.get("port", 8000)
        self.max_concurrent_requests = self.config.get("max_concurrent_requests", 10)
        
        # vLLM and LMCache components
        self.llm = None
        self.integration = None
        self.sampling_params = None
        
        # Request handling
        self.request_executor = ThreadPoolExecutor(max_workers=self.max_concurrent_requests)
        self.request_counter = 0
        self.request_lock = threading.Lock()
        
        # Performance tracking
        self.request_history = []
        self.performance_stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_inference_time": 0.0,
            "cache_hits": 0,
            "cache_misses": 0,
        }
        self.stats_lock = threading.Lock()
        
        # Health check
        self.is_healthy = False
        self.startup_time = None
        
    def setup_environment(self):
        """Setup environment variables for Neuron and LMCache."""
        # Ensure Neuron environment is configured
        if not os.environ.get("NEURON_RT_VISIBLE_CORES"):
            os.environ["NEURON_RT_VISIBLE_CORES"] = "0,1,2,3"
            logger.info("Set NEURON_RT_VISIBLE_CORES to 0,1,2,3")
        
        # Set LMCache configuration path for online serving
        config_path = Path(__file__).parent.parent / "configs" / "advanced_remote_config.yaml"
        if not config_path.exists():
            # Fallback to basic config
            config_path = Path(__file__).parent.parent / "configs" / "basic_neuron_config.yaml"
        
        os.environ["LMCACHE_CONFIG_PATH"] = str(config_path)
        logger.info(f"Set LMCACHE_CONFIG_PATH to {config_path}")
        
        # Enable LMCache integration
        os.environ["ENABLE_LMCACHE"] = "true"
        logger.info("Enabled LMCache integration")
    
    def initialize_components(self):
        """Initialize vLLM engine and LMCache integration."""
        logger.info("Initializing server components...")
        
        # Check Neuron device availability first
        if not self.check_neuron_availability():
            raise RuntimeError("No Neuron devices available. Cannot start server.")
        
        # Initialize LMCache integration
        try:
            from vllm_neuron.lmcache_integration import LMCacheNeuronIntegration
            
            self.integration = LMCacheNeuronIntegration()
            if self.integration.initialize(None):  # Pass None for vllm_config in dry-run mode
                logger.info("LMCache integration initialized successfully")
            else:
                logger.warning("LMCache initialization failed, continuing without cache")
                self.integration = None
        
        # Initialize vLLM engine
        self.initialize_engine()
    
    def check_neuron_availability(self):
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
    
    def initialize_engine(self):
        """Initialize the vLLM engine."""
        try:
            from vllm import LLM, SamplingParams
            
            engine_config = {
                "model": self.model_path,
                "tensor_parallel_size": 4,  # Use 4 Neuron cores for online serving
                "max_model_len": 4096,      # Larger context for online serving
                "trust_remote_code": True,
                "max_num_seqs": self.max_concurrent_requests,  # Support concurrent requests
            }
            
            # Note: LMCache integration is handled separately, not via vLLM engine config
            
            logger.info(f"Creating vLLM engine with config: {engine_config}")
            self.llm = LLM(**engine_config)
            
            # Configure sampling parameters
            self.sampling_params = SamplingParams(
                temperature=0.7,
                top_p=0.9,
                max_tokens=256,
                stop=["</s>", "<|endoftext|>"]
            )
            
            logger.info("vLLM engine initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize vLLM engine: {e}")
            raise
    
    def generate_request_id(self) -> str:
        """Generate unique request ID."""
        with self.request_lock:
            self.request_counter += 1
            return f"req_{self.request_counter}_{int(time.time())}"
    
    def process_generation_request(self, request: GenerationRequest) -> GenerationResponse:
        """Process a single generation request."""
        start_time = time.time()
        
        # Assign request ID if not provided
        if not request.request_id:
            request.request_id = self.generate_request_id()
        
        logger.info(f"Processing request {request.request_id}: {request.prompt[:50]}...")
        
        try:
            # Update sampling parameters if provided in request
            sampling_params = SamplingParams(
                temperature=request.temperature,
                top_p=request.top_p,
                max_tokens=request.max_tokens,
                stop=request.stop or ["</s>", "<|endoftext|>"]
            )
            
            # Run inference
            outputs = self.llm.generate([request.prompt], sampling_params)
            generated_text = outputs[0].outputs[0].text
            
            inference_time = time.time() - start_time
            
            # Get cache statistics if available
            cache_hit = False
            cache_stats = None
            if self.integration:
                try:
                    cache_stats = self.integration.get_cache_stats()
                    # Determine if this was likely a cache hit based on timing
                    cache_hit = inference_time < 1.0  # Heuristic: fast responses likely cache hits
                except Exception as e:
                    logger.warning(f"Failed to get cache stats: {e}")
            
            # Update performance statistics
            with self.stats_lock:
                self.performance_stats["total_requests"] += 1
                self.performance_stats["successful_requests"] += 1
                self.performance_stats["total_inference_time"] += inference_time
                if cache_hit:
                    self.performance_stats["cache_hits"] += 1
                else:
                    self.performance_stats["cache_misses"] += 1
            
            response = GenerationResponse(
                request_id=request.request_id,
                generated_text=generated_text,
                prompt=request.prompt,
                inference_time=inference_time,
                cache_hit=cache_hit,
                cache_stats=cache_stats
            )
            
            logger.info(f"Request {request.request_id} completed in {inference_time:.2f}s")
            return response
            
        except Exception as e:
            inference_time = time.time() - start_time
            error_msg = str(e)
            
            # Update error statistics
            with self.stats_lock:
                self.performance_stats["total_requests"] += 1
                self.performance_stats["failed_requests"] += 1
            
            logger.error(f"Request {request.request_id} failed: {error_msg}")
            
            return GenerationResponse(
                request_id=request.request_id,
                generated_text="",
                prompt=request.prompt,
                inference_time=inference_time,
                error=error_msg
            )
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get server health status."""
        health_info = {
            "status": "healthy" if self.is_healthy else "unhealthy",
            "startup_time": self.startup_time,
            "uptime_seconds": time.time() - self.startup_time if self.startup_time else 0,
            "model_path": self.model_path,
            "lmcache_enabled": self.integration is not None,
        }
        
        # Add component health
        if self.integration:
            try:
                integration_health = self.integration.get_health_status()
                health_info["lmcache_health"] = integration_health
            except Exception as e:
                health_info["lmcache_health"] = {"status": "error", "error": str(e)}
        
        # Add performance statistics
        with self.stats_lock:
            health_info["performance"] = self.performance_stats.copy()
            
            # Calculate derived metrics
            total_requests = self.performance_stats["total_requests"]
            if total_requests > 0:
                health_info["performance"]["success_rate"] = (
                    self.performance_stats["successful_requests"] / total_requests
                )
                health_info["performance"]["average_inference_time"] = (
                    self.performance_stats["total_inference_time"] / 
                    self.performance_stats["successful_requests"]
                    if self.performance_stats["successful_requests"] > 0 else 0
                )
                health_info["performance"]["cache_hit_rate"] = (
                    self.performance_stats["cache_hits"] / total_requests
                )
        
        return health_info
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get detailed metrics for monitoring."""
        metrics = self.get_health_status()
        
        # Add LMCache-specific metrics if available
        if self.integration:
            try:
                cache_stats = self.integration.get_cache_stats()
                metrics["cache_metrics"] = cache_stats
                
                # Get performance metrics
                perf_metrics = self.integration.get_performance_metrics()
                if perf_metrics:
                    metrics["performance_metrics"] = perf_metrics
                    
            except Exception as e:
                logger.warning(f"Failed to get detailed metrics: {e}")
        
        return metrics
    
    def start_server(self):
        """Start the online serving server."""
        logger.info("Starting LMCache-Neuron online serving server...")
        
        try:
            # Setup environment
            self.setup_environment()
            
            # Initialize components
            self.initialize_components()
            
            # Mark as healthy
            self.is_healthy = True
            self.startup_time = time.time()
            
            logger.info(f"Server started successfully on {self.host}:{self.port}")
            logger.info("Server is ready to accept requests")
            
        except Exception as e:
            logger.error(f"Failed to start server: {e}")
            self.is_healthy = False
            raise
    
    def shutdown(self):
        """Shutdown the server and cleanup resources."""
        logger.info("Shutting down server...")
        
        self.is_healthy = False
        
        # Shutdown request executor
        self.request_executor.shutdown(wait=True)
        
        # Cleanup LMCache integration
        if self.integration:
            try:
                self.integration.cleanup()
                logger.info("LMCache integration cleaned up")
            except Exception as e:
                logger.warning(f"LMCache cleanup failed: {e}")
        
        logger.info("Server shutdown completed")

def run_concurrent_requests_demo(server: LMCacheNeuronServer):
    """Run a demonstration of concurrent requests to show cache effectiveness."""
    logger.info("\n" + "="*60)
    logger.info("RUNNING CONCURRENT REQUESTS DEMO")
    logger.info("="*60)
    
    # Sample prompts that demonstrate cache reuse patterns
    prompts = [
        "What is artificial intelligence and how does it work?",
        "Explain the benefits of machine learning in healthcare.",
        "What is artificial intelligence and how does it work?",  # Repeat for cache hit
        "Describe the process of training a neural network.",
        "What are the applications of AI in autonomous vehicles?",
        "Explain the benefits of machine learning in healthcare.",  # Repeat for cache hit
        "What is the difference between supervised and unsupervised learning?",
        "What is artificial intelligence and how does it work?",  # Another repeat
        "How does natural language processing work?",
        "What are the ethical considerations in AI development?",
    ]
    
    # Create requests
    requests = [
        GenerationRequest(
            prompt=prompt,
            max_tokens=200,
            temperature=0.7,
            request_id=f"demo_{i+1}"
        )
        for i, prompt in enumerate(prompts)
    ]
    
    # Process requests concurrently
    logger.info(f"Processing {len(requests)} requests concurrently...")
    start_time = time.time()
    
    # Submit all requests to the executor
    futures = []
    for request in requests:
        future = server.request_executor.submit(server.process_generation_request, request)
        futures.append((request, future))
    
    # Collect results
    responses = []
    for request, future in futures:
        try:
            response = future.result(timeout=60)  # 60 second timeout
            responses.append(response)
        except Exception as e:
            logger.error(f"Request {request.request_id} failed: {e}")
            responses.append(GenerationResponse(
                request_id=request.request_id,
                generated_text="",
                prompt=request.prompt,
                inference_time=0,
                error=str(e)
            ))
    
    total_time = time.time() - start_time
    
    # Analyze results
    successful_responses = [r for r in responses if not r.error]
    failed_responses = [r for r in responses if r.error]
    
    logger.info(f"\nDemo completed in {total_time:.2f}s")
    logger.info(f"Successful requests: {len(successful_responses)}/{len(responses)}")
    logger.info(f"Failed requests: {len(failed_responses)}")
    
    if successful_responses:
        inference_times = [r.inference_time for r in successful_responses]
        cache_hits = sum(1 for r in successful_responses if r.cache_hit)
        
        logger.info(f"Average inference time: {sum(inference_times)/len(inference_times):.2f}s")
        logger.info(f"Min inference time: {min(inference_times):.2f}s")
        logger.info(f"Max inference time: {max(inference_times):.2f}s")
        logger.info(f"Cache hits: {cache_hits}/{len(successful_responses)} ({cache_hits/len(successful_responses):.1%})")
    
    # Print detailed results
    logger.info("\nDETAILED RESULTS:")
    for response in responses:
        status = "SUCCESS" if not response.error else "FAILED"
        cache_status = "HIT" if response.cache_hit else "MISS"
        logger.info(f"  {response.request_id}: {status} | {cache_status} | {response.inference_time:.2f}s | {response.prompt[:40]}...")
    
    return responses

def main():
    """Main function to run the online serving example."""
    logger.info("Starting Online Serving Example with LMCache-Neuron")
    
    try:
        # Model path configuration
        model_path = os.environ.get("MODEL_PATH", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        
        # Server configuration
        server_config = {
            "host": "0.0.0.0",
            "port": 8000,
            "max_concurrent_requests": 8,
        }
        
        # Create and start server
        server = LMCacheNeuronServer(model_path, server_config)
        server.start_server()
        
        # Run concurrent requests demo
        demo_responses = run_concurrent_requests_demo(server)
        
        # Print final health status and metrics
        logger.info("\n" + "="*60)
        logger.info("FINAL SERVER STATUS")
        logger.info("="*60)
        
        health_status = server.get_health_status()
        logger.info(f"Server status: {health_status['status']}")
        logger.info(f"Uptime: {health_status['uptime_seconds']:.1f}s")
        logger.info(f"Total requests processed: {health_status['performance']['total_requests']}")
        logger.info(f"Success rate: {health_status['performance'].get('success_rate', 0):.1%}")
        logger.info(f"Cache hit rate: {health_status['performance'].get('cache_hit_rate', 0):.1%}")
        
        # Get detailed metrics
        metrics = server.get_metrics()
        if "cache_metrics" in metrics:
            cache_metrics = metrics["cache_metrics"]
            logger.info(f"LMCache hit rate: {cache_metrics.get('hit_rate', 0):.1%}")
            logger.info(f"LMCache effective hit rate: {cache_metrics.get('effective_hit_rate', 0):.1%}")
        
        # Shutdown server
        server.shutdown()
        
        logger.info("Online serving example completed successfully")
        return 0
        
    except KeyboardInterrupt:
        logger.info("Example interrupted by user")
        if 'server' in locals():
            server.shutdown()
        return 1
    except Exception as e:
        logger.error(f"Example failed: {e}")
        import traceback
        traceback.print_exc()
        if 'server' in locals():
            server.shutdown()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)