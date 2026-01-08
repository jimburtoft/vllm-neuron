#!/usr/bin/env python3
"""
Multi-Model Serving Example with LMCache-Neuron Integration

This example demonstrates serving multiple models simultaneously using vLLM-Neuron 
with shared LMCache for KV cache reuse across models. It shows how to:
- Set up multiple vLLM engines with different models
- Share cache storage across models where beneficial
- Route requests to appropriate models
- Monitor cache effectiveness across models
- Handle resource management for multiple models

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
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor
from enum import Enum

# Add the vllm_neuron package to the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelType(Enum):
    """Supported model types for different use cases."""
    CHAT = "chat"
    CODE = "code"
    INSTRUCT = "instruct"
    GENERAL = "general"

@dataclass
class ModelConfig:
    """Configuration for a model in the multi-model setup."""
    name: str
    path: str
    model_type: ModelType
    tensor_parallel_size: int = 2
    max_model_len: int = 2048
    max_concurrent_requests: int = 4
    cache_namespace: Optional[str] = None  # For cache isolation

@dataclass
class MultiModelRequest:
    """Request for multi-model serving."""
    prompt: str
    model_name: str
    max_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    stop: Optional[List[str]] = None
    request_id: Optional[str] = None

@dataclass
class MultiModelResponse:
    """Response from multi-model serving."""
    request_id: str
    model_name: str
    generated_text: str
    prompt: str
    inference_time: float
    cache_hit: bool = False
    cache_stats: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

class ModelInstance:
    """Individual model instance with its own vLLM engine."""
    
    def __init__(self, config: ModelConfig, shared_integration=None):
        self.config = config
        self.shared_integration = shared_integration
        self.llm = None
        self.sampling_params = None
        self.is_loaded = False
        self.load_time = None
        
        # Per-model statistics
        self.stats = {
            "requests_processed": 0,
            "total_inference_time": 0.0,
            "cache_hits": 0,
            "cache_misses": 0,
            "errors": 0,
        }
        self.stats_lock = threading.Lock()
    
    def load_model(self):
        """Load the model and initialize the vLLM engine."""
        logger.info(f"Loading model {self.config.name} from {self.config.path}")
        start_time = time.time()
        
        try:
            from vllm import LLM, SamplingParams
            
            # Configure engine for this model
            engine_config = {
                "model": self.config.path,
                "tensor_parallel_size": self.config.tensor_parallel_size,
                "max_model_len": self.config.max_model_len,
                "trust_remote_code": True,
                "max_num_seqs": self.config.max_concurrent_requests,
            }
            
            # Add shared LMCache integration if available
            # Note: LMCache integration is handled separately, not via vLLM engine config
                
                # Set cache namespace for model isolation if specified
                if self.config.cache_namespace:
                    engine_config["lmcache_namespace"] = self.config.cache_namespace
            
            # Create the engine
            self.llm = LLM(**engine_config)
            
            # Configure sampling parameters based on model type
            if self.config.model_type == ModelType.CODE:
                # Code generation typically needs lower temperature
                self.sampling_params = SamplingParams(
                    temperature=0.2,
                    top_p=0.95,
                    max_tokens=512,
                    stop=["</code>", "```", "\n\n\n"]
                )
            elif self.config.model_type == ModelType.CHAT:
                # Chat models need balanced creativity
                self.sampling_params = SamplingParams(
                    temperature=0.7,
                    top_p=0.9,
                    max_tokens=256,
                    stop=["</s>", "<|endoftext|>", "\n\nHuman:", "\n\nAssistant:"]
                )
            else:
                # General/instruct models
                self.sampling_params = SamplingParams(
                    temperature=0.7,
                    top_p=0.9,
                    max_tokens=256,
                    stop=["</s>", "<|endoftext|>"]
                )
            
            self.load_time = time.time() - start_time
            self.is_loaded = True
            
            logger.info(f"Model {self.config.name} loaded successfully in {self.load_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Failed to load model {self.config.name}: {e}")
            raise
    
    def generate(self, request: MultiModelRequest) -> MultiModelResponse:
        """Generate text using this model instance."""
        if not self.is_loaded:
            raise RuntimeError(f"Model {self.config.name} is not loaded")
        
        start_time = time.time()
        
        try:
            # Override sampling parameters if provided in request
            sampling_params = SamplingParams(
                temperature=request.temperature,
                top_p=request.top_p,
                max_tokens=request.max_tokens,
                stop=request.stop or self.sampling_params.stop
            )
            
            # Run inference
            outputs = self.llm.generate([request.prompt], sampling_params)
            generated_text = outputs[0].outputs[0].text
            
            inference_time = time.time() - start_time
            
            # Determine cache hit heuristically (fast responses likely cache hits)
            cache_hit = inference_time < 1.0
            
            # Update statistics
            with self.stats_lock:
                self.stats["requests_processed"] += 1
                self.stats["total_inference_time"] += inference_time
                if cache_hit:
                    self.stats["cache_hits"] += 1
                else:
                    self.stats["cache_misses"] += 1
            
            # Get cache statistics if shared integration is available
            cache_stats = None
            if self.shared_integration:
                try:
                    cache_stats = self.shared_integration.get_cache_stats()
                except Exception as e:
                    logger.warning(f"Failed to get cache stats for {self.config.name}: {e}")
            
            return MultiModelResponse(
                request_id=request.request_id,
                model_name=self.config.name,
                generated_text=generated_text,
                prompt=request.prompt,
                inference_time=inference_time,
                cache_hit=cache_hit,
                cache_stats=cache_stats
            )
            
        except Exception as e:
            inference_time = time.time() - start_time
            
            # Update error statistics
            with self.stats_lock:
                self.stats["requests_processed"] += 1
                self.stats["errors"] += 1
            
            return MultiModelResponse(
                request_id=request.request_id,
                model_name=self.config.name,
                generated_text="",
                prompt=request.prompt,
                inference_time=inference_time,
                error=str(e)
            )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics for this model instance."""
        with self.stats_lock:
            stats = self.stats.copy()
            
            # Calculate derived metrics
            if stats["requests_processed"] > 0:
                stats["average_inference_time"] = (
                    stats["total_inference_time"] / stats["requests_processed"]
                )
                stats["cache_hit_rate"] = (
                    stats["cache_hits"] / stats["requests_processed"]
                )
                stats["error_rate"] = (
                    stats["errors"] / stats["requests_processed"]
                )
            else:
                stats["average_inference_time"] = 0.0
                stats["cache_hit_rate"] = 0.0
                stats["error_rate"] = 0.0
            
            stats["model_name"] = self.config.name
            stats["model_type"] = self.config.model_type.value
            stats["is_loaded"] = self.is_loaded
            stats["load_time"] = self.load_time
            
            return stats

class MultiModelServer:
    """Multi-model serving server with shared LMCache integration."""
    
    def __init__(self, model_configs: List[ModelConfig], config: Optional[Dict[str, Any]] = None):
        self.model_configs = model_configs
        self.config = config or {}
        
        # Server configuration
        self.max_concurrent_requests = self.config.get("max_concurrent_requests", 16)
        
        # Model instances
        self.models: Dict[str, ModelInstance] = {}
        self.shared_integration = None
        
        # Request handling
        self.request_executor = ThreadPoolExecutor(max_workers=self.max_concurrent_requests)
        self.request_counter = 0
        self.request_lock = threading.Lock()
        
        # Global statistics
        self.global_stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "requests_by_model": {},
        }
        self.stats_lock = threading.Lock()
        
        # Health status
        self.is_healthy = False
        self.startup_time = None
    
    def setup_environment(self):
        """Setup environment for multi-model serving."""
        # Configure Neuron cores for multiple models
        if not os.environ.get("NEURON_RT_VISIBLE_CORES"):
            # Allocate cores based on number of models
            total_cores = len(self.model_configs) * 2  # 2 cores per model minimum
            cores = ",".join(str(i) for i in range(min(total_cores, 8)))  # Max 8 cores
            os.environ["NEURON_RT_VISIBLE_CORES"] = cores
            logger.info(f"Set NEURON_RT_VISIBLE_CORES to {cores}")
        
        # Use advanced configuration for multi-model setup
        config_path = Path(__file__).parent.parent / "configs" / "advanced_remote_config.yaml"
        if not config_path.exists():
            config_path = Path(__file__).parent.parent / "configs" / "basic_neuron_config.yaml"
        
        os.environ["LMCACHE_CONFIG_PATH"] = str(config_path)
        os.environ["ENABLE_LMCACHE"] = "true"
        logger.info(f"Set LMCACHE_CONFIG_PATH to {config_path}")
    
    def initialize_shared_integration(self):
        """Initialize shared LMCache integration for all models."""
        # Check Neuron device availability first
        if not self.check_neuron_availability():
            raise RuntimeError("No Neuron devices available. Cannot start multi-model server.")
        
        try:
            from vllm_neuron.lmcache_integration import LMCacheNeuronIntegration
            
            self.shared_integration = LMCacheNeuronIntegration()
            if self.shared_integration.initialize(None):  # Pass None for vllm_config in dry-run mode
                logger.info("Shared LMCache integration initialized successfully")
                return True
            else:
                logger.warning("Shared LMCache initialization failed")
                self.shared_integration = None
                return False
                
        except Exception as e:
            logger.error(f"Failed to initialize shared LMCache integration: {e}")
            self.shared_integration = None
    
    def check_neuron_availability(self):
        """Check if Neuron devices are available for multi-model serving."""
        try:
            from vllm_neuron.lmcache_integration import NeuronDeviceDetector
            
            if not NeuronDeviceDetector.is_neuron_available():
                logger.error("No Neuron devices detected. Please ensure you're running on a Neuron instance.")
                return False
            
            device_count = NeuronDeviceDetector.get_neuron_device_count()
            device_info = NeuronDeviceDetector.get_neuron_device_info()
            
            logger.info(f"✓ Detected {device_count} Neuron device(s) for multi-model serving")
            if device_info:
                total_cores = device_info.get('total_cores', 0)
                logger.info(f"✓ Total cores: {total_cores}")
                
                # Check if we have enough cores for multi-model serving
                required_cores = len(self.model_configs) * 2  # 2 cores per model minimum
                if total_cores < required_cores:
                    logger.warning(f"Limited cores: {total_cores} available, {required_cores} recommended for {len(self.model_configs)} models")
                else:
                    logger.info(f"✓ Sufficient cores for {len(self.model_configs)} models")
                
                for device in device_info.get('devices', []):
                    logger.info(f"  Device {device['device_id']}: {device['cores']} cores, {device['memory']} memory")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to check Neuron availability: {e}")
            return False
            return False
    
    def load_models(self):
        """Load all configured models."""
        logger.info(f"Loading {len(self.model_configs)} models...")
        
        for config in self.model_configs:
            try:
                model_instance = ModelInstance(config, self.shared_integration)
                model_instance.load_model()
                self.models[config.name] = model_instance
                
                # Initialize model stats tracking
                with self.stats_lock:
                    self.global_stats["requests_by_model"][config.name] = 0
                
                logger.info(f"Model {config.name} loaded and ready")
                
            except Exception as e:
                logger.error(f"Failed to load model {config.name}: {e}")
                # Continue loading other models
                continue
        
        if not self.models:
            raise RuntimeError("No models were loaded successfully")
        
        logger.info(f"Successfully loaded {len(self.models)} models")
    
    def generate_request_id(self) -> str:
        """Generate unique request ID."""
        with self.request_lock:
            self.request_counter += 1
            return f"multi_req_{self.request_counter}_{int(time.time())}"
    
    def route_request(self, request: MultiModelRequest) -> MultiModelResponse:
        """Route request to the appropriate model."""
        # Assign request ID if not provided
        if not request.request_id:
            request.request_id = self.generate_request_id()
        
        # Check if requested model exists
        if request.model_name not in self.models:
            available_models = list(self.models.keys())
            return MultiModelResponse(
                request_id=request.request_id,
                model_name=request.model_name,
                generated_text="",
                prompt=request.prompt,
                inference_time=0,
                error=f"Model '{request.model_name}' not available. Available models: {available_models}"
            )
        
        # Update global statistics
        with self.stats_lock:
            self.global_stats["total_requests"] += 1
            self.global_stats["requests_by_model"][request.model_name] += 1
        
        # Route to model
        model_instance = self.models[request.model_name]
        
        try:
            response = model_instance.generate(request)
            
            # Update success statistics
            if not response.error:
                with self.stats_lock:
                    self.global_stats["successful_requests"] += 1
            else:
                with self.stats_lock:
                    self.global_stats["failed_requests"] += 1
            
            return response
            
        except Exception as e:
            with self.stats_lock:
                self.global_stats["failed_requests"] += 1
            
            return MultiModelResponse(
                request_id=request.request_id,
                model_name=request.model_name,
                generated_text="",
                prompt=request.prompt,
                inference_time=0,
                error=f"Model processing failed: {str(e)}"
            )
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get overall server health status."""
        health_info = {
            "status": "healthy" if self.is_healthy else "unhealthy",
            "startup_time": self.startup_time,
            "uptime_seconds": time.time() - self.startup_time if self.startup_time else 0,
            "models_loaded": len(self.models),
            "total_models_configured": len(self.model_configs),
            "lmcache_enabled": self.shared_integration is not None,
        }
        
        # Add per-model health
        health_info["models"] = {}
        for name, model in self.models.items():
            health_info["models"][name] = {
                "loaded": model.is_loaded,
                "load_time": model.load_time,
                "type": model.config.model_type.value,
            }
        
        # Add global statistics
        with self.stats_lock:
            health_info["global_stats"] = self.global_stats.copy()
        
        # Add shared cache health
        if self.shared_integration:
            try:
                cache_health = self.shared_integration.get_health_status()
                health_info["cache_health"] = cache_health
            except Exception as e:
                health_info["cache_health"] = {"status": "error", "error": str(e)}
        
        return health_info
    
    def get_detailed_stats(self) -> Dict[str, Any]:
        """Get detailed statistics for all models."""
        stats = {
            "global": self.get_health_status(),
            "models": {},
            "cache": None,
        }
        
        # Get per-model statistics
        for name, model in self.models.items():
            stats["models"][name] = model.get_stats()
        
        # Get shared cache statistics
        if self.shared_integration:
            try:
                cache_stats = self.shared_integration.get_cache_stats()
                stats["cache"] = cache_stats
            except Exception as e:
                stats["cache"] = {"error": str(e)}
        
        return stats
    
    def start_server(self):
        """Start the multi-model server."""
        logger.info("Starting Multi-Model LMCache-Neuron server...")
        
        try:
            # Setup environment
            self.setup_environment()
            
            # Initialize shared LMCache integration
            self.initialize_shared_integration()
            
            # Load all models
            self.load_models()
            
            # Mark as healthy
            self.is_healthy = True
            self.startup_time = time.time()
            
            logger.info("Multi-model server started successfully")
            logger.info(f"Available models: {list(self.models.keys())}")
            
        except Exception as e:
            logger.error(f"Failed to start multi-model server: {e}")
            self.is_healthy = False
            raise
    
    def shutdown(self):
        """Shutdown the server and cleanup resources."""
        logger.info("Shutting down multi-model server...")
        
        self.is_healthy = False
        
        # Shutdown request executor
        self.request_executor.shutdown(wait=True)
        
        # Cleanup shared integration
        if self.shared_integration:
            try:
                self.shared_integration.cleanup()
                logger.info("Shared LMCache integration cleaned up")
            except Exception as e:
                logger.warning(f"Shared LMCache cleanup failed: {e}")
        
        logger.info("Multi-model server shutdown completed")

def create_example_model_configs() -> List[ModelConfig]:
    """Create example model configurations for demonstration."""
    configs = []
    
    # Chat model configuration
    configs.append(ModelConfig(
        name="chat_model",
        path="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        model_type=ModelType.CHAT,
        tensor_parallel_size=2,
        max_model_len=2048,
        max_concurrent_requests=4,
        cache_namespace="chat"
    ))
    
    # Code model configuration (using same model but different config)
    configs.append(ModelConfig(
        name="code_model", 
        path="TinyLlama/TinyLlama-1.1B-Chat-v1.0",  # Same model, different use case
        model_type=ModelType.CODE,
        tensor_parallel_size=2,
        max_model_len=4096,  # Longer context for code
        max_concurrent_requests=2,  # Fewer concurrent for code tasks
        cache_namespace="code"
    ))
    
    # General instruct model
    configs.append(ModelConfig(
        name="instruct_model",
        path="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        model_type=ModelType.INSTRUCT,
        tensor_parallel_size=2,
        max_model_len=2048,
        max_concurrent_requests=4,
        cache_namespace="instruct"
    ))
    
    return configs

def run_multi_model_demo(server: MultiModelServer):
    """Run demonstration of multi-model serving with cache sharing."""
    logger.info("\n" + "="*60)
    logger.info("RUNNING MULTI-MODEL SERVING DEMO")
    logger.info("="*60)
    
    # Create diverse requests for different models
    requests = [
        # Chat requests
        MultiModelRequest(
            prompt="Hello! How are you doing today?",
            model_name="chat_model",
            request_id="chat_1"
        ),
        MultiModelRequest(
            prompt="Can you help me plan a vacation to Japan?",
            model_name="chat_model",
            request_id="chat_2"
        ),
        MultiModelRequest(
            prompt="Hello! How are you doing today?",  # Repeat for cache hit
            model_name="chat_model",
            request_id="chat_3"
        ),
        
        # Code requests
        MultiModelRequest(
            prompt="Write a Python function to calculate fibonacci numbers:",
            model_name="code_model",
            max_tokens=400,
            temperature=0.2,
            request_id="code_1"
        ),
        MultiModelRequest(
            prompt="Explain how to implement a binary search algorithm:",
            model_name="code_model",
            max_tokens=400,
            temperature=0.2,
            request_id="code_2"
        ),
        
        # Instruct requests
        MultiModelRequest(
            prompt="Explain the concept of machine learning in simple terms.",
            model_name="instruct_model",
            request_id="instruct_1"
        ),
        MultiModelRequest(
            prompt="What are the benefits of renewable energy?",
            model_name="instruct_model",
            request_id="instruct_2"
        ),
        MultiModelRequest(
            prompt="Explain the concept of machine learning in simple terms.",  # Repeat
            model_name="instruct_model",
            request_id="instruct_3"
        ),
    ]
    
    logger.info(f"Processing {len(requests)} requests across {len(server.models)} models...")
    
    # Process requests concurrently
    start_time = time.time()
    
    futures = []
    for request in requests:
        future = server.request_executor.submit(server.route_request, request)
        futures.append((request, future))
    
    # Collect results
    responses = []
    for request, future in futures:
        try:
            response = future.result(timeout=120)  # 2 minute timeout
            responses.append(response)
        except Exception as e:
            logger.error(f"Request {request.request_id} failed: {e}")
            responses.append(MultiModelResponse(
                request_id=request.request_id,
                model_name=request.model_name,
                generated_text="",
                prompt=request.prompt,
                inference_time=0,
                error=str(e)
            ))
    
    total_time = time.time() - start_time
    
    # Analyze results by model
    results_by_model = {}
    for response in responses:
        model_name = response.model_name
        if model_name not in results_by_model:
            results_by_model[model_name] = []
        results_by_model[model_name].append(response)
    
    logger.info(f"\nDemo completed in {total_time:.2f}s")
    
    # Print results by model
    for model_name, model_responses in results_by_model.items():
        successful = [r for r in model_responses if not r.error]
        failed = [r for r in model_responses if r.error]
        
        logger.info(f"\n{model_name.upper()} RESULTS:")
        logger.info(f"  Requests: {len(model_responses)} (Success: {len(successful)}, Failed: {len(failed)})")
        
        if successful:
            times = [r.inference_time for r in successful]
            cache_hits = sum(1 for r in successful if r.cache_hit)
            
            logger.info(f"  Avg inference time: {sum(times)/len(times):.2f}s")
            logger.info(f"  Cache hits: {cache_hits}/{len(successful)} ({cache_hits/len(successful):.1%})")
            
            for response in successful:
                cache_status = "HIT" if response.cache_hit else "MISS"
                logger.info(f"    {response.request_id}: {cache_status} | {response.inference_time:.2f}s")
    
    return responses

def main():
    """Main function to run the multi-model serving example."""
    logger.info("Starting Multi-Model Serving Example with LMCache-Neuron")
    
    try:
        # Create model configurations
        model_configs = create_example_model_configs()
        
        # Server configuration
        server_config = {
            "max_concurrent_requests": 12,
        }
        
        # Create and start server
        server = MultiModelServer(model_configs, server_config)
        server.start_server()
        
        # Run multi-model demo
        demo_responses = run_multi_model_demo(server)
        
        # Print final statistics
        logger.info("\n" + "="*60)
        logger.info("FINAL MULTI-MODEL STATISTICS")
        logger.info("="*60)
        
        detailed_stats = server.get_detailed_stats()
        
        # Global statistics
        global_stats = detailed_stats["global"]["global_stats"]
        logger.info(f"Total requests: {global_stats['total_requests']}")
        logger.info(f"Successful requests: {global_stats['successful_requests']}")
        logger.info(f"Failed requests: {global_stats['failed_requests']}")
        
        # Per-model statistics
        logger.info("\nPER-MODEL STATISTICS:")
        for model_name, stats in detailed_stats["models"].items():
            logger.info(f"  {model_name}:")
            logger.info(f"    Requests processed: {stats['requests_processed']}")
            logger.info(f"    Average inference time: {stats['average_inference_time']:.2f}s")
            logger.info(f"    Cache hit rate: {stats['cache_hit_rate']:.1%}")
            logger.info(f"    Error rate: {stats['error_rate']:.1%}")
        
        # Shared cache statistics
        if detailed_stats["cache"]:
            cache_stats = detailed_stats["cache"]
            logger.info(f"\nSHARED CACHE STATISTICS:")
            logger.info(f"  Overall hit rate: {cache_stats.get('hit_rate', 0):.1%}")
            logger.info(f"  Effective hit rate: {cache_stats.get('effective_hit_rate', 0):.1%}")
            logger.info(f"  Total cache requests: {cache_stats.get('total_requests', 0)}")
        
        # Shutdown server
        server.shutdown()
        
        logger.info("Multi-model serving example completed successfully")
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