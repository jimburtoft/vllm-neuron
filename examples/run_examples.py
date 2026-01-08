#!/usr/bin/env python3
"""
Run All LMCache-Neuron Integration Examples

This script provides a convenient way to run all the integration examples
in sequence or individually. It handles environment setup and provides
clear output for each example.
"""

import os
import sys
import time
import logging
import subprocess
from pathlib import Path
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_environment():
    """Setup common environment variables for all examples."""
    logger.info("Setting up environment for LMCache-Neuron examples...")
    
    # Activate Neuron virtual environment if available
    neuron_venv = "/opt/aws_neuronx_venv_pytorch_inference_vllm/bin/activate"
    if os.path.exists(neuron_venv):
        logger.info(f"Neuron virtual environment found at {neuron_venv}")
        logger.info("Please ensure you have activated it before running examples:")
        logger.info(f"source {neuron_venv}")
    else:
        logger.warning("Neuron virtual environment not found at expected location")
    
    # Set common environment variables
    if not os.environ.get("NEURON_RT_VISIBLE_CORES"):
        os.environ["NEURON_RT_VISIBLE_CORES"] = "0,1,2,3"
        logger.info("Set NEURON_RT_VISIBLE_CORES to 0,1,2,3")
    
    # Enable LMCache
    os.environ["ENABLE_LMCACHE"] = "true"
    logger.info("Enabled LMCache integration")
    
    # Set Python path
    examples_dir = Path(__file__).parent
    vllm_neuron_dir = examples_dir.parent
    sys.path.insert(0, str(vllm_neuron_dir))
    
    logger.info("Environment setup completed")

def check_prerequisites():
    """Check if all prerequisites are available."""
    logger.info("Checking prerequisites...")
    
    checks = []
    
    # Check if we're on a Neuron instance
    try:
        import glob
        neuron_devices = glob.glob('/dev/neuron*')
        if neuron_devices:
            checks.append(("Neuron devices", True, f"Found {len(neuron_devices)} devices"))
        else:
            checks.append(("Neuron devices", False, "No Neuron devices found"))
    except Exception as e:
        checks.append(("Neuron devices", False, f"Error checking: {e}"))
    
    # Check vLLM-Neuron availability
    try:
        import vllm_neuron
        checks.append(("vLLM-Neuron", True, "Available"))
    except ImportError:
        checks.append(("vLLM-Neuron", False, "Not installed"))
    
    # Check LMCache availability
    try:
        import lmcache
        checks.append(("LMCache", True, "Available"))
    except ImportError:
        checks.append(("LMCache", False, "Not installed"))
    
    # Check configuration files
    config_dir = Path(__file__).parent / "configs"
    basic_config = config_dir / "basic_neuron_config.yaml"
    if basic_config.exists():
        checks.append(("Basic config", True, str(basic_config)))
    else:
        checks.append(("Basic config", False, "Configuration file missing"))
    
    # Print results
    logger.info("Prerequisites check results:")
    all_good = True
    for name, status, details in checks:
        status_str = "✓" if status else "✗"
        logger.info(f"  {status_str} {name}: {details}")
        if not status:
            all_good = False
    
    return all_good

def run_example(example_name: str, script_path: Path) -> Dict[str, Any]:
    """Run a single example and return results."""
    logger.info(f"\n{'='*60}")
    logger.info(f"RUNNING EXAMPLE: {example_name}")
    logger.info(f"{'='*60}")
    
    if not script_path.exists():
        logger.error(f"Example script not found: {script_path}")
        return {
            "name": example_name,
            "success": False,
            "error": "Script not found",
            "duration": 0
        }
    
    start_time = time.time()
    
    try:
        # Run the example script
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout
        )
        
        duration = time.time() - start_time
        
        if result.returncode == 0:
            logger.info(f"Example {example_name} completed successfully in {duration:.1f}s")
            return {
                "name": example_name,
                "success": True,
                "duration": duration,
                "stdout": result.stdout,
                "stderr": result.stderr
            }
        else:
            logger.error(f"Example {example_name} failed with return code {result.returncode}")
            logger.error(f"STDERR: {result.stderr}")
            return {
                "name": example_name,
                "success": False,
                "error": f"Exit code {result.returncode}",
                "duration": duration,
                "stdout": result.stdout,
                "stderr": result.stderr
            }
    
    except subprocess.TimeoutExpired:
        duration = time.time() - start_time
        logger.error(f"Example {example_name} timed out after {duration:.1f}s")
        return {
            "name": example_name,
            "success": False,
            "error": "Timeout",
            "duration": duration
        }
    
    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"Example {example_name} failed with exception: {e}")
        return {
            "name": example_name,
            "success": False,
            "error": str(e),
            "duration": duration
        }

def run_all_examples() -> List[Dict[str, Any]]:
    """Run all integration examples."""
    examples_dir = Path(__file__).parent / "integration"
    
    examples = [
        ("Simple Offline Inference", examples_dir / "simple_offline_inference.py"),
        ("Online Serving", examples_dir / "online_serving_example.py"),
        ("Multi-Model Serving", examples_dir / "multi_model_serving.py"),
    ]
    
    results = []
    
    for example_name, script_path in examples:
        result = run_example(example_name, script_path)
        results.append(result)
        
        # Small delay between examples
        time.sleep(5)
    
    return results

def print_summary(results: List[Dict[str, Any]]):
    """Print summary of all example runs."""
    logger.info(f"\n{'='*60}")
    logger.info("EXAMPLES SUMMARY")
    logger.info(f"{'='*60}")
    
    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]
    
    logger.info(f"Total examples: {len(results)}")
    logger.info(f"Successful: {len(successful)}")
    logger.info(f"Failed: {len(failed)}")
    
    if successful:
        total_time = sum(r["duration"] for r in successful)
        logger.info(f"Total successful runtime: {total_time:.1f}s")
        
        logger.info("\nSuccessful examples:")
        for result in successful:
            logger.info(f"  ✓ {result['name']}: {result['duration']:.1f}s")
    
    if failed:
        logger.info("\nFailed examples:")
        for result in failed:
            logger.info(f"  ✗ {result['name']}: {result['error']}")
    
    logger.info(f"{'='*60}")

def main():
    """Main function to run examples."""
    logger.info("LMCache-Neuron Integration Examples Runner")
    
    try:
        # Setup environment
        setup_environment()
        
        # Check prerequisites
        if not check_prerequisites():
            logger.warning("Some prerequisites are missing. Examples may fail.")
            response = input("Continue anyway? (y/N): ")
            if response.lower() != 'y':
                logger.info("Exiting due to missing prerequisites")
                return 1
        
        # Check if user wants to run specific example
        if len(sys.argv) > 1:
            example_arg = sys.argv[1].lower()
            examples_dir = Path(__file__).parent / "integration"
            
            if example_arg in ["simple", "offline"]:
                result = run_example("Simple Offline Inference", 
                                   examples_dir / "simple_offline_inference.py")
                results = [result]
            elif example_arg in ["online", "serving"]:
                result = run_example("Online Serving", 
                                   examples_dir / "online_serving_example.py")
                results = [result]
            elif example_arg in ["multi", "multimodel"]:
                result = run_example("Multi-Model Serving", 
                                   examples_dir / "multi_model_serving.py")
                results = [result]
            else:
                logger.error(f"Unknown example: {example_arg}")
                logger.info("Available examples: simple, online, multi")
                return 1
        else:
            # Run all examples
            logger.info("Running all integration examples...")
            results = run_all_examples()
        
        # Print summary
        print_summary(results)
        
        # Return appropriate exit code
        failed_count = sum(1 for r in results if not r["success"])
        return 1 if failed_count > 0 else 0
        
    except KeyboardInterrupt:
        logger.info("Examples interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Examples runner failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)