# vLLM-Neuron LMCache Integration Examples

This directory contains example configurations and integration examples for using LMCache with vLLM-Neuron on AWS Neuron devices.

## Directory Structure

- `configs/` - Example configuration files for different use cases
- `integration/` - Complete integration examples with scripts
- `benchmarks/` - Performance benchmarking examples

## Quick Start

1. Choose an appropriate configuration from the `configs/` directory
2. Copy the configuration to your working directory
3. Modify the configuration parameters as needed for your environment
4. Run the integration examples to test your setup

## Configuration Files

### Basic Configuration (`configs/basic_neuron_config.yaml`)
- Simple CPU-based storage configuration
- Suitable for development and testing
- Uses local filesystem storage

### Advanced Configuration (`configs/advanced_remote_config.yaml`)
- Remote storage backend configuration
- Production-ready settings
- Includes monitoring and alerting

### Performance Optimized (`configs/performance_optimized_config.yaml`)
- Optimized for maximum performance
- Tuned cache sizes and parameters
- Advanced memory management settings

## Integration Examples

### Simple Offline Inference (`integration/simple_offline_inference.py`)
- Basic offline inference with LMCache
- Single model, single request processing
- Good for understanding the integration

### Online Serving (`integration/online_serving_example.py`)
- Complete online serving setup
- Multiple concurrent requests
- Production-like configuration

### Multi-Model Serving (`integration/multi_model_serving.py`)
- Serving multiple models simultaneously
- Shared cache configuration
- Resource management examples

## Requirements

- AWS Neuron SDK installed and configured
- vLLM-Neuron plugin installed
- LMCache library installed
- Python 3.8+ with required dependencies

## Environment Setup

Before running examples, ensure your environment is properly configured:

```bash
# Activate the Neuron virtual environment
source /opt/aws_neuronx_venv_pytorch_inference_vllm/bin/activate

# Set environment variables
export NEURON_RT_VISIBLE_CORES=0,1,2,3
export LMCACHE_CONFIG_PATH=./configs/basic_neuron_config.yaml
```

## Support

For issues and questions:
- Check the main vLLM-Neuron documentation
- Review LMCache documentation
- File issues in the project repository