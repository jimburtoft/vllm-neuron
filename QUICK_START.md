# LMCache-Neuron Integration Quick Start

This guide shows you how to quickly spin up and test the LMCache-Neuron integration.

## Prerequisites

1. **AWS Neuron Environment**: Ensure you're running on an AWS Neuron instance (Trainium/Inferentia)
2. **Virtual Environment**: Activate the Neuron virtual environment:
   ```bash
   source /opt/aws_neuronx_venv_pytorch_inference_vllm/bin/activate
   ```
3. **Dependencies**: Ensure LMCache and vLLM are installed

## Quick Test

### Basic Test (Limited LMCache Effectiveness)

Run the comprehensive quick start test with default vLLM-Neuron settings:

```bash
cd vllm-neuron
python examples/quick_start_test.py
```

**Note**: This will run with chunked prefill disabled (vLLM-Neuron default), which limits LMCache to full-sequence matches only.

### Enhanced Test (Full LMCache Effectiveness)

For full LMCache functionality with partial cache hits:

```bash
cd vllm-neuron
export DISABLE_NEURON_CUSTOM_SCHEDULER=1
python examples/quick_start_test.py
```

**Important**: When enabling chunked prefill, ensure your vLLM configuration includes appropriate `num_gpu_blocks_override` settings.

This script will:
- ✅ Check all prerequisites
- ✅ Test integration initialization
- ✅ Validate storage backend functionality
- ✅ Verify platform detection
- ✅ Test performance monitoring
- ✅ Provide usage instructions

## Expected Output

### With Chunked Prefill Disabled (Default)

```
🚀 LMCache-Neuron Integration Quick Start Test
============================================================
🔍 Checking prerequisites...
✓ LMCache is available
✓ vLLM is available
✓ LMCache-Neuron integration modules loaded
📝 Creating test configuration...
✓ Test configuration created: test_lmcache_config.yaml
🚀 Testing integration initialization...
⚠️  CHUNKED PREFILL DISABLED - LIMITED LMCACHE EFFECTIVENESS

Current Status: Chunked prefill is disabled (vLLM-Neuron default)
Impact: LMCache can only cache complete sequences, not partial matches

To Enable Full LMCache Functionality:
1. Set environment variable: DISABLE_NEURON_CUSTOM_SCHEDULER=1
2. Configure num_gpu_blocks_override parameter appropriately
3. Restart your vLLM service
✓ Integration initialized successfully
✓ Chunked prefill status: disabled
✓ Cache strategy: Full sequence caching only
✓ Health status: healthy
[... rest of output ...]
🎉 ALL TESTS PASSED! LMCache-Neuron integration is working correctly.
```

### With Chunked Prefill Enabled

```
🚀 LMCache-Neuron Integration Quick Start Test
============================================================
[... initial output ...]
🚀 Testing integration initialization...
✓ Integration initialized successfully
✓ Chunked prefill status: enabled
✓ Cache strategy: Full chunked prefill caching
✓ Health status: healthy
[... rest of output ...]
🎉 ALL TESTS PASSED! LMCache-Neuron integration is working correctly.
```

## Next Steps

After the quick test passes, you can:

1. **Try the examples**:
   - `examples/integration/simple_offline_inference.py` - Basic offline inference
   - `examples/integration/online_serving_example.py` - Online serving setup
   - `examples/integration/multi_model_serving.py` - Multi-model configuration

2. **Configure for your use case**:
   - `examples/configs/basic_neuron_config.yaml` - Basic setup
   - `examples/configs/advanced_remote_config.yaml` - With remote storage
   - `examples/configs/performance_optimized_config.yaml` - Performance optimized

3. **Monitor performance**:
   - Use the built-in performance monitoring
   - Check cache hit rates and memory usage
   - Monitor TTFT improvements

## Troubleshooting

If the quick test fails:

1. **Check Prerequisites**: Ensure all dependencies are installed
2. **Environment**: Verify you're in the correct virtual environment
3. **Permissions**: Ensure you have write permissions for temporary files
4. **Logs**: Check the detailed logs for specific error messages

## Configuration

The integration uses YAML configuration files. Set the environment variable:

```bash
export LMCACHE_CONFIG_PATH=/path/to/your/config.yaml
```

## Performance Expectations

### With Chunked Prefill Enabled (Full Functionality)

With LMCache-Neuron integration and chunked prefill enabled, you should see:
- **30-50% reduction** in Time To First Token (TTFT) for partial cache hits
- **High cache hit rates** for similar prompts
- **Optimal memory efficiency** with partial caching
- **<10ms latency** for cache operations

### With Chunked Prefill Disabled (Limited Functionality - Default)

With chunked prefill disabled (vLLM-Neuron default), you will see:
- **30-50% reduction** in TTFT for identical prompts only
- **Lower cache hit rates** (full sequence matches only)
- **Reduced memory efficiency** (no partial caching)
- **<10ms latency** for cache operations
- **Recommendation**: Enable chunked prefill for full benefits

### How to Enable Full Functionality

```bash
export DISABLE_NEURON_CUSTOM_SCHEDULER=1
# Also ensure proper num_gpu_blocks_override configuration in your vLLM setup
```

## Support

For issues or questions:
1. Check the examples in `examples/integration/`
2. Review configuration templates in `examples/configs/`
3. Run the quick start test for diagnostics
4. Check the implementation in `vllm_neuron/lmcache_integration.py`