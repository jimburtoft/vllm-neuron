# Isaac-0.5 on AWS Trainium2 (vLLM-Neuron + vLLM-Omni)

Serves [PerceptronAI/Isaac-0.5](https://huggingface.co/PerceptronAI/Isaac-0.5) on one Trainium2 device (trn2.3xlarge,
TP=4, LNC=2) in two ways:

1. **Text / vision-language serving** with vLLM-Neuron 0.29: the 36B hybrid MoE decoder (Gated DeltaNet + gated GQA,
   256 experts top-8 with the null-expert router) plus the Qwen3.5 vision tower.
2. **Robot-action serving** with vLLM-Omni 0.29 over the OpenPI realtime websocket API
   (`/v1/realtime/robot/openpi`): one VLM prefill over the observation followed by the 0.58B DiT flow expert
   (10 Euler steps). Each request returns a 50 x 7 action chunk. Concurrent requests are batched.

Everything runs on public releases:
- the **Neuron SDK 2.33 DLAMI** (`Deep Learning AMI Neuron (Ubuntu 24.04) SDK 2.33.0 20261020`) and its stock venvs
  (`aws_neuronx_venv_pytorch_inference_vllm_0_29_0_1_2_0`, `aws_neuronx_venv_pytorch_inference_vllm_omni_0_29_0_0_2_0`);
- this branch of vllm-neuron, based on tag `v0.29.0.1.2.0`;
- vllm-omni 0.29.0rc1 with vllm-omni-neuron 0.29.0.0.2.0;
- LeRobot (perceptron-ai-inc fork, commit `e12389c`) and mharmony (commit `9a57efb`, built from source).

## What is in this branch

| area | files | what it adds |
|---|---|---|
| Qwen3.5-MoE decoder | `vllm_neuron/model/qwen3_5_moe/` | GDN + GQA + MoE text model, null-expert routing, VL view, packed multi-prompt prefill, decode megakernel (`kernels/mk016`), token-based KV admission |
| Job-list MoE prefill kernel | `kernels/moe_prefill_jobs.py` | NKI kernel for small-T, sparse MoE prefill (see below) |
| Action path | `qwen3_5_moe/action.py`, `contrib/isaac05_omni/` | VLM prefill to final hidden states (single and packed batch), vLLM-Omni `IsaacActionPipeline`, device vision tower, batched DiT |
| Tools | `contrib/isaac05_omni/openpi_*.py`, `tests/` | OpenPI accuracy gate and load generator; simulator and hardware tests for the MoE kernel |

### Job-list MoE prefill kernel
At the action-path prompt length (L = 462 tokens), the null router drops about half of the expert picks. Each used
expert then sees only 4 to 7 tokens (median), and a block-based MoE kernel runs at about 5 % fill. The job-list kernel
works as follows:
- **Jobs.** It processes each used expert once per layer, as a job of up to 128 tokens; experts with more tokens get
  extra overflow jobs.
- **Unused experts** move no weights.
- **Combine.** It writes route-compact staging rows and then gather-accumulates them per token in fp32, with no
  read-modify-write chain.
- **Lane tables.** These are built on chip.
- **Pipelining.** The job loop is software-pipelined two deep, with prefetched weights.

One layer at T = 512 takes 0.90 ms per rank, against 4.05 ms for the block kernel at block size 128. Enable it with
`ISAAC_MOE_PREFILL_IMPL=jobs`; the action pipeline turns it on by default.

## Setup

```bash
# on a trn2.3xlarge with the SDK 2.33 DLAMI (500 GB disk: FP32 download + BF16 copy + compile caches)
git clone -b isaac05-action https://github.com/jimburtoft/vllm-neuron.git ~/vllm-neuron
C=~/vllm-neuron/contrib/isaac05_omni
source /opt/aws_neuronx_venv_pytorch_inference_vllm_omni_0_29_0_0_2_0/bin/activate
# 1) BF16 checkpoint (the published checkpoint is FP32; the model runs in BF16)
hf download PerceptronAI/Isaac-0.5 --revision 6e397429f0ab91528ebac915004eba5e763fcdd4 --local-dir ~/models/Isaac-0.5-fp32
python $C/scripts/convert_bf16.py ~/models/Isaac-0.5-fp32 ~/models/Isaac-0.5-bf16
# 2) vLLM views of the checkpoint for text / vision-language serving (symlinks + rewritten config)
python $C/scripts/make_vllm_dir.py ~/models/Isaac-0.5-bf16 ~/models/Isaac-0.5-vllm text
python $C/scripts/make_vllm_dir.py ~/models/Isaac-0.5-bf16 ~/models/Isaac-0.5-vllm-vl vl
# 3) action-path extras into an overlay dir (the SDK venvs stay untouched)
git clone https://github.com/perceptron-ai-inc/lerobot.git ~/lerobot && git -C ~/lerobot checkout e12389c1f8f591ad05dced4e284d4e92e48c5df4
git clone https://github.com/perceptron-ai-inc/mharmony.git ~/mharmony && git -C ~/mharmony checkout 9a57efbf8ada2d73641fc40081f4e4fa6ff2ab92
curl -sSf https://sh.rustup.rs | sh -s -- -y && source ~/.cargo/env          # mharmony is a Rust extension
python3 -m venv ~/maturin && ~/maturin/bin/pip install maturin
(cd ~/mharmony && ~/maturin/bin/maturin build --release -o /tmp/mhwheel)
mkdir -p ~/isaac05_overlay
pip install --no-deps --target ~/isaac05_overlay /tmp/mhwheel/*.whl draccus gymnasium Farama-Notifications \
    mergedeep termcolor toml typing-inspect
ln -s ~/lerobot/src/lerobot ~/isaac05_overlay/lerobot
pip install --no-deps --target ~/isaac05_overlay $C
python -c "from huggingface_hub import hf_hub_download as d; [d('Qwen/Qwen3.5-0.8B', f) for f in ['vocab.json','merges.txt','tokenizer.json','tokenizer_config.json','config.json','chat_template.jinja']]"
```
Add a 64 GB swap file before the first compile. The compiler can need more host memory than the instance has.

## Serving

Text / VL (vLLM-Neuron, OpenAI API):
```bash
MODEL=~/models/Isaac-0.5-vllm PROFILE=t2k bash ~/vllm-neuron/contrib/isaac05_omni/scripts/serve_text.sh      # text, 2K
MODEL=~/models/Isaac-0.5-vllm PROFILE=t8k bash ~/vllm-neuron/contrib/isaac05_omni/scripts/serve_text.sh      # text, 8K
MODEL=~/models/Isaac-0.5-vllm-vl PROFILE=img bash ~/vllm-neuron/contrib/isaac05_omni/scripts/serve_text.sh   # images
```

Robot actions (vLLM-Omni, OpenPI websocket):
```bash
ISAAC05_MODEL=~/models/Isaac-0.5-bf16 DEPLOY=isaac05_action_tp4_batch.yaml \
  XENV="ISAAC05_MAXB=16 ISAAC05_BATCH_BUCKETS=1,2,4,8,16" bash ~/vllm-neuron/contrib/isaac05_omni/serve_isaac.sh
```
Clients use the standard OpenPI websocket protocol (msgpack-numpy), with the LeRobot keys `observation.images.*`,
`observation.state` and `task`. Send camera frames as uint8 HWC/CHW arrays: float frames are accepted, but they
cost about 30 ms more per request in transfer. Request batching needs `max_num_seqs` > 1 in the deploy YAML and
`ISAAC05_MAXB` set to the same value. The first request for each new batch size compiles its graphs, which takes a long
time; warm up every batch bucket before you measure.

## Performance (one trn2.3xlarge, BF16)

**Robot actions**:
- Setup: OpenPI websocket, 2 cameras at 256 x 256 (3-step observation window), joint state and instruction (462
  prompt tokens), 50 x 7 action chunk, 10 flow steps.
- Measurement: concurrent clients, 60 s windows. Latency is the full websocket round trip.

| concurrent clients | 1 | 2 | 4 | 8 | 16 | 24 | 32 | 48 |
|---|---|---|---|---|---|---|---|---|
| plans/s | 4.50 | 4.88 | 6.40 | 8.00 | 9.62 | 10.82 | 11.47 | 11.48 |
| p50 latency (ms) | 222 | 408 | 624 | 993 | 1641 | 2182 | 2766 | 4143 |

Single-plan latency breakdown:

| stage | time |
|---|---|
| render | 10 ms |
| vision tower | 36 ms |
| VLM prefill | 85 to 100 ms |
| DiT | about 50 ms |
| request transfer | 15 ms (uint8 frames) |

Accuracy:
- Action chunks match the reference implementation run with the same initial noise. Cosine similarity is
  0.998 / 0.999 / 0.999 on the three test observations, both one at a time and in batches of 16.
- `contrib/isaac05_omni/openpi_gate.py` and `openpi_batch_gate.py` reproduce this check.

**Text and vision-language** (`vllm bench serve`, random prompts, output tokens/s at the throughput plateau):

| shape (input / output) | peak output tok/s | at concurrency |
|---|---|---|
| 256 / 128 | 1531 | 256 |
| 256 / 1024 | 2296 | 256 |
| 1024 / 128 | 650 | 512 |
| 2048 / 1024 | 762 | 64 |
| 7168 / 1024 | 396 | 32 |
| 2 images (384 x 384) + 128 / 128 | 700 | 256 |

Single-request decode is 6.1 ms per token at 2K context and 7.0 ms per token at 8K.

## Known limits
- Action path:
  - One observation layout (2 cameras at 256 x 256, 3-step window, L <= 512) per compiled graph set.
  - Requests are stateless: the 3-step window is filled from the current frame (no history across requests).
  - Batch size per engine step is capped at 16; B = 32 fails to compile with neuronx-cc 2.28 (NCC_INLA001).
  - Real-time chunking (action-prefix conditioning) is not wired in.
- Text serving: `max_num_seqs` is capped at 128 by the decode megakernel.
