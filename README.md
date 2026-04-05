# local-llm

Local LLM inference server running **Qwen3.5-4B** with [TurboQuant](https://github.com/TheTom/llama-cpp-turboquant) KV cache compression on consumer GPUs.

TurboQuant uses Walsh-Hadamard Transform rotation + Lloyd-Max polar quantization to compress the KV cache, enabling **64k context on a 4GB GPU** with zero speed loss.

## Performance (GTX 1650, 4GB VRAM)

| Metric | Value |
|--------|-------|
| Model | Qwen3.5-4B IQ2_M (1.7 GB) |
| Context | 65,536 tokens |
| Prompt processing | ~195 t/s |
| Generation | ~29-33 t/s |
| KV cache | 731 MiB (turbo2) vs 2,048 MiB (f16) |
| API | OpenAI-compatible (`/v1/chat/completions`) |
| Port | 8899 |

Without TurboQuant, the same model OOMs at just ~8k context with f16 KV cache.

## Quick Start

### 1. Clone and build

```bash
git clone https://github.com/ftulabs/local-llm.git
cd local-llm

# Create venv and install dependencies
python3 -m venv venv
source venv/bin/activate
pip install huggingface-hub

# Clone and build TurboQuant fork of llama.cpp
git clone --branch feature/turboquant-kv-cache \
    https://github.com/TheTom/llama-cpp-turboquant.git turboquant
cd turboquant

# Build with CUDA (change CMAKE_CUDA_ARCHITECTURES to match your GPU)
# Common values: 75 (GTX 16xx/RTX 20xx), 86 (RTX 30xx), 89 (RTX 40xx)
cmake -B build \
    -DGGML_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES=75 \
    -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
cd ..
```

### 2. Download the model

```bash
source venv/bin/activate
python3 -c "
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id='unsloth/Qwen3.5-4B-GGUF',
    filename='Qwen3.5-4B-UD-IQ2_M.gguf',
    local_dir='models'
)
"
```

### 3. Run the server

```bash
./llama-server-start.sh
```

Or manually:

```bash
./turboquant/build/bin/llama-server \
    -m models/Qwen3.5-4B-UD-IQ2_M.gguf \
    --host 0.0.0.0 --port 8899 \
    -ngl 99 -fa on -c 65536 \
    -ctk q8_0 -ctv turbo2 \
    -t 6 --reasoning off
```

### 4. Test

```bash
curl http://localhost:8899/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "qwen",
        "messages": [{"role": "user", "content": "Hello!"}],
        "max_tokens": 64
    }'
```

## systemd Service (auto-start on boot)

### Install

```bash
# Edit llama-server.env to match your paths if needed
nano llama-server.env

# Install and enable the service
sudo cp llama-server.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable llama-server
sudo systemctl start llama-server
```

### Manage

```bash
# Check status
sudo systemctl status llama-server

# View logs
journalctl -u llama-server -f

# Restart after config changes
sudo systemctl restart llama-server

# Stop
sudo systemctl stop llama-server

# Disable auto-start
sudo systemctl disable llama-server
```

## Configuration

Edit `llama-server.env` to customize:

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | `models/Qwen3.5-4B-UD-IQ2_M.gguf` | Path to GGUF model |
| `PORT` | `8899` | Server port |
| `CTX_SIZE` | `65536` | Context window size |
| `GPU_LAYERS` | `99` | Layers offloaded to GPU |
| `THREADS` | `6` | CPU threads |
| `CACHE_TYPE_K` | `q8_0` | Key cache quantization |
| `CACHE_TYPE_V` | `turbo2` | Value cache quantization (turbo2/turbo3/turbo4) |

### TurboQuant KV cache types

| Type | Bits per value | Compression vs f16 | Quality impact |
|------|---------------|---------------------|----------------|
| `turbo4` | 4.25 bpv | 2.51x | negligible |
| `turbo3` | 3.125 bpv | 5.12x | minimal |
| `turbo2` | 2.125 bpv | 7.53x | small (recommended) |

Key precision (`-ctk`) should stay at `q8_0` — keys are more sensitive than values due to softmax amplification.

## GPU Compatibility

Tested on NVIDIA GTX 1650 (compute 7.5). Should work on any CUDA GPU with sufficient VRAM.

Adjust `CMAKE_CUDA_ARCHITECTURES` when building:

| GPU Family | Compute | Architecture flag |
|------------|---------|-------------------|
| GTX 16xx / RTX 20xx | 7.5 | `75` |
| RTX 30xx | 8.6 | `86` |
| RTX 40xx | 8.9 | `89` |
| Jetson Xavier | 7.2 | `72` |

## Agent with Tools (Search + RAG)

A lightweight Python agent that adds web search and document RAG to the LLM server.

### Install agent dependencies

```bash
source venv/bin/activate
pip install -r requirements.txt
```

### Usage

```bash
source venv/bin/activate

# Interactive mode
python agent.py

# Single query
python agent.py "Search for the latest news about AI"

# Verbose mode (shows tool calls)
python agent.py -v "What is the population of Japan?"
```

### Available tools

| Tool | Description |
|------|-------------|
| `web_search` | Search the web via DuckDuckGo (no API key needed) |
| `rag_search` | Search ingested documents using ChromaDB + all-MiniLM-L6-v2 embeddings |
| `rag_ingest` | Add text files to the knowledge base |

### Ingest documents

```bash
# Via CLI shortcut (interactive mode)
python agent.py
You: /ingest /path/to/document.txt

# Or ask the agent
python agent.py "Ingest the file /path/to/my_notes.txt into the knowledge base"

# Then query
python agent.py "What does the document say about X?"
```

### Custom server URL

```bash
python agent.py --url http://192.168.1.100:8899/v1 "Hello"
```

The RAG database is stored in `rag_db/` and persists between sessions.

## References

- [TurboQuant paper (ICLR 2026)](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/)
- [llama-cpp-turboquant fork](https://github.com/TheTom/llama-cpp-turboquant)
- [Qwen3.5-4B GGUF](https://huggingface.co/unsloth/Qwen3.5-4B-GGUF)
- [llama.cpp](https://github.com/ggml-org/llama.cpp)
