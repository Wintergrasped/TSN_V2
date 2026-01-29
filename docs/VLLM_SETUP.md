# vLLM Configuration Guide

## Problem: "All vLLM endpoints failed"

If you see errors like this in your logs:

```
{"error": "All vLLM endpoints failed", "event": "transcript_smoothing_failed"}
{"base_url": "http://host.docker.internal:8001/v1/chat/completions", "error": "[Errno -2] Name or service not known"}
```

This means the TSN server can't reach your vLLM service for AI analysis.

## Quick Fix

**TSN will continue to work without vLLM** - ingestion and transcription run independently. However, you won't get:
- AI-powered transcript smoothing
- Callsign extraction
- Net detection
- Topic analysis
- Profile generation

## vLLM Setup Options

### Option 1: Disable vLLM (Basic Transcription Only)

If you don't need AI features, you can ignore the errors. TSN will:
- ✅ Ingest audio files
- ✅ Transcribe using Whisper
- ❌ No AI analysis/smoothing

The errors are just warnings - transcription continues normally.

### Option 2: Run Local vLLM Server

1. **Install vLLM:**
```bash
pip install vllm
```

2. **Start vLLM server:**
```bash
vllm serve Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4 \
    --port 8001 \
    --host 0.0.0.0 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.9 \
    --dtype auto \
    --quantization gptq
```

3. **Update TSN configuration:**

Edit your `.env` file:
```bash
# For Docker containers
TSN_VLLM_BASE_URL=http://host.docker.internal:8001

# For bare metal / same host
TSN_VLLM_BASE_URL=http://127.0.0.1:8001
```

4. **Restart TSN:**
```bash
docker-compose restart tsn_server
```

### Option 3: Use External vLLM Server

If you have vLLM running on another machine:

```bash
# In .env
TSN_VLLM_BASE_URL=http://your-vllm-server-ip:8001
```

### Option 4: Use OpenAI API (Alternative)

Instead of local vLLM, use OpenAI:

```bash
# In .env
TSN_VLLM_BASE_URL=https://api.openai.com/v1
TSN_VLLM_API_KEY=sk-your-openai-api-key
TSN_VLLM_MODEL=gpt-4o-mini
```

## Verifying vLLM Connection

Test if vLLM is reachable:

```bash
# From TSN server container
docker exec tsn_server curl -s http://host.docker.internal:8001/v1/models

# Expected output:
{"object": "list", "data": [{"id": "Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4", ...}]}
```

If this fails, vLLM isn't accessible from the container.

## Docker Network Issues

### Problem: `host.docker.internal` Not Working

On Linux, `host.docker.internal` may not work. Solutions:

**Solution 1: Use host network mode**

```yaml
# In docker-compose.yml
services:
  tsn_server:
    network_mode: host
```

**Solution 2: Use actual host IP**

```bash
# Find your host IP
ip addr show docker0 | grep inet

# In .env
TSN_VLLM_BASE_URL=http://172.17.0.1:8001
```

**Solution 3: Run vLLM in Docker**

```yaml
# Add to docker-compose.yml
services:
  vllm:
    image: vllm/vllm-openai:latest
    ports:
      - "8001:8000"
    environment:
      - MODEL=Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

Then use:
```bash
TSN_VLLM_BASE_URL=http://vllm:8000
```

## Configuration Reference

All vLLM settings in `.env`:

```bash
# Required
TSN_VLLM_BASE_URL=http://host.docker.internal:8001  # vLLM endpoint

# Optional
TSN_VLLM_API_KEY=sk-no-auth                         # API key (if required)
TSN_VLLM_MODEL=Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4  # Model name
TSN_VLLM_TIMEOUT_SEC=60                             # Request timeout
TSN_VLLM_MAX_CONCURRENT=10                          # Parallel requests
```

## Monitoring vLLM Usage

Check vLLM logs:
```bash
docker-compose logs -f tsn_server | grep vllm
```

Good:
```
{"event": "vllm_call_success", "latency_ms": 234}
{"event": "transcript_smoothing_applied", "chunk": 5}
```

Bad:
```
{"event": "vllm_call_failed_all_endpoints"}
{"error": "All vLLM endpoints failed"}
```

## Performance Tips

1. **Use GPTQ quantized models** for faster inference
2. **Increase `gpu-memory-utilization`** to 0.95 if you have VRAM headroom
3. **Set `TSN_VLLM_MAX_CONCURRENT`** based on GPU memory (1-3 for 8GB, 5-10 for 16GB+)
4. **Monitor GPU utilization** - aim for 80-95% during analysis

## Still Having Issues?

1. **Check vLLM is running:**
   ```bash
   curl http://localhost:8001/v1/models
   ```

2. **Check Docker networking:**
   ```bash
   docker exec tsn_server ping -c 3 host.docker.internal
   ```

3. **Check firewall:**
   ```bash
   sudo ufw allow 8001/tcp
   ```

4. **Review TSN logs:**
   ```bash
   docker-compose logs tsn_server | grep -E "vllm|analyzer"
   ```

## Impact on System

| Component | Works without vLLM? |
|-----------|---------------------|
| Audio ingestion | ✅ Yes |
| Whisper transcription | ✅ Yes |
| Basic audio playback | ✅ Yes |
| Transcript smoothing | ❌ No |
| Callsign extraction | ❌ No |
| Net detection | ❌ No |
| Topic analysis | ❌ No |
| Profile generation | ❌ No |

**Bottom line:** TSN gracefully degrades without vLLM. Transcription continues, but advanced AI features are disabled.
