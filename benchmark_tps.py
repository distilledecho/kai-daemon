#!/usr/bin/env python3
"""mlx-kv-server token throughput benchmark.

Usage:
    cd ~/dev/kai-daemon
    uv run python3 benchmark_tps.py

Measures tokens/sec for generate() calls with thinking disabled.
Runs 3 rounds and reports per-round and average tps.
"""

import os
import sys
import time

SOCKET_PATH = os.environ.get("MLX_KV_SOCKET_PATH", "/tmp/mlx-kv-server.sock")
CACHE_ID = "kai-benchmark"

PROMPTS = [
    "Reply in exactly one sentence: what is the capital of France?",
    "Reply in exactly one sentence: what is 12 multiplied by 8?",
    "Reply in exactly one sentence: name one planet in our solar system.",
]


def main():
    try:
        from mlx_kv_client import MlxKvClient
    except ImportError:
        print("ERROR: mlx_kv_client not installed. Run from kai-daemon with uv run.")
        sys.exit(1)

    try:
        from transformers import AutoTokenizer
    except ImportError:
        print("ERROR: transformers not installed.")
        sys.exit(1)

    client = MlxKvClient(SOCKET_PATH)

    try:
        status = client.status()
    except Exception as e:
        print(f"ERROR: cannot reach mlx-kv-server at {SOCKET_PATH}: {e}")
        sys.exit(1)

    print(f"model : {status.model}")
    print(
        f"cache : {status.cache_used_tokens}/{status.cache_capacity_tokens} tokens used"
    )
    print()

    tokenizer = AutoTokenizer.from_pretrained(status.model, local_files_only=True)

    results = []

    for i, prompt in enumerate(PROMPTS):
        messages = [{"role": "user", "content": prompt}]
        formatted = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        tokens = tokenizer.encode(formatted, add_special_tokens=False)

        # evict any leftover cache from previous round
        try:
            client.evict(CACHE_ID)
        except Exception:
            pass

        client.prefill(tokens[:-1], CACHE_ID)

        token_count = 0
        output_tokens = []
        t0 = time.perf_counter()

        for tok in client.generate([tokens[-1]], CACHE_ID):
            output_tokens.append(tok)
            token_count += 1

        elapsed = time.perf_counter() - t0
        tps = token_count / elapsed if elapsed > 0 else 0
        decoded = tokenizer.decode(output_tokens, skip_special_tokens=True).strip()

        results.append(tps)
        print(f"round {i + 1}: {token_count} tokens in {elapsed:.2f}s = {tps:.1f} tps")
        print(f"  response: {decoded[:120]}")
        print()

    avg = sum(results) / len(results)
    print(f"average: {avg:.1f} tps across {len(results)} rounds")

    # cleanup
    try:
        client.evict(CACHE_ID)
    except Exception:
        pass


if __name__ == "__main__":
    main()
