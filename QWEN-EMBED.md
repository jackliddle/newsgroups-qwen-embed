# QWEN-3-Embedding Usage Guide

A comprehensive guide to using QWEN-3-embedding models via SiliconFlow API, based on production patterns from a high-scale vector embedding system.

## Table of Contents

- [Overview](#overview)
- [SiliconFlow Setup](#siliconflow-setup)
- [Basic Usage](#basic-usage)
- [Batching Strategies](#batching-strategies)
- [Multi-Task Instructions](#multi-task-instructions)
- [Rate Limiting](#rate-limiting)
- [Error Handling](#error-handling)
- [Production Patterns](#production-patterns)
- [Reusable Code Templates](#reusable-code-templates)
- [Reference](#reference)

---

## Overview

**QWEN-3-Embedding** is a family of text embedding models from Alibaba, available via SiliconFlow's API. These models convert text into dense vector representations suitable for semantic search, clustering, and similarity tasks.

### Model Variants

| Model | Max Dimensions | Use Case |
|-------|----------------|----------|
| `Qwen/Qwen3-Embedding-0.6B` | 1024 | Fast, cost-effective embeddings |
| `Qwen/Qwen3-Embedding-4B` | 2048 | Balanced performance |
| `Qwen/Qwen3-Embedding-8B` | 4096 | Highest quality embeddings |

### Valid Dimension Sizes

Each model supports specific dimension outputs:
- **0.6B:** 64, 128, 256, 512, 768, 1024
- **4B:** 64, 128, 256, 512, 768, 1024, 2048
- **8B:** 64, 128, 256, 512, 768, 1024, 2048, 4096

---

## SiliconFlow Setup

### API Configuration

```python
API_BASE_URL = "https://api.siliconflow.com/v1"
EMBEDDINGS_ENDPOINT = f"{API_BASE_URL}/embeddings"
```

### Authentication

```python
import requests

session = requests.Session()
session.headers.update({
    "Authorization": f"Bearer {YOUR_API_KEY}",
    "Content-Type": "application/json"
})
```

### Environment Setup

Create a `.env` file:
```bash
SILICONFLOW_API_KEY=your_api_key_here
```

Load in your application:
```python
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("SILICONFLOW_API_KEY")
```

---

## Basic Usage

### Simple Synchronous Embedding

```python
import requests

def embed_texts(texts, api_key, model="Qwen/Qwen3-Embedding-0.6B", dimensions=1024):
    """
    Embed a list of texts using QWEN-3-Embedding.

    Args:
        texts: List of strings to embed
        api_key: SiliconFlow API key
        model: Model identifier
        dimensions: Output dimension size

    Returns:
        numpy array of shape (len(texts), dimensions)
    """
    url = "https://api.siliconflow.com/v1/embeddings"

    payload = {
        "model": model,
        "input": texts,
        "dimensions": dimensions,
        "encoding_format": "float"
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    response = requests.post(url, json=payload, headers=headers)
    response.raise_for_status()

    result = response.json()
    embeddings = [item['embedding'] for item in result['data']]

    return embeddings
```

### Usage Example

```python
texts = [
    "The quick brown fox jumps over the lazy dog",
    "Machine learning is a subset of artificial intelligence",
    "Python is a popular programming language"
]

embeddings = embed_texts(texts, api_key="your_key_here")
print(f"Generated {len(embeddings)} embeddings of dimension {len(embeddings[0])}")
```

---

## Batching Strategies

### Why Batch?

Batching improves throughput and reduces API overhead. Recommended batch size: **32 texts per request**.

### Synchronous Batching

```python
import time
import numpy as np

def embed_large_dataset(texts, batch_size=32, delay=0.5):
    """
    Embed a large dataset with batching and rate limiting.

    Args:
        texts: List of all texts to embed
        batch_size: Number of texts per API call
        delay: Delay between batches (seconds) to avoid rate limits

    Returns:
        numpy array of all embeddings
    """
    all_embeddings = []
    total_batches = (len(texts) + batch_size - 1) // batch_size

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        batch_num = i // batch_size + 1

        print(f"Processing batch {batch_num}/{total_batches} ({len(batch_texts)} texts)...")

        # Embed this batch
        batch_embeddings = embed_texts(batch_texts, api_key)
        all_embeddings.extend(batch_embeddings)

        # Rate limiting delay (except for last batch)
        if i + batch_size < len(texts):
            time.sleep(delay)

    return np.array(all_embeddings)
```

### Async Batching with Concurrency

For maximum throughput, use async requests with controlled concurrency:

```python
import asyncio
import aiohttp
from aiolimiter import AsyncLimiter

class AsyncEmbedder:
    def __init__(self, api_key, model="Qwen/Qwen3-Embedding-0.6B", dimensions=1024):
        self.api_key = api_key
        self.model = model
        self.dimensions = dimensions
        self.url = "https://api.siliconflow.com/v1/embeddings"

        # Rate limiters (for Tier L0: 1M TPM, 2K RPM)
        self.tpm_limiter = AsyncLimiter(16666, 1)  # ~1M tokens/min = 16.6K/sec
        self.rpm_limiter = AsyncLimiter(33, 1)     # 2K requests/min = ~33/sec

    async def embed_batch(self, session, texts):
        """Embed a single batch with rate limiting."""
        # Estimate tokens (rough: 4 chars = 1 token)
        estimated_tokens = sum(len(t) // 4 for t in texts)

        # Acquire rate limit tokens
        await self.rpm_limiter.acquire(1)
        await self.tpm_limiter.acquire(estimated_tokens)

        payload = {
            "model": self.model,
            "input": texts,
            "dimensions": self.dimensions,
            "encoding_format": "float"
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        async with session.post(self.url, json=payload, headers=headers) as response:
            response.raise_for_status()
            result = await response.json()
            return [item['embedding'] for item in result['data']]

    async def embed_parallel(self, texts, batch_size=32, concurrency=10):
        """
        Embed texts with parallel batching.

        Args:
            texts: List of texts to embed
            batch_size: Texts per batch
            concurrency: Max concurrent requests

        Returns:
            List of embeddings (same order as input)
        """
        # Split into batches
        batches = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batches.append((i // batch_size, batch_texts))

        all_embeddings = [None] * len(batches)

        async with aiohttp.ClientSession() as session:
            # Process in chunks to limit concurrency
            for chunk_start in range(0, len(batches), concurrency):
                chunk = batches[chunk_start:chunk_start + concurrency]

                tasks = [
                    self.embed_batch(session, batch_texts)
                    for batch_num, batch_texts in chunk
                ]

                chunk_results = await asyncio.gather(*tasks)

                # Store results in order
                for idx, (batch_num, _) in enumerate(chunk):
                    all_embeddings[batch_num] = chunk_results[idx]

        # Flatten results
        return [emb for batch in all_embeddings for emb in batch]

# Usage
async def main():
    embedder = AsyncEmbedder(api_key="your_key")
    texts = ["text1", "text2", ...]  # Your texts here
    embeddings = await embedder.embed_parallel(texts, concurrency=10)
    print(f"Generated {len(embeddings)} embeddings")

asyncio.run(main())
```

**Performance:** With `concurrency=10` and `batch_size=32`, expect ~50-60 texts/second.

---

## Multi-Task Instructions

QWEN-3-Embedding supports **task-specific instructions** that condition the embeddings for specialized retrieval tasks.

### Instruction Format

```python
# Format: "Instruct: {task_instruction}\nQuery:{your_text}"

def format_text_with_instruction(text, instruction):
    """Apply task instruction to text (Qwen3 format)."""
    if instruction:
        return f"Instruct: {instruction}\nQuery:{text}"
    else:
        return text
```

### Example Use Cases

```python
# Define task instructions
TASKS = {
    "general": None,  # No instruction - baseline semantic search

    "reviews": "Retrieve comprehensive review articles and surveys that provide broad overviews of this topic",

    "technical": "Retrieve technical documentation and implementation details related to this topic",

    "troubleshooting": "Retrieve debugging guides, error solutions, and troubleshooting resources for this issue",

    "comparisons": "Retrieve comparative analyses and benchmark studies related to this topic"
}

# Apply instruction at embedding time
def embed_with_task(text, task_name, api_key):
    instruction = TASKS.get(task_name)
    formatted_text = format_text_with_instruction(text, instruction)
    return embed_texts([formatted_text], api_key)[0]
```

### Important: Search-Time Matching

**You must use the SAME instruction at search time** as you used during indexing:

```python
# During indexing
document = "Python is a programming language"
instruction = "Retrieve technical documentation..."
doc_embedding = embed_with_task(document, "technical", api_key)

# During search (must match!)
query = "programming language syntax"
query_embedding = embed_with_task(query, "technical", api_key)

# Now you can compute similarity
similarity = cosine_similarity(query_embedding, doc_embedding)
```

### Multi-Task Architecture

For advanced use cases, you can maintain separate vector indices per task:

```python
# Index documents with different task embeddings
tasks = ["general", "technical", "troubleshooting"]

for task in tasks:
    instruction = TASKS[task]
    task_embeddings = []

    for doc in documents:
        formatted = format_text_with_instruction(doc, instruction)
        emb = embed_texts([formatted], api_key)[0]
        task_embeddings.append(emb)

    # Store in separate index/collection
    save_to_index(f"index_{task}", task_embeddings)

# At search time, choose the appropriate task
query = "how to fix error X"
task = "troubleshooting"  # User-selected or auto-detected
query_emb = embed_with_task(query, task, api_key)
results = search_index(f"index_{task}", query_emb)
```

---

## Rate Limiting

### SiliconFlow Tier L0 Limits

| Limit Type | Value | Per |
|------------|-------|-----|
| **TPM** (Tokens) | 1,000,000 | minute |
| **RPM** (Requests) | 2,000 | minute |

### Rate Limiting Strategy

**1. Conservative Rate Limiting (Async):**

```python
from aiolimiter import AsyncLimiter

# Set limits slightly below max to avoid edge cases
tpm_limiter = AsyncLimiter(16666, 1)  # 1M tokens/min ÷ 60 sec ≈ 16.6K/sec
rpm_limiter = AsyncLimiter(33, 1)     # 2K requests/min ÷ 60 sec ≈ 33/sec

async def rate_limited_request(texts):
    # Estimate tokens
    estimated_tokens = sum(len(t) // 4 for t in texts)

    # Acquire both limits before making request
    await rpm_limiter.acquire(1)
    await tpm_limiter.acquire(estimated_tokens)

    # Make request...
```

**2. Simple Delay (Sync):**

```python
import time

# With 32-text batches (~8K tokens each):
# 112 batches/min = 896K TPM (under 1M limit)
# Requires ~0.5s delay between batches

for i in range(0, len(texts), batch_size):
    batch = texts[i:i + batch_size]
    embeddings = embed_texts(batch, api_key)

    # Delay to stay under limits
    if i + batch_size < len(texts):
        time.sleep(0.5)
```

### HTTP 429 Handling (Exponential Backoff)

```python
import time
import requests

def embed_with_retry(texts, api_key, max_retries=5):
    """Embed texts with exponential backoff for 429 errors."""
    base_delay = 1

    for attempt in range(max_retries):
        try:
            response = requests.post(
                "https://api.siliconflow.com/v1/embeddings",
                json={"model": "Qwen/Qwen3-Embedding-0.6B", "input": texts},
                headers={"Authorization": f"Bearer {api_key}"}
            )

            if response.status_code == 429:
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)  # 1, 2, 4, 8, 16 seconds
                    print(f"Rate limited (429), retrying in {delay}s (attempt {attempt + 1}/{max_retries})")
                    time.sleep(delay)
                    continue
                else:
                    response.raise_for_status()

            response.raise_for_status()
            result = response.json()
            return [item['embedding'] for item in result['data']]

        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                print(f"Request failed: {e}, retrying in {delay}s...")
                time.sleep(delay)
            else:
                raise
```

**Backoff pattern:** 1s → 2s → 4s → 8s → 16s

---

## Error Handling

### Circuit Breaker Pattern

Protect against cascading failures with a circuit breaker:

```python
import asyncio

class CircuitBreakerEmbedder:
    def __init__(self, api_key, max_consecutive_failures=10, breaker_delay=60):
        self.api_key = api_key
        self.consecutive_failures = 0
        self.max_consecutive_failures = max_consecutive_failures
        self.circuit_breaker_delay = breaker_delay

    async def embed_with_circuit_breaker(self, texts):
        """Embed with circuit breaker protection."""
        # Check circuit breaker
        if self.consecutive_failures >= self.max_consecutive_failures:
            print(f"⚠️  Circuit breaker triggered! {self.consecutive_failures} consecutive failures.")
            print(f"Pausing for {self.circuit_breaker_delay}s to allow network recovery...")
            await asyncio.sleep(self.circuit_breaker_delay)
            self.consecutive_failures = 0
            print("Circuit breaker reset, resuming...")

        try:
            # Attempt to embed
            embeddings = await self.embed_batch(texts)

            # Success - reset failure counter
            self.consecutive_failures = 0
            return embeddings

        except Exception as e:
            # Failure - increment counter
            self.consecutive_failures += 1
            print(f"Embedding failed (consecutive failures: {self.consecutive_failures}): {e}")
            raise
```

### Input Validation

```python
def validate_config(model_name, dimensions):
    """Validate model and dimension configuration."""
    VALID_CONFIGS = {
        "Qwen/Qwen3-Embedding-0.6B": [64, 128, 256, 512, 768, 1024],
        "Qwen/Qwen3-Embedding-4B": [64, 128, 256, 512, 768, 1024, 2048],
        "Qwen/Qwen3-Embedding-8B": [64, 128, 256, 512, 768, 1024, 2048, 4096],
    }

    if model_name not in VALID_CONFIGS:
        raise ValueError(
            f"Invalid model '{model_name}'. "
            f"Valid models: {list(VALID_CONFIGS.keys())}"
        )

    if dimensions not in VALID_CONFIGS[model_name]:
        raise ValueError(
            f"Invalid dimension {dimensions} for model {model_name}. "
            f"Valid dimensions: {VALID_CONFIGS[model_name]}"
        )
```

### Comprehensive Retry Logic

```python
from tenacity import retry, wait_random_exponential, stop_after_attempt, retry_if_exception_type

@retry(
    wait=wait_random_exponential(min=1, max=60),
    stop=stop_after_attempt(6),
    retry=retry_if_exception_type(requests.exceptions.RequestException),
    reraise=True
)
def robust_embed(texts, api_key):
    """Embed with automatic retry on transient failures."""
    response = requests.post(
        "https://api.siliconflow.com/v1/embeddings",
        json={
            "model": "Qwen/Qwen3-Embedding-0.6B",
            "input": texts,
            "dimensions": 1024,
            "encoding_format": "float"
        },
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=30
    )
    response.raise_for_status()
    result = response.json()
    return [item['embedding'] for item in result['data']]
```

---

## Production Patterns

### Checkpoint-Based Processing

For large datasets, implement checkpointing to enable crash recovery:

```python
import json
import os

def embed_with_checkpoints(texts, checkpoint_interval=10000, checkpoint_dir="checkpoints"):
    """
    Embed large dataset with checkpoint-based recovery.

    Args:
        texts: List of all texts to embed
        checkpoint_interval: Save progress every N texts
        checkpoint_dir: Directory to store checkpoint files
    """
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Load existing embeddings if resuming
    processed_ids = set()
    all_embeddings = {}

    if os.path.exists(f"{checkpoint_dir}/progress.json"):
        with open(f"{checkpoint_dir}/progress.json", 'r') as f:
            progress = json.load(f)
            processed_ids = set(progress.get("processed_ids", []))
            print(f"Resuming from checkpoint: {len(processed_ids)} texts already processed")

    # Process in chunks
    total_chunks = (len(texts) + checkpoint_interval - 1) // checkpoint_interval

    for chunk_idx in range(total_chunks):
        chunk_start = chunk_idx * checkpoint_interval
        chunk_end = min(chunk_start + checkpoint_interval, len(texts))
        chunk_texts = texts[chunk_start:chunk_end]

        print(f"Checkpoint {chunk_idx + 1}/{total_chunks}: Texts {chunk_start + 1}-{chunk_end}")

        # Filter out already-processed texts
        new_texts = [(i, t) for i, t in enumerate(chunk_texts, start=chunk_start) if i not in processed_ids]

        if not new_texts:
            print(f"  ✓ Checkpoint {chunk_idx + 1} already complete, skipping")
            continue

        try:
            # Embed new texts
            indices, texts_to_embed = zip(*new_texts)
            chunk_embeddings = embed_large_dataset(list(texts_to_embed))

            # Store results
            for idx, emb in zip(indices, chunk_embeddings):
                all_embeddings[idx] = emb
                processed_ids.add(idx)

            # Save checkpoint
            print(f"  💾 Saving checkpoint {chunk_idx + 1}/{total_chunks}...")
            with open(f"{checkpoint_dir}/embeddings_{chunk_idx}.npy", 'wb') as f:
                np.save(f, chunk_embeddings)

            with open(f"{checkpoint_dir}/progress.json", 'w') as f:
                json.dump({"processed_ids": list(processed_ids)}, f)

            print(f"  ✓ Checkpoint {chunk_idx + 1} saved")

        except Exception as e:
            print(f"  ❌ ERROR in checkpoint {chunk_idx + 1}: {e}")
            print(f"Progress saved. Resume by running the same command again.")
            raise

    # Return embeddings in original order
    return [all_embeddings[i] for i in range(len(texts))]
```

### Deduplication

Prevent re-processing of already-embedded texts:

```python
def deduplicate_texts(texts, text_ids, already_processed_ids):
    """
    Filter out already-processed texts.

    Args:
        texts: List of text strings
        text_ids: Unique identifier for each text
        already_processed_ids: Set of IDs already embedded

    Returns:
        Tuple of (new_texts, new_ids, skipped_count)
    """
    new_texts = []
    new_ids = []
    skipped = 0

    for text, text_id in zip(texts, text_ids):
        if text_id in already_processed_ids:
            skipped += 1
        else:
            new_texts.append(text)
            new_ids.append(text_id)

    if skipped > 0:
        print(f"Skipped {skipped} already-processed texts")

    return new_texts, new_ids, skipped
```

### Progress Monitoring

```python
import time
from datetime import datetime

class ProgressTracker:
    def __init__(self, total_items):
        self.total_items = total_items
        self.processed_items = 0
        self.start_time = time.time()

    def update(self, batch_size):
        """Update progress after processing a batch."""
        self.processed_items += batch_size
        elapsed = time.time() - self.start_time
        rate = self.processed_items / elapsed if elapsed > 0 else 0
        eta_seconds = (self.total_items - self.processed_items) / rate if rate > 0 else 0

        print(f"Progress: {self.processed_items}/{self.total_items} "
              f"({self.processed_items / self.total_items * 100:.1f}%) | "
              f"Rate: {rate:.1f} texts/sec | "
              f"ETA: {eta_seconds / 60:.1f} min")

    def summary(self):
        """Print final summary."""
        elapsed = time.time() - self.start_time
        rate = self.processed_items / elapsed if elapsed > 0 else 0

        print(f"\n✓ Complete!")
        print(f"  Total: {self.processed_items} texts")
        print(f"  Time: {elapsed / 60:.1f} minutes")
        print(f"  Avg rate: {rate:.1f} texts/sec")

# Usage
tracker = ProgressTracker(total_items=len(texts))
for batch in batches:
    embeddings = embed_texts(batch, api_key)
    tracker.update(len(batch))
tracker.summary()
```

---

## Reusable Code Templates

### Complete Production-Ready Embedder Class

```python
import os
import asyncio
import aiohttp
import numpy as np
from aiolimiter import AsyncLimiter
from dotenv import load_dotenv
from typing import List, Optional

class QwenEmbedder:
    """
    Production-ready QWEN-3-Embedding client with:
    - Rate limiting (TPM + RPM)
    - Exponential backoff
    - Circuit breaker
    - Batch processing
    - Multi-task instructions
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "Qwen/Qwen3-Embedding-0.6B",
        dimensions: int = 1024,
        batch_size: int = 32,
        max_consecutive_failures: int = 10,
        circuit_breaker_delay: int = 60
    ):
        load_dotenv()
        self.api_key = api_key or os.getenv("SILICONFLOW_API_KEY")
        if not self.api_key:
            raise ValueError("API key required (set SILICONFLOW_API_KEY env var)")

        self.model = model
        self.dimensions = dimensions
        self.batch_size = batch_size
        self.url = "https://api.siliconflow.com/v1/embeddings"

        # Validate configuration
        self._validate_config()

        # Rate limiters (Tier L0: 1M TPM, 2K RPM)
        self.tpm_limiter = AsyncLimiter(16666, 1)  # tokens/sec
        self.rpm_limiter = AsyncLimiter(33, 1)     # requests/sec

        # Circuit breaker
        self.consecutive_failures = 0
        self.max_consecutive_failures = max_consecutive_failures
        self.circuit_breaker_delay = circuit_breaker_delay

    def _validate_config(self):
        """Validate model and dimension configuration."""
        valid_configs = {
            "Qwen/Qwen3-Embedding-0.6B": [64, 128, 256, 512, 768, 1024],
            "Qwen/Qwen3-Embedding-4B": [64, 128, 256, 512, 768, 1024, 2048],
            "Qwen/Qwen3-Embedding-8B": [64, 128, 256, 512, 768, 1024, 2048, 4096],
        }

        if self.model not in valid_configs:
            raise ValueError(f"Invalid model: {self.model}")

        if self.dimensions not in valid_configs[self.model]:
            raise ValueError(
                f"Invalid dimension {self.dimensions} for {self.model}. "
                f"Valid: {valid_configs[self.model]}"
            )

    def format_with_instruction(self, text: str, instruction: Optional[str] = None) -> str:
        """Apply task instruction to text (Qwen3 format)."""
        if instruction:
            return f"Instruct: {instruction}\nQuery:{text}"
        return text

    async def _embed_batch_async(
        self,
        session: aiohttp.ClientSession,
        texts: List[str],
        batch_num: int
    ) -> tuple:
        """Embed a single batch with rate limiting and circuit breaker."""
        # Circuit breaker check
        if self.consecutive_failures >= self.max_consecutive_failures:
            print(f"⚠️  Circuit breaker triggered! Pausing for {self.circuit_breaker_delay}s...")
            await asyncio.sleep(self.circuit_breaker_delay)
            self.consecutive_failures = 0

        # Estimate tokens for rate limiting
        estimated_tokens = sum(len(t) // 4 for t in texts)

        # Retry logic with exponential backoff
        max_retries = 5
        base_delay = 1

        for attempt in range(max_retries):
            try:
                # Acquire rate limits
                await self.rpm_limiter.acquire(1)
                await self.tpm_limiter.acquire(estimated_tokens)

                payload = {
                    "model": self.model,
                    "input": texts,
                    "dimensions": self.dimensions,
                    "encoding_format": "float"
                }

                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }

                async with session.post(self.url, json=payload, headers=headers) as response:
                    if response.status == 429:
                        if attempt < max_retries - 1:
                            delay = base_delay * (2 ** attempt)
                            print(f"Rate limited (429), retrying in {delay}s (attempt {attempt + 1}/{max_retries})")
                            await asyncio.sleep(delay)
                            continue

                    response.raise_for_status()
                    result = await response.json()
                    embeddings = [item['embedding'] for item in result['data']]

                    # Success - reset failure counter
                    self.consecutive_failures = 0
                    return (batch_num, embeddings)

            except Exception as e:
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    await asyncio.sleep(delay)
                else:
                    # Max retries exceeded
                    self.consecutive_failures += 1
                    raise

    async def embed_async(
        self,
        texts: List[str],
        instruction: Optional[str] = None,
        concurrency: int = 10
    ) -> np.ndarray:
        """
        Embed texts asynchronously with batching.

        Args:
            texts: List of texts to embed
            instruction: Optional task instruction
            concurrency: Max concurrent requests

        Returns:
            numpy array of shape (len(texts), dimensions)
        """
        # Apply instruction if provided
        if instruction:
            texts = [self.format_with_instruction(t, instruction) for t in texts]

        # Split into batches
        batches = []
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            batches.append((i // self.batch_size, batch_texts))

        all_embeddings = [None] * len(batches)

        async with aiohttp.ClientSession() as session:
            # Process in chunks to limit concurrency
            for chunk_start in range(0, len(batches), concurrency):
                chunk = batches[chunk_start:chunk_start + concurrency]

                tasks = [
                    self._embed_batch_async(session, batch_texts, batch_num)
                    for batch_num, batch_texts in chunk
                ]

                chunk_results = await asyncio.gather(*tasks)

                # Store results in order
                for batch_num, embeddings in chunk_results:
                    all_embeddings[batch_num] = embeddings

        # Flatten and convert to numpy array
        flat_embeddings = [emb for batch in all_embeddings for emb in batch]
        return np.array(flat_embeddings)

    def embed(
        self,
        texts: List[str],
        instruction: Optional[str] = None,
        concurrency: int = 10
    ) -> np.ndarray:
        """Synchronous wrapper for embed_async."""
        return asyncio.run(self.embed_async(texts, instruction, concurrency))


# Usage Example
if __name__ == "__main__":
    # Initialize embedder
    embedder = QwenEmbedder(
        model="Qwen/Qwen3-Embedding-0.6B",
        dimensions=1024,
        batch_size=32
    )

    # Simple embedding
    texts = ["Hello world", "Machine learning", "Python programming"]
    embeddings = embedder.embed(texts)
    print(f"Generated {embeddings.shape[0]} embeddings of dimension {embeddings.shape[1]}")

    # With task instruction
    instruction = "Retrieve technical documentation related to this topic"
    tech_embeddings = embedder.embed(texts, instruction=instruction)
    print(f"Generated task-specific embeddings: {tech_embeddings.shape}")
```

---

## Reference

### Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model` | `Qwen/Qwen3-Embedding-0.6B` | Model variant to use |
| `dimensions` | 1024 | Output embedding dimension |
| `batch_size` | 32 | Texts per API request |
| `concurrency` | 10 | Max concurrent async requests |
| `timeout` | 30 | Request timeout (seconds) |
| `max_retries` | 5 | Max retry attempts on failure |
| `circuit_breaker_failures` | 10 | Consecutive failures before pause |
| `circuit_breaker_delay` | 60 | Circuit breaker pause duration (sec) |

### Performance Characteristics

Based on production testing with 20,000+ texts:

| Metric | Value |
|--------|-------|
| **Throughput** | 50-60 texts/sec (async, concurrency=10) |
| **Latency** | ~0.2s per batch (32 texts) |
| **API Call Overhead** | ~50-100ms per request |
| **Optimal Batch Size** | 32 texts |
| **Optimal Concurrency** | 10 concurrent requests |

### Troubleshooting

**Problem:** Rate limiting errors (429)
- **Solution:** Reduce `concurrency` or increase delays between batches
- **Check:** Verify your SiliconFlow tier limits

**Problem:** Circuit breaker frequently triggering
- **Solution:** Check network stability, reduce concurrency, increase retry delays
- **Check:** Verify API key is valid and has sufficient quota

**Problem:** Slow embedding speed
- **Solution:** Increase `concurrency` (if not hitting rate limits)
- **Try:** Larger `batch_size` (up to 32-64 depending on text length)

**Problem:** Out of memory
- **Solution:** Process in smaller chunks, reduce batch size
- **Consider:** Checkpoint-based processing for very large datasets

**Problem:** Dimension validation errors
- **Solution:** Verify dimension is valid for your chosen model (see table above)

### API Response Format

Successful response structure:
```json
{
  "object": "list",
  "data": [
    {
      "object": "embedding",
      "embedding": [0.123, -0.456, ...],  // Array of floats
      "index": 0
    },
    ...
  ],
  "model": "Qwen/Qwen3-Embedding-0.6B",
  "usage": {
    "prompt_tokens": 150,
    "total_tokens": 150
  }
}
```

### Common HTTP Status Codes

| Code | Meaning | Action |
|------|---------|--------|
| 200 | Success | Continue |
| 400 | Bad request | Check payload format, model name, dimensions |
| 401 | Unauthorized | Verify API key |
| 429 | Rate limited | Exponential backoff, reduce rate |
| 500 | Server error | Retry with backoff |
| 503 | Service unavailable | Wait and retry |

### Dependencies

```bash
pip install requests aiohttp aiolimiter numpy python-dotenv tenacity
```

Or with `requirements.txt`:
```txt
requests>=2.31.0
aiohttp>=3.9.0
aiolimiter>=1.1.0
numpy>=1.24.0
python-dotenv>=1.0.0
tenacity>=8.2.0
```

---

## Additional Resources

- **SiliconFlow Documentation:** https://docs.siliconflow.com
- **QWEN Model Card:** Check Alibaba/Qwen documentation for model details
- **Rate Limits:** Verify current tier limits in your SiliconFlow dashboard

---

**Generated from production patterns in a high-scale vector embedding system (20K+ documents validated).**
