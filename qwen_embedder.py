"""
QWEN-3 Embeddings API Client
Production-ready embedder with async batching, rate limiting, and error handling.
"""

import os
import asyncio
import time
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum

import aiohttp
from aiolimiter import AsyncLimiter
from tenacity import retry, stop_after_attempt, wait_exponential
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class QwenModel(Enum):
    """Available QWEN-3 embedding models."""
    SMALL = "Qwen/Qwen3-Embedding-0.6B"  # Max 1024 dims
    MEDIUM = "Qwen/Qwen3-Embedding-4B"   # Max 2048 dims
    LARGE = "Qwen/Qwen3-Embedding-8B"    # Max 4096 dims


@dataclass
class EmbeddingConfig:
    """Configuration for embedding requests."""
    model: QwenModel = QwenModel.SMALL
    dimensions: Optional[int] = None
    encoding_format: str = "float"
    batch_size: int = 32
    max_concurrent: int = 10


class QwenEmbedder:
    """
    Production-ready QWEN-3 embeddings client with:
    - Async batch processing
    - Rate limiting
    - Automatic retries with exponential backoff
    - Progress tracking
    """

    BASE_URL = "https://api.siliconflow.com/v1/embeddings"

    def __init__(
        self,
        api_key: Optional[str] = None,
        config: Optional[EmbeddingConfig] = None
    ):
        """
        Initialize the embedder.

        Args:
            api_key: SiliconFlow API key (defaults to SILICONFLOW_API_KEY env var)
            config: Embedding configuration (defaults to QwenModel.SMALL)
        """
        self.api_key = api_key or os.getenv("SILICONFLOW_API_KEY")
        if not self.api_key:
            raise ValueError(
                "API key required. Set SILICONFLOW_API_KEY env var or pass api_key parameter"
            )

        self.config = config or EmbeddingConfig()

        # Rate limiter: 2000 requests/min = ~33 req/sec
        # Conservative: 10 concurrent requests
        self.rate_limiter = AsyncLimiter(33, 1)

        self._session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        """Async context manager entry."""
        self._session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self._session:
            await self._session.close()

    @retry(
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1, min=1, max=16),
        reraise=True
    )
    async def _embed_batch(
        self,
        texts: List[str],
        task_instruction: Optional[str] = None
    ) -> List[List[float]]:
        """
        Embed a single batch of texts with retry logic.

        Args:
            texts: List of texts to embed (max batch_size)
            task_instruction: Optional task instruction for specialized embeddings

        Returns:
            List of embedding vectors
        """
        if not self._session:
            raise RuntimeError("Session not initialized. Use async context manager.")

        # Format texts with task instruction if provided
        if task_instruction:
            formatted_texts = [
                f"Instruct: {task_instruction}\nQuery: {text}"
                for text in texts
            ]
        else:
            formatted_texts = texts

        payload = {
            "model": self.config.model.value,
            "input": formatted_texts,
            "encoding_format": self.config.encoding_format
        }

        if self.config.dimensions:
            payload["dimensions"] = self.config.dimensions

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        async with self.rate_limiter:
            async with self._session.post(
                self.BASE_URL,
                json=payload,
                headers=headers
            ) as response:
                response.raise_for_status()
                result = await response.json()

                # Extract embeddings in order
                embeddings = [
                    item["embedding"]
                    for item in sorted(result["data"], key=lambda x: x["index"])
                ]

                return embeddings

    async def embed_async(
        self,
        texts: List[str],
        task_instruction: Optional[str] = None,
        show_progress: bool = True
    ) -> List[List[float]]:
        """
        Embed a list of texts asynchronously with batching.

        Args:
            texts: List of texts to embed
            task_instruction: Optional task instruction for specialized embeddings
            show_progress: Whether to print progress updates

        Returns:
            List of embedding vectors (same order as input texts)
        """
        if not texts:
            return []

        # Split into batches
        batches = [
            texts[i:i + self.config.batch_size]
            for i in range(0, len(texts), self.config.batch_size)
        ]

        if show_progress:
            print(f"Embedding {len(texts)} texts in {len(batches)} batches...")

        start_time = time.time()
        all_embeddings = []

        # Process batches with concurrency control
        for i in range(0, len(batches), self.config.max_concurrent):
            batch_group = batches[i:i + self.config.max_concurrent]

            # Process this group of batches concurrently
            tasks = [
                self._embed_batch(batch, task_instruction)
                for batch in batch_group
            ]

            group_embeddings = await asyncio.gather(*tasks)

            # Flatten results
            for embeddings in group_embeddings:
                all_embeddings.extend(embeddings)

            if show_progress:
                progress = min(len(all_embeddings), len(texts))
                elapsed = time.time() - start_time
                rate = progress / elapsed if elapsed > 0 else 0
                print(f"  Progress: {progress}/{len(texts)} ({rate:.1f} texts/sec)")

        if show_progress:
            elapsed = time.time() - start_time
            rate = len(texts) / elapsed if elapsed > 0 else 0
            print(f"Completed in {elapsed:.1f}s ({rate:.1f} texts/sec)")

        return all_embeddings

    def embed(
        self,
        texts: List[str],
        task_instruction: Optional[str] = None,
        show_progress: bool = True
    ) -> List[List[float]]:
        """
        Synchronous wrapper for embed_async.

        Args:
            texts: List of texts to embed
            task_instruction: Optional task instruction for specialized embeddings
            show_progress: Whether to print progress updates

        Returns:
            List of embedding vectors (same order as input texts)
        """
        async def _run():
            async with self:
                return await self.embed_async(texts, task_instruction, show_progress)

        return asyncio.run(_run())


# Convenience function for quick embedding
async def embed_texts(
    texts: List[str],
    model: QwenModel = QwenModel.SMALL,
    task_instruction: Optional[str] = None,
    api_key: Optional[str] = None
) -> List[List[float]]:
    """
    Quick function to embed texts.

    Args:
        texts: List of texts to embed
        model: QWEN model to use
        task_instruction: Optional task instruction
        api_key: API key (defaults to env var)

    Returns:
        List of embedding vectors
    """
    config = EmbeddingConfig(model=model)
    async with QwenEmbedder(api_key=api_key, config=config) as embedder:
        return await embedder.embed_async(texts, task_instruction)
