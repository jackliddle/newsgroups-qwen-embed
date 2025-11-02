"""
Quick test script to verify the setup works correctly.
"""

import asyncio
from qwen_embedder import QwenEmbedder, EmbeddingConfig, QwenModel

async def test_embedder():
    """Test the embedder with a few sample texts."""
    sample_texts = [
        "Machine learning is a branch of artificial intelligence.",
        "Python is a popular programming language.",
        "The quick brown fox jumps over the lazy dog."
    ]

    config = EmbeddingConfig(
        model=QwenModel.SMALL,
        batch_size=32,
        max_concurrent=10
    )

    print("Testing QWEN-3 Embedder...")
    print(f"Sample texts: {len(sample_texts)}")

    async with QwenEmbedder(config=config) as embedder:
        embeddings = await embedder.embed_async(sample_texts, show_progress=True)

    print(f"\nSuccess! Generated {len(embeddings)} embeddings")
    print(f"Embedding dimension: {len(embeddings[0])}")
    print("\nSetup is working correctly!")

    return embeddings

if __name__ == "__main__":
    asyncio.run(test_embedder())
