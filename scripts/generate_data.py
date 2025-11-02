"""
Generate embeddings data for GitHub Pages visualization.

This script:
1. Loads 20 newsgroups dataset (10 categories, ~800 samples)
2. Embeds texts with 2 task instructions (default, sentiment)
3. Applies UMAP dimensionality reduction to each task's embeddings
4. Outputs JSON file with coordinates for web visualization
"""

import asyncio
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_20newsgroups
import umap

# Add parent directory to path to import qwen_embedder
sys.path.insert(0, str(Path(__file__).parent.parent))
from qwen_embedder import QwenEmbedder, EmbeddingConfig, QwenModel


# Configuration
CATEGORIES = [
    'alt.atheism',
    'comp.graphics',
    'sci.med',
    'talk.religion.misc',
    'rec.sport.baseball',
    'sci.space',
    'talk.politics.guns',
    'talk.politics.mideast',
    'rec.autos',
    'sci.crypt'
]

TASKS = {
    'default': None,
    'sentiment': "Classify the sentiment of the given text as positive, negative, or neutral"
}

N_SAMPLES = 80  # Per category
MIN_TEXT_LENGTH = 100
MAX_TEXT_LENGTH = 5000
PREVIEW_LENGTH = 200


def load_and_preprocess_data():
    """Load newsgroups data and preprocess."""
    print(f"Loading 20 Newsgroups dataset ({len(CATEGORIES)} categories)...")

    newsgroups = fetch_20newsgroups(
        subset='all',
        categories=CATEGORIES,
        remove=('headers', 'footers', 'quotes'),
        random_state=42
    )

    # Create DataFrame
    df = pd.DataFrame({
        'text': newsgroups.data,
        'category': [newsgroups.target_names[i] for i in newsgroups.target]
    })

    # Clean text
    df['text_clean'] = df['text'].str.replace(r'\s+', ' ', regex=True).str.strip()
    df = df[df['text_clean'].str.len() >= MIN_TEXT_LENGTH].copy()
    df['text_clean'] = df['text_clean'].str[:MAX_TEXT_LENGTH]

    # Stratified sampling
    df_sampled = df.groupby('category', group_keys=False).apply(
        lambda x: x.sample(n=min(N_SAMPLES, len(x)), random_state=42)
    ).reset_index(drop=True)

    # Create text preview for hover tooltips
    df_sampled['text_preview'] = df_sampled['text_clean'].str[:PREVIEW_LENGTH] + '...'

    print(f"Preprocessed {len(df_sampled)} documents")
    print(f"Categories: {df_sampled['category'].value_counts().to_dict()}")

    return df_sampled


async def generate_embeddings(df):
    """Generate embeddings for all tasks."""
    print("\nGenerating embeddings for 2 tasks...")

    config = EmbeddingConfig(
        model=QwenModel.SMALL,
        batch_size=32,
        max_concurrent=10
    )

    all_embeddings = {}

    async with QwenEmbedder(config=config) as embedder:
        for task_name, task_instruction in TASKS.items():
            print(f"\n{'='*60}")
            print(f"Task: {task_name.upper()}")
            if task_instruction:
                print(f"Instruction: {task_instruction}")
            print(f"{'='*60}")

            embeddings = await embedder.embed_async(
                df['text_clean'].tolist(),
                task_instruction=task_instruction,
                show_progress=True
            )

            all_embeddings[task_name] = np.array(embeddings)
            print(f"✓ Generated {len(embeddings)} embeddings (dim={len(embeddings[0])})")

    return all_embeddings


def apply_umap(all_embeddings):
    """Apply UMAP to each task's embeddings."""
    print("\nApplying UMAP dimensionality reduction...")

    umap_coords = {}

    for task_name, embeddings in all_embeddings.items():
        print(f"  Processing {task_name}...")

        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=15,
            min_dist=0.1,
            random_state=42,
            verbose=False
        )

        coords = reducer.fit_transform(embeddings)
        umap_coords[task_name] = coords

        print(f"  ✓ {task_name}: {coords.shape}")

    return umap_coords


def generate_json_output(df, umap_coords, output_path):
    """Generate JSON file for web visualization."""
    print(f"\nGenerating JSON output...")

    # Prepare data
    data = []
    for idx, row in df.iterrows():
        doc = {
            'text': row['text_clean'],
            'text_preview': row['text_preview'],
            'category': row['category']
        }

        # Add coordinates for each task
        for task_name in TASKS.keys():
            doc[f'{task_name}_x'] = float(umap_coords[task_name][idx, 0])
            doc[f'{task_name}_y'] = float(umap_coords[task_name][idx, 1])

        data.append(doc)

    # Save to file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"✓ Saved {len(data)} documents to {output_path}")
    print(f"  File size: {file_size_mb:.2f} MB")


async def main():
    """Main execution flow."""
    print("="*60)
    print("QWEN-3 Embeddings Data Generation")
    print("="*60)

    # Load data
    df = load_and_preprocess_data()

    # Generate embeddings
    all_embeddings = await generate_embeddings(df)

    # Apply UMAP
    umap_coords = apply_umap(all_embeddings)

    # Generate JSON output
    output_path = Path(__file__).parent.parent / 'docs' / 'data.json'
    generate_json_output(df, umap_coords, output_path)

    print("\n" + "="*60)
    print("✓ Data generation complete!")
    print("="*60)
    print(f"\nNext steps:")
    print(f"1. Open docs/index.html in your browser to test")
    print(f"2. Commit changes and push to GitHub")
    print(f"3. Enable GitHub Pages (Settings → Pages → Source: main branch, /docs folder)")


if __name__ == "__main__":
    asyncio.run(main())
