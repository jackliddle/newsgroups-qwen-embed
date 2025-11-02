# QWEN-3 Multi-Task Embeddings Demo

**Interactive web demo** exploring how task-specific instructions reshape QWEN-3 embeddings using the 20 Newsgroups dataset.

👉 **[View Live Demo](https://yourusername.github.io/newsgroups-qwen-embed/)**

## What It Does

This demo shows how the **same documents** embedded with **different task instructions** create fundamentally different embedding spaces:

- **Default**: General-purpose embeddings (no instruction)
- **Sentiment**: Optimized for sentiment classification
- **Topic**: Optimized for topic identification
- **Toxicity**: Optimized for toxicity detection

## Features

- 🎯 **Interactive web app** - Click buttons to switch between tasks
- 📊 **Real-time visualization** - See how embeddings cluster differently per task
- 💬 **Hover tooltips** - Read document previews on mouseover
- 📱 **Single-page design** - No scrolling, clean interface
- 🚀 **Fast & efficient** - All data pre-generated, instant switching

## Quick Start

### View the Demo

Simply open `docs/index.html` in your browser, or deploy to GitHub Pages:

1. Push to GitHub
2. Go to Settings → Pages
3. Select Source: **Deploy from branch**
4. Branch: **main**, Folder: **/docs**
5. Save and wait for deployment

### Development Setup

```bash
# Install dependencies
poetry install

# Create .env file with your API key
echo "SILICONFLOW_API_KEY=your_key_here" > .env

# Generate embeddings data (optional - already included)
poetry run python scripts/generate_data.py

# Explore the tutorial notebook
poetry run jupyter notebook newsgroups_embeddings_demo.ipynb
```

## Project Structure

```
newsgroups-qwen-embed/
├── docs/                           # GitHub Pages web app
│   ├── index.html                  # Single-page app
│   ├── data.json                   # Pre-generated embeddings (1.18 MB)
│   └── assets/
│       ├── css/style.css           # Styling
│       └── js/app.js               # Plotly visualization logic
├── scripts/
│   └── generate_data.py            # Data generation script
├── qwen_embedder.py                # QWEN-3 embedding client
├── newsgroups_embeddings_demo.ipynb # Tutorial notebook
├── task_prompts.json               # Example task instructions
├── QWEN-EMBED.md                   # API documentation
├── pyproject.toml                  # Poetry dependencies
└── README.md                       # This file
```

## How It Works

### 1. Data Generation (`scripts/generate_data.py`)

The generation script:
- Loads 800 documents from 10 newsgroup categories
- Embeds each document 4 times with different task instructions
- Applies UMAP dimensionality reduction (1024D → 2D)
- Saves coordinates and text previews to `docs/data.json`

**Run time**: ~30 seconds for 800 documents × 4 tasks

### 2. Web App (`docs/index.html`)

The web app:
- Loads pre-generated data from `data.json`
- Renders interactive Plotly scatter plots
- Switches between tasks on button click
- Shows task-specific explanations

**Performance**: Instant switching (no recomputation needed)

### 3. Tutorial Notebook (`newsgroups_embeddings_demo.ipynb`)

The notebook:
- Explains the concept of task-specific embeddings
- Loads pre-generated data from `docs/data.json`
- Shows how to visualize and analyze the results
- Provides educational context and exploration ideas

## Configuration

### Regenerating Data

Edit `scripts/generate_data.py` to customize:

```python
# Modify tasks
TASKS = {
    'default': None,
    'your_task': "Your custom instruction here"
}

# Adjust sample size
N_SAMPLES = 80  # Per category

# Change UMAP parameters
umap.UMAP(
    n_components=2,
    n_neighbors=15,  # Adjust for tighter/looser clusters
    min_dist=0.1,    # Minimum distance between points
    random_state=42
)
```

### API Configuration

In `qwen_embedder.py`:

```python
config = EmbeddingConfig(
    model=QwenModel.SMALL,      # 0.6B or MEDIUM (1.5B)
    batch_size=32,               # Texts per API request
    max_concurrent=10            # Concurrent requests
)
```

## Technical Details

- **Model**: QWEN-3-Embedding-0.6B (1024 dimensions)
- **API**: SiliconFlow
- **Dataset**: 20 Newsgroups (10 categories, 80 docs each)
- **Reduction**: UMAP (n_neighbors=15, min_dist=0.1)
- **Cost**: ~640K tokens for full generation
- **Performance**: Async batching with rate limiting (33 req/sec)

## Key Insights

### What to Look For

1. **Topic Task**: Best separation of newsgroup categories (task aligns with classification goal)
2. **Sentiment Task**: Groups by emotional tone rather than subject matter
3. **Toxicity Task**: Reveals discourse patterns (political vs technical groups)
4. **Default**: Balanced, general-purpose organization

### Questions to Explore

- Which categories cluster together vs separate?
- How do political/religious groups differ from technical ones in toxicity?
- Does sentiment correlate with topic, or are they orthogonal?
- When would you choose task-specific vs default embeddings?

## Extending the Demo

### Try Different Tasks

Edit `scripts/generate_data.py` and modify the `TASKS` dictionary:

```python
TASKS = {
    'default': None,
    'emotion': "Classify the emotion expressed in the given text into one of the six emotions: anger, fear, joy, love, sadness, and surprise",
    'query': "Retrieve semantically similar text",
    'summary': "Generate a concise summary of the given text"
}
```

See [task_prompts.json](./task_prompts.json) for more QWEN-tested instructions.

### Add More Categories

Expand the newsgroup categories in `scripts/generate_data.py`:

```python
CATEGORIES = [
    'alt.atheism',
    'comp.graphics',
    'comp.sys.mac.hardware',  # Add more categories
    'comp.windows.x',
    # ... up to 20 categories available
]
```

## References

- [QWEN-3 Embeddings Documentation](./QWEN-EMBED.md)
- [SiliconFlow API](https://siliconflow.com)
- [20 Newsgroups Dataset](https://scikit-learn.org/stable/datasets/real_world.html#newsgroups-dataset)
