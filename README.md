# QWEN-3 Multi-Task Embeddings Demo

**Interactive web demo** exploring how task-specific instructions reshape QWEN-3 embeddings using the 20 Newsgroups dataset.

👉 **[View Live Demo](https://yourusername.github.io/newsgroups-qwen-embed/)**

## What It Does

This demo shows how the **same documents** embedded with **different task instructions** create fundamentally different embedding spaces:

- **Default**: General-purpose embeddings (no instruction)
- **Sentiment**: Optimized for sentiment classification

By comparing just two tasks, users can clearly see how task-specific instructions reshape the embedding space.

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

# Copy .env.example and add your API key
cp .env.example .env
# Edit .env and add your SiliconFlow API key

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
│   ├── data.json                   # Pre-generated embeddings (~600 KB)
│   └── assets/
│       ├── css/style.css           # Styling
│       └── js/app.js               # Plotly visualization logic
├── scripts/
│   └── generate_data.py            # Data generation script
├── qwen_embedder.py                # QWEN-3 embedding client
├── newsgroups_embeddings_demo.ipynb # Hands-on tutorial notebook
├── .env.example                    # API key template
├── pyproject.toml                  # Poetry dependencies
└── README.md                       # This file
```

## How It Works

### 1. Data Generation (`scripts/generate_data.py`)

The generation script:
- Loads 800 documents from 10 newsgroup categories
- Embeds each document 2 times with different task instructions (default + sentiment)
- Applies UMAP dimensionality reduction (1024D → 2D)
- Saves coordinates and text previews to `docs/data.json`

**Run time**: ~5 minutes for 800 documents × 2 tasks
**Cost**: ~$0.80 in API credits (640K tokens)

### 2. Web App (`docs/index.html`)

The web app:
- Loads pre-generated data from `data.json`
- Renders interactive Plotly scatter plots
- Switches between tasks on button click
- Shows task-specific explanations

**Performance**: Instant switching (no recomputation needed)

### 3. Tutorial Notebook (`newsgroups_embeddings_demo.ipynb`)

The notebook:
- **Hands-on tutorial** - Actually runs the embedding code (not just loads data)
- Explains the concept of task-specific embeddings
- Embeds 800 documents with 2 tasks (default + sentiment)
- Applies UMAP and creates comparison visualizations
- Provides educational context and exploration ideas

**Note**: Requires SiliconFlow API key in `.env` file

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
- **Dataset**: 20 Newsgroups (10 categories, 80 docs each = 800 total)
- **Tasks**: 2 (Default + Sentiment)
- **Reduction**: UMAP (n_neighbors=15, min_dist=0.1)
- **Cost**: ~$0.80 USD (640K tokens = 800 docs × 2 tasks × ~400 tokens/doc)
- **Performance**: Async batching with rate limiting (33 req/sec)

## Key Insights

### What to Look For

1. **Default Task**: General-purpose organization based on overall semantic similarity
2. **Sentiment Task**: Reorganizes by emotional tone rather than subject matter
3. **Comparison**: The same documents occupy completely different positions in the two spaces!

### Questions to Explore

- Which categories cluster together in each space?
- Do political/religious groups show different sentiment patterns than technical ones?
- Does sentiment correlate with topic, or are they independent?
- When would you choose task-specific vs default embeddings?
- Try modifying the notebook to compare other tasks (topic, toxicity, emotion, etc.)

## Extending the Demo

### Try Different Tasks

Edit the notebook or `scripts/generate_data.py` and modify the task instructions:

```python
# Topic classification
"Identify the topic or theme of the given text"

# Toxicity detection
"Classify the given text as either toxic or not toxic"

# Emotion recognition
"Classify the emotion expressed in the given text into one of the six emotions: anger, fear, joy, love, sadness, and surprise"

# Query-document matching
"Given a web search query, retrieve relevant passages that answer the query"
```

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

- [QWEN-3 Embedding arXiv Paper](https://arxiv.org/abs/2506.05176)
- [Official QWEN Repository](https://github.com/QwenLM/Qwen3-Embedding)
- [SiliconFlow API](https://siliconflow.com)
- [20 Newsgroups Dataset](https://scikit-learn.org/stable/datasets/real_world.html#newsgroups-dataset)
- [UMAP Documentation](https://umap-learn.readthedocs.io/)
