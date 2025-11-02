# QWEN-3 Multi-Task Embeddings Exploration: Newsgroups Dataset

Interactive demo exploring how different task instructions affect QWEN-3 embeddings using the 20 Newsgroups dataset.

## Features

- **Multi-task comparison**: Embed texts with different task instructions (Default, Sentiment, Topic, Toxicity)
- Embed texts using QWEN-3-Embedding-0.6B via SiliconFlow API
- Dimensionality reduction with UMAP
- Interactive 4-panel Plotly visualizations comparing task-specific embeddings
- Text tooltips on hover to inspect individual documents
- Efficient async batching with rate limiting
- Embedding caching to avoid redundant API calls

## Setup

### 1. Install Dependencies with Poetry

```bash
poetry install
```

### 2. Activate Virtual Environment

```bash
poetry shell
```

### 3. Environment Configuration

The `.env` file is already configured with your SiliconFlow API key:
```
SILICONFLOW_API_KEY=sk-tvhriqmcldjcnbszonnphkrbdkcloupvqryqbjmdihcqhrva
```

## Usage

### Run the Jupyter Notebook

```bash
jupyter notebook newsgroups_embeddings_demo.ipynb
```

Or use JupyterLab:

```bash
jupyter lab newsgroups_embeddings_demo.ipynb
```

### Notebook Workflow

The notebook will:

1. **Load Data**: Fetch 20 Newsgroups dataset (10 categories, ~800 samples)
2. **Multi-Task Embedding**: Create 4 embedding sets using QWEN-3-Embedding-0.6B:
   - Default (no instruction)
   - Sentiment task: "Classify the sentiment of the given text as positive, negative, or neutral"
   - Topic task: "Identify the topic or theme of the given text"
   - Toxicity task: "Classify the given text as either toxic or not toxic"
3. **Reduce Dimensions**: Apply UMAP to each embedding set (2D projection)
4. **Visualize**: Create 4-panel comparison + individual plots with hover tooltips
5. **Save**: Export visualizations as HTML files

### Expected Output

- **embeddings/**: Cached multi-task embeddings (to avoid re-running API calls)
- **visualizations/**: Interactive HTML plots
  - `multitask_comparison.html` - 4-panel side-by-side comparison
  - `default_embeddings.html` - Default task only
  - `sentiment_embeddings.html` - Sentiment task only
  - `topic_embeddings.html` - Topic task only
  - `toxicity_embeddings.html` - Toxicity task only

## Project Structure

```
newsgroups-qwen-embed/
├── pyproject.toml              # Poetry dependencies
├── .env                        # API key configuration
├── .gitignore                  # Exclude sensitive/generated files
├── qwen_embedder.py            # QWEN-3 embedding client
├── newsgroups_embeddings_demo.ipynb  # Main demo notebook
├── QWEN-EMBED.md              # API documentation
└── README.md                   # This file
```

## Configuration

### Embedding Settings

In `qwen_embedder.py`:

```python
config = EmbeddingConfig(
    model=QwenModel.SMALL,      # 0.6B model
    batch_size=32,               # Texts per API request
    max_concurrent=10            # Concurrent requests
)
```

### Dimensionality Reduction

In the notebook:

```python
# UMAP (applied to each task's embeddings)
umap_reducer = umap.UMAP(
    n_components=2,
    n_neighbors=15,
    min_dist=0.1,
    random_state=42
)
```

## API Costs

Using QWEN-3-Embedding-0.6B (check SiliconFlow pricing):
- ~800 documents × 4 tasks × ~200 tokens = ~640K tokens
- Embeddings are cached after first run
- Multi-task embeddings allow comparing how instructions reshape the embedding space

## Troubleshooting

### Import Errors
```bash
poetry install
poetry shell
```

### API Rate Limits
The embedder includes automatic rate limiting (33 req/sec) and retry logic with exponential backoff.

### Memory Issues
Reduce `n_samples` in the notebook to embed fewer documents.

## Next Steps

### Analyze the Results

After running the notebook, compare the four visualizations:

1. **Which task creates the best newsgroup category separation?**
   - Topic task should perform best (aligned with classification task)
   - Sentiment may group by emotional tone instead
   - Toxicity should reveal discourse civility patterns across categories
   - Default provides balanced general-purpose representation

2. **How do task instructions reshape the embedding space?**
   - Observe how clusters form differently across tasks
   - Check if some categories separate better with specific instructions
   - Compare political/religious groups vs technical groups in toxicity task

3. **Toxicity patterns across categories:**
   - Do political newsgroups (talk.politics.guns, talk.politics.mideast) show different toxicity patterns?
   - Are technical groups (sci.crypt, comp.graphics) less toxic than controversial ones?
   - Does toxicity correlate with sentiment, or are they orthogonal?

### Try Additional Tasks

Modify the `tasks` dictionary in the notebook to explore other instructions:

```python
tasks = {
    'default': None,
    'topic': "Identify the topic or theme of the given text",
    'emotion': "Classify the emotion expressed in the given text into one of the six emotions: anger, fear, joy, love, sadness, and surprise",
    'similarity': "Retrieve semantically similar text"
}
```

See [task_prompts.json](./task_prompts.json) for more QWEN-tested instructions.

## References

- [QWEN-3 Embeddings Documentation](./QWEN-EMBED.md)
- [SiliconFlow API](https://siliconflow.com)
- [20 Newsgroups Dataset](https://scikit-learn.org/stable/datasets/real_world.html#newsgroups-dataset)
