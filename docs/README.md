# QWEN-3 Multi-Task Embeddings Web Demo

This directory contains the **GitHub Pages web app** for the QWEN-3 multi-task embeddings demo.

## Files

- **index.html** - Single-page application (main entry point)
- **data.json** - Pre-generated embeddings data (1.18 MB, 800 documents × 4 tasks)
- **assets/css/style.css** - Styling and layout
- **assets/js/app.js** - Visualization logic (Plotly.js integration)

## How It Works

### Data Structure (`data.json`)

Each document in the dataset contains:

```json
{
  "text": "Full document text",
  "text_preview": "First 200 characters...",
  "category": "sci.space",
  "default_x": -1.234,
  "default_y": 5.678,
  "sentiment_x": 2.345,
  "sentiment_y": -3.456,
  "topic_x": 0.123,
  "topic_y": 4.567,
  "toxicity_x": -2.345,
  "toxicity_y": 1.234
}
```

The `*_x` and `*_y` coordinates are UMAP-reduced embeddings (1024D → 2D) for each task.

### Web App Features

1. **Task Switcher**: Four buttons (Default, Sentiment, Topic, Toxicity)
   - Active button: Blue background
   - Inactive buttons: Green background
   - Click to switch between embedding spaces

2. **Interactive Plot**: Plotly.js scatter plot
   - Each point = one document
   - Color = newsgroup category (10 colors)
   - Hover = text preview tooltip

3. **Details Panel**: Right sidebar
   - Explains current task instruction
   - Shows what the visualization reveals
   - Updates when task changes

4. **Info Overlay**: "What is this?" button
   - Comprehensive project explanation
   - Technical details
   - Usage instructions

## Local Testing

Simply open `index.html` in a web browser:

```bash
# From project root
open docs/index.html

# Or use Python's built-in server
cd docs
python -m http.server 8000
# Visit http://localhost:8000
```

## Deployment

### GitHub Pages

1. Push to GitHub
2. Go to repository Settings → Pages
3. Select:
   - **Source**: Deploy from a branch
   - **Branch**: main
   - **Folder**: /docs
4. Save and wait for deployment
5. Access at: `https://yourusername.github.io/reponame/`

### Custom Domain (Optional)

Add a `CNAME` file to this directory:

```
your-domain.com
```

Then configure DNS settings to point to GitHub Pages.

## Regenerating Data

To update the embeddings:

```bash
# From project root
poetry run python scripts/generate_data.py
```

This will overwrite `docs/data.json` with fresh embeddings. Commit and push to update the live site.

## Browser Compatibility

- ✅ Chrome/Edge (recommended)
- ✅ Firefox
- ✅ Safari
- ⚠️ IE 11 (not tested, may have issues)

Requires JavaScript enabled.

## Dependencies

The web app uses CDN-hosted libraries (no npm/build step required):

- **Plotly.js v2.27.0** - Interactive plotting
- No other dependencies

## Performance

- **Data size**: 1.18 MB
- **Rendering**: Plotly.js handles 800 points efficiently
- **Task switching**: Instant (no recomputation)

## License

See main project README for license information.
