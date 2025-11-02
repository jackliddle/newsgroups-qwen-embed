// Task definitions
const TASKS = {
    default: {
        name: 'Default',
        instruction: null,
        description: 'Default embeddings with no specific task instruction. This provides a general-purpose representation of the text, capturing overall semantic meaning without being optimized for any specific downstream task.'
    },
    sentiment: {
        name: 'Sentiment',
        instruction: 'Classify the sentiment of the given text as positive, negative, or neutral',
        description: 'Embeddings optimized for sentiment analysis. The model focuses on emotional tone and polarity, grouping texts by whether they express positive, negative, or neutral sentiment.'
    },
    topic: {
        name: 'Topic',
        instruction: 'Identify the topic or theme of the given text',
        description: 'Embeddings optimized for topic classification. The model emphasizes subject matter and thematic content, making it easier to distinguish between different topics like science, politics, sports, and religion.'
    },
    toxicity: {
        name: 'Toxicity',
        instruction: 'Classify the given text as either toxic or not toxic',
        description: 'Embeddings optimized for toxicity detection. The model focuses on discourse patterns, inflammatory language, and potentially harmful content, helping identify civil vs. uncivil discussions.'
    }
};

// Category colors (matching plotly default palette)
const CATEGORY_COLORS = {
    'alt.atheism': '#636EFA',
    'comp.graphics': '#EF553B',
    'sci.med': '#00CC96',
    'talk.religion.misc': '#AB63FA',
    'rec.sport.baseball': '#FFA15A',
    'sci.space': '#19D3F3',
    'talk.politics.guns': '#FF6692',
    'talk.politics.mideast': '#B6E880',
    'rec.autos': '#FF97FF',
    'sci.crypt': '#FECB52'
};

// Global state
let data = null;
let currentTask = 'default';

// Initialize the app
async function init() {
    try {
        // Load data
        const response = await fetch('data.json');
        data = await response.json();

        // Set up event listeners
        document.querySelectorAll('.task-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const task = e.target.dataset.task;
                selectTask(task);
            });
        });

        document.getElementById('what-is-this-btn').addEventListener('click', showOverlay);
        document.getElementById('close-overlay').addEventListener('click', hideOverlay);
        document.querySelector('.overlay').addEventListener('click', (e) => {
            if (e.target.classList.contains('overlay')) {
                hideOverlay();
            }
        });

        // Initial render
        selectTask('default'); // Start with default active

    } catch (error) {
        console.error('Error loading data:', error);
        document.getElementById('plot').innerHTML = '<div class="loading">Error loading data. Please try again.</div>';
    }
}

// Select a task
function selectTask(task) {
    currentTask = task;

    // Update button states
    document.querySelectorAll('.task-btn').forEach(btn => {
        if (btn.dataset.task === task) {
            btn.classList.remove('inactive');
            btn.classList.add('active');
        } else {
            btn.classList.remove('active');
            btn.classList.add('inactive');
        }
    });

    // Update details panel
    updateDetailsPanel(task);

    // Render plot
    renderPlot(task);
}

// Update details panel
function updateDetailsPanel(task) {
    const taskInfo = TASKS[task];
    const detailsHtml = `
        <h2>Details about the prompt and what you see here</h2>
        <p>${taskInfo.description}</p>
        ${taskInfo.instruction ? `
            <div class="task-instruction">
                <h3>Task Instruction</h3>
                <p>"${taskInfo.instruction}"</p>
            </div>
        ` : `
            <div class="task-instruction">
                <h3>Task Instruction</h3>
                <p>No specific instruction provided</p>
            </div>
        `}
        <div class="legend-info">
            <h3>About the Visualization</h3>
            <p>Each point represents a document sampled from 10 categories of the 20 Newsgroups dataset. Colors indicate different categories. Hover over points to see the text preview. The UMAP projection shows how the task instruction reshapes the embedding space.</p>
        </div>
    `;

    document.querySelector('.details-panel').innerHTML = detailsHtml;
}

// Render plot
function renderPlot(task) {
    if (!data) return;

    const xField = `${task}_x`;
    const yField = `${task}_y`;

    // Group data by category
    const categories = [...new Set(data.map(d => d.category))];

    const traces = categories.map(category => {
        const categoryData = data.filter(d => d.category === category);

        return {
            x: categoryData.map(d => d[xField]),
            y: categoryData.map(d => d[yField]),
            mode: 'markers',
            type: 'scatter',
            name: category,
            text: categoryData.map(d => d.text_preview),
            hovertemplate: '<b>%{text}</b><br><b>Category:</b> ' + category + '<extra></extra>',
            marker: {
                size: 8,
                color: CATEGORY_COLORS[category],
                opacity: 0.7,
                line: {
                    color: 'white',
                    width: 0.5
                }
            }
        };
    });

    const layout = {
        xaxis: {
            showgrid: true,
            zeroline: false,
            showticklabels: false
        },
        yaxis: {
            showgrid: true,
            zeroline: false,
            showticklabels: false
        },
        hovermode: 'closest',
        showlegend: true,
        legend: {
            x: 1.02,
            y: 1,
            xanchor: 'left',
            yanchor: 'top',
            font: { size: 10 }
        },
        margin: { t: 20, r: 200, b: 20, l: 20 },
        plot_bgcolor: '#fafafa',
        paper_bgcolor: 'white'
    };

    const config = {
        responsive: true,
        displayModeBar: true,
        modeBarButtonsToRemove: ['lasso2d', 'select2d'],
        displaylogo: false
    };

    Plotly.newPlot('plot', traces, layout, config);
}

// Show overlay
function showOverlay() {
    document.querySelector('.overlay').classList.add('active');
}

// Hide overlay
function hideOverlay() {
    document.querySelector('.overlay').classList.remove('active');
}

// Start the app
init();
