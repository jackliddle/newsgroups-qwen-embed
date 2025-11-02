// Task definitions
const TASKS = {
    default: {
        name: '🔷 Default',
        instruction: null,
        description: 'The Baseline — We observe good separation between some categories, but there\'s significant overlap and bleed-through across topics. This general-purpose embedding captures broad semantic similarities without optimization for any specific task. But watch what happens when we add task instructions...',
        explanation: 'What does this show? The baseline embedding space organizes documents by overall semantic similarity, revealing natural groupings but without the sharpness that task-specific optimization provides.'
    },
    topic: {
        name: '🎯 Topic',
        instruction: 'Identify the topic or theme of the given text',
        description: 'Remarkable! The categories show dramatically improved separation with minimal bleed-through. Documents cluster tightly by their subject matter, making topic boundaries crystal clear. This task-specific optimization would significantly boost performance for downstream classification tasks. Perfect for document routing and content categorization.',
        explanation: 'What does this show? When optimized for topic identification, the embedding space reorganizes to emphasize subject matter, creating distinct, well-separated clusters that align perfectly with newsgroup categories.'
    },
    toxicity: {
        name: '⚠️ Toxicity',
        instruction: 'Classify the given text as either toxic or not toxic',
        description: 'A Window Into Discourse Patterns — The visualization reveals distinct patterns in discourse civility across communities. Discussions in alt.atheism and talk.politics.mideast cluster toward higher toxicity, while rec.autos and comp.graphics show notably more civil discourse. The embedding space reorganizes around communication style rather than subject matter. This shows embeddings can detect style, not just content.',
        explanation: 'What does this show? Optimized for toxicity detection, the embedding space highlights discourse patterns and communication style, revealing that some discussion areas naturally exhibit more contentious language than others.'
    },
    sentiment: {
        name: '😊 Sentiment',
        instruction: 'Classify the sentiment of the given text as positive, negative, or neutral',
        description: 'A Stunning Transformation — Notice how the well-defined category clusters dissolve into a smooth gradient. Rather than grouping by topic, documents now arrange themselves along a continuous spectrum of emotional tone—from negative through neutral to positive sentiment. The embedding space has fundamentally reshaped around affective content.',
        explanation: 'What does this show? When optimized for sentiment, the embedding space completely reorganizes around emotional tone rather than topic, creating a gradient from negative to positive affect that cuts across subject-matter boundaries.'
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
        <h2><strong>${taskInfo.name}</strong></h2>
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
        <div class="task-explanation">
            <h3>What does this show?</h3>
            <p>${taskInfo.description}</p>
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
