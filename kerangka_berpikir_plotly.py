#!/usr/bin/env python3
"""
Generate Kerangka Berpikir diagram using Plotly
Clean, professional flowchart for thesis
"""

import plotly.graph_objects as go

# Define nodes
nodes = [
    "Start",
    "Data Selection\n(1,200 Samples)",
    "Data Preprocessing",
    "Labelling",
    "Data Split\n(80% Train, 20% Test)",
    "5 Model Training",
    "Evaluation",
    "Top 3 Models Selected",
    "Streamlit Development",
    "Docker Deployment",
    "End"
]

# Define connections (from, to)
edges = [
    (0, 1), (1, 2), (2, 3), (3, 4), (4, 5),
    (5, 6), (6, 7), (7, 8), (8, 9), (9, 10)
]

# Y positions for vertical layout
y_pos = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
x_pos = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

# Color mapping - blue for Phase 1, green for Phase 2
colors = [
    '#E8F4F8',  # Start
    '#4A90E2',  # Data Selection
    '#4A90E2',  # Preprocessing
    '#4A90E2',  # Labelling
    '#FFD700',  # Data Split (highlight)
    '#4A90E2',  # Training
    '#4A90E2',  # Evaluation
    '#FF6B6B',  # Top 3 (highlight)
    '#50C878',  # Streamlit
    '#50C878',  # Docker
    '#E8F4F8',  # End
]

# Create edge traces
edge_x = []
edge_y = []
for edge in edges:
    x0, y0 = x_pos[edge[0]], y_pos[edge[0]]
    x1, y1 = x_pos[edge[1]], y_pos[edge[1]]
    edge_x += [x0, x1, None]
    edge_y += [y0, y1, None]

edge_trace = go.Scatter(
    x=edge_x, y=edge_y,
    mode='lines',
    line=dict(width=2, color='#555'),
    hoverinfo='none',
    showlegend=False
)

# Create node trace
node_trace = go.Scatter(
    x=x_pos, y=y_pos,
    mode='markers+text',
    text=nodes,
    textposition='middle center',
    textfont=dict(size=9, color='black', family='Arial'),
    hoverinfo='text',
    marker=dict(
        size=40,
        color=colors,
        line=dict(width=2, color='black'),
        opacity=0.9
    ),
    showlegend=False
)

# Create figure
fig = go.Figure(data=[edge_trace, node_trace])

fig.update_layout(
    title={
        'text': 'Gambar 3.1 Kerangka Berpikir Penelitian',
        'x': 0.5,
        'xanchor': 'center',
        'font': {'size': 14, 'family': 'Arial', 'color': '#333'}
    },
    showlegend=False,
    hovermode='closest',
    margin=dict(b=20, l=5, r=5, t=40),
    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    plot_bgcolor='white',
    paper_bgcolor='white',
    width=1200,
    height=900,
)

# Add annotations for phases
fig.add_annotation(
    x=-0.5, y=7.5,
    text="<b>Fase 1: Evaluasi</b>",
    showarrow=False,
    font=dict(size=12, color='#4A90E2', family='Arial'),
    bgcolor='#E8F4F8',
    bordercolor='#4A90E2',
    borderwidth=2,
    borderpad=8
)

fig.add_annotation(
    x=-0.5, y=1.5,
    text="<b>Fase 2: Implementasi</b>",
    showarrow=False,
    font=dict(size=12, color='#50C878', family='Arial'),
    bgcolor='#E8F4F8',
    bordercolor='#50C878',
    borderwidth=2,
    borderpad=8
)

# Save as PNG
fig.write_image('kerangka_berpikir_clean.png', width=1200, height=900)
print("✅ Gambar saved: kerangka_berpikir_clean.png")
