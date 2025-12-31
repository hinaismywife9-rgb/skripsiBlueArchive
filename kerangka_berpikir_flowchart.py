#!/usr/bin/env python3
"""
Generate Kerangka Berpikir with proper flowchart shapes
Rectangles for process, Diamonds for decision, Ellipse for start/end
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Polygon
import numpy as np

fig, ax = plt.subplots(1, 1, figsize=(10, 14))
ax.set_xlim(0, 10)
ax.set_ylim(0, 16)
ax.axis('off')

# Color scheme
color_start_end = '#E8F4F8'
color_process = '#4A90E2'
color_decision = '#FFD700'
color_highlight = '#FF6B6B'
color_implementation = '#50C878'

# Helper function to draw rounded rectangle (process)
def draw_process(ax, x, y, width, height, text, color=color_process):
    box = FancyBboxPatch((x-width/2, y-height/2), width, height,
                         boxstyle="round,pad=0.1", 
                         edgecolor='black', facecolor=color, linewidth=2)
    ax.add_patch(box)
    ax.text(x, y, text, ha='center', va='center', fontsize=9, fontweight='bold', 
            wrap=True, multialignment='center')

# Helper function to draw diamond (decision)
def draw_diamond(ax, x, y, width, height, text, color=color_decision):
    diamond = Polygon([(x, y+height/2), (x+width/2, y), 
                       (x, y-height/2), (x-width/2, y)],
                      facecolor=color, edgecolor='black', linewidth=2)
    ax.add_patch(diamond)
    ax.text(x, y, text, ha='center', va='center', fontsize=8, fontweight='bold')

# Helper function to draw ellipse (start/end)
def draw_ellipse(ax, x, y, width, height, text, color=color_start_end):
    ellipse = mpatches.Ellipse((x, y), width, height, 
                               facecolor=color, edgecolor='black', linewidth=2)
    ax.add_patch(ellipse)
    ax.text(x, y, text, ha='center', va='center', fontsize=9, fontweight='bold')

# Helper function to draw arrow
def draw_arrow(ax, x1, y1, x2, y2, label='', color='black'):
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                           arrowstyle='->', mutation_scale=20, 
                           linewidth=2, color=color)
    ax.add_patch(arrow)
    if label:
        mid_x, mid_y = (x1+x2)/2, (y1+y2)/2
        ax.text(mid_x+0.3, mid_y, label, fontsize=8, color=color, fontweight='bold')

# Title
ax.text(5, 15.5, 'Gambar 3.1 Kerangka Berpikir Penelitian', 
        ha='center', fontsize=13, fontweight='bold')

# Phase 1 label
ax.text(0.5, 14.8, 'Fase 1: Pengumpulan Data - Evaluasi', 
        fontsize=10, fontweight='bold', 
        bbox=dict(boxstyle='round', facecolor='#4A90E2', alpha=0.8, edgecolor='black'),
        color='white')

# Start
y = 14.2
draw_ellipse(ax, 5, y, 1, 0.6, 'START')
draw_arrow(ax, 5, y-0.3, 5, y-0.7)

# Data Selection
y = 13.3
draw_process(ax, 5, y, 2.2, 0.6, 'Data Selection\n(1,200 Samples)')
draw_arrow(ax, 5, y-0.3, 5, y-0.7)

# Data Preprocessing
y = 12.4
draw_process(ax, 5, y, 2.2, 0.6, 'Data Preprocessing\n(Cleaning, Tokenization)')
draw_arrow(ax, 5, y-0.3, 5, y-0.7)

# Labelling
y = 11.5
draw_process(ax, 5, y, 2.2, 0.6, 'Labelling\n(Positive/Negative)')
draw_arrow(ax, 5, y-0.3, 5, y-0.7)

# Data Split
y = 10.6
draw_process(ax, 5, y, 2.4, 0.7, 'Data Split\n(80% Train = 960\n20% Test = 240)', 
             color='#FFE6CC')
draw_arrow(ax, 5, y-0.35, 5, y-0.75)

# 5 Model Training
y = 9.5
draw_process(ax, 5, y, 2.4, 0.7, '5 Model Training\nBERT, DistilBERT\nRoBERTa, ALBERT, XLNET',
             color=color_process)
draw_arrow(ax, 5, y-0.35, 5, y-0.75)

# Evaluation
y = 8.4
draw_process(ax, 5, y, 2.2, 0.6, 'Evaluation\n(Metrics & Performance)')
draw_arrow(ax, 5, y-0.3, 5, y-0.7)

# Decision Diamond: Model Performance OK?
y = 7.3
draw_diamond(ax, 5, y, 1.4, 1, 'Performance\nOK?', color=color_decision)

# No path - loop back
draw_arrow(ax, 5.7, y, 7.2, y, color='red')
draw_arrow(ax, 7.2, y, 7.2, 9.5, color='red')
draw_arrow(ax, 7.2, 9.5, 5.6, 9.5, label='No', color='red')

# Yes path - down
draw_arrow(ax, 5, y-0.5, 5, y-1, label='Yes')

# Select Top 3 Models
y = 5.8
draw_process(ax, 5, y, 2.6, 0.7, 'Select Top 3 Models\nDistilBERT, XLNET, BERT',
             color=color_highlight)
draw_arrow(ax, 5, y-0.35, 5, y-0.75)

# Phase 2 label
y = 4.9
ax.text(0.5, y, 'Fase 2: Implementasi', 
        fontsize=10, fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='#50C878', alpha=0.8, edgecolor='black'),
        color='white')
draw_arrow(ax, 5, y+0.3, 5, y-0.2)

# Streamlit Development
y = 4.1
draw_process(ax, 5, y, 2.4, 0.7, 'Develop Web Application\n(Streamlit)', 
             color=color_implementation)
draw_arrow(ax, 5, y-0.35, 5, y-0.75)

# Features Box
y = 3.1
draw_process(ax, 5, y, 2.6, 0.8, 'Features: Single Prediction\nBatch Processing, Model Compare\nWordCloud, History Tracking',
             color='#E6F9E6')
draw_arrow(ax, 5, y-0.4, 5, y-0.8)

# Docker Deployment
y = 2.0
draw_process(ax, 5, y, 2.4, 0.7, 'Deploy (Docker)\nContainerization & Cloud',
             color=color_implementation)
draw_arrow(ax, 5, y-0.35, 5, y-0.75)

# End
y = 0.8
draw_ellipse(ax, 5, y, 1, 0.6, 'END')

# Legend
legend_y = 15.8
ax.text(0.3, legend_y, '■', fontsize=14, color=color_process)
ax.text(0.7, legend_y, 'Proses', fontsize=9, va='center')

ax.text(2.0, legend_y, '◆', fontsize=14, color=color_decision)
ax.text(2.4, legend_y, 'Keputusan', fontsize=9, va='center')

ax.text(3.8, legend_y, '◯', fontsize=14, color=color_start_end)
ax.text(4.2, legend_y, 'Start/End', fontsize=9, va='center')

ax.text(6.0, legend_y, '—→', fontsize=12, color='red')
ax.text(6.5, legend_y, 'Loop (Tidak OK)', fontsize=9, va='center', color='red')

plt.tight_layout()
plt.savefig('kerangka_berpikir_flowchart.png', dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
print("✅ Gambar saved: kerangka_berpikir_flowchart.png")
plt.close()
