#!/usr/bin/env python3
"""
Generate Kerangka Berpikir - Sentiment Analysis Style (FIXED)
Clean layout, no overlap, properly sized
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Polygon, Rectangle
import numpy as np

fig, ax = plt.subplots(1, 1, figsize=(18, 14))
ax.set_xlim(0, 18)
ax.set_ylim(0, 14)
ax.axis('off')

# Color scheme
color_start_end = '#E8F4F8'
color_process = '#4A90E2'
color_subprocess = '#B3D9FF'
color_decision = '#FFD700'
color_training = '#FF9F43'
color_evaluation = '#50C878'
color_implementation = '#00B4D8'

# Helper functions
def draw_box(ax, x, y, w, h, text, color, fontsize=8, bold=True):
    """Draw rounded rectangle"""
    box = FancyBboxPatch((x-w/2, y-h/2), w, h, boxstyle="round,pad=0.08",
                         edgecolor='black', facecolor=color, linewidth=1.5)
    ax.add_patch(box)
    weight = 'bold' if bold else 'normal'
    ax.text(x, y, text, ha='center', va='center', fontsize=fontsize, 
            fontweight=weight, multialignment='center')

def draw_diamond(ax, x, y, w, h, text, color=color_decision):
    """Draw decision diamond"""
    diamond = Polygon([(x, y+h/2), (x+w/2, y), (x, y-h/2), (x-w/2, y)],
                      facecolor=color, edgecolor='black', linewidth=1.5)
    ax.add_patch(diamond)
    ax.text(x, y, text, ha='center', va='center', fontsize=7, fontweight='bold')

def draw_ellipse(ax, x, y, w, h, text, color=color_start_end):
    """Draw ellipse (start/end)"""
    ellipse = mpatches.Ellipse((x, y), w, h, facecolor=color, 
                               edgecolor='black', linewidth=2)
    ax.add_patch(ellipse)
    ax.text(x, y, text, ha='center', va='center', fontsize=9, fontweight='bold')

def draw_arrow(ax, x1, y1, x2, y2, label='', color='black', style='-'):
    """Draw arrow"""
    arrow = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='->', 
                           mutation_scale=15, linewidth=1.5, color=color, linestyle=style)
    ax.add_patch(arrow)
    if label:
        mid_x, mid_y = (x1+x2)/2, (y1+y2)/2
        ax.text(mid_x+0.2, mid_y+0.15, label, fontsize=7, color=color, fontweight='bold')

def draw_subprocess_box(ax, x, y, w, h, items, color):
    """Draw box with sub-processes"""
    main_box = Rectangle((x-w/2, y-h/2), w, h, facecolor=color, 
                         edgecolor='black', linewidth=2, alpha=0.3)
    ax.add_patch(main_box)
    
    # Draw sub-items
    item_height = h / (len(items) + 1)
    for i, item in enumerate(items):
        item_y = y + h/2 - (i+1)*item_height
        draw_box(ax, x, item_y, w-0.3, item_height*0.8, item, '#FFFFFF', fontsize=6.5, bold=False)

# TITLE
ax.text(9, 13.5, 'Gambar 3.1 Kerangka Berpikir Penelitian', 
        ha='center', fontsize=13, fontweight='bold')
ax.text(9, 13.0, 'Sentiment Analysis menggunakan Transformer Models', 
        ha='center', fontsize=10, style='italic', color='#666')

# ========== LEFT SIDE: PHASE 1 DATA ==========
phase1_x = 2.5

# Phase 1 Header
ax.text(phase1_x, 12.3, 'Fase 1: Pengumpulan Data & Evaluasi', fontsize=9, fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='#4A90E2', alpha=0.9, edgecolor='black', linewidth=1.5),
        color='white')

# START
draw_ellipse(ax, phase1_x, 11.8, 0.8, 0.5, 'START')
draw_arrow(ax, phase1_x, 11.5, phase1_x, 11.0)

# DATA SELECTION
draw_box(ax, phase1_x, 10.5, 1.6, 0.6, 'Data Selection\n(1,200 Samples)', color_process)
draw_arrow(ax, phase1_x, 10.2, phase1_x, 9.6)

# DATA PREPROCESSING (subprocess box)
y_preproc = 8.7
draw_box(ax, phase1_x, y_preproc, 2.2, 1.4, 
         'Data Preprocessing\n\nRemove NULL\nRemove Duplicates\nTokenization\nLowercase\nRemove Punctuation', 
         color_subprocess, fontsize=7)
draw_arrow(ax, phase1_x, 7.95, phase1_x, 7.4)

# LABELLING
draw_box(ax, phase1_x, 6.9, 1.6, 0.6, 'Labelling\n(Positive/Negative)', color_process)
draw_arrow(ax, phase1_x, 6.6, phase1_x, 6.0)

# DATA SPLIT (80-20)
y_split = 5.4
draw_box(ax, phase1_x, y_split, 2.2, 0.9, 'Data Split\n(80% Train = 960\n20% Test = 240)', 
         '#FFE6CC', fontsize=8)

# ========== MIDDLE: MODEL TRAINING ==========
train_x = 9

# Arrow from split to training
draw_arrow(ax, phase1_x+1.1, y_split, train_x-1.5, y_split)
draw_arrow(ax, train_x-1.5, y_split, train_x-1.5, 4.8)
draw_arrow(ax, train_x-1.5, 4.8, train_x-0.8, 4.8)

# 5 MODEL TRAINING (main box)
draw_box(ax, train_x, 4.8, 1.4, 0.6, '5 Model Training', color_training, fontsize=9, bold=True)

# Models in parallel (side by side)
models = ['BERT', 'DistilBERT', 'RoBERTa', 'ALBERT', 'XLNET']
model_x_positions = [6.2, 7.4, 8.6, 9.8, 11.0]
y_model = 3.6

for model, x_pos in zip(models, model_x_positions):
    draw_box(ax, x_pos, y_model, 1.0, 0.5, model, color_subprocess, fontsize=7.5, bold=False)
    draw_arrow(ax, x_pos, 3.8, x_pos, 4.3)

# Arrow from main training to models
draw_arrow(ax, train_x, 4.5, train_x, 3.9)

# EVALUATION METRICS (separate row)
y_eval = 2.4
eval_label_x = 8.5
ax.text(eval_label_x, 2.9, 'Evaluation & Metrics', ha='center', fontsize=8, fontweight='bold',
        bbox=dict(boxstyle='round', facecolor=color_evaluation, alpha=0.6, edgecolor='black'))

metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
metrics_x = [6.2, 7.4, 8.6, 9.8, 11.0]

for metric, x_pos in zip(metrics, metrics_x):
    draw_box(ax, x_pos, y_eval, 0.95, 0.45, metric, color_evaluation, fontsize=7, bold=False)
    draw_arrow(ax, x_pos, 3.4, x_pos, 2.65)

# ========== DECISION POINTS ==========
dec_x = 9

# Decision: All models trained?
y_dec1 = 1.4
draw_diamond(ax, dec_x, y_dec1, 1.2, 0.9, 'All Models\nTrained?', color_decision)

# Arrow from evaluation to decision
draw_arrow(ax, eval_label_x, 2.7, dec_x, 1.85)

# No path - retry (loop)
draw_arrow(ax, dec_x-0.6, y_dec1, dec_x-2.0, y_dec1, label='No', color='red')
draw_arrow(ax, dec_x-2.0, y_dec1, dec_x-2.0, 4.8, color='red', style='--')
draw_arrow(ax, dec_x-2.0, 4.8, train_x-0.8, 4.8, color='red', style='--')

# Yes path
draw_arrow(ax, dec_x, y_dec1-0.45, dec_x, 0.9, label='Yes')

# Decision: Performance Acceptable?
y_dec2 = 0.4
draw_diamond(ax, dec_x, y_dec2, 1.4, 0.8, 'Performance\nAcceptable?', color_decision)

# No - retrain
draw_arrow(ax, dec_x+0.7, y_dec2, dec_x+2.5, y_dec2, label='No', color='red')
draw_arrow(ax, dec_x+2.5, y_dec2, dec_x+2.5, 4.8, color='red')
draw_arrow(ax, dec_x+2.5, 4.8, train_x+0.8, 4.8, color='red')

# Yes - continue to phase 2
draw_arrow(ax, dec_x, y_dec2-0.4, 13.5, y_dec2-0.4, label='Yes')

# SELECT TOP 3 MODELS (between phases)
draw_box(ax, 13.5, 1.2, 2.2, 0.7, 'Select Top 3 Models\nDistilBERT, XLNET, BERT', 
         '#FFB6C1', fontsize=8, bold=True)

# ========== RIGHT SIDE: PHASE 2 IMPLEMENTATION ==========
impl_x = 15.5

# Phase 2 Header
ax.text(impl_x, 12.3, 'Fase 2: Implementasi', fontsize=9, fontweight='bold',
        bbox=dict(boxstyle='round', facecolor=color_implementation, alpha=0.9, edgecolor='black', linewidth=1.5),
        color='white')

# Arrow from select top 3 to phase 2
draw_arrow(ax, 14.6, 1.2, impl_x-1.2, 11.8)

# Streamlit Development
y_impl = 11.3
draw_box(ax, impl_x, y_impl, 1.8, 0.6, 'Develop Web App\n(Streamlit)', color_implementation, fontsize=8)
draw_arrow(ax, impl_x, y_impl-0.3, impl_x, y_impl-0.7)

# Features (subprocess)
y_features = 10.0
draw_box(ax, impl_x, y_features, 2.0, 1.2,
         'Dashboard Features\n\nSingle Prediction\nBatch Processing\nModel Comparison\nWordCloud\nHistory',
         color_implementation, fontsize=7)
draw_arrow(ax, impl_x, y_features-0.6, impl_x, y_features-0.9)

# Testing
y_test = 8.5
draw_box(ax, impl_x, y_test, 1.8, 0.6, 'Testing & Validation\n(Unit, Integration, Load)',
         color_implementation, fontsize=8)
draw_arrow(ax, impl_x, y_test-0.3, impl_x, y_test-0.7)

# Docker Deployment
y_docker = 7.0
draw_box(ax, impl_x, y_docker, 1.8, 0.6, 'Deploy (Docker)\nCloud Deployment',
         color_implementation, fontsize=8)
draw_arrow(ax, impl_x, y_docker-0.3, impl_x, y_docker-0.7)

# Monitoring
y_monitor = 5.5
draw_box(ax, impl_x, y_monitor, 1.8, 0.6, 'Monitoring\nHealth & Performance',
         color_implementation, fontsize=8)
draw_arrow(ax, impl_x, y_monitor-0.3, impl_x, y_monitor-0.7)

# Production
y_prod = 4.0
draw_box(ax, impl_x, y_prod, 1.8, 0.6, 'Production\n(Live Service)',
         color_implementation, fontsize=8)
draw_arrow(ax, impl_x, y_prod-0.3, impl_x, y_prod-0.7)

# END
y_end = 3.0
draw_ellipse(ax, impl_x, y_end, 0.8, 0.5, 'END')

# ========== LEGEND ==========
leg_y = 0.8
leg_x = 0.5
ax.text(leg_x, leg_y, '■ Proses', fontsize=7, 
        bbox=dict(boxstyle='round', facecolor=color_process, alpha=0.6, pad=0.3))
ax.text(leg_x+2.0, leg_y, '◆ Keputusan', fontsize=7,
        bbox=dict(boxstyle='round', facecolor=color_decision, alpha=0.6, pad=0.3))
ax.text(leg_x+4.2, leg_y, '◯ Start/End', fontsize=7,
        bbox=dict(boxstyle='round', facecolor=color_start_end, alpha=0.6, pad=0.3))
ax.text(leg_x+6.2, leg_y, '→ Flow Normal', fontsize=7, color='black',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, pad=0.3, edgecolor='black'))
ax.text(leg_x+8.8, leg_y, '--→ Loop/Retry', fontsize=7, color='red',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, pad=0.3, edgecolor='red'))

plt.tight_layout()
plt.savefig('kerangka_berpikir_sentiment.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none', pad_inches=0.3)
print("✅ Gambar saved: kerangka_berpikir_sentiment.png")
plt.close()
