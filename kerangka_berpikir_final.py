#!/usr/bin/env python3
"""
Kerangka Berpikir - Ultra Clean Linear Design
No overlapping, super simple, professional
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Polygon
from matplotlib.patches import Rectangle
import numpy as np

fig = plt.figure(figsize=(12, 16))
ax = fig.add_subplot(111)
ax.set_xlim(0, 10)
ax.set_ylim(0, 20)
ax.axis('off')

# Colors
c_start = '#E8F4F8'
c_process = '#4A90E2'
c_sub = '#B3D9FF'
c_decision = '#FFD700'
c_train = '#FF9F43'
c_eval = '#50C878'
c_impl = '#00B4D8'
c_select = '#FFB6C1'

def box(x, y, w, h, txt, color, size=9):
    rect = FancyBboxPatch((x-w/2, y-h/2), w, h, boxstyle="round,pad=0.05",
                          edgecolor='black', facecolor=color, linewidth=1.5)
    ax.add_patch(rect)
    ax.text(x, y, txt, ha='center', va='center', fontsize=size, fontweight='bold')

def ellipse(x, y, w, h, txt):
    el = mpatches.Ellipse((x, y), w, h, facecolor=c_start, edgecolor='black', linewidth=2)
    ax.add_patch(el)
    ax.text(x, y, txt, ha='center', va='center', fontsize=9, fontweight='bold')

def diamond(x, y, w, h, txt):
    d = Polygon([(x, y+h/2), (x+w/2, y), (x, y-h/2), (x-w/2, y)],
               facecolor=c_decision, edgecolor='black', linewidth=1.5)
    ax.add_patch(d)
    ax.text(x, y, txt, ha='center', va='center', fontsize=7, fontweight='bold')

def arrow(x1, y1, x2, y2, label='', color='black'):
    arr = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='->', 
                         mutation_scale=20, linewidth=2, color=color)
    ax.add_patch(arr)
    if label:
        ax.text((x1+x2)/2 + 0.3, (y1+y2)/2, label, fontsize=7, 
               color=color, fontweight='bold')

# ===== TITLE =====
ax.text(5, 19.5, 'Gambar 3.1 Kerangka Berpikir Penelitian', 
       ha='center', fontsize=12, fontweight='bold')
ax.text(5, 19, 'Sentiment Analysis dengan Transformer Models',
       ha='center', fontsize=9, style='italic')

y = 18.3

# ===== PHASE 1 =====
ax.text(5, y, 'FASE 1: PENGUMPULAN DATA & EVALUASI', ha='center', fontsize=9, fontweight='bold',
       bbox=dict(boxstyle='round', facecolor=c_process, alpha=0.8, edgecolor='black', linewidth=1.5),
       color='white')
y -= 0.6

# START
ellipse(5, y, 0.8, 0.5, 'START')
arrow(5, y-0.25, 5, y-0.55)
y -= 1.0

# Data Selection
box(5, y, 2.0, 0.6, 'Data Selection\n(1,200 Samples)', c_process, 8)
arrow(5, y-0.3, 5, y-0.7)
y -= 1.0

# Data Preprocessing
box(5, y, 2.2, 0.8, 'Data Preprocessing\n(Clean, Tokenize, Lowercase)', c_sub, 8)
arrow(5, y-0.4, 5, y-0.8)
y -= 1.0

# Labelling
box(5, y, 2.0, 0.6, 'Labelling\n(Positive/Negative)', c_process, 8)
arrow(5, y-0.3, 5, y-0.7)
y -= 1.0

# Data Split
box(5, y, 2.4, 0.8, 'Data Split\n80% Train (960) | 20% Test (240)', '#FFE6CC', 8)
arrow(5, y-0.4, 5, y-0.8)
y -= 1.0

# ===== MODEL TRAINING =====
ax.text(5, y, 'MODEL TRAINING', ha='center', fontsize=8, fontweight='bold',
       bbox=dict(boxstyle='round', facecolor=c_train, alpha=0.7, edgecolor='black'))
y -= 0.5

box(5, y, 1.8, 0.6, '5 Model Training', c_train, 9)
arrow(5, y-0.3, 5, y-0.7)
y -= 1.0

# 5 Models side by side
models_x = [2.5, 3.8, 5.0, 6.2, 7.5]
models = ['BERT', 'DistilBERT', 'RoBERTa', 'ALBERT', 'XLNET']
for i, (mx, model) in enumerate(zip(models_x, models)):
    box(mx, y, 1.0, 0.5, model, c_sub, 7)
    arrow(mx, y-0.25, mx, y-0.55)

arrow(5, y+0.3, 2.5, y+0.3)
arrow(5, y+0.3, 7.5, y+0.3)
y -= 1.0

# ===== EVALUATION =====
ax.text(5, y, 'EVALUATION', ha='center', fontsize=8, fontweight='bold',
       bbox=dict(boxstyle='round', facecolor=c_eval, alpha=0.7, edgecolor='black'))
y -= 0.5

# Metrics side by side
metrics_x = [2.5, 3.8, 5.0, 6.2, 7.5]
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
for i, (mx, metric) in enumerate(zip(metrics_x, metrics)):
    box(mx, y, 1.0, 0.45, metric, c_eval, 7)

# Collect to center
for mx in metrics_x:
    arrow(mx, y-0.225, 5, y-0.5)

y -= 0.8

# ===== DECISION 1 =====
diamond(5, y, 1.2, 0.8, 'All Models\nTrained?')
arrow(5, y-0.4, 5, y-0.8)

# Loop back (No path)
arrow(5.6, y, 8.5, y, label='No', color='red')
arrow(8.5, y, 8.5, y+3.5, color='red', label='')
arrow(8.5, y+3.5, 5.8, y+3.5, color='red', label='')

y -= 1.0

# ===== DECISION 2 =====
diamond(5, y, 1.4, 0.8, 'Performance\nGood?')
arrow(5, y-0.4, 5, y-0.8)

# Loop back (No path)
arrow(5.7, y, 9.0, y, label='No', color='red')
arrow(9.0, y, 9.0, y+5, color='red')
arrow(9.0, y+5, 6.0, y+5, color='red')

y -= 1.0

# ===== SELECT TOP 3 =====
box(5, y, 2.4, 0.7, 'Select Top 3 Models\nDistilBERT, XLNET, BERT', c_select, 8)
arrow(5, y-0.35, 5, y-0.7)
y -= 0.8

# ===== PHASE 2 =====
ax.text(5, y, 'FASE 2: IMPLEMENTASI', ha='center', fontsize=9, fontweight='bold',
       bbox=dict(boxstyle='round', facecolor=c_impl, alpha=0.8, edgecolor='black', linewidth=1.5),
       color='white')
y -= 0.6

# Streamlit
box(5, y, 2.0, 0.6, 'Develop Web App\n(Streamlit)', c_impl, 8)
arrow(5, y-0.3, 5, y-0.7)
y -= 1.0

# Features
box(5, y, 2.4, 0.8, 'Features: Single & Batch Prediction\nModel Comparison, WordCloud, History', 
   c_impl, 7)
arrow(5, y-0.4, 5, y-0.8)
y -= 1.0

# Testing
box(5, y, 2.0, 0.6, 'Testing & Validation\n(Unit, Integration, Load)', c_impl, 8)
arrow(5, y-0.3, 5, y-0.7)
y -= 1.0

# Docker
box(5, y, 2.0, 0.6, 'Deploy (Docker)\nCloud Deployment', c_impl, 8)
arrow(5, y-0.3, 5, y-0.7)
y -= 1.0

# Production
box(5, y, 2.0, 0.6, 'Production\n(Live Service)', c_impl, 8)
arrow(5, y-0.3, 5, y-0.7)
y -= 0.8

# END
ellipse(5, y, 0.8, 0.5, 'END')

# ===== LEGEND =====
leg_y = 0.5
ax.text(0.5, leg_y, '■ Process', fontsize=7, bbox=dict(boxstyle='round', facecolor=c_process, alpha=0.6))
ax.text(2.2, leg_y, '◆ Decision', fontsize=7, bbox=dict(boxstyle='round', facecolor=c_decision, alpha=0.6))
ax.text(3.8, leg_y, '◯ Start/End', fontsize=7, bbox=dict(boxstyle='round', facecolor=c_start, alpha=0.6))
ax.text(5.5, leg_y, '→ Normal Flow', fontsize=7)
ax.text(7.3, leg_y, '--→ Loop', fontsize=7, color='red')

plt.tight_layout()
plt.savefig('kerangka_berpikir_final.png', dpi=300, bbox_inches='tight', 
           facecolor='white', pad_inches=0.4)
print("✅ Gambar saved: kerangka_berpikir_final.png")
plt.close()
