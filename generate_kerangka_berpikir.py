"""
Generate Kerangka Berpikir (Research Framework Diagram)
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

fig, ax = plt.subplots(figsize=(14, 10))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

# ============================================================
# COLOR SCHEME
# ============================================================
color_process = '#E8F4F8'
color_decision = '#FFF9E6'
color_impl = '#F0F8E8'
color_start_end = '#E8E8E8'

# ============================================================
# PHASE 1: PENGUMPULAN DATA - EVALUASI
# ============================================================

# Title
title_box = FancyBboxPatch((0.2, 8.8), 4, 0.6, 
                           boxstyle="round,pad=0.1", 
                           edgecolor='black', facecolor='#4A90E2', 
                           linewidth=2)
ax.add_patch(title_box)
ax.text(2.2, 9.1, 'Fase 1: Pengumpulan Data - Evaluasi', 
        ha='center', va='center', fontsize=11, fontweight='bold', color='white')

# Start
start = mpatches.Circle((0.7, 8), 0.25, facecolor=color_start_end, edgecolor='black', linewidth=2)
ax.add_patch(start)
ax.text(0.7, 8, 'Start', ha='center', va='center', fontsize=9, fontweight='bold')

# Arrow
ax.arrow(0.95, 8, 0.4, 0, head_width=0.15, head_length=0.1, fc='black', ec='black')

# Data Selection
box1 = FancyBboxPatch((1.5, 7.7), 1.2, 0.6, 
                      boxstyle="round,pad=0.05", 
                      edgecolor='black', facecolor=color_process, linewidth=1.5)
ax.add_patch(box1)
ax.text(2.1, 8, 'Data\nSelection', ha='center', va='center', fontsize=9)

# Arrow
ax.arrow(2.7, 8, 0.4, 0, head_width=0.15, head_length=0.1, fc='black', ec='black')

# Data Preprocessing
box2 = FancyBboxPatch((3.2, 7.7), 1.2, 0.6, 
                      boxstyle="round,pad=0.05", 
                      edgecolor='black', facecolor=color_process, linewidth=1.5)
ax.add_patch(box2)
ax.text(3.8, 8, 'Data\nPreprocessing', ha='center', va='center', fontsize=9)

# Arrow
ax.arrow(4.4, 8, 0.4, 0, head_width=0.15, head_length=0.1, fc='black', ec='black')

# Labelling
box3 = FancyBboxPatch((4.9, 7.7), 1.0, 0.6, 
                      boxstyle="round,pad=0.05", 
                      edgecolor='black', facecolor=color_process, linewidth=1.5)
ax.add_patch(box3)
ax.text(5.4, 8, 'Labelling', ha='center', va='center', fontsize=9)

# Arrow
ax.arrow(5.9, 8, 0.4, 0, head_width=0.15, head_length=0.1, fc='black', ec='black')

# Data Split
box4 = FancyBboxPatch((6.4, 7.5), 1.5, 0.8, 
                      boxstyle="round,pad=0.05", 
                      edgecolor='black', facecolor=color_process, linewidth=1.5)
ax.add_patch(box4)
ax.text(7.15, 7.9, 'Data Split\n(80% Train,\n20% Test)', 
        ha='center', va='center', fontsize=8)

# Arrow down
ax.arrow(7.15, 7.5, 0, -0.4, head_width=0.15, head_length=0.1, fc='black', ec='black')

# Model Training (5 Models)
box6 = FancyBboxPatch((4.9, 6.3), 1.4, 1.0, 
                      boxstyle="round,pad=0.05", 
                      edgecolor='black', facecolor=color_process, linewidth=1.5)
ax.add_patch(box6)
ax.text(5.6, 6.9, '5 Model\nTraining:', ha='center', va='center', fontsize=8, fontweight='bold')
ax.text(5.6, 6.45, 'BERT, DistilBERT\nRoBERTa, ALBERT\nXLNET', 
        ha='center', va='center', fontsize=7)

# Arrow
ax.arrow(4.9, 6.8, -0.4, 0, head_width=0.15, head_length=0.1, fc='black', ec='black')

# Evaluation
box7 = FancyBboxPatch((4.0, 6.3), 0.8, 1.0, 
                      boxstyle="round,pad=0.05", 
                      edgecolor='black', facecolor=color_process, linewidth=1.5)
ax.add_patch(box7)
ax.text(4.4, 6.8, 'Evaluation\n(Metrics)', ha='center', va='center', fontsize=8)

# Arrow down
ax.arrow(4.4, 6.3, 0, -0.4, head_width=0.15, head_length=0.1, fc='black', ec='black')

# Decision Diamond - Last Fold?
diamond1_x = [4.4, 4.9, 4.4, 3.9, 4.4]
diamond1_y = [5.0, 4.4, 3.8, 4.4, 5.0]
ax.plot(diamond1_x, diamond1_y, 'k-', linewidth=1.5)
ax.fill(diamond1_x, diamond1_y, color=color_decision)
ax.text(4.4, 4.4, 'Last\nFold?', ha='center', va='center', fontsize=8, fontweight='bold')

# Arrow No (loop back)
ax.annotate('', xy=(5.6, 6.5), xytext=(4.85, 4.7),
            arrowprops=dict(arrowstyle='->', lw=1.5, color='red', 
                          connectionstyle="arc3,rad=.5"))
ax.text(5.3, 5.5, 'No', fontsize=8, color='red', fontweight='bold')

# Arrow Yes (down)
ax.arrow(4.4, 3.8, 0, -0.4, head_width=0.15, head_length=0.1, fc='black', ec='black')
ax.text(4.6, 3.6, 'Yes', fontsize=8, fontweight='bold')

# Decision Diamond - Last K?
diamond2_x = [4.4, 4.9, 4.4, 3.9, 4.4]
diamond2_y = [3.2, 2.6, 2.0, 2.6, 3.2]
ax.plot(diamond2_x, diamond2_y, 'k-', linewidth=1.5)
ax.fill(diamond2_x, diamond2_y, color=color_decision)
ax.text(4.4, 2.6, 'Last K?', ha='center', va='center', fontsize=8, fontweight='bold')

# Arrow No (loop back to Data Transformation)
ax.annotate('', xy=(6.4, 6.65), xytext=(3.85, 2.9),
            arrowprops=dict(arrowstyle='->', lw=1.5, color='red',
                          connectionstyle="arc3,rad=.3"))
ax.text(2.8, 4.5, 'No', fontsize=8, color='red', fontweight='bold')

# Arrow Yes (down)
ax.arrow(4.4, 2.0, 0, -0.4, head_width=0.15, head_length=0.1, fc='black', ec='black')
ax.text(4.6, 1.8, 'Yes', fontsize=8, fontweight='bold')

# Choose Top 3 Models
box8 = FancyBboxPatch((3.7, 0.5), 1.4, 0.9, 
                      boxstyle="round,pad=0.05", 
                      edgecolor='black', facecolor='#FFE6E6', linewidth=2)
ax.add_patch(box8)
ax.text(4.4, 0.95, 'Choose Top 3 Models:\nDistilBERT\nXLNET, BERT', 
        ha='center', va='center', fontsize=8, fontweight='bold')

# ============================================================
# PHASE 2: IMPLEMENTASI
# ============================================================

# Title
title_box2 = FancyBboxPatch((6.0, 3.5), 3.5, 0.6, 
                            boxstyle="round,pad=0.1", 
                            edgecolor='black', facecolor='#90EE90', 
                            linewidth=2)
ax.add_patch(title_box2)
ax.text(7.75, 3.8, 'Fase 2: Implementasi', 
        ha='center', va='center', fontsize=11, fontweight='bold')

# Arrow from Phase 1 to Phase 2
ax.annotate('', xy=(6.0, 0.95), xytext=(5.15, 0.95),
            arrowprops=dict(arrowstyle='->', lw=2, color='black'))

# Develop Web App
box9 = FancyBboxPatch((6.2, 2.3), 3.1, 0.9, 
                      boxstyle="round,pad=0.05", 
                      edgecolor='black', facecolor=color_impl, linewidth=1.5)
ax.add_patch(box9)
ax.text(7.75, 2.95, 'Develop Web Application (Streamlit)', 
        ha='center', va='center', fontsize=9, fontweight='bold')
ax.text(7.75, 2.55, 'Dashboard + Sentiment Predictor', 
        ha='center', va='center', fontsize=8)

# Arrow
ax.arrow(7.75, 2.3, 0, -0.4, head_width=0.15, head_length=0.1, fc='black', ec='black')

# Deploy Web App
box10 = FancyBboxPatch((6.2, 1.3), 3.1, 0.8, 
                       boxstyle="round,pad=0.05", 
                       edgecolor='black', facecolor=color_impl, linewidth=1.5)
ax.add_patch(box10)
ax.text(7.75, 1.7, 'Deploy Web App (Docker)', 
        ha='center', va='center', fontsize=9, fontweight='bold')

# Arrow
ax.arrow(7.75, 1.3, 0, -0.4, head_width=0.15, head_length=0.1, fc='black', ec='black')

# End
end = mpatches.Circle((7.75, 0.55), 0.25, facecolor=color_start_end, edgecolor='black', linewidth=2)
ax.add_patch(end)
ax.text(7.75, 0.55, 'End', ha='center', va='center', fontsize=9, fontweight='bold')

# ============================================================
# LEGEND
# ============================================================
ax.text(0.3, 0.3, 'Legenda:', fontsize=9, fontweight='bold')
rect1 = mpatches.Rectangle((0.3, -0.1), 0.3, 0.25, facecolor=color_process, 
                            edgecolor='black', linewidth=1)
ax.add_patch(rect1)
ax.text(0.8, 0.025, 'Data Processing', fontsize=8)

rect2 = mpatches.Rectangle((2.0, -0.1), 0.3, 0.25, facecolor=color_decision, 
                            edgecolor='black', linewidth=1)
ax.add_patch(rect2)
ax.text(2.5, 0.025, 'Decision/Control', fontsize=8)

rect3 = mpatches.Rectangle((3.9, -0.1), 0.3, 0.25, facecolor=color_impl, 
                            edgecolor='black', linewidth=1)
ax.add_patch(rect3)
ax.text(4.4, 0.025, 'Implementation', fontsize=8)

# Title
fig.suptitle('Gambar 3.1 Kerangka Berpikir Penelitian', 
             fontsize=14, fontweight='bold', y=0.98)

plt.tight_layout()
plt.savefig('kerangka_berpikir_penelitian.png', dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
print("✅ Gambar saved: kerangka_berpikir_penelitian.png")
plt.show()
