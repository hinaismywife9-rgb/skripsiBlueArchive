import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib import gridspec

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['legend.fontsize'] = 9

# Load results
with open('training_results.json', 'r') as f:
    data = json.load(f)

# Extract data
models = list(data.keys())
metrics = {
    'Accuracy': [data[m]['acc'] for m in models],
    'Precision': [data[m]['prec'] for m in models],
    'Recall': [data[m]['rec'] for m in models],
    'F1-Score': [data[m]['f1'] for m in models]
}

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

# ============================================================
# FIGURE 1: BAR CHART - All metrics from 0.0
# ============================================================
fig, ax = plt.subplots(figsize=(12, 6))

x = np.arange(len(models))
width = 0.2

for i, (metric_name, values) in enumerate(metrics.items()):
    ax.bar(x + i * width, values, width, label=metric_name, alpha=0.85)

ax.set_xlabel('Model', fontweight='bold', fontsize=12)
ax.set_ylabel('Score', fontweight='bold', fontsize=12)
ax.set_title('Performance Metrics Comparison - All Transformer Models (Y-axis: 0.0 to 1.0)', 
             fontweight='bold', fontsize=13, pad=15)
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(models, fontweight='bold')
ax.legend(loc='lower right', ncol=2)
ax.set_ylim(0.0, 1.05)  # Start from 0.0
ax.grid(axis='y', alpha=0.3, linestyle='--')

# Add value labels on bars
for i, (metric_name, values) in enumerate(metrics.items()):
    for j, v in enumerate(values):
        ax.text(j + i * width, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig('fig1_metrics_bars_full_scale.png', dpi=300, bbox_inches='tight')
print('✓ Saved: fig1_metrics_bars_full_scale.png')
plt.close()

# ============================================================
# FIGURE 2: LINE CHART - All metrics from 0.0
# ============================================================
fig, ax = plt.subplots(figsize=(12, 6))

for i, (metric_name, values) in enumerate(metrics.items()):
    ax.plot(models, values, marker='o', linewidth=2.5, markersize=8, 
            label=metric_name, color=colors[i])

ax.set_xlabel('Model', fontweight='bold', fontsize=12)
ax.set_ylabel('Score', fontweight='bold', fontsize=12)
ax.set_title('Performance Metrics Trends - All Transformer Models (Y-axis: 0.0 to 1.0)', 
             fontweight='bold', fontsize=13, pad=15)
ax.legend(loc='lower right', ncol=2)
ax.set_ylim(0.0, 1.05)  # Start from 0.0
ax.grid(True, alpha=0.3, linestyle='--')

# Add value labels
for i, (metric_name, values) in enumerate(metrics.items()):
    for j, v in enumerate(values):
        ax.text(j, v + 0.02, f'{v:.3f}', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig('fig2_metrics_lines_full_scale.png', dpi=300, bbox_inches='tight')
print('✓ Saved: fig2_metrics_lines_full_scale.png')
plt.close()

# ============================================================
# FIGURE 3: F1-SCORE RANKING (Horizontal) - from 0.0
# ============================================================
fig, ax = plt.subplots(figsize=(10, 6))

f1_scores = metrics['F1-Score']
sorted_indices = sorted(range(len(f1_scores)), key=lambda i: f1_scores[i], reverse=True)
sorted_models = [models[i] for i in sorted_indices]
sorted_f1 = [f1_scores[i] for i in sorted_indices]
sorted_colors = [colors[i] for i in sorted_indices]

bars = ax.barh(sorted_models, sorted_f1, color=sorted_colors, alpha=0.85, height=0.6)

ax.set_xlabel('F1-Score', fontweight='bold', fontsize=12)
ax.set_title('Model Ranking by F1-Score (Y-axis: 0.0 to 1.0)', 
             fontweight='bold', fontsize=13, pad=15)
ax.set_xlim(0.0, 1.05)  # Start from 0.0

# Add value labels
for i, (model, f1) in enumerate(zip(sorted_models, sorted_f1)):
    ax.text(f1 + 0.01, i, f'{f1:.4f}', va='center', fontsize=10, fontweight='bold')

# Add rank badges
for i in range(len(sorted_models)):
    ax.text(-0.08, i, f'#{i+1}', ha='center', va='center', fontweight='bold', 
            fontsize=11, bbox=dict(boxstyle='circle', facecolor='yellow', alpha=0.7))

ax.grid(axis='x', alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig('fig3_f1_ranking_full_scale.png', dpi=300, bbox_inches='tight')
print('✓ Saved: fig3_f1_ranking_full_scale.png')
plt.close()

# ============================================================
# FIGURE 4: COMPREHENSIVE DASHBOARD - Full scale
# ============================================================
fig = plt.figure(figsize=(14, 10))
gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.3)

# Subplot 1: Accuracy comparison
ax1 = fig.add_subplot(gs[0, 0])
acc_values = metrics['Accuracy']
bars1 = ax1.bar(models, acc_values, color=colors, alpha=0.85, width=0.6)
ax1.set_ylabel('Accuracy', fontweight='bold')
ax1.set_title('Accuracy by Model', fontweight='bold', fontsize=11)
ax1.set_ylim(0.0, 1.05)
ax1.grid(axis='y', alpha=0.3)
for i, v in enumerate(acc_values):
    ax1.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=9)
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

# Subplot 2: F1-Score comparison
ax2 = fig.add_subplot(gs[0, 1])
f1_values = metrics['F1-Score']
bars2 = ax2.bar(models, f1_values, color=colors, alpha=0.85, width=0.6)
ax2.set_ylabel('F1-Score', fontweight='bold')
ax2.set_title('F1-Score by Model', fontweight='bold', fontsize=11)
ax2.set_ylim(0.0, 1.05)
ax2.grid(axis='y', alpha=0.3)
for i, v in enumerate(f1_values):
    ax2.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=9)
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

# Subplot 3: Precision vs Recall
ax3 = fig.add_subplot(gs[1, 0])
x_pos = np.arange(len(models))
width = 0.35
ax3.bar(x_pos - width/2, metrics['Precision'], width, label='Precision', color='#1f77b4', alpha=0.85)
ax3.bar(x_pos + width/2, metrics['Recall'], width, label='Recall', color='#ff7f0e', alpha=0.85)
ax3.set_ylabel('Score', fontweight='bold')
ax3.set_title('Precision vs Recall', fontweight='bold', fontsize=11)
ax3.set_xticks(x_pos)
ax3.set_xticklabels(models, rotation=45, ha='right')
ax3.set_ylim(0.0, 1.05)
ax3.legend()
ax3.grid(axis='y', alpha=0.3)

# Subplot 4: All metrics line chart
ax4 = fig.add_subplot(gs[1, 1])
for metric_name, values in metrics.items():
    ax4.plot(models, values, marker='o', linewidth=2, markersize=6, label=metric_name)
ax4.set_ylabel('Score', fontweight='bold')
ax4.set_title('All Metrics Trends', fontweight='bold', fontsize=11)
ax4.set_ylim(0.0, 1.05)
ax4.legend(fontsize=9, loc='lower right')
ax4.grid(True, alpha=0.3)
plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha='right')

# Subplot 5: Performance Summary Table
ax5 = fig.add_subplot(gs[2, :])
ax5.axis('tight')
ax5.axis('off')

table_data = []
for model in models:
    table_data.append([
        model,
        f"{metrics['Accuracy'][models.index(model)]:.4f}",
        f"{metrics['Precision'][models.index(model)]:.4f}",
        f"{metrics['Recall'][models.index(model)]:.4f}",
        f"{metrics['F1-Score'][models.index(model)]:.4f}"
    ])

# Sort by F1-Score (descending)
table_data.sort(key=lambda x: float(x[4]), reverse=True)

table = ax5.table(cellText=table_data, 
                  colLabels=['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score'],
                  cellLoc='center',
                  loc='center',
                  colWidths=[0.15, 0.2, 0.2, 0.2, 0.2])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2)

# Style header
for i in range(5):
    table[(0, i)].set_facecolor('#4472C4')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Alternate row colors
for i in range(1, len(table_data) + 1):
    color = '#E7E6E6' if i % 2 == 0 else 'white'
    for j in range(5):
        table[(i, j)].set_facecolor(color)

fig.suptitle('Comprehensive Training Results Dashboard - All Transformer Models (Full Scale: 0.0 to 1.0)', 
             fontsize=14, fontweight='bold', y=0.98)

plt.savefig('fig4_dashboard_full_scale.png', dpi=300, bbox_inches='tight')
print('✓ Saved: fig4_dashboard_full_scale.png')
plt.close()

# ============================================================
# FIGURE 5: SCATTER PLOT - Accuracy vs F1-Score
# ============================================================
fig, ax = plt.subplots(figsize=(10, 7))

accuracies = metrics['Accuracy']
f1_scores = metrics['F1-Score']

scatter = ax.scatter(accuracies, f1_scores, s=300, c=colors, alpha=0.7, edgecolors='black', linewidth=2)

# Add model labels
for i, model in enumerate(models):
    ax.annotate(model, (accuracies[i], f1_scores[i]), 
                xytext=(10, 10), textcoords='offset points',
                fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

ax.set_xlabel('Accuracy', fontweight='bold', fontsize=12)
ax.set_ylabel('F1-Score', fontweight='bold', fontsize=12)
ax.set_title('Model Performance: Accuracy vs F1-Score (Both axes: 0.0 to 1.0)', 
             fontweight='bold', fontsize=13, pad=15)
ax.set_xlim(0.0, 1.05)
ax.set_ylim(0.0, 1.05)
ax.grid(True, alpha=0.3, linestyle='--')

# Add diagonal reference line (perfect correlation)
ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, linewidth=1.5, label='Perfect Correlation')
ax.legend()

plt.tight_layout()
plt.savefig('fig5_accuracy_vs_f1_full_scale.png', dpi=300, bbox_inches='tight')
print('✓ Saved: fig5_accuracy_vs_f1_full_scale.png')
plt.close()

# ============================================================
# Print Summary
# ============================================================
print("\n" + "="*70)
print("VISUALIZATION COMPLETE - FULL SCALE (0.0 to 1.0)")
print("="*70)
print("\n📊 Generated Figures:")
print("  1. fig1_metrics_bars_full_scale.png    - Bar chart of all metrics")
print("  2. fig2_metrics_lines_full_scale.png   - Line chart of all metrics")
print("  3. fig3_f1_ranking_full_scale.png      - F1-Score ranking (horizontal)")
print("  4. fig4_dashboard_full_scale.png       - Comprehensive 4-panel dashboard")
print("  5. fig5_accuracy_vs_f1_full_scale.png  - Scatter plot comparison")
print("\n🏆 Best Model: DistilBERT")
print(f"   └─ F1-Score: {metrics['F1-Score'][models.index('DistilBERT')]:.4f} (99.15%)")
print(f"   └─ Accuracy: {metrics['Accuracy'][models.index('DistilBERT')]:.4f} (99.17%)")
print("\n📈 All Y-axes now start from 0.0 for better comparison")
print("="*70)
