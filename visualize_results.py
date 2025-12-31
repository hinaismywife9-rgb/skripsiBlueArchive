"""
Visualize Training Results - Create Charts and Save Images
"""

import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import numpy as np

# Load results
with open('training_results.json', 'r') as f:
    results = json.load(f)

# Prepare data
models = list(results.keys())
accuracies = [results[m]['acc'] for m in models]
precisions = [results[m]['prec'] for m in models]
recalls = [results[m]['rec'] for m in models]
f1_scores = [results[m]['f1'] for m in models]

# Sort by F1 score
sorted_idx = np.argsort(f1_scores)[::-1]
models_sorted = [models[i] for i in sorted_idx]
acc_sorted = [accuracies[i] for i in sorted_idx]
prec_sorted = [precisions[i] for i in sorted_idx]
rec_sorted = [recalls[i] for i in sorted_idx]
f1_sorted = [f1_scores[i] for i in sorted_idx]

# Color palette
colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E']

print("Creating visualizations...")

# Figure 1: Line Chart - Model Performance Comparison
fig1, ax1 = plt.subplots(figsize=(12, 6))

x_pos = np.arange(len(models_sorted))
width = 0.2

bars1 = ax1.bar(x_pos - 1.5*width, acc_sorted, width, label='Accuracy', color='#2E86AB', alpha=0.8)
bars2 = ax1.bar(x_pos - 0.5*width, prec_sorted, width, label='Precision', color='#A23B72', alpha=0.8)
bars3 = ax1.bar(x_pos + 0.5*width, rec_sorted, width, label='Recall', color='#F18F01', alpha=0.8)
bars4 = ax1.bar(x_pos + 1.5*width, f1_sorted, width, label='F1-Score', color='#6A994E', alpha=0.8)

ax1.set_xlabel('Model', fontsize=12, fontweight='bold')
ax1.set_ylabel('Score', fontsize=12, fontweight='bold')
ax1.set_title('Transformer Models - Performance Metrics Comparison', fontsize=14, fontweight='bold')
ax1.set_xticks(x_pos)
ax1.set_xticklabels(models_sorted)
ax1.legend(loc='lower right', fontsize=10)
ax1.set_ylim([0.90, 1.0])
ax1.grid(axis='y', alpha=0.3, linestyle='--')

# Add value labels on bars
for bars in [bars1, bars2, bars3, bars4]:
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig('model_comparison_bars.png', dpi=300, bbox_inches='tight')
print("✓ Saved: model_comparison_bars.png")
plt.close()

# Figure 2: Line Chart - F1 Score Trend
fig2, ax2 = plt.subplots(figsize=(12, 6))

# Plot lines for each metric
line1 = ax2.plot(x_pos, acc_sorted, marker='o', linewidth=2.5, markersize=8, 
                 label='Accuracy', color='#2E86AB')
line2 = ax2.plot(x_pos, prec_sorted, marker='s', linewidth=2.5, markersize=8, 
                 label='Precision', color='#A23B72')
line3 = ax2.plot(x_pos, rec_sorted, marker='^', linewidth=2.5, markersize=8, 
                 label='Recall', color='#F18F01')
line4 = ax2.plot(x_pos, f1_sorted, marker='D', linewidth=2.5, markersize=8, 
                 label='F1-Score', color='#6A994E')

ax2.set_xlabel('Model', fontsize=12, fontweight='bold')
ax2.set_ylabel('Score', fontsize=12, fontweight='bold')
ax2.set_title('Model Performance - Line Chart Comparison', fontsize=14, fontweight='bold')
ax2.set_xticks(x_pos)
ax2.set_xticklabels(models_sorted)
ax2.legend(loc='lower right', fontsize=11)
ax2.set_ylim([0.90, 1.0])
ax2.grid(True, alpha=0.3, linestyle='--')

# Add value labels on points
for i, model in enumerate(models_sorted):
    ax2.text(i, acc_sorted[i] + 0.002, f'{acc_sorted[i]:.4f}', ha='center', fontsize=8)
    ax2.text(i, f1_sorted[i] - 0.003, f'{f1_sorted[i]:.4f}', ha='center', fontsize=8)

plt.tight_layout()
plt.savefig('model_performance_lines.png', dpi=300, bbox_inches='tight')
print("✓ Saved: model_performance_lines.png")
plt.close()

# Figure 3: F1-Score Only - Ranking
fig3, ax3 = plt.subplots(figsize=(10, 6))

bars = ax3.barh(models_sorted, f1_sorted, color=colors, alpha=0.85, edgecolor='black', linewidth=1.5)

ax3.set_xlabel('F1-Score', fontsize=12, fontweight='bold')
ax3.set_title('Model Rankings by F1-Score', fontsize=14, fontweight='bold')
ax3.set_xlim([0.94, 0.995])
ax3.grid(axis='x', alpha=0.3, linestyle='--')

# Add ranking numbers and scores
for i, (model, f1) in enumerate(zip(models_sorted, f1_sorted)):
    ax3.text(f1 + 0.0005, i, f'#{i+1} ({f1:.4f})', va='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('model_f1_ranking.png', dpi=300, bbox_inches='tight')
print("✓ Saved: model_f1_ranking.png")
plt.close()

# Figure 4: Comprehensive Dashboard
fig4 = plt.figure(figsize=(16, 10))
gs = GridSpec(2, 2, figure=fig4, hspace=0.3, wspace=0.3)

# Subplot 1: Bar chart
ax4_1 = fig4.add_subplot(gs[0, :])
x_pos = np.arange(len(models_sorted))
width = 0.2
ax4_1.bar(x_pos - 1.5*width, acc_sorted, width, label='Accuracy', color='#2E86AB', alpha=0.8)
ax4_1.bar(x_pos - 0.5*width, prec_sorted, width, label='Precision', color='#A23B72', alpha=0.8)
ax4_1.bar(x_pos + 0.5*width, rec_sorted, width, label='Recall', color='#F18F01', alpha=0.8)
ax4_1.bar(x_pos + 1.5*width, f1_sorted, width, label='F1-Score', color='#6A994E', alpha=0.8)
ax4_1.set_ylabel('Score', fontweight='bold')
ax4_1.set_title('Performance Metrics Comparison', fontweight='bold')
ax4_1.set_xticks(x_pos)
ax4_1.set_xticklabels(models_sorted)
ax4_1.legend(loc='lower right')
ax4_1.set_ylim([0.90, 1.0])
ax4_1.grid(axis='y', alpha=0.3)

# Subplot 2: F1 Score ranking
ax4_2 = fig4.add_subplot(gs[1, 0])
bars2 = ax4_2.barh(models_sorted, f1_sorted, color=colors, alpha=0.85)
ax4_2.set_xlabel('F1-Score', fontweight='bold')
ax4_2.set_title('F1-Score Ranking', fontweight='bold')
ax4_2.set_xlim([0.94, 0.995])
for i, (model, f1) in enumerate(zip(models_sorted, f1_sorted)):
    ax4_2.text(f1 + 0.0005, i, f'{f1:.4f}', va='center', fontsize=9)
ax4_2.grid(axis='x', alpha=0.3)

# Subplot 3: Summary Table
ax4_3 = fig4.add_subplot(gs[1, 1])
ax4_3.axis('tight')
ax4_3.axis('off')

table_data = []
table_data.append(['Rank', 'Model', 'Accuracy', 'F1-Score'])
for i, model in enumerate(models_sorted, 1):
    idx = models.index(model)
    table_data.append([
        f'#{i}',
        model,
        f'{accuracies[idx]:.4f}',
        f'{f1_scores[idx]:.4f}'
    ])

table = ax4_3.table(cellText=table_data, cellLoc='center', loc='center',
                    colWidths=[0.15, 0.30, 0.25, 0.25])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2)

# Style header row
for i in range(4):
    table[(0, i)].set_facecolor('#2E86AB')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Style data rows with alternating colors
for i in range(1, len(table_data)):
    color = '#F0F0F0' if i % 2 == 0 else 'white'
    for j in range(4):
        table[(i, j)].set_facecolor(color)

ax4_3.set_title('Performance Summary', fontweight='bold', pad=20)

plt.suptitle('Transformer Models - Sentiment Analysis Results', 
             fontsize=16, fontweight='bold', y=0.98)

plt.savefig('training_results_dashboard.png', dpi=300, bbox_inches='tight')
print("✓ Saved: training_results_dashboard.png")
plt.close()

# Figure 5: Accuracy vs F1-Score Scatter
fig5, ax5 = plt.subplots(figsize=(10, 7))

scatter = ax5.scatter(acc_sorted, f1_sorted, s=300, alpha=0.7, c=range(len(models_sorted)), 
                     cmap='viridis', edgecolors='black', linewidth=2)

for i, model in enumerate(models_sorted):
    ax5.annotate(model, (acc_sorted[i], f1_sorted[i]), 
                xytext=(5, 5), textcoords='offset points', fontsize=11, fontweight='bold')

ax5.set_xlabel('Accuracy', fontsize=12, fontweight='bold')
ax5.set_ylabel('F1-Score', fontsize=12, fontweight='bold')
ax5.set_title('Accuracy vs F1-Score Relationship', fontsize=14, fontweight='bold')
ax5.grid(True, alpha=0.3, linestyle='--')
ax5.set_xlim([0.935, 1.0])
ax5.set_ylim([0.945, 1.0])

plt.tight_layout()
plt.savefig('accuracy_vs_f1_scatter.png', dpi=300, bbox_inches='tight')
print("✓ Saved: accuracy_vs_f1_scatter.png")
plt.close()

print("\n" + "=" * 80)
print("✓ ALL VISUALIZATIONS CREATED SUCCESSFULLY!")
print("=" * 80)
print("\nGenerated Files:")
print("  1. model_comparison_bars.png - Bar chart comparing all metrics")
print("  2. model_performance_lines.png - Line chart showing trends")
print("  3. model_f1_ranking.png - Horizontal bar chart ranking by F1")
print("  4. training_results_dashboard.png - Comprehensive dashboard")
print("  5. accuracy_vs_f1_scatter.png - Scatter plot comparison")
print("\n" + "=" * 80)

# Print summary
print("\nTRAINING RESULTS SUMMARY:")
print("-" * 80)
for i, model in enumerate(models_sorted, 1):
    idx = models.index(model)
    print(f"{i}. {model:15} | Acc: {accuracies[idx]:.4f} | "
          f"Prec: {precisions[idx]:.4f} | Rec: {recalls[idx]:.4f} | F1: {f1_scores[idx]:.4f}")

print("\n" + "=" * 80)
print(f"✓ BEST MODEL: {models_sorted[0]} (F1-Score: {f1_sorted[0]:.4f})")
print("=" * 80)
