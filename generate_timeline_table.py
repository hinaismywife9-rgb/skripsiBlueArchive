import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np

# Data
activities = [
    "Identifikasi Masalah",
    "Latar Belakang",
    "Rumusan Masalah",
    "Pengumpulan Data",
    "Data Scrapping",
    "Pengembangan",
    "Data Preprocessing",
    "Data Labelling",
    "Models Training",
    "Evaluasi",
    "Models Testing",
    "Implementasi",
    "Develop Aplikasi Web",
    "Deploy Aplikasi Web"
]

phases = [
    "TAHAP 1: ANALISIS & PERSIAPAN",
    "TAHAP 1: ANALISIS & PERSIAPAN",
    "TAHAP 1: ANALISIS & PERSIAPAN",
    "TAHAP 2: PENGUMPULAN DATA",
    "TAHAP 2: PENGUMPULAN DATA",
    "TAHAP 3: PENGEMBANGAN & PREPROCESSING",
    "TAHAP 3: PENGEMBANGAN & PREPROCESSING",
    "TAHAP 3: PENGEMBANGAN & PREPROCESSING",
    "TAHAP 4: MODEL TRAINING & EVALUATION",
    "TAHAP 4: MODEL TRAINING & EVALUATION",
    "TAHAP 4: MODEL TRAINING & EVALUATION",
    "TAHAP 5: IMPLEMENTASI & DEPLOYMENT",
    "TAHAP 5: IMPLEMENTASI & DEPLOYMENT",
    "TAHAP 5: IMPLEMENTASI & DEPLOYMENT"
]

# Start month (0=Sep) and duration in weeks
timelines = [
    (0, 4, "Sep 1-4"),      # 1
    (0, 6, "Sep 1-4, Okt 1-2"),  # 2
    (2, 6, "Okt 1-4, Nov 1-2"),  # 3
    (2, 8, "Okt 1-4, Nov 1-4"),  # 4
    (2, 10, "Okt 1-4, Nov 1-4, Des 1-2"),  # 5
    (4, 8, "Nov 1-4, Des 1-4"),  # 6
    (4, 10, "Nov 1-4, Des 1-4, Jan 1-2"),  # 7
    (5, 10, "Des 1-4, Jan 1-4, Feb 1-2"),  # 8
    (6, 10, "Jan 1-4, Feb 1-4, Mar 1-2"),  # 9
    (7, 10, "Feb 1-4, Mar 1-4, Apr 1-2"),  # 10
    (8, 10, "Mar 1-4, Apr 1-4, Mei 1-2"),  # 11
    (9, 10, "Apr 1-4, Mei 1-4, Jun 1-2"),  # 12
    (9, 14, "Apr 1-4, Mei 1-4, Jun 1-4, Jul 1-2"),  # 13
    (11, 8, "Jun 1-4, Jul 1-4"),  # 14
]

durations = ["4 minggu", "6 minggu", "6 minggu", "8 minggu", "10 minggu", 
             "8 minggu", "10 minggu", "10 minggu", "10 minggu", "10 minggu", 
             "10 minggu", "10 minggu", "14 minggu", "8 minggu"]

# Colors for phases
phase_colors = {
    "TAHAP 1: ANALISIS & PERSIAPAN": "#FF6B6B",
    "TAHAP 2: PENGUMPULAN DATA": "#4ECDC4",
    "TAHAP 3: PENGEMBANGAN & PREPROCESSING": "#45B7D1",
    "TAHAP 4: MODEL TRAINING & EVALUATION": "#FFA07A",
    "TAHAP 5: IMPLEMENTASI & DEPLOYMENT": "#98D8C8"
}

# Create figure
fig, ax = plt.subplots(figsize=(16, 12))
ax.set_xlim(0, 20)
ax.set_ylim(-1, len(activities) + 1)
ax.axis('off')

# Title
fig.text(0.5, 0.98, 'TABEL 3.5.1 PERENCANAAN PENELITIAN', 
         ha='center', fontsize=20, fontweight='bold')
fig.text(0.5, 0.945, 'Sentiment Analysis Sistem Berbasis Transformer Models', 
         ha='center', fontsize=12, style='italic')

# Header
months = ['Sep', 'Okt', 'Nov', 'Des', 'Jan', 'Feb', 'Mar', 'Apr', 'Mei', 'Jun', 'Jul']
header_y = len(activities) + 0.5

# Draw header row
for i, month in enumerate(months):
    x = 2.5 + (i * 1.5)
    ax.text(x, header_y, month, ha='center', va='bottom', fontweight='bold', fontsize=11)

# Draw month columns
for i in range(11):
    x = 2.5 + (i * 1.5)
    ax.plot([x-0.7, x-0.7], [-0.5, len(activities)], 'k-', linewidth=0.5, alpha=0.3)

# Row counter
current_phase = None
row = len(activities) - 1

# Draw each activity row
for idx, (activity, phase, timeline, duration) in enumerate(zip(activities, phases, timelines, durations)):
    y = row
    
    # Phase header (grouped)
    if phase != current_phase:
        # Draw phase background
        ax.add_patch(FancyBboxPatch((0.1, y-0.35), 18.8, 0.6, 
                                   boxstyle="round,pad=0.05", 
                                   edgecolor='black', facecolor=phase_colors[phase], 
                                   alpha=0.3, linewidth=1.5))
        ax.text(0.3, y, phase, fontsize=11, fontweight='bold', va='center')
        current_phase = phase
        row -= 1
        y = row
    
    # Row number
    ax.text(0.5, y, str(idx+1), ha='center', va='center', fontweight='bold', fontsize=10)
    
    # Activity name
    ax.text(1.3, y, activity, ha='left', va='center', fontsize=10)
    
    # Timeline
    start_month, weeks, timeline_str = timeline
    ax.text(2.5 + start_month * 1.5, y, '█' * (weeks // 2), 
            ha='left', va='center', fontsize=14, color='#2E86AB')
    
    # Duration
    ax.text(18.5, y, duration, ha='left', va='center', fontsize=9, style='italic')
    
    # Draw row separator
    ax.plot([0.1, 18.9], [y-0.4, y-0.4], 'k-', linewidth=0.5)
    
    row -= 1

# Draw border
ax.add_patch(FancyBboxPatch((0.1, -0.5), 18.8, len(activities) + 0.9, 
                           boxstyle="round,pad=0.05", 
                           edgecolor='black', facecolor='none', linewidth=2))

# Legend
legend_y = -0.7
ax.text(0.5, legend_y, 'Keterangan:', fontweight='bold', fontsize=10)
ax.text(0.5, legend_y - 0.35, '█ = 2 minggu', fontsize=9)
ax.text(3, legend_y - 0.35, 'Critical Path: Data Collection → Preprocessing → Labelling → Training → Evaluation → Deployment', 
        fontsize=9, style='italic')

# Milestone markers
milestone_y = -1.5
ax.text(0.5, milestone_y, 'MILESTONE:', fontweight='bold', fontsize=10)
milestones_text = [
    "Akhir Oktober: Data collection selesai",
    "Akhir November: Preprocessing & labelling complete",
    "Akhir Januari: Training & evaluation complete",
    "Akhir Juni: Web app developed",
    "Akhir Juli: Production deployment ready"
]
for i, milestone in enumerate(milestones_text):
    ax.text(0.5, milestone_y - 0.35 - (i * 0.35), f"• {milestone}", fontsize=9)

plt.tight_layout()
plt.savefig('tabel_perencanaan_penelitian.png', dpi=300, bbox_inches='tight', facecolor='white')
print("✅ Tabel saved: tabel_perencanaan_penelitian.png")
plt.show()
