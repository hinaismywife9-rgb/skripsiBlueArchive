import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Data untuk September - Januari (5 bulan)
data = {
    'No': ['', '1', '2', '3', '', '4', '5', '', '6', '7', '8', '', '9', '10', '11', '', '12', '13', '14'],
    'Aktivitas': [
        'TAHAP 1: ANALISIS & PERSIAPAN',
        'Identifikasi Masalah',
        'Latar Belakang', 
        'Rumusan Masalah',
        'TAHAP 2: PENGUMPULAN DATA',
        'Pengumpulan Data',
        'Data Scrapping',
        'TAHAP 3: PENGEMBANGAN & PREPROCESSING',
        'Pengembangan',
        'Data Preprocessing',
        'Data Labelling',
        'TAHAP 4: MODEL TRAINING & EVALUATION',
        'Models Training',
        'Evaluasi',
        'Models Testing',
        'TAHAP 5: IMPLEMENTASI & DEPLOYMENT',
        'Implementasi',
        'Develop Aplikasi Web',
        'Deploy Aplikasi Web'
    ],
    'Sep': ['', '●', '●', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', ''],
    'Okt': ['', '', '●', '●', '', '●', '●', '', '', '', '', '', '', '', '', '', '', '', ''],
    'Nov': ['', '', '', '●', '', '●', '●', '', '●', '●', '', '', '', '', '', '', '', '', ''],
    'Des': ['', '', '', '', '', '', '●', '', '●', '●', '●', '', '●', '', '', '', '', '', ''],
    'Jan': ['', '', '', '', '', '', '', '', '', '●', '●', '', '●', '●', '●', '', '●', '●', '●'],
    'Durasi': ['', '4 mgg', '6 mgg', '6 mgg', '', '8 mgg', '10 mgg', '', '6 mgg', '8 mgg', '6 mgg', '', '6 mgg', '4 mgg', '4 mgg', '', '4 mgg', '4 mgg', '4 mgg']
}

# Create figure with larger size
fig, ax = plt.subplots(figsize=(14, 12))
ax.axis('off')

# Title
fig.suptitle('Tabel 3.5.1 Perencanaan Penelitian\nSentiment Analysis Berbasis Transformer Models', 
             fontsize=16, fontweight='bold', y=0.98)

# Create table
columns = ['No', 'Aktivitas', 'Sep', 'Okt', 'Nov', 'Des', 'Jan', 'Durasi']
cell_text = []
cell_colors = []

# Phase header colors
phase_color = '#E8F4FD'
normal_color = 'white'
header_color = '#2E86AB'

for i in range(len(data['No'])):
    row = [data[col][i] for col in columns]
    cell_text.append(row)
    
    # Determine row color
    if data['No'][i] == '' and 'TAHAP' in data['Aktivitas'][i]:
        cell_colors.append([phase_color] * len(columns))
    else:
        cell_colors.append([normal_color] * len(columns))

# Create table
table = ax.table(
    cellText=cell_text,
    colLabels=columns,
    cellColours=cell_colors,
    colColours=[header_color] * len(columns),
    cellLoc='center',
    loc='center',
    colWidths=[0.05, 0.35, 0.08, 0.08, 0.08, 0.08, 0.08, 0.1]
)

# Style the table
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.2, 1.8)

# Make header text white
for j in range(len(columns)):
    table[(0, j)].get_text().set_color('white')
    table[(0, j)].get_text().set_fontweight('bold')

# Make phase headers bold
for i in range(len(data['No'])):
    if data['No'][i] == '' and 'TAHAP' in data['Aktivitas'][i]:
        for j in range(len(columns)):
            table[(i+1, j)].get_text().set_fontweight('bold')
            table[(i+1, j)].get_text().set_fontsize(8)

# Left align activity column
for i in range(len(data['No']) + 1):
    table[(i, 1)].get_text().set_ha('left')

# Add legend
legend_text = """
Keterangan:
● = Aktivitas berlangsung pada bulan tersebut
Total Durasi: 5 bulan (September 2025 - Januari 2026)
Critical Path: Data Collection → Preprocessing → Labelling → Training → Evaluation → Deployment
"""
fig.text(0.1, 0.02, legend_text, fontsize=9, verticalalignment='bottom',
         bbox=dict(boxstyle='round', facecolor='#f0f0f0', alpha=0.8))

plt.tight_layout()
plt.subplots_adjust(top=0.92, bottom=0.12)
plt.savefig('tabel_perencanaan_penelitian.png', dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
print("✅ Tabel saved: tabel_perencanaan_penelitian.png")
plt.show()
