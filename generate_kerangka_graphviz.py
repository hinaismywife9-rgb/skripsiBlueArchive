#!/usr/bin/env python3
"""
Generate Kerangka Berpikir diagram using Graphviz
Clean, professional, for thesis use
"""

from graphviz import Digraph

# Create digraph
dot = Digraph(comment='Kerangka Berpikir Penelitian', format='png', engine='dot')
dot.attr('graph', rankdir='TB', splines='ortho', nodesep='0.5', ranksep='0.7')
dot.attr('node', shape='box', style='rounded,filled', fillcolor='white', color='black', fontname='Arial', fontsize='10')
dot.attr('edge', color='black', fontsize='9')

# Phase 1 header
dot.node('phase1', 'Fase 1: Pengumpulan Data - Evaluasi', shape='box', style='filled', fillcolor='#4A90E2', fontcolor='white', fontweight='bold')

# Start
dot.node('start', 'Start', shape='ellipse', fillcolor='#E8F4F8')

# Data Selection
dot.node('data_selection', 'Data Selection\n(1,200 Samples)', fillcolor='#E8F4F8')

# Data Preprocessing
dot.node('preprocessing', 'Data Preprocessing\n(Cleaning, Tokenization)', fillcolor='#E8F4F8')

# Labelling
dot.node('labelling', 'Labelling\n(Positive/Negative)', fillcolor='#E8F4F8')

# Data Split
dot.node('split', 'Data Split\n(80% Train = 960\n20% Test = 240)', fillcolor='#FFF4E6')

# 5 Model Training
dot.node('training', '5 Model Training\nBERT | DistilBERT\nRoBERTa | ALBERT | XLNET', shape='box', fillcolor='#F0E6FF')

# Evaluation
dot.node('evaluation', 'Evaluation\n(Accuracy, Precision\nRecall, F1-Score)', fillcolor='#F0E6FF')

# Select Top 3
dot.node('top3', 'Select Top 3 Models\nDistilBERT | XLNET | BERT', shape='box', fillcolor='#FFE6E6', fontweight='bold')

# Phase 2 header
dot.node('phase2', 'Fase 2: Implementasi', shape='box', style='filled', fillcolor='#50C878', fontcolor='white', fontweight='bold')

# Streamlit Development
dot.node('streamlit', 'Develop Web Application\n(Streamlit)', fillcolor='#E6F9E6')

# Dashboard & Features
dot.node('features', 'Features:\n• Single Prediction\n• Batch Processing\n• Model Comparison\n• WordCloud\n• History', fillcolor='#E6F9E6')

# Docker Deployment
dot.node('docker', 'Deploy (Docker)\nContainerization & Cloud', fillcolor='#E6F9E6')

# End
dot.node('end', 'End', shape='ellipse', fillcolor='#E8F4F8')

# Edges - Flow
dot.edge('start', 'phase1')
dot.edge('phase1', 'data_selection')
dot.edge('data_selection', 'preprocessing')
dot.edge('preprocessing', 'labelling')
dot.edge('labelling', 'split')
dot.edge('split', 'training')
dot.edge('training', 'evaluation')
dot.edge('evaluation', 'top3')
dot.edge('top3', 'phase2')
dot.edge('phase2', 'streamlit')
dot.edge('streamlit', 'features')
dot.edge('features', 'docker')
dot.edge('docker', 'end')

# Render
output_path = 'kerangka_berpikir_graphviz'
dot.render(output_path, cleanup=True, quiet=False)
print(f"✅ Gambar saved: {output_path}.png")
