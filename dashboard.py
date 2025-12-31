"""
Sentiment Analysis Dashboard
Interactive dashboard untuk visualisasi dan testing sentiment analysis models
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve, auc,
    precision_recall_curve, average_precision_score
)
import os
import json
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from sentiment_utils import SentimentAnalyzer, compare_models, ensemble_predict
import warnings

warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Sentiment Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        padding: 20px;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .header-title {
        color: #1f77b4;
        font-size: 2.5em;
        font-weight: bold;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="header-title">📊 Sentiment Analysis Dashboard</div>', unsafe_allow_html=True)

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select Page",
    ["📈 Overview", "🤖 Model Training", "🧪 Test Model", "📊 Model Comparison", "🎯 Predictions", "📉 Metrics Analysis"]
)

# ============================================================================
# PAGE 1: OVERVIEW
# ============================================================================
if page == "📈 Overview":
    st.header("Project Overview")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("📁 Data Status", "Ready", "Balanced & Cleaned")
    
    with col2:
        st.metric("🤖 Models", "5", "Configured")
    
    with col3:
        st.metric("📊 Dashboard", "Live", "Streamlit")
    
    st.divider()
    
    # Data Overview
    st.subheader("Data Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Original Data Statistics:**")
        original_stats = {
            'Total Samples': 1688,
            'Negative': 805,
            'Positive': 883,
            'Train-Test Split': '80-20',
            'Max Length': 128
        }
        for key, value in original_stats.items():
            st.write(f"• {key}: {value}")
    
    with col2:
        st.write("**Data Balancing Methods:**")
        balancing_methods = {
            'Original': 1688,
            'Oversampling': 1610,
            'Undersampling': 1610,
            'Hybrid': 1200
        }
        for method, size in balancing_methods.items():
            st.write(f"• {method}: {size} samples")
    
    st.divider()
    
    # Models Overview
    st.subheader("5 Transformer Models")
    
    models_info = {
        'BERT': {
            'Base': 'bert-base-uncased',
            'Size': '440MB',
            'Speed': 'Moderate',
            'Best For': 'General NLP',
            'Parameters': '110M'
        },
        'DistilBERT': {
            'Base': 'distilbert-base-uncased',
            'Size': '268MB',
            'Speed': 'Very Fast',
            'Best For': 'Fast Inference',
            'Parameters': '66M'
        },
        'RoBERTa': {
            'Base': 'roberta-base',
            'Size': '498MB',
            'Speed': 'Moderate',
            'Best For': 'Best Sentiment',
            'Parameters': '125M'
        },
        'ALBERT': {
            'Base': 'albert-base-v2',
            'Size': '48MB',
            'Speed': 'Very Fast',
            'Best For': 'Mobile/Edge',
            'Parameters': '11.7M'
        },
        'XLNET': {
            'Base': 'xlnet-base-cased',
            'Size': '1,360MB',
            'Speed': 'Moderate',
            'Best For': 'Complex Context',
            'Parameters': '340M'
        }
    }
    
    for model_name, info in models_info.items():
        with st.expander(f"🤖 {model_name} - {info['Best For']}"):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write(f"**Base Model:** {info['Base']}")
                st.write(f"**Parameters:** {info['Parameters']}")
            with col2:
                st.write(f"**Model Size:** {info['Size']}")
                st.write(f"**Speed:** {info['Speed']}")
            with col3:
                st.write(f"**Best For:** {info['Best For']}")

# ============================================================================
# PAGE 2: MODEL TRAINING
# ============================================================================
elif page == "🤖 Model Training":
    st.header("Model Training Status")
    
    st.info("""
    To train models, run in terminal:
    ```bash
    python train_transformer_models.py
    ```
    Training takes 30-60 minutes depending on GPU availability.
    """)
    
    # Check for training results
    results_file = 'training_results.json'
    comparison_file = 'model_performance_comparison.csv'
    
    if os.path.exists(results_file) and os.path.exists(comparison_file):
        st.success("✅ Models have been trained!")
        
        # Load results
        with open(results_file) as f:
            results = json.load(f)
        
        comparison_df = pd.read_csv(comparison_file)
        
        # Display results
        st.subheader("Training Results")
        
        col1, col2, col3, col4 = st.columns(4)
        
        # Sort by F1-Score
        comparison_df = comparison_df.sort_values('F1-Score', ascending=False)
        
        for idx, row in comparison_df.iterrows():
            with col1 if idx == 0 else col2 if idx == 1 else col3 if idx == 2 else col4:
                st.metric(
                    row['Model'],
                    f"{row['F1-Score']:.4f}",
                    f"Acc: {row['Accuracy']:.4f}"
                )
        
        # Detailed comparison
        st.subheader("Detailed Model Performance")
        st.dataframe(
            comparison_df,
            use_container_width=True,
            hide_index=True
        )
        
        # Visualizations
        st.subheader("Performance Metrics Visualization")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Accuracy comparison
            fig = px.bar(
                comparison_df.sort_values('Accuracy', ascending=True),
                x='Accuracy',
                y='Model',
                orientation='h',
                title='Accuracy by Model',
                color='Accuracy',
                color_continuous_scale='Blues'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # F1-Score comparison
            fig = px.bar(
                comparison_df.sort_values('F1-Score', ascending=True),
                x='F1-Score',
                y='Model',
                orientation='h',
                title='F1-Score by Model',
                color='F1-Score',
                color_continuous_scale='Greens'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Precision vs Recall
            fig = go.Figure()
            for idx, row in comparison_df.iterrows():
                fig.add_trace(go.Scatter(
                    x=[row['Precision']],
                    y=[row['Recall']],
                    mode='markers+text',
                    name=row['Model'],
                    text=row['Model'],
                    textposition='top center',
                    marker=dict(size=15)
                ))
            fig.update_layout(
                title='Precision vs Recall',
                xaxis_title='Precision',
                yaxis_title='Recall',
                hovermode='closest'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # All metrics radar
            fig = px.bar(
                comparison_df.sort_values('Model'),
                x='Model',
                y=['Accuracy', 'Precision', 'Recall', 'F1-Score'],
                title='All Metrics Comparison',
                barmode='group',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.warning("⚠️ Models not trained yet. Run training script first.")

# ============================================================================
# PAGE 3: TEST MODEL
# ============================================================================
elif page == "🧪 Test Model":
    st.header("Test Individual Model")
    
    # Check for trained models
    models_dir = './sentiment_models'
    if not os.path.exists(models_dir):
        st.error("❌ No trained models found. Train models first.")
    else:
        available_models = [d for d in os.listdir(models_dir) if os.path.isdir(os.path.join(models_dir, d))]
        
        if not available_models:
            st.error("❌ No trained models found in sentiment_models/")
        else:
            col1, col2 = st.columns([3, 1])
            
            with col1:
                selected_model = st.selectbox(
                    "Select Model",
                    available_models
                )
            
            with col2:
                use_gpu = st.checkbox("Use GPU", value=True)
            
            # Load model
            @st.cache_resource
            def load_model(model_name):
                model_path = f'./sentiment_models/{model_name}'
                return SentimentAnalyzer(model_path, use_gpu=use_gpu)
            
            try:
                analyzer = load_model(selected_model)
                
                st.success(f"✅ {selected_model} loaded successfully!")
                
                # Test input
                st.subheader("Make Predictions")
                
                input_type = st.radio("Input Type", ["Single Text", "Multiple Texts"])
                
                if input_type == "Single Text":
                    text_input = st.text_area(
                        "Enter text to analyze:",
                        height=150,
                        placeholder="Type your text here..."
                    )
                    
                    if st.button("🔍 Analyze", use_container_width=True):
                        if text_input:
                            result = analyzer.predict(text_input)
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.write("**Prediction Result:**")
                                st.markdown(f"""
                                **Sentiment:** {result['sentiment'].upper()}
                                **Confidence:** {result['confidence']:.4f}
                                """)
                            
                            with col2:
                                # Confidence gauge
                                fig = go.Figure(go.Indicator(
                                    mode="gauge+number+delta",
                                    value=result['confidence'] * 100,
                                    title={'text': "Confidence (%)"},
                                    domain={'x': [0, 1], 'y': [0, 1]},
                                    gauge={
                                        'axis': {'range': [0, 100]},
                                        'bar': {'color': "darkblue"},
                                        'steps': [
                                            {'range': [0, 50], 'color': "lightgray"},
                                            {'range': [50, 100], 'color': "gray"}
                                        ],
                                        'threshold': {
                                            'line': {'color': "red", 'width': 4},
                                            'thickness': 0.75,
                                            'value': 90
                                        }
                                    }
                                ))
                                st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.warning("Please enter some text!")
                
                else:  # Multiple texts
                    uploaded_file = st.file_uploader(
                        "Upload CSV file with 'text' column",
                        type=['csv']
                    )
                    
                    if uploaded_file:
                        df = pd.read_csv(uploaded_file)
                        
                        if 'text' not in df.columns:
                            st.error("CSV must have 'text' column")
                        else:
                            if st.button("🔍 Analyze All", use_container_width=True):
                                st.info("Processing texts...")
                                
                                predictions = analyzer.predict_batch(df['text'].tolist())
                                
                                results_df = df.copy()
                                results_df['predicted_sentiment'] = [p.get('sentiment', 'error') for p in predictions]
                                results_df['confidence'] = [p.get('confidence', 0) for p in predictions]
                                
                                st.success(f"✅ Processed {len(results_df)} texts!")
                                
                                st.dataframe(results_df, use_container_width=True)
                                
                                # Download results
                                csv = results_df.to_csv(index=False)
                                st.download_button(
                                    label="📥 Download Results",
                                    data=csv,
                                    file_name="predictions.csv",
                                    mime="text/csv"
                                )
            
            except Exception as e:
                st.error(f"❌ Error loading model: {str(e)}")

# ============================================================================
# PAGE 4: MODEL COMPARISON
# ============================================================================
elif page == "📊 Model Comparison":
    st.header("Compare All Models")
    
    comparison_file = 'model_performance_comparison.csv'
    
    if not os.path.exists(comparison_file):
        st.warning("⚠️ Run model training first to generate comparison data.")
    else:
        comparison_df = pd.read_csv(comparison_file).sort_values('F1-Score', ascending=False)
        
        st.subheader("Metrics Summary")
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)
        
        st.divider()
        
        # Detailed visualizations
        st.subheader("Detailed Performance Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Accuracy
            fig = px.bar(
                comparison_df,
                x='Model',
                y='Accuracy',
                title='Model Accuracy Comparison',
                color='Accuracy',
                color_continuous_scale='Blues',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # F1-Score
            fig = px.bar(
                comparison_df,
                x='Model',
                y='F1-Score',
                title='Model F1-Score Comparison',
                color='F1-Score',
                color_continuous_scale='Greens',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Precision
            fig = px.bar(
                comparison_df,
                x='Model',
                y='Precision',
                title='Model Precision Comparison',
                color='Precision',
                color_continuous_scale='Oranges',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Recall
            fig = px.bar(
                comparison_df,
                x='Model',
                y='Recall',
                title='Model Recall Comparison',
                color='Recall',
                color_continuous_scale='Reds',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Radar chart comparison
        st.subheader("Multi-Metric Radar Chart")
        
        fig = go.Figure()
        
        for idx, row in comparison_df.iterrows():
            fig.add_trace(go.Scatterpolar(
                r=[row['Accuracy'], row['Precision'], row['Recall'], row['F1-Score']],
                theta=['Accuracy', 'Precision', 'Recall', 'F1-Score'],
                fill='toself',
                name=row['Model']
            ))
        
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=True,
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# PAGE 5: MAKE PREDICTIONS
# ============================================================================
elif page == "🎯 Predictions":
    st.header("Batch Predictions & Ensemble")
    
    models_dir = './sentiment_models'
    
    if not os.path.exists(models_dir):
        st.error("❌ No trained models found.")
    else:
        available_models = [d for d in os.listdir(models_dir) if os.path.isdir(os.path.join(models_dir, d))]
        
        if not available_models:
            st.error("❌ No trained models available.")
        else:
            prediction_type = st.radio("Prediction Type", ["Single Model", "Ensemble Voting"])
            
            if prediction_type == "Single Model":
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    selected_model = st.selectbox("Select Model", available_models, key="single")
                
                with col2:
                    use_gpu = st.checkbox("Use GPU", value=True, key="gpu_single")
                
                text_input = st.text_area("Enter text:", height=200)
                
                if st.button("Predict", use_container_width=True):
                    if text_input:
                        try:
                            analyzer = SentimentAnalyzer(f'./sentiment_models/{selected_model}', use_gpu=use_gpu)
                            result = analyzer.predict(text_input)
                            
                            st.success("Prediction Complete!")
                            
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric("Model", selected_model)
                            
                            with col2:
                                st.metric("Sentiment", result['sentiment'].upper())
                            
                            with col3:
                                st.metric("Confidence", f"{result['confidence']:.2%}")
                            
                            # Confidence bar
                            st.progress(result['confidence'])
                        
                        except Exception as e:
                            st.error(f"Error: {str(e)}")
                    else:
                        st.warning("Please enter text!")
            
            else:  # Ensemble
                selected_models = st.multiselect(
                    "Select Models for Ensemble",
                    available_models,
                    default=available_models[:3]
                )
                
                voting_method = st.radio("Voting Method", ["majority", "confidence"])
                
                text_input = st.text_area("Enter text:", height=200)
                
                if st.button("Predict with Ensemble", use_container_width=True):
                    if text_input and selected_models:
                        try:
                            model_paths = {
                                model: f'./sentiment_models/{model}'
                                for model in selected_models
                            }
                            
                            result = ensemble_predict(text_input, model_paths, voting=voting_method)
                            
                            st.success("Ensemble Prediction Complete!")
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.metric("Voting Method", voting_method.upper())
                                st.metric("Ensemble Result", result['ensemble_prediction'].upper())
                            
                            with col2:
                                st.write("**Individual Model Predictions:**")
                                for model, pred in result['individual_predictions'].items():
                                    if 'error' not in pred:
                                        st.write(f"• {model}: {pred['sentiment'].upper()} ({pred['confidence']:.2%})")
                                    else:
                                        st.write(f"• {model}: ERROR")
                        
                        except Exception as e:
                            st.error(f"Error: {str(e)}")
                    else:
                        st.warning("Please select models and enter text!")

# ============================================================================
# PAGE 6: METRICS ANALYSIS
# ============================================================================
elif page == "📉 Metrics Analysis":
    st.header("Advanced Metrics Analysis")
    
    st.info("""
    This page shows detailed metrics analysis including:
    - ROC/AUC curves
    - Precision-Recall curves
    - Confusion matrices
    - Detailed performance breakdowns
    """)
    
    comparison_file = 'model_performance_comparison.csv'
    
    if not os.path.exists(comparison_file):
        st.warning("⚠️ Train models first to generate metrics.")
    else:
        comparison_df = pd.read_csv(comparison_file).sort_values('F1-Score', ascending=False)
        
        st.subheader("Performance Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Best Model", comparison_df.iloc[0]['Model'])
        
        with col2:
            st.metric("Best Accuracy", f"{comparison_df['Accuracy'].max():.4f}")
        
        with col3:
            st.metric("Best F1-Score", f"{comparison_df['F1-Score'].max():.4f}")
        
        with col4:
            st.metric("Best Precision", f"{comparison_df['Precision'].max():.4f}")
        
        st.divider()
        
        # Detailed metrics table
        st.subheader("Detailed Metrics Table")
        
        metrics_df = comparison_df[[
            'Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score'
        ]].copy()
        
        # Format as percentages
        for col in ['Accuracy', 'Precision', 'Recall', 'F1-Score']:
            metrics_df[col] = metrics_df[col].apply(lambda x: f"{x:.4f}")
        
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)
        
        st.divider()
        
        # Export results
        st.subheader("Export Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📊 Download Comparison CSV"):
                csv = comparison_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download CSV",
                    data=csv,
                    file_name="model_comparison.csv",
                    mime="text/csv"
                )
        
        with col2:
            if st.button("📈 Download as JSON"):
                if os.path.exists('training_results.json'):
                    with open('training_results.json', 'r') as f:
                        json_data = f.read()
                    st.download_button(
                        label="📥 Download JSON",
                        data=json_data,
                        file_name="training_results.json",
                        mime="application/json"
                    )

# Footer
st.divider()
st.markdown("""
---
**Sentiment Analysis Dashboard** | Version 1.0 | December 2025
Made with ❤️ using Streamlit
""", unsafe_allow_html=True)
