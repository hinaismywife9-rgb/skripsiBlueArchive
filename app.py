import streamlit as st
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import json
from datetime import datetime
import os

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="🎯 Sentiment Analysis Dashboard",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS
# ============================================================
st.markdown("""
    <style>
    .main-title {
        text-align: center;
        color: #1f77b4;
        font-size: 3em;
        font-weight: bold;
        margin-bottom: 10px;
    }
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1.2em;
        margin-bottom: 30px;
    }
    .prediction-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    .confidence-high {
        color: #2ca02c;
        font-weight: bold;
    }
    .confidence-medium {
        color: #ff7f0e;
        font-weight: bold;
    }
    .confidence-low {
        color: #d62728;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================
# LOAD MODELS (CACHED)
# ============================================================
@st.cache_resource
def load_models():
    """Load all 5 trained models"""
    models = {}
    tokenizers = {}
    
    model_names = ['BERT', 'DistilBERT', 'RoBERTa', 'ALBERT', 'XLNET']
    model_paths = {
        'BERT': './sentiment_models/BERT',
        'DistilBERT': './sentiment_models/DistilBERT',
        'RoBERTa': './sentiment_models/RoBERTa',
        'ALBERT': './sentiment_models/ALBERT',
        'XLNET': './sentiment_models/XLNET'
    }
    
    try:
        for model_name in model_names:
            path = model_paths[model_name]
            if os.path.exists(path):
                tokenizers[model_name] = AutoTokenizer.from_pretrained(path)
                models[model_name] = AutoModelForSequenceClassification.from_pretrained(path)
                models[model_name].eval()
    except Exception as e:
        st.error(f"Error loading models: {e}")
    
    return models, tokenizers

# ============================================================
# PREDICTION FUNCTION
# ============================================================
def predict_sentiment(text, model_name, tokenizer, model):
    """Predict sentiment for a single text"""
    try:
        inputs = tokenizer(text, return_tensors="pt", max_length=128, 
                          truncation=True, padding=True)
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1)
        
        prediction = torch.argmax(logits, dim=1).item()
        confidence = probabilities.max().item()
        
        sentiment = "POSITIVE" if prediction == 1 else "NEGATIVE"
        
        return {
            'sentiment': sentiment,
            'confidence': confidence,
            'prob_negative': probabilities[0][0].item(),
            'prob_positive': probabilities[0][1].item()
        }
    except Exception as e:
        return None

# ============================================================
# LOAD PREDICTION HISTORY
# ============================================================
def load_history():
    """Load prediction history from JSON file"""
    history_file = 'prediction_history.json'
    if os.path.exists(history_file):
        with open(history_file, 'r') as f:
            return json.load(f)
    return []

def save_history(history):
    """Save prediction history to JSON file"""
    with open('prediction_history.json', 'w') as f:
        json.dump(history, f, indent=2)

# ============================================================
# MAIN APP
# ============================================================

# Title
st.markdown('<h1 class="main-title">🎯 SENTIMENT ANALYSIS DASHBOARD</h1>', 
            unsafe_allow_html=True)
st.markdown('<p class="subtitle">Analyze sentiment using 5 transformer models</p>', 
            unsafe_allow_html=True)

# Load models
with st.spinner('Loading models...'):
    models, tokenizers = load_models()

if not models:
    st.error("❌ No models found. Please ensure models are trained and saved.")
    st.stop()

st.success(f"✅ Loaded {len(models)} models: {', '.join(models.keys())}")

# ============================================================
# SIDEBAR - MODEL SELECTION & SETTINGS
# ============================================================
with st.sidebar:
    st.header("⚙️ Settings")
    
    selected_model = st.selectbox(
        "Select Model",
        list(models.keys()),
        help="Choose which model to use for prediction"
    )
    
    st.divider()
    
    # Model info
    st.subheader("📊 Model Performance")
    model_perf = {
        'BERT': {'F1': 98.70, 'Acc': 98.75},
        'DistilBERT': {'F1': 99.15, 'Acc': 99.17},
        'RoBERTa': {'F1': 95.12, 'Acc': 95.00},
        'ALBERT': {'F1': 98.28, 'Acc': 98.33},
        'XLNET': {'F1': 99.14, 'Acc': 99.17}
    }
    
    perf = model_perf[selected_model]
    col1, col2 = st.columns(2)
    col1.metric("F1-Score", f"{perf['F1']}%")
    col2.metric("Accuracy", f"{perf['Acc']}%")
    
    st.divider()
    
    # Instructions
    st.subheader("📖 How to Use")
    st.markdown("""
    1. **Single Prediction**: Enter text in the main area
    2. **Batch Prediction**: Upload a CSV file
    3. **View History**: See past predictions
    4. **Model Comparison**: Compare all models side-by-side
    """)

# ============================================================
# MAIN TABS
# ============================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔮 Single Prediction",
    "📁 Batch Prediction",
    "📈 History & Analytics",
    "🔄 Model Comparison",
    "☁️ WordCloud"
])

# ============================================================
# TAB 1: SINGLE PREDICTION
# ============================================================
with tab1:
    st.header("🔮 Single Text Prediction")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        text_input = st.text_area(
            "Enter text to analyze:",
            placeholder="Type or paste your text here...",
            height=150,
            key="single_text"
        )
    
    with col2:
        st.write("")
        st.write("")
        st.write("")
        predict_button = st.button("🔍 Predict", key="predict_btn", use_container_width=True)
    
    if predict_button and text_input:
        result = predict_sentiment(
            text_input,
            selected_model,
            tokenizers[selected_model],
            models[selected_model]
        )
        
        if result:
            # Save to history
            history = load_history()
            history.append({
                'timestamp': datetime.now().isoformat(),
                'text': text_input,
                'model': selected_model,
                'sentiment': result['sentiment'],
                'confidence': round(result['confidence'], 4)
            })
            save_history(history)
            
            # Display result
            st.divider()
            
            col1, col2, col3 = st.columns(3)
            
            # Sentiment badge
            with col1:
                if result['sentiment'] == 'POSITIVE':
                    st.success(f"😊 {result['sentiment']}", icon="✅")
                else:
                    st.error(f"😞 {result['sentiment']}", icon="❌")
            
            # Confidence
            with col2:
                confidence_pct = result['confidence'] * 100
                st.metric("Confidence", f"{confidence_pct:.2f}%")
            
            # Model used
            with col3:
                st.info(f"Model: {selected_model}")
            
            # Probability distribution
            st.divider()
            st.subheader("📊 Prediction Probabilities")
            
            fig, ax = plt.subplots(figsize=(10, 4))
            sentiments = ['Negative', 'Positive']
            probs = [result['prob_negative'], result['prob_positive']]
            colors = ['#d62728', '#2ca02c']
            
            bars = ax.barh(sentiments, probs, color=colors, alpha=0.7)
            ax.set_xlabel('Probability', fontweight='bold')
            ax.set_xlim(0, 1)
            
            # Add percentage labels
            for i, (bar, prob) in enumerate(zip(bars, probs)):
                ax.text(prob + 0.02, i, f'{prob*100:.2f}%', 
                       va='center', fontweight='bold')
            
            ax.grid(axis='x', alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)

# ============================================================
# TAB 2: BATCH PREDICTION
# ============================================================
with tab2:
    st.header("📁 Batch Prediction")
    
    st.markdown("""
    Upload a CSV file with a 'text' column to predict sentiment for multiple texts.
    """)
    
    uploaded_file = st.file_uploader("Choose CSV file", type=['csv'])
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        
        if 'text' not in df.columns:
            st.error("❌ CSV must have a 'text' column")
        else:
            st.info(f"📊 Loaded {len(df)} texts for prediction")
            
            if st.button("🔍 Predict All"):
                predictions = []
                progress_bar = st.progress(0)
                
                for idx, row in df.iterrows():
                    result = predict_sentiment(
                        row['text'],
                        selected_model,
                        tokenizers[selected_model],
                        models[selected_model]
                    )
                    
                    if result:
                        predictions.append({
                            'text': row['text'],
                            'sentiment': result['sentiment'],
                            'confidence': round(result['confidence'], 4)
                        })
                    
                    progress_bar.progress((idx + 1) / len(df))
                
                # Display results
                st.success(f"✅ Prediction complete!")
                results_df = pd.DataFrame(predictions)
                
                # Show statistics
                col1, col2, col3 = st.columns(3)
                positive_count = len(results_df[results_df['sentiment'] == 'POSITIVE'])
                negative_count = len(results_df[results_df['sentiment'] == 'NEGATIVE'])
                avg_confidence = results_df['confidence'].mean()
                
                col1.metric("Positive", positive_count)
                col2.metric("Negative", negative_count)
                col3.metric("Avg Confidence", f"{avg_confidence:.2%}")
                
                # Show table
                st.dataframe(results_df, use_container_width=True)
                
                # Download results
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results",
                    data=csv,
                    file_name=f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )

# ============================================================
# TAB 3: HISTORY & ANALYTICS
# ============================================================
with tab3:
    st.header("📈 Prediction History & Analytics")
    
    history = load_history()
    
    if not history:
        st.info("No predictions yet. Make some predictions to see history!")
    else:
        # Sentiment distribution
        history_df = pd.DataFrame(history)
        
        col1, col2, col3 = st.columns(3)
        
        total_predictions = len(history_df)
        positive_count = len(history_df[history_df['sentiment'] == 'POSITIVE'])
        negative_count = len(history_df[history_df['sentiment'] == 'NEGATIVE'])
        
        col1.metric("Total Predictions", total_predictions)
        col2.metric("Positive", positive_count)
        col3.metric("Negative", negative_count)
        
        st.divider()
        
        # Charts
        col1, col2 = st.columns(2)
        
        # Sentiment pie chart
        with col1:
            st.subheader("Sentiment Distribution")
            sentiment_counts = history_df['sentiment'].value_counts()
            fig, ax = plt.subplots(figsize=(8, 6))
            colors = ['#2ca02c', '#d62728']
            ax.pie(sentiment_counts.values, labels=sentiment_counts.index, 
                   autopct='%1.1f%%', colors=colors, startangle=90)
            st.pyplot(fig)
        
        # Confidence over time
        with col2:
            st.subheader("Confidence Over Time")
            history_df['timestamp'] = pd.to_datetime(history_df['timestamp'])
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(range(len(history_df)), history_df['confidence'], marker='o')
            ax.set_xlabel('Prediction #')
            ax.set_ylabel('Confidence')
            ax.set_ylim(0, 1)
            ax.grid(alpha=0.3)
            st.pyplot(fig)
        
        st.divider()
        
        # Recent predictions table
        st.subheader("Recent Predictions")
        display_df = history_df[['timestamp', 'text', 'model', 'sentiment', 'confidence']].copy()
        display_df['text'] = display_df['text'].str[:50] + '...'
        display_df = display_df.sort_values('timestamp', ascending=False).head(10)
        st.dataframe(display_df, use_container_width=True)

# ============================================================
# TAB 4: MODEL COMPARISON
# ============================================================
with tab4:
    st.header("🔄 Model Comparison")
    
    st.markdown("""
    Compare how different models predict the same text.
    """)
    
    comparison_text = st.text_area(
        "Enter text to compare across all models:",
        placeholder="Type or paste text here...",
        height=100,
        key="comparison_text"
    )
    
    if st.button("🔄 Compare All Models", key="compare_btn"):
        if comparison_text:
            results = {}
            
            for model_name in models.keys():
                result = predict_sentiment(
                    comparison_text,
                    model_name,
                    tokenizers[model_name],
                    models[model_name]
                )
                results[model_name] = result
            
            # Create comparison dataframe
            comparison_df = pd.DataFrame({
                'Model': list(results.keys()),
                'Sentiment': [results[m]['sentiment'] for m in results.keys()],
                'Confidence': [round(results[m]['confidence'], 4) for m in results.keys()],
                'Negative Prob': [round(results[m]['prob_negative'], 4) for m in results.keys()],
                'Positive Prob': [round(results[m]['prob_positive'], 4) for m in results.keys()]
            })
            
            st.dataframe(comparison_df, use_container_width=True)
            
            # Confidence comparison chart
            fig, ax = plt.subplots(figsize=(10, 6))
            models_list = comparison_df['Model'].tolist()
            confidences = comparison_df['Confidence'].tolist()
            colors = ['#2ca02c' if s == 'POSITIVE' else '#d62728' 
                     for s in comparison_df['Sentiment']]
            
            bars = ax.bar(models_list, confidences, color=colors, alpha=0.7)
            ax.set_ylabel('Confidence', fontweight='bold')
            ax.set_ylim(0, 1)
            ax.set_title('Model Confidence Comparison')
            
            # Add value labels
            for bar, conf in zip(bars, confidences):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{conf:.2%}', ha='center', va='bottom', fontweight='bold')
            
            ax.grid(axis='y', alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)

# ============================================================
# TAB 5: WORDCLOUD
# ============================================================
with tab5:
    st.header("☁️ WordCloud Visualization")
    
    st.markdown("""
    Explore word frequency patterns in our dataset or generate custom word clouds.
    """)
    
    # Subtabs for WordCloud
    wc_tab1, wc_tab2 = st.tabs(["📊 Dataset WordClouds", "🎨 Custom WordCloud"])
    
    # Dataset WordClouds
    with wc_tab1:
        st.subheader("📊 WordCloud dari Dataset Balance Kita")
        st.markdown("""
        Visualisasi kata-kata paling sering muncul dalam 1,200 samples balanced data.
        
        **Dataset**: 
        - Total samples: 1,200 (600 POSITIVE + 600 NEGATIVE)
        - Total words: 11,967
        - Average words per sample: 10.0
        """)
        
        # Select which wordcloud to display
        wc_type = st.radio(
            "Choose WordCloud Type:",
            ["All Data (Viridis)", "Positive Only (Green)", "Negative Only (Red)", 
             "Comparison (Side-by-side)", "Plasma", "Inferno", "Magma"],
            horizontal=True
        )
        
        wc_file_map = {
            "All Data (Viridis)": "wordcloud_dataset_basic.png",
            "Positive Only (Green)": "wordcloud_dataset_positive.png",
            "Negative Only (Red)": "wordcloud_dataset_negative.png",
            "Comparison (Side-by-side)": "wordcloud_dataset_comparison.png",
            "Plasma": "wordcloud_dataset_plasma.png",
            "Inferno": "wordcloud_dataset_inferno.png",
            "Magma": "wordcloud_dataset_magma.png"
        }
        
        wc_file = wc_file_map[wc_type]
        
        if os.path.exists(wc_file):
            st.image(wc_file, width=700)
            
            # Download button
            with open(wc_file, 'rb') as f:
                st.download_button(
                    label=f"📥 Download {wc_type}",
                    data=f.read(),
                    file_name=wc_file,
                    mime="image/png"
                )
        else:
            st.warning(f"⚠️ WordCloud file not found: {wc_file}")
            st.info("Run `wordcloud_dataset.py` to generate dataset word clouds.")
        
        # Info about dataset
        st.divider()
        st.subheader("📈 Dataset Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Samples", "1,200")
        col2.metric("Positive", "600 (50%)")
        col3.metric("Negative", "600 (50%)")
        col4.metric("Total Words", "11,967")
    
    # Custom WordCloud
    with wc_tab2:
        st.subheader("🎨 Create Custom WordCloud")
        st.markdown("""
        Generate a word cloud from your own text to visualize word frequency.
        """)
        
        wordcloud_text = st.text_area(
            "Enter text for word cloud:",
            placeholder="Paste a large text here...",
            height=150,
            key="wordcloud_text"
        )
        
        if st.button("☁️ Generate WordCloud", key="wordcloud_btn"):
            if wordcloud_text and len(wordcloud_text) > 10:
                # Generate wordcloud
                wordcloud = WordCloud(width=1000, height=600, 
                                     background_color='white',
                                     colormap='viridis').generate(wordcloud_text)
                
                fig, ax = plt.subplots(figsize=(12, 7))
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.axis('off')
                plt.tight_layout()
                
                st.pyplot(fig, use_container_width=True)
                
                # Save option
                if st.button("💾 Save WordCloud"):
                    wordcloud.to_file('wordcloud_custom.png')
                    st.success("✅ WordCloud saved as 'wordcloud_custom.png'")
            else:
                st.error("Please enter longer text (at least 10 characters)")

# ============================================================
# FOOTER
# ============================================================
st.divider()
st.markdown("""
    <div style='text-align: center; color: gray; font-size: 0.8em;'>
    <p>🎯 Sentiment Analysis Dashboard | Powered by Streamlit & Transformers</p>
    <p>Models: BERT, DistilBERT, RoBERTa, ALBERT, XLNet</p>
    </div>
""", unsafe_allow_html=True)
