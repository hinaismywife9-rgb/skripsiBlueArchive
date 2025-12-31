# 🚀 QUICK START GUIDE - Sentiment Analysis dengan 5 Transformer Models

## ⚡ 5 Langkah Cepat

### Step 1: Setup Environment
```bash
python setup.py
```
Ini akan:
- ✓ Check Python version (3.8+)
- ✓ Check GPU availability
- ✓ Install semua dependencies
- ✓ Create necessary folders

**Waktu**: ~3-5 menit

---

### Step 2: Clean & Balance Data
```bash
python balance_sentiment.py
```
Ini akan:
- ✓ Standardisasi label sentiment
- ✓ Balance data dengan 3 metode (Oversampling, Undersampling, Hybrid)
- ✓ Generate `sentiment analysis BA_CLEANED.xlsx`

**Output**: File Excel dengan 4 sheet
- Original_Cleaned (1,688 samples)
- Oversampling (1,610 samples - balanced)
- Undersampling (1,610 samples - balanced)
- Hybrid (1,200 samples - balanced)

**Waktu**: ~1 menit

---

### Step 3: Train 5 Models
```bash
python train_transformer_models.py
```
Ini akan melatih:
1. 🔵 **BERT** - Balanced, reliable
2. 🟢 **DistilBERT** - Fast (60% lebih cepat)
3. 🔴 **RoBERTa** - Best performance (usually)
4. 🟡 **ALBERT** - Memory efficient (90% lebih kecil)
5. 🟣 **XLNET** - Best for complex context

**Output Files**:
- `sentiment_models/` - Folder dengan 5 trained models
- `training_results.json` - Hasil training
- `model_performance_comparison.csv` - Comparison table
- `model_training_report.txt` - Detailed report

**Waktu**: 30-60 menit (tergantung GPU)

---

### Step 4: Test & Evaluate
```bash
python inference_sentiment.py
```
Ini akan:
- ✓ Load semua 5 models
- ✓ Test dengan contoh texts
- ✓ Generate `example_predictions.csv`
- ✓ Show usage examples

**Output**: Example predictions CSV dan usage instructions

**Waktu**: ~2-5 menit

---

### Step 5: Use Models (Production)
```python
from sentiment_utils import SentimentAnalyzer

# Load best model (RoBERTa)
analyzer = SentimentAnalyzer('./sentiment_models/RoBERTa')

# Single prediction
result = analyzer.predict("I love this product!")
print(result)
# Output: {'text': '...', 'sentiment': 'positive', 'confidence': 0.9999}

# Batch prediction
texts = ["Great!", "Bad...", "Okay"]
results = analyzer.predict_batch(texts)
```

---

## 📊 Model Performance Guide

Setelah Step 3 (training), Anda akan melihat hasil seperti:

```
                Rank      Model  Accuracy  Precision    Recall  F1-Score
                   1    RoBERTa    0.9234     0.9201    0.9234    0.9215
                   2       BERT    0.9102     0.9078    0.9102    0.9089
                   3      XLNET    0.9187     0.9165    0.9187    0.9175
                   4  DistilBERT    0.8945     0.8912    0.8945    0.8928
                   5     ALBERT    0.8756     0.8723    0.8756    0.8739
```

### Recommended Models by Use Case:

| Use Case | Model | Why |
|----------|-------|-----|
| 🏆 **Best Accuracy** | RoBERTa | ~92-93% accuracy, best for sentiment |
| ⚡ **Fastest Inference** | DistilBERT | 60% faster, 89-90% accuracy |
| 📱 **Mobile/Edge** | ALBERT | 90% smaller, very fast |
| 🧠 **Complex Context** | XLNet | Best understanding of nuance |
| ⚖️ **Balanced** | BERT | Good all-around performance |

---

## 💡 Common Usage Examples

### Example 1: Single Text Prediction
```python
from sentiment_utils import SentimentAnalyzer

analyzer = SentimentAnalyzer('./sentiment_models/RoBERTa')
result = analyzer.predict("This product is amazing!")

print(f"Sentiment: {result['sentiment']}")  # Output: positive
print(f"Confidence: {result['confidence']:.2%}")  # Output: 99.95%
```

### Example 2: Batch Processing (CSV File)
```python
from sentiment_utils import batch_predict_to_csv

# Predict sentiment for CSV file
batch_predict_to_csv(
    model_path='./sentiment_models/RoBERTa',
    input_csv='my_reviews.csv',
    output_csv='my_reviews_predicted.csv',
    text_column='review_text'
)

# my_reviews_predicted.csv akan memiliki kolom:
# - review_text
# - predicted_sentiment
# - confidence
```

### Example 3: Ensemble Voting (Multiple Models)
```python
from sentiment_utils import ensemble_predict

models = {
    'RoBERTa': './sentiment_models/RoBERTa',
    'BERT': './sentiment_models/BERT',
    'DistilBERT': './sentiment_models/DistilBERT',
}

result = ensemble_predict(
    "I like this but it could be better",
    models,
    voting='confidence'  # or 'majority'
)

print(f"Ensemble: {result['ensemble_prediction']}")
print(f"Individual predictions: {result['individual_predictions']}")
```

### Example 4: Model Comparison
```python
from sentiment_utils import compare_models

test_texts = ["I love it!", "Hate it.", "It's okay"]
test_labels = ['positive', 'negative', 'neutral']

model_paths = {
    'RoBERTa': './sentiment_models/RoBERTa',
    'BERT': './sentiment_models/BERT',
    'DistilBERT': './sentiment_models/DistilBERT',
}

comparison_df = compare_models(test_texts, test_labels, model_paths)
print(comparison_df)

# Output: DataFrame dengan Accuracy, Precision, Recall, F1-Score untuk setiap model
```

---

## 🎯 Optimization Tips

### Untuk Akurasi Maksimal
```python
# Use RoBERTa (best accuracy)
analyzer = SentimentAnalyzer('./sentiment_models/RoBERTa')

# Use ensemble voting
results = ensemble_predict(text, all_models, voting='confidence')
```

### Untuk Kecepatan Maksimal
```python
# Use DistilBERT (60% faster)
analyzer = SentimentAnalyzer('./sentiment_models/DistilBERT')
```

### Untuk Memory Efficiency
```python
# Use ALBERT (90% lebih kecil)
analyzer = SentimentAnalyzer('./sentiment_models/ALBERT')
```

---

## ⚠️ Troubleshooting

### Error: "CUDA out of memory"
**Solution**: 
- Gunakan DistilBERT atau ALBERT (lebih kecil)
- Kurangi batch size
- Gunakan CPU saja

### Error: "Model not found"
**Solution**: 
Pastikan sudah menjalankan `train_transformer_models.py` terlebih dahulu

### Training terlalu lambat
**Solution**:
- Install GPU support: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`
- Kurangi epochs di config
- Gunakan DistilBERT (lebih cepat)

### Bad predictions
**Solution**:
- Training mungkin belum selesai sempurna
- Try dengan model lain (RoBERTa atau XLNET)
- Use ensemble voting untuk hasil lebih robust

---

## 📁 Project Structure

```
james BA/
├── sentiment_models/          # Trained models
│   ├── BERT/
│   ├── DistilBERT/
│   ├── RoBERTa/
│   ├── ALBERT/
│   └── XLNET/
│
├── results/                   # Training results
│   ├── BERT/
│   ├── DistilBERT/
│   └── ...
│
├── sentiment analysis BA.xlsx # Original data
├── sentiment analysis BA_CLEANED.xlsx  # Cleaned & balanced
│
├── train_transformer_models.py   # Training script
├── inference_sentiment.py        # Inference script
├── sentiment_utils.py            # Utility functions
├── balance_sentiment.py          # Data cleaning script
├── check_sentiment.py            # Data analysis script
├── setup.py                      # Setup script
├── requirements.txt              # Dependencies
├── README.md                     # Full documentation
└── QUICKSTART.md                 # This file
```

---

## 🔄 Full Workflow

```
Data Collection
    ↓
Analyze (check_sentiment.py)
    ↓
Clean & Balance (balance_sentiment.py)
    ↓
Train Models (train_transformer_models.py)
    ↓
Evaluate (inference_sentiment.py)
    ↓
Production Use (sentiment_utils.py)
```

---

## 📞 Support Files

- 📖 **README.md** - Dokumentasi lengkap
- 🚀 **QUICKSTART.md** - Ini file
- 🔧 **requirements.txt** - Dependencies
- 📊 **training_results.json** - Training metrics
- 📈 **model_performance_comparison.csv** - Model comparison

---

## ✅ Checklist

- [ ] Run `python setup.py`
- [ ] Run `python balance_sentiment.py`
- [ ] Run `python train_transformer_models.py`
- [ ] Run `python inference_sentiment.py`
- [ ] Review results in `model_performance_comparison.csv`
- [ ] Choose best model for your use case
- [ ] Integrate into production using `sentiment_utils.py`

---

## 🎉 Done!

Anda sekarang memiliki 5 transformer models yang siap untuk sentiment analysis!

Untuk mulai menggunakan, cukup:
```python
from sentiment_utils import SentimentAnalyzer

analyzer = SentimentAnalyzer('./sentiment_models/RoBERTa')
print(analyzer.predict("Your text here"))
```

Good luck! 🚀
