# 5 Transformer Models untuk Sentiment Analysis

Proyek ini melatih dan membandingkan 5 model transformer terbaik untuk sentiment analysis.

## 📋 Model-Model yang Digunakan

### 1. **BERT** (Bidirectional Encoder Representations from Transformers)
- **Model Base**: `bert-base-uncased`
- **Ukuran**: ~110M parameters
- **Kecepatan**: Moderate
- **Akurasi**: Very Good
- **Best for**: General purpose NLP tasks
- **Keunggulan**: Bidirectional context understanding

### 2. **DistilBERT** (Distilled BERT)
- **Model Base**: `distilbert-base-uncased`
- **Ukuran**: ~66M parameters (40% lebih kecil)
- **Kecepatan**: Fast (60% lebih cepat)
- **Akurasi**: Good
- **Best for**: Resource-constrained environments
- **Keunggulan**: Inference cepat dengan minimal performance loss

### 3. **RoBERTa** (Robustly Optimized BERT)
- **Model Base**: `roberta-base`
- **Ukuran**: ~125M parameters
- **Kecepatan**: Moderate
- **Akurasi**: Excellent (biasanya terbaik untuk sentiment)
- **Best for**: Sentiment analysis, text classification
- **Keunggulan**: Pelatihan ulang yang lebih baik, lebih robust

### 4. **ALBERT** (A Lite BERT)
- **Model Base**: `albert-base-v2`
- **Ukuran**: ~11M parameters (90% lebih kecil)
- **Kecepatan**: Very Fast
- **Akurasi**: Good
- **Best for**: Mobile deployment, real-time inference
- **Keunggulan**: Parameter sharing, memory efficient

### 5. **XLNet** (eXtreme MultiLingual Net)
- **Model Base**: `xlnet-base-cased`
- **Ukuran**: ~340M parameters
- **Kecepatan**: Moderate
- **Akurasi**: Excellent
- **Best for**: Complex context, nuanced understanding
- **Keunggulan**: Permutation language modeling, better context

---

## 🚀 Cara Menggunakan

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Data

Pastikan file Excel sudah dibersihkan:
```bash
python balance_sentiment.py
```

Ini akan menghasilkan `sentiment analysis BA_CLEANED.xlsx` dengan data yang sudah balanced.

### 3. Train All 5 Models

```bash
python train_transformer_models.py
```

Proses training akan:
- Membuat folder `sentiment_models/` berisi semua model terlatih
- Generate file `training_results.json` dengan hasil training
- Generate file `model_performance_comparison.csv` dengan perbandingan
- Generate file `model_training_report.txt` dengan laporan lengkap

**Waktu training estimate**: 30-60 menit tergantung GPU/CPU

### 4. Run Inference

```bash
python inference_sentiment.py
```

Script ini akan:
- Load semua model yang sudah dilatih
- Test dengan contoh texts
- Generate `example_predictions.csv`

---

## 📊 Output Files

### Training
- `sentiment_models/BERT/` - Model BERT terlatih
- `sentiment_models/DistilBERT/` - Model DistilBERT terlatih
- `sentiment_models/RoBERTa/` - Model RoBERTa terlatih
- `sentiment_models/ALBERT/` - Model ALBERT terlatih
- `sentiment_models/XLNET/` - Model XLNet terlatih

### Results
- `training_results.json` - Hasil training dalam format JSON
- `model_performance_comparison.csv` - Tabel perbandingan performance
- `model_training_report.txt` - Laporan training lengkap
- `example_predictions.csv` - Contoh prediksi

---

## 🎯 Recommendation

Berdasarkan trade-off antara accuracy, speed, dan resource usage:

| Use Case | Recommended Model | Reason |
|----------|-------------------|--------|
| **Best Accuracy** | RoBERTa | Consistently best performance on sentiment tasks |
| **Fast Inference** | DistilBERT | Good balance: 60% faster, 95% accuracy |
| **Mobile/Edge** | ALBERT | 90% smaller, very fast |
| **Complex Context** | XLNet | Best for nuanced understanding |
| **General Purpose** | BERT | Balanced, reliable, widely used |

---

## 💻 Usage Examples

### Predict Single Text

```python
from transformers import pipeline

# Load model
pipe = pipeline("text-classification", model="./sentiment_models/RoBERTa")

# Predict
result = pipe("I love this product!")
print(result)
# Output: [{'label': 'POSITIVE', 'score': 0.9999}]
```

### Batch Prediction

```python
texts = [
    "I love this!",
    "This is terrible",
    "It's okay"
]

results = pipe(texts)
for text, result in zip(texts, results):
    print(f"{text} → {result['label']} ({result['score']:.4f})")
```

### With Custom Tokenization

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

model_path = "./sentiment_models/RoBERTa"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

text = "This product is amazing!"
inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)

with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    prediction = torch.argmax(logits, dim=-1).item()
    score = torch.softmax(logits, dim=-1).max().item()

print(f"Prediction: {prediction}, Score: {score:.4f}")
```

---

## 📈 Training Details

### Data Split
- Training: 80%
- Validation: 20%

### Training Hyperparameters
- **Epochs**: 3
- **Batch Size**: 16
- **Learning Rate**: 5e-5 (default)
- **Warmup Steps**: 100
- **Weight Decay**: 0.01

### Metrics
- Accuracy
- Precision (weighted)
- Recall (weighted)
- F1-Score (weighted)

---

## 🔍 Performance Comparison

Hasil training akan ditampilkan dalam format:

```
TRAINING RESULTS SUMMARY
════════════════════════════════════════════════════════════════════════════════

   Rank          Model  Accuracy  Precision    Recall  F1-Score
   1           RoBERTa    0.9234     0.9201    0.9234    0.9215
   2              BERT    0.9102     0.9078    0.9102    0.9089
   3         DistilBERT    0.8945     0.8912    0.8945    0.8928
   4             XLNET    0.9187     0.9165    0.9187    0.9175
   5            ALBERT    0.8756     0.8723    0.8756    0.8739
```

---

## ⚙️ System Requirements

### Minimum
- Python 3.8+
- 8GB RAM
- CPU (training akan lambat)

### Recommended
- Python 3.9+
- 16GB+ RAM
- NVIDIA GPU with CUDA support (untuk faster training)

### For GPU Support (Optional)
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

---

## 🛠️ Troubleshooting

### Out of Memory Error
```python
# Kurangi batch size di training_config
'per_device_train_batch_size': 8  # dari 16
```

### Slow Training
- Gunakan GPU (install CUDA support)
- Kurangi epochs
- Kurangi jumlah models (jalankan individually)

### Model Not Found
Pastikan sudah menjalankan `train_transformer_models.py` terlebih dahulu

---

## 📚 References

- [BERT Paper](https://arxiv.org/abs/1810.04805)
- [RoBERTa Paper](https://arxiv.org/abs/1907.11692)
- [DistilBERT Paper](https://arxiv.org/abs/1910.01108)
- [ALBERT Paper](https://arxiv.org/abs/1909.11942)
- [XLNet Paper](https://arxiv.org/abs/1906.08237)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/)

---

## 📝 License

Semua model menggunakan lisensi dari Hugging Face Model Hub.

---

## 🤝 Support

Jika ada pertanyaan atau masalah, silakan buat issue atau contact development team.

Generated: 2025-12-24
