"""
WordCloud visualization dari dataset balance kita
Membuat word cloud dari semua text dalam data training (1,200 samples)
Menunjukkan kata-kata paling sering muncul dalam sentiment data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import os

print("=" * 70)
print("☁️  WORDCLOUD DARI DATASET BALANCE KITA")
print("=" * 70)

# Load data
try:
    excel_file = "sentiment analysis BA_CLEANED.xlsx"
    print(f"\n📂 Loading: {excel_file}")
    df = pd.read_excel(excel_file, sheet_name="Hybrid")
    print(f"✓ Loaded {len(df)} samples dari sheet 'Hybrid'")
except Exception as e:
    print(f"✗ Error loading: {e}")
    exit(1)

# Check columns
print(f"\n📋 Columns: {df.columns.tolist()}")

# Identify text column
text_column = None
for col in df.columns:
    if col.lower() in ['text', 'content', 'review', 'comment', 'message']:
        text_column = col
        break

if text_column is None:
    # Try first column
    text_column = df.columns[0]
    print(f"⚠️  Text column not found by name, using first column: '{text_column}'")
else:
    print(f"✓ Text column: '{text_column}'")

# Get all text
print(f"\n🔤 Combining all text ({len(df)} samples)...")
all_text = " ".join(df[text_column].astype(str).values)
print(f"✓ Total text length: {len(all_text):,} characters")

# Count words
word_count = len(all_text.split())
print(f"✓ Total words: {word_count:,}")

# ========================================================
# WORDCLOUD 1: Basic WordCloud
# ========================================================
print("\n" + "=" * 70)
print("1️⃣  Creating Basic WordCloud...")
print("=" * 70)

fig, ax = plt.subplots(figsize=(14, 9), dpi=300)

wordcloud = WordCloud(
    width=1400,
    height=900,
    background_color='white',
    colormap='viridis',
    max_words=100,
    relative_scaling=0.5,
    min_font_size=10
).generate(all_text)

ax.imshow(wordcloud, interpolation='bilinear')
ax.axis('off')
ax.set_title('WordCloud - All Dataset Text (1,200 Samples)', 
             fontsize=16, fontweight='bold', pad=20)

plt.tight_layout()
filename1 = 'wordcloud_dataset_basic.png'
plt.savefig(filename1, dpi=300, bbox_inches='tight')
print(f"✓ Saved: {filename1}")
plt.close()

# ========================================================
# WORDCLOUD 2: Positive Sentiment Only
# ========================================================
print("\n" + "=" * 70)
print("2️⃣  Creating WordCloud - POSITIVE Sentiment Only...")
print("=" * 70)

# Identify sentiment column
sentiment_column = None
for col in df.columns:
    if col.lower() in ['sentiment', 'label', 'class', 'target']:
        sentiment_column = col
        break

if sentiment_column:
    print(f"✓ Sentiment column: '{sentiment_column}'")
    
    # Get positive text
    positive_df = df[df[sentiment_column].astype(str).str.upper() == 'POSITIVE']
    positive_text = " ".join(positive_df[text_column].astype(str).values)
    
    print(f"✓ Positive samples: {len(positive_df)}")
    print(f"✓ Positive text length: {len(positive_text):,} characters")
    
    fig, ax = plt.subplots(figsize=(14, 9), dpi=300)
    
    wordcloud_pos = WordCloud(
        width=1400,
        height=900,
        background_color='white',
        colormap='Greens',
        max_words=100,
        relative_scaling=0.5,
        min_font_size=10
    ).generate(positive_text)
    
    ax.imshow(wordcloud_pos, interpolation='bilinear')
    ax.axis('off')
    ax.set_title('WordCloud - POSITIVE Sentiment (600 Samples)', 
                 fontsize=16, fontweight='bold', pad=20, color='green')
    
    plt.tight_layout()
    filename2 = 'wordcloud_dataset_positive.png'
    plt.savefig(filename2, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {filename2}")
    plt.close()
    
    # ========================================================
    # WORDCLOUD 3: Negative Sentiment Only
    # ========================================================
    print("\n" + "=" * 70)
    print("3️⃣  Creating WordCloud - NEGATIVE Sentiment Only...")
    print("=" * 70)
    
    # Get negative text
    negative_df = df[df[sentiment_column].astype(str).str.upper() == 'NEGATIVE']
    negative_text = " ".join(negative_df[text_column].astype(str).values)
    
    print(f"✓ Negative samples: {len(negative_df)}")
    print(f"✓ Negative text length: {len(negative_text):,} characters")
    
    fig, ax = plt.subplots(figsize=(14, 9), dpi=300)
    
    wordcloud_neg = WordCloud(
        width=1400,
        height=900,
        background_color='white',
        colormap='Reds',
        max_words=100,
        relative_scaling=0.5,
        min_font_size=10
    ).generate(negative_text)
    
    ax.imshow(wordcloud_neg, interpolation='bilinear')
    ax.axis('off')
    ax.set_title('WordCloud - NEGATIVE Sentiment (600 Samples)', 
                 fontsize=16, fontweight='bold', pad=20, color='red')
    
    plt.tight_layout()
    filename3 = 'wordcloud_dataset_negative.png'
    plt.savefig(filename3, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {filename3}")
    plt.close()

# ========================================================
# WORDCLOUD 4: Side-by-side Comparison
# ========================================================
if sentiment_column:
    print("\n" + "=" * 70)
    print("4️⃣  Creating Side-by-Side Comparison...")
    print("=" * 70)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 9), dpi=300)
    
    # Positive
    ax1.imshow(wordcloud_pos, interpolation='bilinear')
    ax1.axis('off')
    ax1.set_title('POSITIVE (600 Samples)', fontsize=14, fontweight='bold', color='green')
    
    # Negative
    ax2.imshow(wordcloud_neg, interpolation='bilinear')
    ax2.axis('off')
    ax2.set_title('NEGATIVE (600 Samples)', fontsize=14, fontweight='bold', color='red')
    
    fig.suptitle('WordCloud Comparison - Balanced Dataset', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    filename4 = 'wordcloud_dataset_comparison.png'
    plt.savefig(filename4, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {filename4}")
    plt.close()

# ========================================================
# WORDCLOUD 5: Custom Colormaps (All)
# ========================================================
print("\n" + "=" * 70)
print("5️⃣  Creating Multiple Colormap Variants...")
print("=" * 70)

colormaps = ['plasma', 'inferno', 'magma']
wordcloud_all = WordCloud(
    width=1400,
    height=900,
    background_color='white',
    max_words=100,
    relative_scaling=0.5,
    min_font_size=10
).generate(all_text)

for colormap in colormaps:
    fig, ax = plt.subplots(figsize=(14, 9), dpi=300)
    
    # Re-generate with different colormap
    wc = WordCloud(
        width=1400,
        height=900,
        background_color='white',
        colormap=colormap,
        max_words=100,
        relative_scaling=0.5,
        min_font_size=10
    ).generate(all_text)
    
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    ax.set_title(f'WordCloud - {colormap.upper()} Colormap (1,200 Samples)', 
                 fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    filename = f'wordcloud_dataset_{colormap}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {filename}")
    plt.close()

# ========================================================
# DATA SUMMARY
# ========================================================
print("\n" + "=" * 70)
print("📊 DATASET SUMMARY")
print("=" * 70)

print(f"\nTotal Samples: {len(df)}")
print(f"Text Column: '{text_column}'")

if sentiment_column:
    print(f"Sentiment Column: '{sentiment_column}'")
    print(f"\nSentiment Distribution:")
    print(f"  POSITIVE: {len(positive_df)} samples (50%)")
    print(f"  NEGATIVE: {len(negative_df)} samples (50%)")

print(f"\nText Statistics:")
print(f"  Total Characters: {len(all_text):,}")
print(f"  Total Words: {word_count:,}")
print(f"  Average Words per Sample: {word_count / len(df):.1f}")

# ========================================================
# FILES SUMMARY
# ========================================================
print("\n" + "=" * 70)
print("📁 FILES GENERATED")
print("=" * 70)

files_generated = [
    'wordcloud_dataset_basic.png',
    'wordcloud_dataset_positive.png',
    'wordcloud_dataset_negative.png',
    'wordcloud_dataset_comparison.png',
    'wordcloud_dataset_plasma.png',
    'wordcloud_dataset_inferno.png',
    'wordcloud_dataset_magma.png'
]

for i, filename in enumerate(files_generated, 1):
    if os.path.exists(filename):
        size_kb = os.path.getsize(filename) / 1024
        print(f"{i}. ✓ {filename} ({size_kb:.1f} KB)")
    else:
        print(f"{i}. ✗ {filename} (not found)")

print("\n" + "=" * 70)
print("✅ ALL WORDCLOUDS GENERATED SUCCESSFULLY!")
print("=" * 70)
print("\n🎨 Colormaps used:")
print("  • Basic: Viridis (blue → yellow)")
print("  • Positive: Greens (light → dark green)")
print("  • Negative: Reds (light → dark red)")
print("  • Variants: Plasma, Inferno, Magma")

print("\n📈 Size: 1400x900 pixels, 300 DPI")
print("🔤 Max words: 100 per cloud")

print("\n" + "=" * 70)
