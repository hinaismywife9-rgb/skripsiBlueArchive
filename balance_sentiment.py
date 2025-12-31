import pandas as pd
import numpy as np
from sklearn.utils import resample
import matplotlib.pyplot as plt

# Read the original file
df = pd.read_excel('sentiment analysis BA.xlsx')

print("=" * 70)
print("DATA CLEANING & BALANCING")
print("=" * 70)

print("\n[1] STANDARDISASI LABEL")
print("-" * 70)
print("Before:")
print(df['sentiment'].value_counts())

# Standardize labels
df['sentiment'] = df['sentiment'].str.lower().str.strip()

print("\nAfter:")
print(df['sentiment'].value_counts())

print("\n[2] BALANCING DATA - METODE OVERSAMPLING")
print("-" * 70)

# Separate by sentiment
negative_df = df[df['sentiment'] == 'negative']
positive_df = df[df['sentiment'] == 'positive']

print(f"Negative samples: {len(negative_df)}")
print(f"Positive samples: {len(positive_df)}")

# Oversample minority class to match majority
max_samples = len(negative_df)
positive_upsampled = resample(positive_df, 
                               replace=True,  # Allow sampling with replacement
                               n_samples=max_samples,
                               random_state=42)

print(f"\nAfter oversampling:")
print(f"Negative samples: {len(negative_df)}")
print(f"Positive samples: {len(positive_upsampled)}")

# Combine
df_balanced_oversample = pd.concat([negative_df, positive_upsampled], ignore_index=True)
df_balanced_oversample = df_balanced_oversample.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"\nBalanced dataset size: {len(df_balanced_oversample)}")
print("\nDistribusi:")
print(df_balanced_oversample['sentiment'].value_counts())

print("\n[3] BALANCING DATA - METODE UNDERSAMPLING")
print("-" * 70)

# Undersample majority class to match minority
max_class = max(len(negative_df), len(positive_df))
min_class = min(len(negative_df), len(positive_df))
min_samples = min_class

negative_downsampled = resample(negative_df,
                                replace=False,  # Without replacement
                                n_samples=min_samples,
                                random_state=42)

print(f"Negative samples (downsampled): {len(negative_downsampled)}")
print(f"Positive samples (undersampled): {len(positive_df[:min_samples])}")

# Undersample positive to match negative
positive_downsampled = resample(positive_df,
                                replace=False,
                                n_samples=min_samples,
                                random_state=42)

# Combine - use class with fewer samples
if len(negative_df) < len(positive_df):
    df_balanced_undersample = pd.concat([negative_df, positive_downsampled], ignore_index=True)
else:
    df_balanced_undersample = pd.concat([negative_downsampled, positive_df], ignore_index=True)
df_balanced_undersample = df_balanced_undersample.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"\nBalanced dataset size: {len(df_balanced_undersample)}")
print("\nDistribusi:")
print(df_balanced_undersample['sentiment'].value_counts())

print("\n[4] BALANCING DATA - METODE HYBRID (Oversampling + Undersampling)")
print("-" * 70)

# Target 600 samples per class (middle ground)
target_samples = 600

negative_hybrid = resample(negative_df,
                           replace=True,
                           n_samples=target_samples,
                           random_state=42)

positive_hybrid = resample(positive_df,
                           replace=True,
                           n_samples=target_samples,
                           random_state=42)

df_balanced_hybrid = pd.concat([negative_hybrid, positive_hybrid], ignore_index=True)
df_balanced_hybrid = df_balanced_hybrid.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"Negative samples: {len(negative_hybrid)}")
print(f"Positive samples: {len(positive_hybrid)}")
print(f"\nBalanced dataset size: {len(df_balanced_hybrid)}")
print("\nDistribusi:")
print(df_balanced_hybrid['sentiment'].value_counts())

print("\n[5] SUMMARY")
print("=" * 70)
summary_data = {
    'Method': ['Original (Cleaned)', 'Oversampling', 'Undersampling', 'Hybrid'],
    'Total Size': [len(df), len(df_balanced_oversample), len(df_balanced_undersample), len(df_balanced_hybrid)],
    'Negative': [len(df[df['sentiment']=='negative']), len(negative_df), len(negative_downsampled), len(negative_hybrid)],
    'Positive': [len(df[df['sentiment']=='positive']), len(positive_upsampled), len(positive_df), len(positive_hybrid)],
}

summary_df = pd.DataFrame(summary_data)
print(summary_df.to_string(index=False))

# Save all versions
print("\n[6] MENYIMPAN FILE")
print("-" * 70)

with pd.ExcelWriter('sentiment analysis BA_CLEANED.xlsx', engine='openpyxl') as writer:
    df.to_excel(writer, sheet_name='Original_Cleaned', index=False)
    print("✓ Sheet 'Original_Cleaned' - Data yang sudah dibersihkan (standardisasi label)")
    
    df_balanced_oversample.to_excel(writer, sheet_name='Oversampling', index=False)
    print("✓ Sheet 'Oversampling' - Oversampling (replikasi data minority)")
    
    df_balanced_undersample.to_excel(writer, sheet_name='Undersampling', index=False)
    print("✓ Sheet 'Undersampling' - Undersampling (mengurangi data majority)")
    
    df_balanced_hybrid.to_excel(writer, sheet_name='Hybrid', index=False)
    print("✓ Sheet 'Hybrid' - Hybrid approach (target 600 per class)")
    
    summary_df.to_excel(writer, sheet_name='Summary', index=False)
    print("✓ Sheet 'Summary' - Ringkasan perbandingan")

print("\n" + "=" * 70)
print("✓ SELESAI! File tersimpan: sentiment analysis BA_CLEANED.xlsx")
print("=" * 70)
print("\nREKOMENDASI PENGGUNAAN:")
print("- Untuk training umum: Gunakan 'Hybrid' (balanced dan ukuran reasonable)")
print("- Untuk data yang banyak: Gunakan 'Oversampling' (1610 samples)")
print("- Untuk data yang sedikit: Gunakan 'Undersampling' (604 samples)")
print("- Untuk fair comparison: Gunakan 'Original_Cleaned' + weighted loss")
