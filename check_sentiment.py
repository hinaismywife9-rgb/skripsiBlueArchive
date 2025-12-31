import pandas as pd
import os

# Change to the correct directory
excel_file = 'sentiment analysis BA.xlsx'

# Read the Excel file
df = pd.read_excel(excel_file)

print("=" * 60)
print("ANALISIS SENTIMENT BALANCE")
print("=" * 60)

print(f"\nUkuran Data: {df.shape[0]} baris, {df.shape[1]} kolom")
print(f"\nNama Kolom: {df.columns.tolist()}")

print("\n" + "=" * 60)
print("DISTRIBUSI SENTIMENT:")
print("=" * 60)

if 'sentiment' in df.columns:
    sentiment_counts = df['sentiment'].value_counts()
    sentiment_pct = df['sentiment'].value_counts(normalize=True) * 100
    
    for sentiment in sentiment_counts.index:
        count = sentiment_counts[sentiment]
        pct = sentiment_pct[sentiment]
        print(f"{sentiment:20s}: {count:4d} ({pct:6.2f}%)")
    
    print("\n" + "=" * 60)
    print("ANALISIS BALANCE:")
    print("=" * 60)
    
    max_count = sentiment_counts.max()
    min_count = sentiment_counts.min()
    ratio = max_count / min_count if min_count > 0 else 0
    
    print(f"Total data: {len(df)}")
    print(f"Class terbanyak: {sentiment_counts.idxmax()} ({max_count})")
    print(f"Class tersedikit: {sentiment_counts.idxmin()} ({min_count})")
    print(f"Ratio (max/min): {ratio:.2f}")
    
    if ratio <= 1.2:
        print("\n✓ DATA SUDAH BALANCE (Ratio ≤ 1.2)")
    elif ratio <= 2.0:
        print("\n⚠ DATA CUKUP BALANCE (Ratio ≤ 2.0)")
    else:
        print("\n✗ DATA TIDAK BALANCE (Ratio > 2.0)")
else:
    print("Kolom 'sentiment' tidak ditemukan!")
    print(f"Kolom yang tersedia: {df.columns.tolist()}")

print("\n" + "=" * 60)
print("PREVIEW DATA:")
print("=" * 60)
print(df.head(10))
