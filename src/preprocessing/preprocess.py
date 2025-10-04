#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))


from src.ingestion.ingest_data import load_data
def preprocess(df):
    X = df.drop(columns=['target'])
    y = df['target']
    return X, y


if __name__ == "__main__":
    df = load_data()
    X, y = preprocess(df)
    print(X.head(), y.head())