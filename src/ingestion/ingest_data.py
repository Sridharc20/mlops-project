#!/usr/bin/env python3


from sklearn.datasets import load_iris

import pandas as pd

def load_data():
    iris_data = load_iris(as_frame=True)
    df = iris_data.frame
    print(f"loaded frame {len(df)} records")
    return df


if __name__ == "__main__":
    print("Data Ingestion")
    data = load_data()
    print(data.head())
    