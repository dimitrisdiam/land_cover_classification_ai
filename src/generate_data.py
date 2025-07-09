"""
Generates a synthetic land cover classification dataset using simulated satellite spectral indices.
"""

import numpy as np
import pandas as pd
from pathlib import Path

def generate_sample_data(n_samples: int = 2000, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)
    land_types = ['forest', 'oil_palm', 'cocoa', 'urban', 'water']
    data = []

    for _ in range(n_samples):
        label = np.random.choice(land_types, p=[0.3, 0.2, 0.2, 0.2, 0.1])
        if label == 'forest':
            ndvi = np.random.uniform(0.6, 0.9)
            nir = np.random.uniform(0.7, 0.9)
            red = np.random.uniform(0.1, 0.3)
            blue = np.random.uniform(0.05, 0.15)
            swir = np.random.uniform(0.2, 0.4)
        elif label == 'oil_palm':
            ndvi = np.random.uniform(0.4, 0.6)
            nir = np.random.uniform(0.6, 0.75)
            red = np.random.uniform(0.2, 0.4)
            blue = np.random.uniform(0.1, 0.2)
            swir = np.random.uniform(0.3, 0.45)
        elif label == 'cocoa':
            ndvi = np.random.uniform(0.4, 0.65)
            nir = np.random.uniform(0.55, 0.7)
            red = np.random.uniform(0.2, 0.35)
            blue = np.random.uniform(0.1, 0.2)
            swir = np.random.uniform(0.35, 0.5)
        elif label == 'urban':
            ndvi = np.random.uniform(0.1, 0.3)
            nir = np.random.uniform(0.2, 0.4)
            red = np.random.uniform(0.4, 0.6)
            blue = np.random.uniform(0.3, 0.5)
            swir = np.random.uniform(0.5, 0.7)
        else:  # water
            ndvi = np.random.uniform(-0.1, 0.1)
            nir = np.random.uniform(0.05, 0.2)
            red = np.random.uniform(0.05, 0.2)
            blue = np.random.uniform(0.4, 0.6)
            swir = np.random.uniform(0.01, 0.1)

        data.append([ndvi, nir, red, blue, swir, label])

    return pd.DataFrame(data, columns=['NDVI', 'NIR', 'RED', 'BLUE', 'SWIR', 'label'])

def main():
    df = generate_sample_data()
    Path('data').mkdir(parents=True, exist_ok=True)
    df.to_csv('land_cover_classification_ai/data/synthetic_land_cover.csv', index=False)
    

if __name__ == '__main__':
    main()

