from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
file_path = PROJECT_ROOT / "data/raw/crypto/XRP.csv"


try:
    df = pd.read_csv(file_path)

    print("Available columns:")
    print(list(df.columns))
    print("-" * 30)

    column_name = 'Close'

    if column_name in df.columns:
        plt.figure(figsize=(10, 6))
        plt.plot(df[column_name], marker="o", linestyle="-", color="b")

        plt.title(f"Graph of {column_name}")
        plt.xlabel("Index")
        plt.ylabel(column_name)
        plt.grid(True)

        plt.show()
    else:
        print(f"Error: Column '{column_name}' not found.")

except FileNotFoundError:
    print(f"Error: File '{file_path}' not found.")
except Exception as e:
    print(f"An error occurred: {e}")
