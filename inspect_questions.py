
from pathlib import Path
import pandas as pd

csv_path = Path("test-data/level2.csv")

if not csv_path.exists():
    raise FileNotFoundError(
        f"Cannot find {csv_path}. Check the filename and folder."
    )

df = pd.read_csv(csv_path)

print("Columns:", df.columns.tolist())
print("Total questions:", len(df))

if "bill_id" in df.columns:
    print("\nQuestions per bill:")
    print(df.groupby("bill_id").size().to_string())

if {"bill_id", "question_id", "question"}.issubset(df.columns):
    print("\nSample questions:")
    print(
        df[["bill_id", "question_id", "question"]]
        .head(12)
        .to_string(index=False)
    )
else:
    print("\nFirst 5 rows:")
    print(df.head().to_string(index=False))
