
import json
import os
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import Image

# Run this script from the backend folder.
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "test-data"
CSV_PATH = DATA_DIR / "level2.csv"
OUTPUT_PATH = DATA_DIR / "level2_answered.csv"

load_dotenv(BASE_DIR / ".env")

API_KEY = os.getenv("GEMINI_API_KEY")
MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")

if not API_KEY:
    raise RuntimeError("GEMINI_API_KEY is missing from backend/.env")

client = genai.Client(api_key=API_KEY)

SYSTEM_PROMPT = """
You analyze Pakistani electricity bills and answer questions about them.

Inspect the entire image, especially the 12-month consumption history,
monthly amounts, FPA/FCA, net-metering/import/export readings,
current charges, taxes, due dates, and late-payment amounts.

Answer the supplied question using evidence visible in the bill.
Do not invent missing information. If the image does not contain enough
evidence, say that the bill does not provide enough information.

For monthly history:
- Report the month and units exactly as printed.
- For comparisons, use only months actually visible.
- Do not claim a payment was made just because it appears on a bill.
- Distinguish billed amount from amount actually paid.
- Clearly label estimates and explain assumptions.
- For summer versus winter, compare the available June-August months
  against December-February months. State if months are missing.
- For average monthly units, use all readable monthly-history values.
- Return a concise, useful answer in plain text.

Return JSON with exactly one property: "answer" (a string).
"""

def image_for_bill(bill_id: str) -> Path:
    matches = list(DATA_DIR.glob(f"{bill_id}.*"))
    matches = [
        p for p in matches
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    ]
    if not matches:
        raise FileNotFoundError(f"No image found for bill_id {bill_id}")
    return matches[0]


def answer_question(image_path: Path, question: str) -> str:
    image_bytes = image_path.read_bytes()

    with Image.open(image_path) as img:
        img_format = img.format
        img.verify()

    mime_types = {
        "JPEG": "image/jpeg",
        "PNG": "image/png",
        "WEBP": "image/webp",
    }
    mime_type = mime_types.get(img_format)
    if not mime_type:
        return "Unable to process: unsupported image format."

    prompt = f"""
{SYSTEM_PROMPT}

Question to answer:
{question}
"""

    response = client.models.generate_content(
        model=MODEL,
        contents=[
            prompt,
            types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
        ],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_json_schema={
                "type": "OBJECT",
                "properties": {
                    "answer": {"type": "STRING"}
                },
                "required": ["answer"],
            },
            temperature=0,
        ),
    )

    if not response.text:
        return "Unable to answer: model returned an empty response."

    data = json.loads(response.text)
    return str(data.get("answer", "")).strip() or (
        "Unable to answer from the available bill image."
    )


def main():
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"CSV not found: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH, dtype={"answer": "string"})
    df["answer"] = df["answer"].fillna("").astype(str) 

    required = {"bill_id", "question_id", "question", "answer"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing columns: {sorted(missing)}")

    # Reuse an answer if the script is rerun and the row is already complete.
    for index, row in df.iterrows():
        existing = row.get("answer")
        if pd.notna(existing) and str(existing).strip():
            continue

        bill_id = str(row["bill_id"]).strip()
        question_id = str(row["question_id"]).strip()
        question = str(row["question"]).strip()

        try:
            image_path = image_for_bill(bill_id)
            print(
                f"[{index + 1}/{len(df)}] "
                f"{bill_id} {question_id} - analyzing..."
            )
            answer = answer_question(image_path, question)
            df.at[index, "answer"] = answer

        except Exception as exc:
            print(f"  Error: {type(exc).__name__}: {exc}")
            df.at[index, "answer"] = (
                "Unable to answer automatically; manual review required."
            )

        # Save progress after every row so interrupted runs can resume.
        df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

        # A small pause between API requests.
        time.sleep(1)

    print(f"\nFinished. Results saved to:\n{OUTPUT_PATH}")


if __name__ == "__main__":
    main()
