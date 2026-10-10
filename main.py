
import json
import os
from io import BytesIO

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from google import genai
from google.genai import types
from PIL import Image, UnidentifiedImageError
from pydantic import ValidationError

from schema import BillExtraction


load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_NAME = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.5-flash-lite",
)
app = FastAPI(
    title="WTQ Utility Bill Decoder",
    description="Extract structured information from Pakistani electricity bills.",
    version="1.0.0",
)

# Allow the local React frontend to connect during development.
# Restrict this to your actual frontend origin in production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB

ALLOWED_FORMATS = {
    "JPEG": "image/jpeg",
    "PNG": "image/png",
    "WEBP": "image/webp",
}

EXTRACTION_PROMPT = """
You are a careful electricity-bill data extraction system for Pakistan.

Read the attached electricity bill image and extract its printed billing
information for the WTQ Utility Bill Decoder challenge.

Supported providers:
- K-Electric, represented as KE
- LESCO
- IESCO

Rules:

1. Extract values exactly as printed. Do not recalculate or correct amounts.
2. Return only fields defined in the supplied JSON schema.
3. Never include customer names, addresses, CNICs, account numbers,
   reference numbers, consumer numbers, or meter numbers.
4. Do not copy personal identifiers into other fields.
5. Use null for scalar values that are missing, hidden, covered, or unreadable.
6. Use an empty array when no charge or tax entries can be identified.
7. Return money as numeric PKR amounts without currency symbols or commas.
8. Preserve negative credits and subsidies as negative numbers.
9. Normalize K-Electric to provider KE.
10. Normalize dates to YYYY-MM-DD and billing months to YYYY-MM.
11. Classify charges and taxes using the supplied schema.
12. The printed section determines whether an entry is a charge or a tax.
13. Skip blank and zero-valued charge and tax lines.
14. Preserve separate entries when a charge type appears more than once.
15. Use the printed subtotal when available. Do not invent subtotals.
16. If only a combined tax amount is printed, use an empty taxes array
    and put the combined amount in total_taxes.
17. If multiple late-payment amounts are printed, use the highest amount
    for payable_after_due_date.
18. Do not guess unreadable digits or infer values from identifiers.
19. If the provider cannot be identified, do not invent one. The application
    will report that the bill requires manual review.

Carefully inspect the complete image, including small text, tables,
dates, meter readings, adjustment lines, and totals.
"""

client = genai.Client(api_key=API_KEY) if API_KEY else None


@app.get("/")
def root():
    return {
        "message": "WTQ Utility Bill Decoder API",
        "docs": "/docs",
        "status": "running",
    }


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "gemini_configured": client is not None,
        "model": MODEL_NAME,
    }


@app.post("/api/bills/analyze")
def analyze_bill(file: UploadFile = File(...)):
    if client is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Gemini is not configured. Add GEMINI_API_KEY "
                "to the backend .env file and restart the server."
            ),
        )

    # Read at most 10 MB plus one byte, to detect oversized uploads.
    image_bytes = file.file.read(MAX_FILE_SIZE + 1)

    if not image_bytes:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is empty.",
        )

    if len(image_bytes) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="The image is too large. Maximum size is 10 MB.",
        )

    # Check the actual image content, not just the filename.
    try:
        with Image.open(BytesIO(image_bytes)) as image:
            image_format = image.format
            image.verify()

        if image_format not in ALLOWED_FORMATS:
            raise HTTPException(
                status_code=415,
                detail="Please upload a JPG, PNG, or WebP image.",
            )

        mime_type = ALLOWED_FORMATS[image_format]

    except (UnidentifiedImageError, OSError, ValueError):
        raise HTTPException(
            status_code=400,
            detail="This file is not a valid supported image.",
        )

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=[
                EXTRACTION_PROMPT,
                types.Part.from_bytes(
                    data=image_bytes,
                    mime_type=mime_type,
                ),
            ],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_json_schema=(
                    BillExtraction.model_json_schema()
                ),
                temperature=0,
            ),
        )

        if not response.text:
            raise HTTPException(
                status_code=502,
                detail="Gemini returned an empty response. Please retry.",
            )

        # Parse and validate the JSON against our Pydantic schema.
        bill = BillExtraction.model_validate_json(response.text)

        # Explicitly return all schema fields, including null values.
        return bill.model_dump(mode="json")

    except HTTPException:
        raise

    except ValidationError:
        raise HTTPException(
            status_code=502,
            detail=(
                "The extracted data did not match the required schema. "
                "Please retry or inspect the bill manually."
            ),
        )

    except Exception as exc:
        # Do not return provider errors, credentials, or stack traces
        # to the user.
        print(f"Bill extraction failed: {type(exc).__name__}")

        raise HTTPException(
            status_code=502,
            detail=(
                "Bill analysis failed. Check the API key, model "
                "availability, internet connection, and API quota."
            ),
        )
