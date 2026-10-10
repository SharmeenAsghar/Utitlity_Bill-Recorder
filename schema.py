
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


Provider = Literal["KE", "LESCO", "IESCO"]

ChargeType = Literal[
    "energy",
    "fixed",
    "fpa",
    "quarterly_adjustment",
    "surcharge",
    "meter_rent",
    "subsidy",
    "other",
]

TaxType = Literal[
    "gst",
    "electricity_duty",
    "income_tax",
    "municipal_tax",
    "other_tax",
]


class Charge(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: ChargeType
    amount: float


class Tax(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: TaxType
    amount: float


class BillExtraction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: Provider | None
    tariff: str | None
    sanctioned_load_kw: float | None
    bill_month: str | None
    reading_date: str | None
    issue_date: str | None
    due_date: str | None
    previous_reading: float | None
    current_reading: float | None
    units_consumed: float | None
    charges: list[Charge] = Field(default_factory=list)
    total_charges: float | None
    taxes: list[Tax] = Field(default_factory=list)
    total_taxes: float | None
    current_bill: float | None
    arrears: float | None
    payable_within_due_date: float | None
    payable_after_due_date: float | None
