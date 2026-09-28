"""Pydantic request and response models for the inference API."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

YesNo = Literal["Yes", "No"]
InternetOption = Literal["DSL", "Fiber optic", "No"]
ServiceOption = Literal["Yes", "No", "No internet service"]


class CustomerRequest(BaseModel):
    """Customer inputs; omitted source fields receive documented neutral defaults."""

    model_config = ConfigDict(extra="forbid")

    customer_id: str | None = Field(default=None, max_length=128)
    gender: Literal["Female", "Male"] = "Female"
    senior_citizen: int = Field(default=0, ge=0, le=1)
    partner: YesNo = "No"
    dependents: YesNo = "No"
    tenure: int = Field(default=12, ge=0, le=72)
    phone_service: YesNo = "Yes"
    multiple_lines: Literal["Yes", "No", "No phone service"] = "No"
    internet_service: InternetOption = "DSL"
    online_security: ServiceOption = "No"
    online_backup: ServiceOption = "No"
    device_protection: ServiceOption = "No"
    tech_support: ServiceOption = "No"
    streaming_tv: ServiceOption = "No"
    streaming_movies: ServiceOption = "No"
    contract: Literal["Month-to-month", "One year", "Two year"] = "Month-to-month"
    paperless_billing: YesNo = "Yes"
    payment_method: Literal[
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ] = "Electronic check"
    monthly_charges: float = Field(default=70.0, ge=0, le=1000, allow_inf_nan=False)
    total_charges: float | None = Field(default=840.0, ge=0, le=100000, allow_inf_nan=False)

    @model_validator(mode="before")
    @classmethod
    def default_services_for_internet_choice(cls, values: object) -> object:
        """Fill only omitted internet-service fields with matching Telco values."""
        if not isinstance(values, dict) or values.get("internet_service", "DSL") != "No":
            result = values.copy() if isinstance(values, dict) else values
        else:
            result = values.copy()
            for name in (
                "online_security",
                "online_backup",
                "device_protection",
                "tech_support",
                "streaming_tv",
                "streaming_movies",
            ):
                result.setdefault(name, "No internet service")
        if isinstance(result, dict) and result.get("phone_service", "Yes") == "No":
            result.setdefault("multiple_lines", "No phone service")
        return result

    @model_validator(mode="after")
    def validate_internet_service_fields(self) -> "CustomerRequest":
        """Require the Telco category used for services when no internet is active."""
        internet_services = (
            self.online_security,
            self.online_backup,
            self.device_protection,
            self.tech_support,
            self.streaming_tv,
            self.streaming_movies,
        )
        if self.internet_service == "No" and any(
            value != "No internet service" for value in internet_services
        ):
            raise ValueError(
                "Internet-dependent services must be 'No internet service' when "
                "Internet Service is No."
            )
        if self.internet_service != "No" and any(
            value == "No internet service" for value in internet_services
        ):
            raise ValueError(
                "Use Yes or No for internet-dependent services when Internet Service is active."
            )
        if self.phone_service == "No" and self.multiple_lines != "No phone service":
            raise ValueError("Multiple Lines must be 'No phone service' when Phone Service is No.")
        if self.phone_service == "Yes" and self.multiple_lines == "No phone service":
            raise ValueError(
                "Multiple Lines cannot be 'No phone service' when Phone Service is Yes."
            )
        return self

    def to_model_features(self) -> dict[str, object]:
        """Map API-friendly snake_case names to the existing training column names."""
        return {
            "gender": self.gender,
            "SeniorCitizen": self.senior_citizen,
            "Partner": self.partner,
            "Dependents": self.dependents,
            "tenure": self.tenure,
            "PhoneService": self.phone_service,
            "MultipleLines": self.multiple_lines,
            "InternetService": self.internet_service,
            "OnlineSecurity": self.online_security,
            "OnlineBackup": self.online_backup,
            "DeviceProtection": self.device_protection,
            "TechSupport": self.tech_support,
            "StreamingTV": self.streaming_tv,
            "StreamingMovies": self.streaming_movies,
            "Contract": self.contract,
            "PaperlessBilling": self.paperless_billing,
            "PaymentMethod": self.payment_method,
            "MonthlyCharges": self.monthly_charges,
            "TotalCharges": self.total_charges,
        }


class PredictionResponse(BaseModel):
    """Safe inference response without customer input values."""

    prediction: Literal[0, 1]
    churn_probability: float = Field(ge=0, le=1)
    risk_level: Literal["High Risk", "Medium Risk", "Low Risk"]
    model_version: str


class BatchPredictionRequest(BaseModel):
    """Bounded list request to keep batch inference memory use predictable."""

    model_config = ConfigDict(extra="forbid")

    customers: list[CustomerRequest] = Field(min_length=1, max_length=1000)


class BatchPredictionItem(PredictionResponse):
    """Minimal per-row batch output with the caller's optional key."""

    customer_id: str | None = None


class BatchPredictionResponse(BaseModel):
    """Batch outputs and model version used for every row."""

    predictions: list[BatchPredictionItem]
    model_version: str
