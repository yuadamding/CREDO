"""Structured validation reports for study and storage contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

IssueLocationPart: TypeAlias = str | int


@dataclass(frozen=True)
class ValidationIssue:
    """One machine-readable validation finding."""

    level: Literal["error", "warning"]
    code: str
    message: str
    location: tuple[IssueLocationPart, ...] = ()

    def __post_init__(self) -> None:
        if self.level not in {"error", "warning"}:
            raise ValueError("ValidationIssue.level must be 'error' or 'warning'.")
        if not str(self.code) or not str(self.message):
            raise ValueError("ValidationIssue code and message must be nonempty.")
        object.__setattr__(self, "code", str(self.code))
        object.__setattr__(self, "message", str(self.message))
        object.__setattr__(self, "location", tuple(self.location))
        if any(not isinstance(part, (str, int)) for part in self.location):
            raise TypeError("ValidationIssue.location parts must be strings or integers.")


@dataclass(frozen=True)
class ValidationReport:
    """Collection of validation findings with an explicit failure boundary."""

    issues: tuple[ValidationIssue, ...] = ()

    @property
    def errors(self) -> tuple[ValidationIssue, ...]:
        return tuple(issue for issue in self.issues if issue.level == "error")

    @property
    def warnings(self) -> tuple[ValidationIssue, ...]:
        return tuple(issue for issue in self.issues if issue.level == "warning")

    @property
    def valid(self) -> bool:
        return not self.errors

    def merged(self, *reports: ValidationReport) -> ValidationReport:
        return ValidationReport(
            self.issues + tuple(issue for report in reports for issue in report.issues)
        )

    def raise_for_errors(self) -> None:
        if self.errors:
            detail = "; ".join(
                f"{issue.code}{_format_location(issue.location)}: {issue.message}"
                for issue in self.errors
            )
            raise ValueError(f"Study validation failed: {detail}")


def _format_location(location: tuple[IssueLocationPart, ...]) -> str:
    return "" if not location else f" at {'.'.join(map(str, location))}"


__all__ = ["ValidationIssue", "ValidationReport"]
