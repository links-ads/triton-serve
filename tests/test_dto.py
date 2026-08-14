"""Regression tests for #134: three field validators were stacked `@classmethod` above
`@field_validator`, which makes pydantic skip them entirely rather than fail loudly."""

import pytest
from pydantic import ValidationError

from triton_serve.api.dto import APIKeyUpdateBody, ModelUpdateBody, ServiceCreateResources


@pytest.mark.parametrize(
    "model, field",
    [
        (ServiceCreateResources, "validate_units"),
        (ModelUpdateBody, "validate_name"),
        (APIKeyUpdateBody, "validate_project"),
    ],
)
def test_field_validators_are_registered(model, field):
    assert field in model.__pydantic_decorators__.field_validators


@pytest.mark.parametrize("value, expected", [("2G", 2048), ("512M", 512), ("1g", 1024), (256, 256)])
def test_size_units_are_converted_to_megabytes(value, expected):
    assert ServiceCreateResources(shm_size=value).shm_size == expected
    assert ServiceCreateResources(mem_size=value).mem_size == expected


def test_unknown_size_unit_is_rejected():
    with pytest.raises(ValidationError, match="Invalid unit"):
        ServiceCreateResources(shm_size="2X")


@pytest.mark.parametrize("name", ["", "   ", "caffè"])
def test_blank_or_non_ascii_model_name_is_rejected(name):
    with pytest.raises(ValidationError):
        ModelUpdateBody(name=name)


def test_blank_project_is_rejected():
    with pytest.raises(ValidationError, match="Project name cannot be empty"):
        APIKeyUpdateBody(project="   ")
