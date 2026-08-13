import hashlib
import json
import re
from collections.abc import Iterable
from dataclasses import dataclass

from packaging.requirements import InvalidRequirement, Requirement

APT_NAME = re.compile(r"^[a-z0-9][a-z0-9+._-]*$")


@dataclass(frozen=True)
class BuildSpec:
    """A validated, normalized description of an image to build. Content-addressed by `image_hash`."""

    base_image: str
    apt_packages: tuple[str, ...]
    pip_packages: tuple[str, ...]

    @property
    def is_empty(self) -> bool:
        return not self.apt_packages and not self.pip_packages

    @property
    def image_hash(self) -> str:
        canonical = json.dumps(
            {
                "base_image": self.base_image,
                "apt_packages": list(self.apt_packages),
                "pip_packages": list(self.pip_packages),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(canonical.encode()).hexdigest()


def validated_apt(packages: Iterable[str]) -> tuple[str, ...]:
    """Sorted, deduplicated apt package names.

    Raises:
        ValueError: If a name is not a valid Debian package name.
    """
    names = sorted({p.strip() for p in packages if p.strip()})
    for name in names:
        if not APT_NAME.match(name):
            raise ValueError(f"invalid apt package name: {name!r}")
    return tuple(names)


def validated_pip(packages: Iterable[str]) -> tuple[str, ...]:
    """Sorted, deduplicated pip requirements.

    Raises:
        ValueError: If a line is not a valid PEP 508 requirement.
    """
    lines = sorted({p.strip() for p in packages if p.strip()})
    for line in lines:
        try:
            Requirement(line)
        except InvalidRequirement as e:
            raise ValueError(f"invalid pip requirement: {line!r} ({e})") from e
    return tuple(lines)


def make_build_spec(base_image: str, apt_packages: Iterable[str], pip_packages: Iterable[str]) -> BuildSpec:
    """Validates and normalizes raw user input into a hashable spec.

    Args:
        base_image (str): The image to build on top of.
        apt_packages (Iterable[str]): System package names.
        pip_packages (Iterable[str]): PEP 508 requirement strings.

    Returns:
        BuildSpec: The normalized spec, with packages sorted and deduplicated.

    Raises:
        ValueError: If any package name or requirement is invalid.
    """
    if not base_image.strip():
        raise ValueError("base image is required")
    return BuildSpec(
        base_image=base_image.strip(),
        apt_packages=validated_apt(apt_packages),
        pip_packages=validated_pip(pip_packages),
    )
