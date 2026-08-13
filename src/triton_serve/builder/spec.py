import hashlib
import json
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from urllib.parse import urlparse

from packaging.requirements import InvalidRequirement, Requirement

APT_NAME = re.compile(r"^[a-z0-9][a-z0-9+._-]*$")


@dataclass(frozen=True)
class BuildSpec:
    """A validated, normalized description of an image to build. Content-addressed by `image_hash`."""

    base_image: str
    apt_packages: tuple[str, ...]
    pip_packages: tuple[str, ...]
    pip_index_url: str | None
    pip_extra_index_urls: tuple[str, ...]

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
                "pip_index_url": self.pip_index_url,
                "pip_extra_index_urls": list(self.pip_extra_index_urls),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(canonical.encode()).hexdigest()


def _validated_apt(packages: Iterable[str]) -> tuple[str, ...]:
    names = sorted({p.strip() for p in packages if p.strip()})
    for name in names:
        if not APT_NAME.match(name):
            raise ValueError(f"invalid apt package name: {name!r}")
    return tuple(names)


def _validated_pip(packages: Iterable[str]) -> tuple[str, ...]:
    lines = sorted({p.strip() for p in packages if p.strip()})
    for line in lines:
        try:
            Requirement(line)
        except InvalidRequirement as e:
            raise ValueError(f"invalid pip requirement: {line!r} ({e})") from e
    return tuple(lines)


def _validated_index(url: str, allowed_hosts: Sequence[str]) -> str:
    parsed = urlparse(url)
    if parsed.scheme != "https":
        raise ValueError(f"index url must use https: {url!r}")
    if parsed.hostname not in allowed_hosts:
        raise ValueError(f"disallowed index host: {parsed.hostname!r}")
    return url


def make_build_spec(
    base_image: str,
    apt_packages: Iterable[str],
    pip_packages: Iterable[str],
    allowed_index_hosts: Sequence[str],
    pip_index_url: str | None = None,
    pip_extra_index_urls: Iterable[str] = (),
) -> BuildSpec:
    """Validates and normalizes raw user input into a hashable spec.

    Args:
        base_image (str): The image to build on top of.
        apt_packages (Iterable[str]): System package names.
        pip_packages (Iterable[str]): PEP 508 requirement strings.
        allowed_index_hosts (Sequence[str]): Hosts index urls may point at.
        pip_index_url (str | None): Replacement primary index.
        pip_extra_index_urls (Iterable[str]): Additional indexes.

    Returns:
        BuildSpec: The normalized spec, with packages sorted and deduplicated.

    Raises:
        ValueError: If any package name, requirement, or index url is invalid.
    """
    if not base_image.strip():
        raise ValueError("base image is required")
    return BuildSpec(
        base_image=base_image.strip(),
        apt_packages=_validated_apt(apt_packages),
        pip_packages=_validated_pip(pip_packages),
        pip_index_url=_validated_index(pip_index_url, allowed_index_hosts) if pip_index_url else None,
        pip_extra_index_urls=tuple(
            _validated_index(u, allowed_index_hosts) for u in sorted(set(pip_extra_index_urls))
        ),
    )
