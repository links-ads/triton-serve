from pathlib import Path

from triton_serve.builder.spec import BuildSpec

APT_BLOCK = """RUN apt-get update \\
    && apt-get install -y --no-install-recommends {packages} \\
    && rm -rf /var/lib/apt/lists/*
"""

PIP_BLOCK = """COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt \\
    && rm /tmp/requirements.txt
"""


def render_dockerfile(spec: BuildSpec) -> str:
    """Renders the Dockerfile for a spec.

    Requirements are installed from a file in the build context rather than as command-line
    arguments, which is what keeps unvalidated strings out of the pip invocation entirely.

    Args:
        spec (BuildSpec): The validated spec to render.

    Returns:
        str: The Dockerfile text.
    """
    blocks = [f"FROM {spec.base_image}\n"]
    if spec.apt_packages:
        blocks.append(APT_BLOCK.format(packages=" ".join(spec.apt_packages)))
    if spec.pip_packages:
        blocks.append(PIP_BLOCK)
    return "\n".join(blocks)


def render_requirements(spec: BuildSpec) -> str:
    """Renders the requirements.txt placed in the build context."""
    return "".join(f"{package}\n" for package in spec.pip_packages)


def write_build_context(spec: BuildSpec, path: Path) -> None:
    """Writes the Dockerfile and requirements.txt for a spec into an existing directory."""
    (path / "Dockerfile").write_text(render_dockerfile(spec))
    (path / "requirements.txt").write_text(render_requirements(spec))
