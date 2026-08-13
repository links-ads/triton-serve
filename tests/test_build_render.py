from pathlib import Path

from triton_serve.builder.render import render_dockerfile, render_requirements, write_build_context


def test_pip_only_dockerfile(build_spec):
    spec = build_spec(pip_packages=["numpy==1.26.0", "pillow==10.0.0"])
    assert render_dockerfile(spec) == (
        f"FROM {spec.base_image}\n"
        "\n"
        "COPY requirements.txt /tmp/requirements.txt\n"
        "RUN pip install --no-cache-dir -r /tmp/requirements.txt \\\n"
        "    && rm /tmp/requirements.txt\n"
    )


def test_apt_block_is_emitted_before_pip(build_spec):
    spec = build_spec(apt_packages=["libgl1"], pip_packages=["numpy==1.26.0"])
    rendered = render_dockerfile(spec)
    assert rendered.index("apt-get install") < rendered.index("pip install")
    assert "apt-get install -y --no-install-recommends libgl1" in rendered
    assert "rm -rf /var/lib/apt/lists/*" in rendered


def test_apt_only_omits_the_pip_block(build_spec):
    rendered = render_dockerfile(build_spec(apt_packages=["libgl1"]))
    assert "pip install" not in rendered
    assert "COPY requirements.txt" not in rendered


def test_requirements_are_one_per_line_sorted(build_spec):
    spec = build_spec(pip_packages=["pillow==10.0.0", "numpy==1.26.0"])
    assert render_requirements(spec) == "numpy==1.26.0\npillow==10.0.0\n"


def test_write_build_context_writes_both_files(build_spec, tmp_path: Path):
    spec = build_spec(pip_packages=["numpy==1.26.0"])
    write_build_context(spec, tmp_path)
    assert (tmp_path / "Dockerfile").read_text() == render_dockerfile(spec)
    assert (tmp_path / "requirements.txt").read_text() == render_requirements(spec)
