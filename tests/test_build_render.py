from pathlib import Path

from triton_serve.builder.render import render_dockerfile, render_requirements, write_build_context
from triton_serve.builder.spec import make_build_spec

HOSTS = ["pypi.org", "files.pythonhosted.org", "download.pytorch.org"]
BASE = "ghcr.io/links-ads/serve-triton:23.07-py3"


def _spec(**kwargs):
    params = {"base_image": BASE, "apt_packages": [], "pip_packages": [], "allowed_index_hosts": HOSTS}
    return make_build_spec(**{**params, **kwargs})


def test_pip_only_dockerfile():
    spec = _spec(pip_packages=["numpy==1.26.0", "pillow==10.0.0"])
    assert render_dockerfile(spec) == (
        f"FROM {BASE}\n"
        "\n"
        "COPY requirements.txt /tmp/requirements.txt\n"
        "RUN pip install --no-cache-dir -r /tmp/requirements.txt \\\n"
        "    && rm /tmp/requirements.txt\n"
    )


def test_apt_block_is_emitted_before_pip():
    spec = _spec(apt_packages=["libgl1"], pip_packages=["numpy==1.26.0"])
    rendered = render_dockerfile(spec)
    assert rendered.index("apt-get install") < rendered.index("pip install")
    assert "apt-get install -y --no-install-recommends libgl1" in rendered
    assert "rm -rf /var/lib/apt/lists/*" in rendered


def test_apt_only_omits_the_pip_block():
    rendered = render_dockerfile(_spec(apt_packages=["libgl1"]))
    assert "pip install" not in rendered
    assert "COPY requirements.txt" not in rendered


def test_index_urls_become_pip_flags():
    spec = _spec(
        pip_packages=["torch==2.1.0"],
        pip_index_url="https://pypi.org/simple",
        pip_extra_index_urls=["https://download.pytorch.org/whl/cu121"],
    )
    rendered = render_dockerfile(spec)
    assert "--index-url https://pypi.org/simple" in rendered
    assert "--extra-index-url https://download.pytorch.org/whl/cu121" in rendered


def test_requirements_are_one_per_line_sorted():
    spec = _spec(pip_packages=["pillow==10.0.0", "numpy==1.26.0"])
    assert render_requirements(spec) == "numpy==1.26.0\npillow==10.0.0\n"


def test_write_build_context_writes_both_files(tmp_path: Path):
    spec = _spec(pip_packages=["numpy==1.26.0"])
    write_build_context(spec, tmp_path)
    assert (tmp_path / "Dockerfile").read_text() == render_dockerfile(spec)
    assert (tmp_path / "requirements.txt").read_text() == render_requirements(spec)
