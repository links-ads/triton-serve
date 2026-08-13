import pytest

from triton_serve.builder.spec import make_build_spec

HOSTS = ["pypi.org", "files.pythonhosted.org", "download.pytorch.org"]
BASE = "ghcr.io/links-ads/serve-triton:23.07-py3"


def _spec(**kwargs):
    params = {"base_image": BASE, "apt_packages": [], "pip_packages": [], "allowed_index_hosts": HOSTS}
    return make_build_spec(**{**params, **kwargs})


def test_hash_is_sha256_hex():
    assert len(_spec(pip_packages=["numpy==1.26.0"]).image_hash) == 64


def test_hash_is_order_invariant():
    a = _spec(pip_packages=["numpy==1.26.0", "torch==2.1.0"])
    b = _spec(pip_packages=["torch==2.1.0", "numpy==1.26.0"])
    assert a.image_hash == b.image_hash


def test_packages_are_deduplicated():
    spec = _spec(pip_packages=["numpy==1.26.0", "numpy==1.26.0"])
    assert spec.pip_packages == ("numpy==1.26.0",)


def test_different_base_image_changes_the_hash():
    assert _spec().image_hash != _spec(base_image="python:3.12-slim").image_hash


def test_empty_spec_is_flagged_empty():
    assert _spec().is_empty
    assert not _spec(apt_packages=["libgl1"]).is_empty


def test_rejects_bad_apt_name():
    with pytest.raises(ValueError, match="apt package"):
        _spec(apt_packages=["libgl1; rm -rf /"])


def test_rejects_unparseable_pip_requirement():
    with pytest.raises(ValueError, match="requirement"):
        _spec(pip_packages=["--extra-index-url http://attacker.example"])


def test_rejects_index_url_on_disallowed_host():
    with pytest.raises(ValueError, match="index host"):
        _spec(pip_index_url="https://attacker.example/simple")


def test_rejects_non_https_index_url():
    with pytest.raises(ValueError, match="https"):
        _spec(pip_index_url="http://pypi.org/simple")


def test_accepts_allowed_index_host():
    spec = _spec(pip_extra_index_urls=["https://download.pytorch.org/whl/cu121"])
    assert spec.pip_extra_index_urls == ("https://download.pytorch.org/whl/cu121",)
