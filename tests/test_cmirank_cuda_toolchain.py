import tarfile

import pytest

from src.cmirank.cuda_toolchain import failed_cuda_source, validate_archive_members


def member(name, kind=tarfile.REGTYPE, link=""):
    value = tarfile.TarInfo(name)
    value.type, value.linkname = kind, link
    return value


def test_official_component_keeps_relative_links_within_single_root():
    assert validate_archive_members([member("compiler/bin/nvcc"),
        member("compiler/lib/libcudart.so", tarfile.SYMTYPE, "libcudart.so.12")]) == "compiler"


@pytest.mark.parametrize("unsafe", [member("/etc/a"), member("compiler/../../a"),
    member("compiler/bin/a", tarfile.SYMTYPE, "../../escape"),
    member("compiler/bin/a", tarfile.SYMTYPE, "/usr/local/cuda/bin/nvcc"),
    member("compiler/device", tarfile.CHRTYPE), member("other/bin/a")])
def test_archive_traversal_and_special_files_fail_closed(unsafe):
    with pytest.raises(ValueError):
        validate_archive_members([member("compiler/bin/nvcc"), unsafe])


def test_compiler_replay_uses_only_exact_failed_generated_source():
    log = "trace\nRuntimeError: #include <x>\nvoid kernel_kernel() {}\n\nCompilation error:\ncp.async.bulk.tensor\nCommand: old"
    assert failed_cuda_source(log) == "#include <x>\nvoid kernel_kernel() {}\n"
    with pytest.raises(ValueError):
        failed_cuda_source(log.replace("cp.async.bulk.tensor", "other"))
