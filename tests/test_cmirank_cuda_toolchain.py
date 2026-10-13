import tarfile
import json

import pytest

from src.cmirank.cuda_toolchain import failed_cuda_source, validate_archive_members, verify_cuda_toolchain


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


def test_reusable_compiler_receipt_requires_its_previously_reviewed_hash(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    private = tmp_path / "private"
    path = root / "configs/cmirank/cuda_toolchain_v1.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"run_id":"compiler-run"}))
    run = private / "runs/compiler-run"
    run.mkdir(parents=True)
    (run / "report.json").write_text('{}')
    monkeypatch.setenv("MEMREC_EXPECTED_COMMIT", "new-consumer")
    with pytest.raises(ValueError, match="Reviewed reusable compiler receipt changed"):
        verify_cuda_toolchain(private, root, {"toolchain_prepare_source_commit":"old-producer",
            "toolchain_prepare_report_sha256":"not-the-actual-review-hash"})
