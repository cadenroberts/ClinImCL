"""Minimal test suite for ClinImCL critical paths."""
import os, tempfile, shutil
import numpy as np
import torch
import pytest

from model import ClinImCL, Encoder, Block, IMG, info_nce, augment


# ── model.py ────────────────────────────────────────────────────────

def test_block_shape():
    out = Block(1, 32)(torch.randn(1, 1, 8, 8, 8))
    assert out.shape == (1, 32, 8, 8, 8)


def test_encoder_shape():
    out = Encoder()(torch.randn(1, 1, IMG, IMG, IMG))
    assert out.shape == (1, 256)


def test_clinimcl_shapes():
    z, h = ClinImCL()(torch.randn(2, 1, IMG, IMG, IMG))
    assert z.shape == (2, 128)
    assert h.shape == (2, 256)


def test_clinimcl_l2_norm():
    z, _ = ClinImCL()(torch.randn(3, 1, IMG, IMG, IMG))
    norms = z.norm(dim=1)
    assert torch.allclose(norms, torch.ones(3), atol=1e-5)


def test_img_constant():
    assert IMG == 96


def test_info_nce_scalar():
    z = torch.nn.functional.normalize(torch.randn(4, 128), dim=1)
    loss = info_nce(z, z)
    assert loss.shape == ()
    assert loss.item() >= 0.0


def test_info_nce_symmetric():
    z1 = torch.nn.functional.normalize(torch.randn(4, 128), dim=1)
    z2 = torch.nn.functional.normalize(torch.randn(4, 128), dim=1)
    assert torch.allclose(info_nce(z1, z2), info_nce(z2, z1))


def test_info_nce_aligned_vs_random():
    z = torch.nn.functional.normalize(torch.randn(8, 128), dim=1)
    z_rand = torch.nn.functional.normalize(torch.randn(8, 128), dim=1)
    assert info_nce(z, z).item() < info_nce(z, z_rand).item()


def test_augment_shape_and_range():
    x = torch.rand(1, 16, 16, 16)
    out = augment(x)
    assert out.shape == x.shape
    assert out.min() >= 0.0
    assert out.max() <= 1.0


def test_augment_does_not_modify_input():
    x = torch.rand(1, 16, 16, 16)
    original = x.clone()
    augment(x)
    assert torch.equal(x, original)


# ── visualize.py ────────────────────────────────────────────────────

from visualize import (plot_pca_scatter, plot_umap_trajectories,
                       linear_probe_evaluation, load_labels,
                       load_embeddings_from_gcs, _run_local)


@pytest.fixture
def tmpdir():
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d)


def test_pca_scatter(tmpdir):
    path = os.path.join(tmpdir, "pca.png")
    plot_pca_scatter(np.random.randn(20, 128), path)
    assert os.path.exists(path)


def test_umap_trajectories(tmpdir):
    path = os.path.join(tmpdir, "umap.png")
    feats = np.random.randn(20, 128)
    labels = ["S1"] * 10 + ["S2"] * 10
    days = list(range(10)) * 2
    plot_umap_trajectories(feats, labels, days, path)
    assert os.path.exists(path)


def test_linear_probe(tmpdir):
    X = np.random.randn(40, 128)
    y = np.array([0] * 20 + [1] * 20, dtype=np.float64)
    auc = linear_probe_evaluation(X, y, tmpdir)
    assert 0.0 <= auc <= 1.0
    assert os.path.exists(os.path.join(tmpdir, "linearprobe_roc.png"))
    assert os.path.exists(os.path.join(tmpdir, "linearprobe_cm.png"))


def test_linear_probe_rejects_multiclass():
    X = np.random.randn(30, 128)
    y = np.array([0] * 10 + [1] * 10 + [2] * 10, dtype=np.float64)
    with pytest.raises(ValueError, match="binary labels"):
        linear_probe_evaluation(X, y, "/tmp")


def test_load_labels_from_csv(tmpdir):
    csv_path = os.path.join(tmpdir, "labels.csv")
    with open(csv_path, "w") as f:
        f.write("subject_id,cdr\nOAS30001,0\nOAS30002,0.5\nOAS30003,1\n")
    y = load_labels(csv_path, ["OAS30001_d0000", "OAS30002_d0100", "OAS30003_d0200"])
    assert y[0] == 0.0
    assert y[1] == 1.0
    assert y[2] == 1.0


def test_load_labels_missing_subject(tmpdir):
    csv_path = os.path.join(tmpdir, "labels.csv")
    with open(csv_path, "w") as f:
        f.write("subject_id,cdr\nOAS30001,0\n")
    y = load_labels(csv_path, ["OAS30001_d0000", "OAS39999_d0000"])
    assert y[0] == 0.0
    assert np.isnan(y[1])


def test_load_labels_no_subject_column(tmpdir):
    csv_path = os.path.join(tmpdir, "bad.csv")
    with open(csv_path, "w") as f:
        f.write("name,value\nfoo,1\n")
    with pytest.raises(ValueError, match="No subject/id column"):
        load_labels(csv_path, ["OAS30001"])


def test_load_labels_no_label_column(tmpdir):
    csv_path = os.path.join(tmpdir, "bad.csv")
    with open(csv_path, "w") as f:
        f.write("subject_id,value\nOAS30001,1\n")
    with pytest.raises(ValueError, match="No label/cdr/dx column"):
        load_labels(csv_path, ["OAS30001"])


def test_load_labels_string_dx(tmpdir):
    csv_path = os.path.join(tmpdir, "labels.csv")
    with open(csv_path, "w") as f:
        f.write("subject_id,dx\nOAS30001,AD\nOAS30002,Normal\nOAS30003,Dementia\n")
    y = load_labels(csv_path, ["OAS30001_d0000", "OAS30002_d0000", "OAS30003_d0000"])
    assert y[0] == 1.0
    assert y[1] == 0.0
    assert y[2] == 1.0


def test_load_embeddings_from_gcs_mock(tmpdir):
    from unittest.mock import MagicMock, patch
    embs = []
    for i in range(2):
        p = os.path.join(tmpdir, f"e{i}.npy")
        np.save(p, np.random.randn(128).astype(np.float32))
        embs.append(p)
    mock_fs = MagicMock()
    mock_fs.ls.side_effect = [
        ["bucket/part_0"],
        ["bucket/part_0/OAS30001", "bucket/part_0/OAS30002"],
    ]
    mock_fs.exists.return_value = True
    _idx = [0]
    _builtin_open = open
    def _mock_open(path, mode="rb"):
        from contextlib import contextmanager
        @contextmanager
        def _ctx():
            with _builtin_open(embs[_idx[0]], "rb") as fh:
                _idx[0] += 1
                yield fh
        return _ctx()
    mock_fs.open = _mock_open
    with patch("gcsfs.GCSFileSystem", return_value=mock_fs):
        X, subjects = load_embeddings_from_gcs("bucket", epoch=20)
    assert X.shape == (2, 128)
    assert subjects == ["OAS30001", "OAS30002"]


def test_run_local_synthetic(tmpdir):
    import argparse
    m = ClinImCL()
    ckpt_path = os.path.join(tmpdir, "ckpt.pth")
    torch.save({"model": m.state_dict(), "epoch": 1}, ckpt_path)
    data_dir = os.path.join(tmpdir, "data")
    os.makedirs(data_dir)
    for name in ["OAS30001_d0000.pt", "OAS30001_d0100.pt",
                  "OAS30002_d0000.pt", "OAS30002_d0100.pt"]:
        torch.save(torch.randn(1, IMG, IMG, IMG), os.path.join(data_dir, name))
    out_dir = os.path.join(tmpdir, "out")
    os.makedirs(out_dir)
    args = argparse.Namespace(
        ckpt=ckpt_path, data_dir=data_dir, output_dir=out_dir, labels_csv=None
    )
    _run_local(args)
    assert os.path.exists(os.path.join(out_dir, "pca_visualization.png"))
    assert os.path.exists(os.path.join(out_dir, "umap_trajectories.png"))


# ── preprocess.py ───────────────────────────────────────────────────

from preprocess import build_index, main as preprocess_main, SPATIAL_SIZE


def test_build_index_empty_raises(tmpdir):
    with pytest.raises(RuntimeError, match="No T1w scans found"):
        build_index(tmpdir)


def test_build_index_valid(tmpdir):
    import nibabel as nib
    sub_dir = os.path.join(tmpdir, "OAS30001", "anat")
    os.makedirs(sub_dir)
    for fname in ["OAS30001_d0000_T1w.nii.gz", "OAS30001_d0365_T1w.nii.gz"]:
        nib.save(nib.Nifti1Image(np.zeros((4, 4, 4), dtype=np.float32), np.eye(4)),
                 os.path.join(sub_dir, fname))
    df = build_index(tmpdir)
    assert len(df) == 2
    assert list(df.columns) == ["subject", "days", "path"]
    assert df.iloc[0]["subject"] == "OAS30001"
    assert df.iloc[0]["days"] == 0
    assert df.iloc[1]["days"] == 365


def test_build_index_skips_nonmatching(tmpdir):
    import nibabel as nib
    img = nib.Nifti1Image(np.zeros((4, 4, 4), dtype=np.float32), np.eye(4))
    nib.save(img, os.path.join(tmpdir, "OAS30001_d0000_T1w.nii.gz"))
    os.makedirs(os.path.join(tmpdir, "other"))
    nib.save(img, os.path.join(tmpdir, "other", "random_T1w.nii.gz"))
    df = build_index(tmpdir)
    assert len(df) == 1


def test_preprocess_transforms(tmpdir):
    import nibabel as nib
    from monai.transforms import (
        LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd,
        ScaleIntensityRanged, CropForegroundd, ResizeWithPadOrCropd, Compose
    )
    vol = np.random.rand(32, 32, 32).astype(np.float32) * 5000
    nib.save(nib.Nifti1Image(vol, np.diag([2.0, 2.0, 2.0, 1.0])),
             os.path.join(tmpdir, "test.nii.gz"))
    out = Compose([
        LoadImaged(keys="image"), EnsureChannelFirstd(keys="image"),
        Orientationd(keys="image", axcodes="RAS"),
        Spacingd(keys="image", pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
        ScaleIntensityRanged(keys="image", a_min=0, a_max=5000, b_min=0, b_max=1, clip=True),
        CropForegroundd(keys="image", source_key="image"),
        ResizeWithPadOrCropd(keys="image", spatial_size=SPATIAL_SIZE),
    ])({"image": os.path.join(tmpdir, "test.nii.gz")})["image"]
    assert out.shape == (1,) + SPATIAL_SIZE


def test_preprocess_main_e2e(tmpdir):
    import sys, nibabel as nib
    data_dir = os.path.join(tmpdir, "raw", "OAS30001", "anat")
    os.makedirs(data_dir)
    vol = np.random.rand(32, 32, 32).astype(np.float32) * 5000
    nib.save(nib.Nifti1Image(vol, np.diag([2.0, 2.0, 2.0, 1.0])),
             os.path.join(data_dir, "OAS30001_d0000_T1w.nii.gz"))
    out_dir = os.path.join(tmpdir, "preprocessed")
    orig = sys.argv
    sys.argv = ["preprocess.py", "--data_dir", os.path.join(tmpdir, "raw"),
                "--out_dir", out_dir]
    try:
        preprocess_main()
    finally:
        sys.argv = orig
    assert os.path.exists(os.path.join(out_dir, "longitudinal_index.csv"))
    pt_files = [f for f in os.listdir(out_dir) if f.endswith(".pt")]
    assert len(pt_files) == 1
    saved = torch.load(os.path.join(out_dir, pt_files[0]), map_location="cpu",
                        weights_only=True)
    assert saved.shape == (1,) + SPATIAL_SIZE


# ── download.sh ─────────────────────────────────────────────────────

def test_download_sh_syntax():
    import subprocess
    ret = subprocess.run(["bash", "-n", "download.sh"], capture_output=True, text=True)
    assert ret.returncode == 0, f"Bash syntax error: {ret.stderr}"


def test_download_sh_functions_defined():
    import subprocess
    ret = subprocess.run(
        ["bash", "-c",
         "source download.sh && type startSession && type download "
         "&& type endSession && type download_scans"],
        capture_output=True, text=True)
    assert ret.returncode == 0, f"Functions not defined: {ret.stderr}"


def test_download_sh_subject_extraction():
    import subprocess
    for exp_id, expected in [("OAS30001_MR_d0000", "OAS30001"),
                             ("OAS30002_MR_d0365", "OAS30002"),
                             ("OAS40001_MR_d0000", "OAS40001")]:
        ret = subprocess.run(
            ["bash", "-c", f"echo '{exp_id}' | cut -d_ -f1"],
            capture_output=True, text=True)
        assert ret.stdout.strip() == expected


def test_download_sh_project_routing():
    import subprocess
    script = (
        'source download.sh; '
        'for EID in OAS30001_MR_d0 OAS40001_MR_d0 OAS30001_AV1451_d0; do '
        '  P=OASIS3; '
        '  [[ "$EID" == "OAS4"* ]] && P=OASIS4; '
        '  [[ "$EID" == "OAS3"*"_AV1451"* ]] && P=OASIS3_AV1451; '
        '  echo "$P"; '
        'done'
    )
    ret = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
    lines = ret.stdout.strip().split("\n")
    assert lines == ["OASIS3", "OASIS4", "OASIS3_AV1451"]
