import gcsfs, torch, nibabel as nib, tempfile, os
from monai.transforms import (
    EnsureChannelFirst, Orientation, Spacing,
    ScaleIntensityRange, CropForeground, NormalizeIntensity, Resize, Compose
)
from tqdm import tqdm

fs = gcsfs.GCSFileSystem(token='google_default')
raw_bucket = "clinimcl-data/OASIS3/raw"
preproc_bucket = "clinimcl-data/OASIS3/preprocessed_new"

preprocess = Compose([
    EnsureChannelFirst(channel_dim='no_channel'),
    Orientation(axcodes="RAS"),
    Spacing(pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
    ScaleIntensityRange(a_min=0, a_max=5000, b_min=0, b_max=1, clip=True),
    CropForeground(),
    Resize((128, 128, 128)),
    NormalizeIntensity(nonzero=True)
])

subjects = fs.ls(raw_bucket)
subjects = [s for s in subjects if s.strip("/") != raw_bucket]

for subject in tqdm(subjects, desc="Preprocessing all T1 Scans"):
    subject_id_date = subject.split("/")[-1]
    print(f"\n Processing subject {subject_id_date}")
    scan_types = fs.ls(subject)
    for scan_type in scan_types:
        scan_type_name = scan_type.split("/")[-1]
        scan_files = fs.ls(scan_type)
        for scan_file in scan_files:
            if "T1w.nii.gz" in scan_file:
                pt_filename = f"{subject_id_date}_{scan_type_name}.pt"
                try:
                    print(f" Processing: {scan_file}")
                    with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=True) as tmp_nii:
                        with fs.open(scan_file, 'rb') as f_gcs:
                            tmp_nii.write(f_gcs.read())
                            tmp_nii.flush()
                        img = nib.load(tmp_nii.name)
                        img_data = img.get_fdata()
                        if img_data.ndim == 4:
                            print(f" Found 4D data: {img_data.shape}, extracting first volume clearly.")
                            img_data = img_data[..., 0]
                        if img_data.ndim != 3:
                            raise ValueError(f"Expected 3D data, got {img_data.ndim}D clearly.")
                        img_tensor = torch.tensor(img_data, dtype=torch.float32)
                    image = preprocess(img_tensor.numpy())
                    assert image.shape == (1,128,128,128), f"Wrong shape: {image.shape}"
                    assert torch.isfinite(image).all(), "Non-finite values detected"
                    with tempfile.NamedTemporaryFile(delete=False) as buffer:
                        torch.save(image, buffer.name)
                        buffer.seek(0)
                        pt_file_path = f"{preproc_bucket}/{pt_filename}"
                        print(f" Uploading: {pt_file_path}")
                        with fs.open(pt_file_path, 'wb') as f:
                            f.write(buffer.read())
                    os.unlink(buffer.name)
                    print(f"Successfully processed: {pt_filename}")
                except Exception as e:
                    print(f"Error processing {scan_file}: {e}")