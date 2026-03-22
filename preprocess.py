import os, glob, re, torch
import pandas as pd
from monai.transforms import (
    LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd,
    ScaleIntensityRanged, CropForegroundd, ResizeWithPadOrCropd, Compose
)
from monai.data import Dataset, DataLoader
from tqdm import tqdm

OUT_DIR = "/data/OASIS3/preprocessed_t1w"
SPATIAL_SIZE = (128, 128, 128)

os.makedirs(OUT_DIR, exist_ok=True)

records = []
for f in sorted(glob.glob("/data/OASIS3/**/*T1w.nii.gz", recursive=True)):
    s = re.search(r"(OAS3\d{4})", f)
    d = re.search(r"_d(\d+)", f)
    if not s or not d:
        print(f"[skip] could not parse subject/day from: {f}")
        continue
    records.append({"subject": s.group(1), "days": int(d.group(1)), "path": f})

pre = pd.DataFrame(records)
if pre.empty:
    raise RuntimeError("No T1w scans found — check glob path and OASIS directory structure")

pre.sort_values(["subject", "days"], inplace=True)
pre.reset_index(drop=True, inplace=True)
pre.to_csv(os.path.join(OUT_DIR, "longitudinal_index.csv"), index=False)
print(f"[index] {len(pre)} scans from {pre['subject'].nunique()} subjects")

transforms = Compose([
    LoadImaged(keys="image"),
    EnsureChannelFirstd(keys="image"),
    Orientationd(keys="image", axcodes="RAS"),
    Spacingd(keys="image", pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
    ScaleIntensityRanged(keys="image", a_min=0, a_max=5000, b_min=0, b_max=1, clip=True),
    CropForegroundd(keys="image", source_key="image"),
    ResizeWithPadOrCropd(keys="image", spatial_size=SPATIAL_SIZE),
])

data_list = [{"image": row.path} for _, row in pre.iterrows()]
ds = Dataset(data=data_list, transform=transforms)
loader = DataLoader(ds, batch_size=1, num_workers=4)

for i, batch in enumerate(tqdm(loader, total=len(pre))):
    row = pre.iloc[i]
    out_path = os.path.join(OUT_DIR, f"{row['subject']}_d{row['days']:04d}.pt")
    try:
        torch.save(batch["image"][0], out_path)
    except Exception as e:
        print(f"[error] scan {i} ({row['subject']}_d{row['days']:04d}): {e}")
