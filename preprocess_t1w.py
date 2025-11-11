import os, glob, re, torch, nibabel as nib
import pandas as pd
from monai.transforms import (
    LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd,
    ScaleIntensityRanged, CropForegroundd, NormalizeIntensityd, Compose
)
from monai.data import Dataset, DataLoader
from tqdm import tqdm

# Locate all T1w images
t1w_files = sorted(glob.glob("/data/OASIS3/**/**/*T1w.nii.gz", recursive=True))
print(f"Found {len(t1w_files)} T1w scans")

# Build metadata
records = []
for f in t1w_files:
    subj = re.search(r"(OAS\d+)", f).group(1)
    days_match = re.search(r"d(\d+)", f)
    days = int(days_match.group(1)) if days_match else 0
    records.append({"subject": subj, "days": days, "path": f})
df = pd.DataFrame(records)
df.sort_values(["subject", "days"], inplace=True)

os.makedirs("/data/OASIS3/preprocessed_t1w", exist_ok=True)
df.to_csv("/data/OASIS3/preprocessed_t1w/longitudinal_index.csv", index=False)

# Define MONAI preprocessing
preprocess = Compose([
    LoadImaged(keys="image"),
    EnsureChannelFirstd(keys="image"),
    Orientationd(keys="image", axcodes="RAS"),
    Spacingd(keys="image", pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
    ScaleIntensityRanged(keys="image", a_min=0, a_max=5000, b_min=0, b_max=1, clip=True),
    CropForegroundd(keys="image", source_key="image"),
    NormalizeIntensityd(keys="image", nonzero=True)
])

data = [{"image": row.path} for _, row in df.iterrows()]
ds = Dataset(data=data, transform=preprocess)
loader = DataLoader(ds, batch_size=1, num_workers=4)

# Save preprocessed tensors
for i, batch in enumerate(tqdm(loader)):
    subj = df.iloc[i]["subject"]
    days = df.iloc[i]["days"]
    tensor = batch["image"][0]
    out = f"/data/OASIS3/preprocessed_t1w/{subj}_d{days:04d}.pt"
    if os.path.exists(out):
        continue
    torch.save(tensor, out)
