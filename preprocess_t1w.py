import os, glob, re, torch
import pandas as pd
from monai.transforms import (
    LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd,
    ScaleIntensityRanged, CropForegroundd, NormalizeIntensityd, Compose
)
from monai.data import Dataset, DataLoader
from tqdm import tqdm
pre = pd.DataFrame([{"subject": s.group(1) if (s := re.search(r"(OAS\d+)", f)) else None, "days": int(d.group(1)) if (d := re.search(r"d(\d+)", f)) else None, "path": f,} for f in sorted(glob.glob("/data/OASIS3/**/**/*T1w.nii.gz", recursive=True))])
pre.sort_values(["subject", "days"], inplace=True)
pre.to_csv("/data/OASIS3/preprocessed_t1w/longitudinal_index.csv", index=False)
for i, tensor in enumerate(tqdm(DataLoader(Dataset(data=[{"image": row.path} for _, row in pre.iterrows()], transform=Compose([ LoadImaged(keys="image"), EnsureChannelFirstd(keys="image"), Orientationd(keys="image", axcodes="RAS"), Spacingd(keys="image", pixdim=(1.0, 1.0, 1.0), mode="bilinear"), ScaleIntensityRanged(keys="image", a_min=0, a_max=5000, b_min=0, b_max=1, clip=True), CropForegroundd(keys="image", source_key="image"), NormalizeIntensityd(keys="image", nonzero=True)])), batch_size=1, num_workers=4))):
    torch.save(tensor['image'][0], f"/data/OASIS3/preprocessed_t1w/{pre.iloc[i]['subject']}_d{pre.iloc[i]['days']:04d}.pt")
