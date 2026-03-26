import os, glob, re, argparse, torch
import pandas as pd
from monai.transforms import (
    LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd,
    ScaleIntensityRanged, CropForegroundd, ResizeWithPadOrCropd, Compose
)
from monai.data import Dataset, DataLoader
from tqdm import tqdm

from model import IMG

SPATIAL_SIZE = (IMG, IMG, IMG)


def build_index(data_dir):
    records = []
    for f in sorted(glob.glob(os.path.join(data_dir, "**", "*T1w.nii.gz"), recursive=True)):
        s = re.search(r"(OAS3\d+)", f)
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
    return pre


def main():
    parser = argparse.ArgumentParser(description="Preprocess OASIS T1w volumes to 96^3 .pt tensors")
    parser.add_argument("--data_dir", default="/data/OASIS3",
                        help="Root directory containing raw OASIS NIfTI scans")
    parser.add_argument("--out_dir", default="/data/OASIS3/preprocessed",
                        help="Output directory for preprocessed .pt files")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    pre = build_index(args.data_dir)
    pre.to_csv(os.path.join(args.out_dir, "longitudinal_index.csv"), index=False)
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

    data_list = [
        {"image": row.path, "tag": f"{row['subject']}_d{row['days']:04d}"}
        for _, row in pre.iterrows()
    ]
    ds = Dataset(data=data_list, transform=transforms)
    loader = DataLoader(ds, batch_size=1, num_workers=0)

    saved, skipped = 0, 0
    it = iter(loader)
    for _ in tqdm(range(len(loader))):
        try:
            batch = next(it)
        except StopIteration:
            break
        except Exception as e:
            print(f"[error] failed to load batch: {e}")
            skipped += 1
            continue
        tag = batch["tag"][0]
        vol = batch["image"][0]
        expected = (1,) + SPATIAL_SIZE
        if vol.shape != expected:
            print(f"[warn] {tag}: unexpected shape {tuple(vol.shape)}, expected {expected}")
        try:
            torch.save(vol.as_subclass(torch.Tensor), os.path.join(args.out_dir, f"{tag}.pt"))
            saved += 1
        except Exception as e:
            print(f"[error] {tag}: {e}")
            skipped += 1

    print(f"[done] saved={saved} skipped={skipped}")


if __name__ == "__main__":
    main()
