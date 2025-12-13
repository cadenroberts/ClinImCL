import torch
import matplotlib.pyplot as plt
import gcsfs
import monai
from monai.data.meta_tensor import MetaTensor
from monai.utils.enums import TraceKeys
import numpy as np

torch.serialization.add_safe_globals({
    MetaTensor,
    TraceKeys,
    np.core.multiarray._reconstruct,
})

fs = gcsfs.GCSFileSystem(token='google_default')
file = "clinimcl-data/OASIS3/preprocessed_new/OAS30001_MR_d0129_anat2.pt"
print("Opening:", file)

with fs.open(file, 'rb') as f:
    img = torch.load(f, weights_only=False)
print("Loaded tensor shape:", img.shape)
img = img[0]
slice_img = img[:, :, 64]
plt.imshow(slice_img, cmap="gray")
plt.axis("off")
save_path = "/root/sample_mri_slice.png"
plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
print("SAVED MRI SLICE TO:", save_path)