import gcsfs
import matplotlib.pyplot as plt
from collections import Counter
import re

fs = gcsfs.GCSFileSystem(token='google_default')
files = fs.ls("clinimcl-data/OASIS3/preprocessed_new")


subjects = []
for f in files:
    name = f.split("/")[-1]  # get filename
    m = re.match(r"(OAS\d+)", name)
    if m:
        subjects.append(m.group(1))

counts = Counter(subjects)
plt.figure(figsize=(10,5))
plt.hist(counts.values(), bins=20, edgecolor='black')
plt.title("Distribution of MRI Scans per Subject (OASIS-3)")
plt.xlabel("# Scans for a Subject")
plt.ylabel("# Subjects")
plt.tight_layout()
plt.savefig("/root/scan_distribution.png")
with fs.open("clinimcl-data/OASIS3/dataset_visuals/scan_distribution.png", "wb") as f:
    plt.savefig(f, format="png")
plt.show()