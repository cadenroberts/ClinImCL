#!/bin/bash
set -e

echo "========================================"
echo "ClinImCL Smoke Test"
echo "========================================"
echo ""

# Check Python availability
if ! command -v python3 &> /dev/null; then
    echo "❌ ERROR: python3 not found"
    exit 1
fi

echo "✓ Python detected: $(python3 --version)"

# Check required packages
echo ""
echo "Checking dependencies..."

python3 -c "import torch" 2>/dev/null || { echo "❌ ERROR: PyTorch not installed"; exit 1; }
echo "✓ PyTorch installed"

python3 -c "import monai" 2>/dev/null || { echo "❌ ERROR: MONAI not installed"; exit 1; }
echo "✓ MONAI installed"

python3 -c "import numpy" 2>/dev/null || { echo "❌ ERROR: NumPy not installed"; exit 1; }
echo "✓ NumPy installed"

# Run preprocessing validation
echo ""
echo "========================================"
echo "Running preprocessing validation..."
echo "========================================"
echo ""

python3 << 'EOF'
import torch
import numpy as np
from monai.transforms import (
    EnsureChannelFirst, Orientation, Spacing,
    ScaleIntensityRange, CropForeground, Resize, NormalizeIntensity, Compose
)

# Define preprocessing pipeline (same as Preprocessing.py)
preprocess = Compose([
    EnsureChannelFirst(channel_dim='no_channel'),
    Orientation(axcodes="RAS"),
    Spacing(pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
    ScaleIntensityRange(a_min=0, a_max=5000, b_min=0, b_max=1, clip=True),
    CropForeground(),
    Resize((128, 128, 128)),
    NormalizeIntensity(nonzero=True)
])

# Create synthetic 3D brain-like volume
np.random.seed(42)
volume = np.random.randn(160, 200, 180) * 1000 + 2500
volume[volume < 0] = 0
volume[10:150, 20:180, 15:165] += 500

print("Input volume shape:", volume.shape)
print("Input intensity range:", f"[{volume.min():.1f}, {volume.max():.1f}]")

# Preprocess
processed = preprocess(volume)

print("\nAfter preprocessing:")
print("  Shape:", processed.shape)
print("  Dtype:", processed.dtype)
print("  Intensity range:", f"[{processed.min():.4f}, {processed.max():.4f}]")
print("  Mean:", f"{processed.mean():.4f}")
print("  Std:", f"{processed.std():.4f}")

# Validations
assert processed.shape == (1, 128, 128, 128), f"Expected (1,128,128,128), got {processed.shape}"
assert torch.isfinite(processed).all(), "Non-finite values detected"
assert processed.dtype == torch.float32, f"Expected float32, got {processed.dtype}"

print("\n✅ All preprocessing checks passed")
EOF

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ SMOKE_FAIL: Preprocessing validation failed"
    exit 1
fi

# Test model forward pass
echo ""
echo "========================================"
echo "Testing model forward pass..."
echo "========================================"
echo ""

python3 << 'EOF'
import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=3), nn.ReLU(), nn.MaxPool3d(2),
            nn.Conv3d(8, 16, kernel_size=3), nn.ReLU(), nn.AdaptiveAvgPool3d(1)
        )
        self.fc = nn.Linear(16, 128)

    def forward(self, x):
        x = self.features(x)
        return self.fc(x.view(x.size(0), -1))

# Create model
model = SimpleCNN()
model.eval()

# Create synthetic input
x = torch.randn(2, 1, 128, 128, 128)

print("Input shape:", x.shape)

# Forward pass
with torch.no_grad():
    embedding = model(x)

print("Embedding shape:", embedding.shape)
print("Embedding dtype:", embedding.dtype)

# Validations
assert embedding.shape == (2, 128), f"Expected (2, 128), got {embedding.shape}"
assert torch.isfinite(embedding).all(), "Non-finite values in embedding"

print("\n✅ Model forward pass successful")
EOF

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ SMOKE_FAIL: Model forward pass failed"
    exit 1
fi

# Final summary
echo ""
echo "========================================"
echo "SMOKE_OK"
echo "========================================"
echo ""
echo "All smoke tests passed successfully."
echo "Core preprocessing and model components validated."
echo ""
echo "Note: This smoke test does not validate:"
echo "  - GCS I/O operations"
echo "  - Full training convergence"
echo "  - Split integrity"
echo "  - Embedding quality metrics"
echo ""
echo "For full validation, see EVAL.md"
