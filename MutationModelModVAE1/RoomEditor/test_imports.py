"""
Quick test script to verify tboi_vae module import in the executable
"""
import sys
import os

print("=== MODULE IMPORT TEST ===")
print(f"Python executable: {sys.executable}")
print(f"Current working directory: {os.getcwd()}")
print(f"Python path: {sys.path}")
print()

# Test basic imports
try:
    import numpy as np
    print("✅ NumPy imported successfully")
except ImportError as e:
    print(f"❌ NumPy import failed: {e}")

try:
    import torch
    print("✅ PyTorch imported successfully")
    print(f"   PyTorch version: {torch.__version__}")
    print(f"   Device available: {torch.cuda.is_available()}")
except ImportError as e:
    print(f"❌ PyTorch import failed: {e}")

try:
    from PIL import Image
    print("✅ PIL imported successfully")
except ImportError as e:
    print(f"❌ PIL import failed: {e}")

# Test our custom modules
try:
    from tboi_bitmap import TBoI_Bitmap, EntityType
    print("✅ tboi_bitmap imported successfully")
    print(f"   EntityType enum available: {hasattr(EntityType, 'FREE_SPACE')}")
except ImportError as e:
    print(f"❌ tboi_bitmap import failed: {e}")

try:
    from tboi_vae import ConvVAE, load_model, DEVICE
    print("✅ tboi_vae imported successfully")
    print(f"   Device: {DEVICE}")
    print("   ConvVAE class available")
    print("   load_model function available")
    
    # Test model creation
    model = ConvVAE(latent_dim=64)
    print(f"   Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
except ImportError as e:
    print(f"❌ tboi_vae import failed: {e}")
except Exception as e:
    print(f"❌ tboi_vae error: {e}")

print("\n🏁 Import test completed!")
print("=== END TEST ===")
input("Press Enter to exit...")

try:
    from tboi_vae import ConvVAE, load_model, DEVICE
    print("✅ tboi_vae imported successfully")
    print(f"   DEVICE: {DEVICE}")
    print(f"   ConvVAE class available: {ConvVAE is not None}")
    print(f"   load_model function available: {load_model is not None}")
except ImportError as e:
    print(f"❌ tboi_vae import failed: {e}")

# Test Data folder access
data_path = "Data"
if os.path.exists(data_path):
    print(f"✅ Data folder found at: {os.path.abspath(data_path)}")
    
    model_path = os.path.join(data_path, "best.pt")
    if os.path.exists(model_path):
        print(f"✅ Model file found: {model_path}")
    else:
        print(f"❌ Model file NOT found: {model_path}")
    
    mutations_path = os.path.join(data_path, "Mutations_Extracted")
    if os.path.exists(mutations_path):
        print(f"✅ Mutations folder found: {mutations_path}")
        # Count .bmp files
        bmp_count = 0
        for root, dirs, files in os.walk(mutations_path):
            bmp_count += len([f for f in files if f.endswith('.bmp')])
        print(f"   Found {bmp_count} .bmp files")
    else:
        print(f"❌ Mutations folder NOT found: {mutations_path}")
else:
    print(f"❌ Data folder NOT found: {data_path}")

print("\n=== TEST COMPLETE ===")
input("Press Enter to exit...")
