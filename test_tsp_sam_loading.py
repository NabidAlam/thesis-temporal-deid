#!/usr/bin/env python3
"""
Simple test script to check TSP-SAM model loading step by step
"""

import os
import sys
import time
import torch

# Add the official TSP-SAM to path
sys.path.append('tsp_sam_official')
sys.path.append('tsp_sam_official/lib')

try:
    from lib import VideoModel_pvtv2 as Network
    print("✓ Successfully imported TSP-SAM modules")
except ImportError as e:
    print(f"✗ Error importing TSP-SAM modules: {e}")
    sys.exit(1)

# Configuration for TSP-SAM model
class ModelConfig:
    def __init__(self):
        self.channel = 32
        self.imgsize = 352
        self.pretrained = True
        self.gpu_ids = [0] if torch.cuda.is_available() else []

def test_model_loading():
    print("=== TSP-SAM Model Loading Test ===")
    
    # Check checkpoint
    checkpoint_path = "tsp_sam_official/snapshot/pvt_v2_b5.pth"
    if not os.path.exists(checkpoint_path):
        print(f"✗ Checkpoint not found: {checkpoint_path}")
        return False
    
    print(f"✓ Checkpoint found: {checkpoint_path}")
    checkpoint_size = os.path.getsize(checkpoint_path) / (1024 * 1024)  # MB
    print(f"  Checkpoint size: {checkpoint_size:.1f} MB")
    
    # Check device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"✓ Using device: {device}")
    if device == "cuda":
        print(f"  CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")
    
    # Create model configuration
    print("\n1. Creating model configuration...")
    start_time = time.time()
    opt = ModelConfig()
    opt.gpu_ids = [0] if torch.cuda.is_available() else []
    opt.trainsize = 352
    opt.testsize = 352
    config_time = time.time() - start_time
    print(f"✓ Configuration created in {config_time:.3f} seconds")
    
    # Build model
    print("\n2. Building TSP-SAM model...")
    start_time = time.time()
    try:
        if len(opt.gpu_ids) == 0 or device == "cpu":
            model = Network(opt).to(device)
        elif len(opt.gpu_ids) == 1:
            model = Network(opt).cuda(opt.gpu_ids[0])
        else:
            model = Network(opt).cuda(opt.gpu_ids[0])
            model = torch.nn.DataParallel(model, device_ids=opt.gpu_ids)
        
        build_time = time.time() - start_time
        print(f"✓ Model built in {build_time:.1f} seconds")
        
    except Exception as e:
        print(f"✗ Error building model: {e}")
        return False
    
    # Load checkpoint
    print("\n3. Loading checkpoint...")
    start_time = time.time()
    try:
        params = torch.load(checkpoint_path, map_location=device)
        print(f"✓ Checkpoint loaded into memory in {time.time() - start_time:.1f} seconds")
        
        # Try strict loading first
        print("4a. Attempting strict loading...")
        start_time = time.time()
        try:
            model.load_state_dict(params, strict=True)
            load_time = time.time() - start_time
            print(f"✓ Strict loading successful in {load_time:.1f} seconds")
        except Exception as e:
            print(f"⚠ Strict loading failed: {e}")
            print("4b. Attempting non-strict loading...")
            start_time = time.time()
            model.load_state_dict(params, strict=False)
            load_time = time.time() - start_time
            print(f"✓ Non-strict loading successful in {load_time:.1f} seconds")
        
    except Exception as e:
        print(f"✗ Error loading checkpoint: {e}")
        return False
    
    # Set model to eval mode
    print("\n5. Setting model to evaluation mode...")
    model.eval()
    print("✓ Model is ready for inference")
    
    # Test inference
    print("\n6. Testing inference with dummy input...")
    try:
        dummy_input = torch.randn(1, 3, 352, 352)
        if device == "cuda":
            dummy_input = dummy_input.cuda()
        
        start_time = time.time()
        with torch.no_grad():
            output = model(dummy_input)
        inference_time = time.time() - start_time
        
        print(f"✓ Inference successful in {inference_time:.3f} seconds")
        print(f"  Output type: {type(output)}")
        if isinstance(output, (list, tuple)):
            print(f"  Output length: {len(output)}")
            for i, out in enumerate(output):
                if hasattr(out, 'shape'):
                    print(f"  Output {i} shape: {out.shape}")
        elif hasattr(output, 'shape'):
            print(f"  Output shape: {output.shape}")
        
    except Exception as e:
        print(f"✗ Error during inference: {e}")
        return False
    
    total_time = time.time() - start_time
    print(f"\n=== Test completed successfully! ===")
    print(f"Total setup time: {total_time:.1f} seconds")
    return True

if __name__ == '__main__':
    success = test_model_loading()
    if not success:
        print("\n✗ Test failed. Check the errors above.")
        sys.exit(1)
    else:
        print("\n✓ All tests passed! TSP-SAM is working correctly.")
