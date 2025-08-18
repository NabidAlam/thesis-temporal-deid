#!/usr/bin/env python3
"""
Test script for temporal consistency functions
"""

import numpy as np
import cv2
import sys
import os

# Add temporal directory to path
sys.path.append(os.path.abspath("temporal"))

def test_temporal_consistency():
    """Test the temporal consistency functions"""
    
    print("🧪 Testing Temporal Consistency Functions...")
    
    # Test 1: Create test masks
    print("\n1. Creating test masks...")
    mask1 = np.zeros((100, 100), dtype=np.uint8)
    mask1[20:80, 20:80] = 255  # Square mask
    
    mask2 = np.zeros((100, 100), dtype=np.uint8)
    mask2[25:85, 25:85] = 255  # Slightly shifted square mask
    
    mask3 = np.zeros((100, 100), dtype=np.uint8)
    mask3[50:90, 50:90] = 255  # Different size and position
    
    print(f"   Mask 1 area: {np.sum(mask1 > 0)}")
    print(f"   Mask 2 area: {np.sum(mask2 > 0)}")
    print(f"   Mask 3 area: {np.sum(mask3 > 0)}")
    
    # Test 2: Import and test temporal consistency
    try:
        from temporal.tsp_sam_complete import (
            compute_temporal_consistency_loss,
            compute_motion_consistency,
            temporal_mask_smoothing
        )
        print("\n✅ Successfully imported temporal consistency functions!")
        
        # Test 3: Test consistency computation
        print("\n2. Testing temporal consistency computation...")
        consistency_1_2 = compute_temporal_consistency_loss(mask1, mask2)
        consistency_1_3 = compute_temporal_consistency_loss(mask1, mask3)
        consistency_2_3 = compute_temporal_consistency_loss(mask2, mask3)
        
        print(f"   Consistency between mask 1 & 2: {consistency_1_2:.3f}")
        print(f"   Consistency between mask 1 & 3: {consistency_1_3:.3f}")
        print(f"   Consistency between mask 2 & 3: {consistency_2_3:.3f}")
        
        # Test 4: Test motion consistency
        print("\n3. Testing motion consistency...")
        motion_1_2 = compute_motion_consistency(mask1, mask2)
        motion_1_3 = compute_motion_consistency(mask1, mask3)
        
        print(f"   Motion consistency 1->2: {motion_1_2:.3f}")
        print(f"   Motion consistency 1->3: {motion_1_3:.3f}")
        
        # Test 5: Test temporal smoothing
        print("\n4. Testing temporal smoothing...")
        mask_sequence = [mask1, mask2, mask3]
        smoothed_masks = temporal_mask_smoothing(mask_sequence, window_size=3)
        
        print(f"   Original sequence length: {len(mask_sequence)}")
        print(f"   Smoothed sequence length: {len(smoothed_masks)}")
        
        for i, (orig, smooth) in enumerate(zip(mask_sequence, smoothed_masks)):
            orig_area = np.sum(orig > 0)
            smooth_area = np.sum(smooth > 0)
            print(f"   Frame {i}: Original area: {orig_area}, Smoothed area: {smooth_area}")
        
        print("\n🎉 All temporal consistency tests passed!")
        return True
        
    except ImportError as e:
        print(f"\n❌ Failed to import temporal functions: {e}")
        return False
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        return False

def test_opencv_optical_flow():
    """Test if OpenCV optical flow works"""
    print("\n🔍 Testing OpenCV optical flow...")
    
    try:
        # Create test images
        img1 = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        img2 = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        
        # Test optical flow
        flow = cv2.calcOpticalFlowFarneback(img1, img2, None, 
                                          pyr_scale=0.5, levels=3, winsize=15, 
                                          iterations=3, poly_n=5, poly_sigma=1.2, 
                                          flags=0)
        
        print(f"   Optical flow computation successful!")
        print(f"   Flow shape: {flow.shape}")
        print(f"   Flow range: {flow.min():.2f} to {flow.max():.2f}")
        return True
        
    except Exception as e:
        print(f"Optical flow failed: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("TEMPORAL CONSISTENCY FUNCTION TEST")
    print("=" * 60)
    
    # Test OpenCV first
    opencv_ok = test_opencv_optical_flow()
    
    # Test temporal functions
    temporal_ok = test_temporal_consistency()
    
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"OpenCV Optical Flow: {'PASS' if opencv_ok else 'FAIL'}")
    print(f"Temporal Functions: {'PASS' if temporal_ok else 'FAIL'}")
    
    if opencv_ok and temporal_ok:
        print("\n🎉 All tests passed! Ready to run full pipeline.")
        print("\nNext steps:")
        print("1. Run: python temporal/tsp_sam_complete.py input/davis2017/JPEGImages/480p/dog output/tsp_sam_test/dog configs/tsp_sam_config.yaml --force")
        print("2. Check the temporal consistency scores in the output")
        print("3. Look for the 'TEMPORAL CONSISTENCY SUMMARY' at the end")
    else:
        print("\n❌ Some tests failed. Please check the errors above.")
    
    print("=" * 60)
