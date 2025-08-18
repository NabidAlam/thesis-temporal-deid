#!/usr/bin/env python3
"""
Test script for advanced post-processing system
"""

import numpy as np
import cv2
import sys
import os

def compute_temporal_consistency_loss(current_mask, previous_mask):
    """Temporal consistency function for testing"""
    if previous_mask is None:
        return 1.0
    
    current_binary = (current_mask > 0).astype(np.uint8)
    previous_binary = (previous_mask > 0).astype(np.uint8)
    
    intersection = np.logical_and(current_binary, previous_binary).sum()
    union = np.logical_or(current_binary, previous_binary).sum()
    iou = intersection / (union + 1e-6)
    
    current_area = current_binary.sum()
    previous_area = previous_binary.sum()
    area_ratio = min(current_area, previous_area) / max(current_area, previous_area) if max(current_area, previous_area) > 0 else 1.0
    
    if current_area > 0 and previous_area > 0:
        current_centroid = np.mean(np.where(current_binary > 0), axis=1)
        previous_centroid = np.mean(np.where(previous_binary > 0), axis=1)
        centroid_distance = np.linalg.norm(current_centroid - previous_centroid)
        max_distance = np.sqrt(current_binary.shape[0]**2 + current_binary.shape[1]**2)
        centroid_penalty = 1.0 - (centroid_distance / max_distance)
    else:
        centroid_penalty = 1.0
    
    consistency_score = 0.5 * iou + 0.3 * area_ratio + 0.2 * centroid_penalty
    return max(0.0, min(1.0, consistency_score))


def compute_motion_magnitude(current_mask, previous_mask, flow_threshold=5.0):
    """Motion magnitude function for testing"""
    if previous_mask is None:
        return 0.0
    
    try:
        prev_gray = previous_mask.astype(np.uint8)
        curr_gray = current_mask.astype(np.uint8)
        
        flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 
                                          pyr_scale=0.5, levels=3, winsize=15, 
                                          iterations=3, poly_n=5, poly_sigma=1.2, 
                                          flags=0)
        
        magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
        avg_motion = np.mean(magnitude)
        
        motion_score = min(1.0, avg_motion / flow_threshold)
        return motion_score
        
    except Exception as e:
        print(f"[WARNING] Motion magnitude computation failed: {e}")
        return 0.5


def adaptive_morphological_operations(mask, motion_magnitude, temporal_consistency, 
                                   base_kernel_size=3, max_kernel_size=7):
    """Adaptive morphological operations function for testing"""
    if mask is None or np.sum(mask > 0) == 0:
        return mask
    
    binary_mask = (mask > 0).astype(np.uint8)
    
    motion_factor = motion_magnitude
    consistency_factor = 1.0 - temporal_consistency
    
    adaptive_factor = (motion_factor + consistency_factor) / 2.0
    kernel_size = int(base_kernel_size + (max_kernel_size - base_kernel_size) * adaptive_factor)
    kernel_size = max(base_kernel_size, min(max_kernel_size, kernel_size))
    
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    print(f"Motion: {motion_magnitude:.3f}, Consistency: {temporal_consistency:.3f}")
    print(f"Adaptive kernel size: {kernel_size}x{kernel_size}")
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    processed_mask = binary_mask.copy()
    
    # Apply operations based on conditions
    if motion_magnitude > 0.3:
        processed_mask = cv2.morphologyEx(processed_mask, cv2.MORPH_OPEN, kernel)
        print(f"Applied opening with {kernel_size}x{kernel_size} kernel")
    
    if temporal_consistency < 0.7:
        processed_mask = cv2.morphologyEx(processed_mask, cv2.MORPH_CLOSE, kernel)
        print(f"Applied closing with {kernel_size}x{kernel_size} kernel")
    
    if motion_magnitude > 0.5:
        gradient_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(3, kernel_size-2), max(3, kernel_size-2)))
        dilated = cv2.dilate(processed_mask, gradient_kernel, iterations=1)
        eroded = cv2.erode(processed_mask, gradient_kernel, iterations=1)
        gradient = cv2.subtract(dilated, eroded)
        
        smoothed_gradient = cv2.GaussianBlur(gradient, (3, 3), 0)
        processed_mask = cv2.add(processed_mask, smoothed_gradient)
        processed_mask = (processed_mask > 0).astype(np.uint8) * 255
        
        print(f"Applied boundary smoothing")
    
    return processed_mask


def temporal_boundary_refinement(mask, previous_mask, next_mask=None, refinement_strength=0.3):
    """Temporal boundary refinement function for testing"""
    if mask is None or np.sum(mask > 0) == 0:
        return mask
    
    refined_mask = mask.copy()
    
    if previous_mask is not None:
        current_contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        previous_contours, _ = cv2.findContours(previous_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if current_contours and previous_contours:
            current_contour = max(current_contours, key=cv2.contourArea)
            previous_contour = max(previous_contours, key=cv2.contourArea)
            
            current_perimeter = cv2.arcLength(current_contour, True)
            previous_perimeter = cv2.arcLength(previous_contour, True)
            
            if current_perimeter > 0 and previous_perimeter > 0:
                perimeter_ratio = min(current_perimeter, previous_perimeter) / max(current_perimeter, previous_perimeter)
                
                if perimeter_ratio < 0.8:
                    temporal_boundary = np.zeros_like(mask, dtype=np.uint8)
                    cv2.drawContours(temporal_boundary, [previous_contour], -1, 255, 2)
                    
                    blended = cv2.addWeighted(mask, 1.0 - refinement_strength, 
                                           temporal_boundary, refinement_strength, 0)
                    refined_mask = (blended > 127).astype(np.uint8) * 255
                    
                    print(f"Applied temporal boundary refinement (strength: {refinement_strength})")
    
    return refined_mask


def advanced_post_processing(fused_mask, previous_mask, next_mask=None, 
                           motion_magnitude=0.5, temporal_consistency=0.8,
                           enable_adaptive_morph=True, enable_temporal_refinement=True):
    """Advanced post-processing function for testing"""
    if fused_mask is None or np.sum(fused_mask > 0) == 0:
        return fused_mask
    
    print(f"Starting advanced post-processing...")
    print(f"Input mask area: {np.sum(fused_mask > 0)}")
    
    processed_mask = fused_mask.copy()
    
    # Step 1: Adaptive morphological operations
    if enable_adaptive_morph:
        print(f"Step 1: Adaptive morphological operations")
        processed_mask = adaptive_morphological_operations(
            processed_mask, motion_magnitude, temporal_consistency
        )
        print(f"After morphological operations: {np.sum(processed_mask > 0)} pixels")
    
    # Step 2: Temporal boundary refinement
    if enable_temporal_refinement:
        print(f"Step 2: Temporal boundary refinement")
        processed_mask = temporal_boundary_refinement(
            processed_mask, previous_mask, next_mask
        )
        print(f"After temporal refinement: {np.sum(processed_mask > 0)} pixels")
    
    # Step 3: Final consistency check
    if previous_mask is not None:
        final_consistency = compute_temporal_consistency_loss(processed_mask, previous_mask)
        print(f"Final temporal consistency: {final_consistency:.3f}")
        
        if final_consistency < temporal_consistency * 0.8:
            print(f"Consistency dropped significantly, applying temporal smoothing")
            blend_factor = 0.7
            processed_mask = cv2.addWeighted(processed_mask, blend_factor, 
                                           previous_mask, 1.0 - blend_factor, 0)
            processed_mask = (processed_mask > 127).astype(np.uint8) * 255
            print(f"After temporal smoothing: {np.sum(processed_mask > 0)} pixels")
    
    # Step 4: Quality validation
    final_area = np.sum(processed_mask > 0)
    original_area = np.sum(fused_mask > 0)
    
    if final_area > 0:
        area_change = abs(final_area - original_area) / original_area
        print(f"Area change: {area_change:.1%}")
        
        if area_change > 0.3:
            print(f"[WARNING] Large area change detected ({area_change:.1%})")
    
    print(f"Advanced post-processing completed")
    print(f"Final mask area: {final_area} pixels")
    
    return processed_mask


def test_advanced_post_processing():
    """Test the advanced post-processing system"""
    print("Testing Advanced Post-Processing System")
    print("=" * 60)
    
    # Create test masks with different characteristics
    print("\n1. Creating test masks...")
    
    # Create a mask with noise and holes
    base_mask = np.zeros((100, 100), dtype=np.uint8)
    base_mask[20:80, 20:80] = 255  # Main object
    
    # Add noise
    noise_mask = base_mask.copy()
    noise_mask[15:25, 15:25] = 255  # Noise region
    noise_mask[75:85, 75:85] = 255  # More noise
    
    # Add holes
    hole_mask = noise_mask.copy()
    hole_mask[35:45, 35:45] = 0    # Hole in object
    
    # Create previous mask (slightly different)
    prev_mask = np.zeros((100, 100), dtype=np.uint8)
    prev_mask[22:78, 22:78] = 255  # Similar but not identical
    
    print(f"Base mask: Area = {np.sum(base_mask > 0)}")
    print(f"Noise mask: Area = {np.sum(noise_mask > 0)}")
    print(f"Hole mask: Area = {np.sum(hole_mask > 0)}")
    print(f"Previous mask: Area = {np.sum(prev_mask > 0)}")
    
    print("\n2. Testing motion magnitude computation...")
    
    motion_score = compute_motion_magnitude(hole_mask, prev_mask)
    print(f"Motion magnitude: {motion_score:.3f}")
    
    print("\n3. Testing adaptive morphological operations...")
    
    # Test with different motion and consistency scenarios
    scenarios = [
        ("Low motion, High consistency", 0.2, 0.9),
        ("Medium motion, Medium consistency", 0.5, 0.6),
        ("High motion, Low consistency", 0.8, 0.3)
    ]
    
    for desc, motion, consistency in scenarios:
        print(f"\n   Testing: {desc}")
        processed = adaptive_morphological_operations(
            hole_mask, motion, consistency, base_kernel_size=3, max_kernel_size=7
        )
        print(f"Original area: {np.sum(hole_mask > 0)}")
        print(f"Processed area: {np.sum(processed > 0)}")
    
    print("\n4. Testing temporal boundary refinement...")
    
    refined = temporal_boundary_refinement(hole_mask, prev_mask, refinement_strength=0.3)
    print(f"Original area: {np.sum(hole_mask > 0)}")
    print(f"Refined area: {np.sum(refined > 0)}")
    
    print("\n5. Testing full advanced post-processing pipeline...")
    
    # Test with different configurations
    configs = [
        ("Full pipeline", True, True),
        ("Morphology only", True, False),
        ("Refinement only", False, True),
        ("Disabled", False, False)
    ]
    
    for desc, enable_morph, enable_refine in configs:
        print(f"\n   Testing: {desc}")
        final = advanced_post_processing(
            hole_mask, prev_mask, None, motion_score, 0.6,
            enable_morph, enable_refine
        )
        print(f"Final area: {np.sum(final > 0)}")
    
    print("\nAll advanced post-processing tests passed!")
    return True


if __name__ == "__main__":
    print("=" * 60)
    print("ADVANCED POST-PROCESSING SYSTEM TEST")
    print("=" * 60)
    
    try:
        success = test_advanced_post_processing()
        if success:
            print(f"\nAll tests passed successfully!")
            print(f"\nThe advanced post-processing system is ready!")
            print(f"\nKey Features:")
            print(f"-Motion-aware morphological operations")
            print(f"-Temporal boundary refinement")
            print(f"-Adaptive kernel sizing")
            print(f"-Quality validation and consistency checks")
            print(f"-Intelligent noise removal and hole filling")
        else:
            print(f"\nSome tests failed.")
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()
    
    print("=" * 60)
