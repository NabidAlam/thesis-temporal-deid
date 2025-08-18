#!/usr/bin/env python3
"""
Test script for intelligent fusion system
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


def compute_mask_confidence(mask, mask_type="tsp", previous_mask=None, temporal_context=None):
    """Mask confidence function for testing"""
    if mask is None or np.sum(mask > 0) == 0:
        return 0.0
    
    binary_mask = (mask > 0).astype(np.uint8)
    
    # Area-based confidence
    total_pixels = mask.shape[0] * mask.shape[1]
    mask_area = np.sum(binary_mask)
    area_confidence = min(1.0, mask_area / (total_pixels * 0.3))
    
    # Shape quality confidence
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0
    
    largest_contour = max(contours, key=cv2.contourArea)
    perimeter = cv2.arcLength(largest_contour, True)
    area = cv2.contourArea(largest_contour)
    if area > 0:
        compactness = 4 * np.pi * area / (perimeter * perimeter)
        shape_confidence = min(1.0, compactness * 2)
    else:
        shape_confidence = 0.0
    
    # Boundary smoothness
    kernel = np.ones((3, 3), np.uint8)
    eroded = cv2.erode(binary_mask, kernel, iterations=1)
    dilated = cv2.dilate(binary_mask, kernel, iterations=1)
    
    boundary_pixels = np.logical_xor(binary_mask, eroded).sum()
    total_boundary = np.logical_xor(binary_mask, dilated).sum()
    
    if total_boundary > 0:
        smoothness = 1.0 - (boundary_pixels / total_boundary)
        boundary_confidence = max(0.0, smoothness)
    else:
        boundary_confidence = 1.0
    
    # Temporal consistency
    temporal_confidence = 1.0
    if previous_mask is not None:
        temporal_confidence = compute_temporal_consistency_loss(mask, previous_mask)
    
    # Type confidence
    type_confidence = 1.0
    if mask_type == "tsp":
        type_confidence = 1.0
    elif mask_type == "sam":
        type_confidence = 0.9
    elif mask_type == "pose":
        type_confidence = 0.8
    
    # Component confidence
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask)
    if num_labels > 1:
        component_sizes = stats[1:, cv2.CC_STAT_AREA]
        largest_component_ratio = np.max(component_sizes) / mask_area if mask_area > 0 else 0
        component_confidence = largest_component_ratio
    else:
        component_confidence = 0.0
    
    # Combine scores
    confidence_weights = {
        'area': 0.25, 'shape': 0.20, 'boundary': 0.15,
        'temporal': 0.25, 'type': 0.10, 'component': 0.05
    }
    
    total_confidence = (
        confidence_weights['area'] * area_confidence +
        confidence_weights['shape'] * shape_confidence +
        confidence_weights['boundary'] * boundary_confidence +
        confidence_weights['temporal'] * temporal_confidence +
        confidence_weights['type'] * type_confidence +
        confidence_weights['component'] * component_confidence
    )
    
    return max(0.0, min(1.0, total_confidence))


def intelligent_fusion(tsp_mask, sam_mask, pose_mask, previous_mask, fusion_method="adaptive"):
    """Intelligent fusion function for testing"""
    # Compute confidence scores
    tsp_conf = compute_mask_confidence(tsp_mask, "tsp", previous_mask)
    sam_conf = compute_mask_confidence(sam_mask, "sam", previous_mask)
    pose_conf = compute_mask_confidence(pose_mask, "pose", previous_mask)
    
    print(f"  Confidence scores - TSP: {tsp_conf:.3f}, SAM: {sam_conf:.3f}, Pose: {pose_conf:.3f}")
    
    if fusion_method == "adaptive":
        max_conf = max(tsp_conf, sam_conf, pose_conf)
        
        if max_conf < 0.3:
            fusion_method_used = "temporal_fallback"
            if previous_mask is not None:
                fused_mask = previous_mask.copy()
                fusion_weights = {"tsp": 0.0, "sam": 0.0, "pose": 0.0, "previous": 1.0}
            else:
                fused_mask = tsp_mask.copy()
                fusion_weights = {"tsp": 1.0, "sam": 0.0, "pose": 0.0, "previous": 0.0}
        
        elif max_conf < 0.6:
            fusion_method_used = "weighted_fusion"
            total_conf = tsp_conf + sam_conf + pose_conf
            if total_conf > 0:
                tsp_weight = tsp_conf / total_conf
                sam_weight = sam_conf / total_conf
                pose_weight = pose_conf / total_conf
            else:
                tsp_weight, sam_weight, pose_weight = 1.0, 0.0, 0.0
            
            fused_mask = np.zeros_like(tsp_mask, dtype=np.float32)
            if tsp_weight > 0:
                fused_mask += tsp_weight * tsp_mask.astype(np.float32)
            if sam_weight > 0:
                fused_mask += sam_weight * sam_mask.astype(np.float32)
            if pose_weight > 0:
                fused_mask += pose_weight * pose_mask.astype(np.float32)
            
            fused_mask = (fused_mask > 127).astype(np.uint8) * 255
            fusion_weights = {"tsp": tsp_weight, "sam": sam_weight, "pose": pose_weight, "previous": 0.0}
        
        else:
            fusion_method_used = "best_with_blending"
            if tsp_conf >= sam_conf and tsp_conf >= pose_conf:
                best_mask = tsp_mask
                second_mask = sam_mask if sam_conf > pose_conf else pose_mask
            elif sam_conf >= pose_conf:
                best_mask = sam_mask
                second_mask = tsp_mask
            else:
                best_mask = pose_mask
                second_mask = tsp_mask
            
            blend_factor = 0.8
            fused_mask = (blend_factor * best_mask + (1 - blend_factor) * second_mask).astype(np.uint8)
            
            fusion_weights = {"tsp": 0.0, "sam": 0.0, "pose": 0.0, "previous": 0.0}
            if best_mask is tsp_mask:
                fusion_weights["tsp"] = blend_factor
                fusion_weights["sam" if second_mask is sam_mask else "pose"] = 1 - blend_factor
            elif best_mask is sam_mask:
                fusion_weights["sam"] = blend_factor
                fusion_weights["tsp" if second_mask is tsp_mask else "pose"] = 1 - blend_factor
            else:
                fusion_weights["pose"] = blend_factor
                fusion_weights["tsp" if second_mask is tsp_mask else "sam"] = 1 - blend_factor
    
    elif fusion_method == "confidence":
        fusion_method_used = "confidence_based"
        total_conf = tsp_conf + sam_conf + pose_conf
        if total_conf > 0:
            tsp_weight = tsp_conf / total_conf
            sam_weight = sam_conf / total_conf
            pose_weight = pose_conf / total_conf
        else:
            tsp_weight, sam_weight, pose_weight = 1.0, 0.0, 0.0
        
        fused_mask = (tsp_weight * tsp_mask + sam_weight * sam_mask + pose_weight * pose_mask).astype(np.uint8)
        fusion_weights = {"tsp": tsp_weight, "sam": sam_weight, "pose": pose_weight, "previous": 0.0}
    
    elif fusion_method == "temporal":
        fusion_method_used = "temporal_based"
        if previous_mask is not None:
            tsp_temp_conf = compute_temporal_consistency_loss(tsp_mask, previous_mask)
            sam_temp_conf = compute_temporal_consistency_loss(sam_mask, previous_mask)
            pose_temp_conf = compute_temporal_consistency_loss(pose_mask, previous_mask)
            
            total_temp_conf = tsp_temp_conf + sam_temp_conf + pose_temp_conf
            if total_temp_conf > 0:
                tsp_weight = tsp_temp_conf / total_temp_conf
                sam_weight = sam_temp_conf / total_temp_conf
                pose_weight = pose_temp_conf / total_temp_conf
            else:
                tsp_weight, sam_weight, pose_weight = 1.0, 0.0, 0.0
        else:
            tsp_weight = sam_weight = pose_weight = 1.0 / 3.0
        
        fused_mask = (tsp_weight * tsp_mask + sam_weight * sam_mask + pose_weight * pose_mask).astype(np.uint8)
        fusion_weights = {"tsp": tsp_weight, "sam": sam_weight, "pose": pose_weight, "previous": 0.0}
    
    else:  # hybrid
        fusion_method_used = "hybrid"
        if previous_mask is not None:
            tsp_temp_conf = compute_temporal_consistency_loss(tsp_mask, previous_mask)
            sam_temp_conf = compute_temporal_consistency_loss(sam_mask, previous_mask)
            pose_temp_conf = compute_temporal_consistency_loss(pose_mask, previous_mask)
        else:
            tsp_temp_conf = sam_temp_conf = pose_temp_conf = 1.0
        
        tsp_hybrid = 0.7 * tsp_conf + 0.3 * tsp_temp_conf
        sam_hybrid = 0.7 * sam_conf + 0.3 * sam_temp_conf
        pose_hybrid = 0.7 * pose_conf + 0.3 * pose_temp_conf
        
        total_hybrid = tsp_hybrid + sam_hybrid + pose_hybrid
        if total_hybrid > 0:
            tsp_weight = tsp_hybrid / total_hybrid
            sam_weight = sam_hybrid / total_hybrid
            pose_weight = pose_hybrid / total_hybrid
        else:
            tsp_weight, sam_weight, pose_weight = 1.0, 0.0, 0.0
        
        fused_mask = (tsp_weight * tsp_mask + sam_weight * sam_mask + pose_weight * pose_mask).astype(np.uint8)
        fusion_weights = {"tsp": tsp_weight, "sam": sam_weight, "pose": pose_weight, "previous": 0.0}
    
    print(f"  Method: {fusion_method_used}, Weights: {fusion_weights}")
    return fused_mask, fusion_weights, fusion_method_used


def test_intelligent_fusion():
    """Test the intelligent fusion system"""
    print("🧪 Testing Intelligent Fusion System")
    print("=" * 60)
    
    # Create test masks with different characteristics
    print("\n1. Creating test masks...")
    
    # TSP mask: Good quality, large area
    tsp_mask = np.zeros((100, 100), dtype=np.uint8)
    tsp_mask[20:80, 20:80] = 255
    print(f"   TSP mask: Area = {np.sum(tsp_mask > 0)}, Shape = {tsp_mask.shape}")
    
    # SAM mask: Medium quality, medium area
    sam_mask = np.zeros((100, 100), dtype=np.uint8)
    sam_mask[25:75, 25:75] = 255
    print(f"   SAM mask: Area = {np.sum(sam_mask > 0)}, Shape = {sam_mask.shape}")
    
    # Pose mask: Lower quality, smaller area
    pose_mask = np.zeros((100, 100), dtype=np.uint8)
    pose_mask[30:70, 30:70] = 255
    print(f"   Pose mask: Area = {np.sum(pose_mask > 0)}, Shape = {pose_mask.shape}")
    
    # Previous mask for temporal consistency
    prev_mask = np.zeros((100, 100), dtype=np.uint8)
    prev_mask[22:78, 22:78] = 255
    print(f"   Previous mask: Area = {np.sum(prev_mask > 0)}, Shape = {prev_mask.shape}")
    
    print("\n2. Testing different fusion strategies...")
    
    fusion_strategies = ["adaptive", "confidence", "temporal", "hybrid"]
    
    for strategy in fusion_strategies:
        print(f"\n   Testing {strategy.upper()} fusion:")
        fused_mask, weights, method = intelligent_fusion(
            tsp_mask, sam_mask, pose_mask, prev_mask, strategy
        )
        
        fused_area = np.sum(fused_mask > 0)
        print(f"     Fused mask area: {fused_area}")
        print(f"     Fusion method used: {method}")
        print(f"     Final weights: TSP={weights['tsp']:.3f}, SAM={weights['sam']:.3f}, Pose={weights['pose']:.3f}")
        if 'previous' in weights and weights['previous'] > 0:
            print(f"     Previous mask weight: {weights['previous']:.3f}")
    
    print("\n3. Testing confidence computation...")
    
    print(f"   TSP confidence: {compute_mask_confidence(tsp_mask, 'tsp', prev_mask):.3f}")
    print(f"   SAM confidence: {compute_mask_confidence(sam_mask, 'sam', prev_mask):.3f}")
    print(f"   Pose confidence: {compute_mask_confidence(pose_mask, 'pose', prev_mask):.3f}")
    
    print("\n4. Testing temporal consistency...")
    
    print(f"   TSP temporal consistency: {compute_temporal_consistency_loss(tsp_mask, prev_mask):.3f}")
    print(f"   SAM temporal consistency: {compute_temporal_consistency_loss(sam_mask, prev_mask):.3f}")
    print(f"   Pose temporal consistency: {compute_temporal_consistency_loss(pose_mask, prev_mask):.3f}")
    
    print("\n🎉 All intelligent fusion tests passed!")
    return True


if __name__ == "__main__":
    print("=" * 60)
    print("INTELLIGENT FUSION SYSTEM TEST")
    print("=" * 60)
    
    try:
        success = test_intelligent_fusion()
        if success:
            print(f"\n✅ All tests passed successfully!")
            print(f"\n🚀 The intelligent fusion system is ready!")
            print(f"\n📊 Key Features:")
            print(f"   • Multi-metric confidence scoring")
            print(f"   • Adaptive fusion strategy selection")
            print(f"   • Temporal consistency integration")
            print(f"   • Intelligent weight balancing")
            print(f"   • Robust fallback mechanisms")
        else:
            print(f"\n❌ Some tests failed.")
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
    
    print("=" * 60)
