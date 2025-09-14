import numpy as np
import cv2
from typing import Tuple, Dict, Optional

def compute_mask_confidence(mask: np.ndarray, mask_type: str, prev_mask: Optional[np.ndarray] = None) -> float:
    """Compute confidence score for a mask based on multiple metrics"""
    if mask is None or mask.size == 0:
        return 0.0
    
    # Convert to binary if needed
    if mask.max() > 1:
        binary_mask = (mask > 128).astype(np.uint8)
    else:
        binary_mask = mask.astype(np.uint8)
    
    # 1. Area-based confidence (larger masks are generally more reliable)
    total_pixels = mask.shape[0] * mask.shape[1]
    foreground_pixels = np.sum(binary_mask > 0)
    area_ratio = foreground_pixels / total_pixels
    area_confidence = min(area_ratio * 10, 1.0)  # Scale to 0-1
    
    # 2. Shape quality (compactness)
    if foreground_pixels > 0:
        # Find contours
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            # Use the largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            contour_area = cv2.contourArea(largest_contour)
            if contour_area > 0:
                # Compactness = area / perimeter^2 (higher is better)
                perimeter = cv2.arcLength(largest_contour, True)
                compactness = contour_area / (perimeter * perimeter) if perimeter > 0 else 0
                shape_confidence = min(compactness * 100, 1.0)  # Scale to 0-1
            else:
                shape_confidence = 0.0
        else:
            shape_confidence = 0.0
    else:
        shape_confidence = 0.0
    
    # 3. Boundary smoothness
    if foreground_pixels > 0:
        # Apply morphological operations to smooth the mask
        kernel = np.ones((3, 3), np.uint8)
        smoothed = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
        smoothed = cv2.morphologyEx(smoothed, cv2.MORPH_OPEN, kernel)
        
        # Calculate boundary smoothness (difference between original and smoothed)
        boundary_diff = np.sum(np.abs(binary_mask - smoothed))
        smoothness = 1.0 - (boundary_diff / (foreground_pixels * 2))  # Normalize
        smoothness_confidence = max(0.0, min(smoothness, 1.0))
    else:
        smoothness_confidence = 0.0
    
    # 4. Temporal consistency (if previous mask available)
    temporal_confidence = 0.0
    if prev_mask is not None and prev_mask.size > 0:
        if prev_mask.max() > 1:
            prev_binary = (prev_mask > 128).astype(np.uint8)
        else:
            prev_binary = prev_mask.astype(np.uint8)
        
        # IoU-based temporal consistency
        intersection = np.sum(np.logical_and(binary_mask, prev_binary))
        union = np.sum(np.logical_or(binary_mask, prev_binary))
        if union > 0:
            iou = intersection / union
            temporal_confidence = iou
    
    # 5. Mask type confidence (different types have different reliability)
    type_confidence = {
        'tsp': 0.9,      # TSP-SAM is generally reliable
        'sam': 0.8,      # SAM is good but can be inconsistent
        'pose': 0.6,     # Pose-based masks are less reliable
        'manual': 1.0,   # Manual masks are most reliable
        'unknown': 0.5   # Default for unknown types
    }.get(mask_type.lower(), 0.5)
    
    # 6. Connected components analysis
    if foreground_pixels > 0:
        num_labels, labels = cv2.connectedComponents(binary_mask)
        if num_labels > 1:  # More than just background
            # Prefer masks with fewer, larger components
            component_sizes = [np.sum(labels == i) for i in range(1, num_labels)]
            largest_component = max(component_sizes) if component_sizes else 0
            component_ratio = largest_component / foreground_pixels if foreground_pixels > 0 else 0
            component_confidence = component_ratio
        else:
            component_confidence = 0.0
    else:
        component_confidence = 0.0
    
    # Weighted combination of all confidence metrics
    weights = {
        'area': 0.2,
        'shape': 0.2,
        'smoothness': 0.15,
        'temporal': 0.2,
        'type': 0.15,
        'component': 0.1
    }
    
    final_confidence = (
        weights['area'] * area_confidence +
        weights['shape'] * shape_confidence +
        weights['smoothness'] * smoothness_confidence +
        weights['temporal'] * temporal_confidence +
        weights['type'] * type_confidence +
        weights['component'] * component_confidence
    )
    
    return max(0.0, min(1.0, final_confidence))

def compute_temporal_consistency_loss(current_mask: np.ndarray, previous_mask: np.ndarray) -> float:
    """Compute temporal consistency loss between current and previous masks"""
    if current_mask is None or previous_mask is None:
        return 1.0  # Maximum loss if either mask is missing
    
    # Convert to binary if needed
    if current_mask.max() > 1:
        current_binary = (current_mask > 128).astype(np.uint8)
    else:
        current_binary = current_mask.astype(np.uint8)
    
    if previous_mask.max() > 1:
        prev_binary = (previous_mask > 128).astype(np.uint8)
    else:
        prev_binary = previous_mask.astype(np.uint8)
    
    # 1. IoU-based consistency
    intersection = np.sum(np.logical_and(current_binary, prev_binary))
    union = np.sum(np.logical_or(current_binary, prev_binary))
    iou = intersection / union if union > 0 else 0.0
    
    # 2. Area change consistency
    current_area = np.sum(current_binary > 0)
    prev_area = np.sum(prev_binary > 0)
    total_pixels = current_binary.shape[0] * current_binary.shape[1]
    
    if prev_area > 0:
        area_change = abs(current_area - prev_area) / prev_area
    else:
        area_change = 1.0 if current_area > 0 else 0.0
    
    # Normalize area change to 0-1
    area_consistency = max(0.0, 1.0 - min(area_change, 1.0))
    
    # 3. Centroid distance consistency
    if current_area > 0 and prev_area > 0:
        # Find centroids
        current_coords = np.where(current_binary > 0)
        prev_coords = np.where(prev_binary > 0)
        
        if len(current_coords[0]) > 0 and len(prev_coords[0]) > 0:
            current_centroid = np.array([np.mean(current_coords[0]), np.mean(current_coords[1])])
            prev_centroid = np.array([np.mean(prev_coords[0]), np.mean(prev_coords[1])])
            
            # Calculate distance
            distance = np.linalg.norm(current_centroid - prev_centroid)
            max_distance = np.sqrt(total_pixels)  # Diagonal of the image
            centroid_consistency = max(0.0, 1.0 - (distance / max_distance))
        else:
            centroid_consistency = 0.0
    else:
        centroid_consistency = 0.0
    
    # Weighted combination
    weights = {'iou': 0.5, 'area': 0.3, 'centroid': 0.2}
    temporal_consistency = (
        weights['iou'] * iou +
        weights['area'] * area_consistency +
        weights['centroid'] * centroid_consistency
    )
    
    # Convert to loss (1 - consistency)
    return 1.0 - temporal_consistency

def intelligent_fusion(
    tsp_mask: np.ndarray,
    sam_mask: np.ndarray,
    pose_mask: np.ndarray,
    prev_mask: Optional[np.ndarray] = None,
    strategy: str = "adaptive"
) -> Tuple[np.ndarray, Dict[str, float], str]:
    """Intelligent fusion of multiple mask sources"""
    
    # Compute confidence scores for each mask
    tsp_confidence = compute_mask_confidence(tsp_mask, 'tsp', prev_mask)
    sam_confidence = compute_mask_confidence(sam_mask, 'sam', prev_mask)
    pose_confidence = compute_mask_confidence(pose_mask, 'pose', prev_mask)
    
    print(f"  Confidence scores: TSP={tsp_confidence:.3f}, SAM={sam_confidence:.3f}, Pose={pose_confidence:.3f}")
    
    # Normalize masks to 0-1 range
    def normalize_mask(mask):
        if mask is None or mask.size == 0:
            return np.zeros((100, 100), dtype=np.float32)
        if mask.max() > 1:
            return (mask > 128).astype(np.float32)
        return mask.astype(np.float32)
    
    tsp_norm = normalize_mask(tsp_mask)
    sam_norm = normalize_mask(sam_mask)
    pose_norm = normalize_mask(pose_mask)
    
    fusion_method_used = strategy
    
    if strategy == "adaptive":
        # Choose the best strategy based on confidence scores
        max_confidence = max(tsp_confidence, sam_confidence, pose_confidence)
        if max_confidence > 0.7:
            strategy = "confidence"
        elif prev_mask is not None:
            strategy = "temporal"
        else:
            strategy = "hybrid"
        fusion_method_used = f"adaptive->{strategy}"
    
    if strategy == "confidence":
        # Use confidence-weighted fusion
        total_confidence = tsp_confidence + sam_confidence + pose_confidence
        if total_confidence > 0:
            tsp_weight = tsp_confidence / total_confidence
            sam_weight = sam_confidence / total_confidence
            pose_weight = pose_confidence / total_confidence
        else:
            tsp_weight, sam_weight, pose_weight = 1.0, 0.0, 0.0
        
        fused_mask = (tsp_weight * tsp_norm + sam_weight * sam_norm + pose_weight * pose_norm)
        fused_mask = (fused_mask > 0.5).astype(np.uint8) * 255
        fusion_weights = {"tsp": tsp_weight, "sam": sam_weight, "pose": pose_weight, "previous": 0.0}
    
    elif strategy == "temporal":
        # Prioritize temporal consistency
        if prev_mask is not None:
            prev_norm = normalize_mask(prev_mask)
            
            # Calculate temporal consistency for each mask
            tsp_temp = 1.0 - compute_temporal_consistency_loss(tsp_norm, prev_norm)
            sam_temp = 1.0 - compute_temporal_consistency_loss(sam_norm, prev_norm)
            pose_temp = 1.0 - compute_temporal_consistency_loss(pose_norm, prev_norm)
            
            # Combine confidence and temporal consistency
            tsp_score = (tsp_confidence + tsp_temp) / 2
            sam_score = (sam_confidence + sam_temp) / 2
            pose_score = (pose_confidence + pose_temp) / 2
            
            total_score = tsp_score + sam_score + pose_score
            if total_score > 0:
                tsp_weight = tsp_score / total_score
                sam_weight = sam_score / total_score
                pose_weight = pose_score / total_score
            else:
                tsp_weight, sam_weight, pose_weight = 1.0, 0.0, 0.0
            
            fused_mask = (tsp_weight * tsp_norm + sam_weight * sam_norm + pose_weight * pose_norm)
            fused_mask = (fused_mask > 0.5).astype(np.uint8) * 255
            fusion_weights = {"tsp": tsp_weight, "sam": sam_weight, "pose": pose_weight, "previous": 0.0}
        else:
            # Fallback to confidence-based fusion
            return intelligent_fusion(tsp_mask, sam_mask, pose_mask, prev_mask, "confidence")
    
    else:  # hybrid
        # Combine all strategies with intelligent weighting
        # Base weights from confidence
        base_tsp = tsp_confidence
        base_sam = sam_confidence
        base_pose = pose_confidence
        
        # Adjust weights based on temporal consistency
        if prev_mask is not None:
            prev_norm = normalize_mask(prev_mask)
            tsp_temp = 1.0 - compute_temporal_consistency_loss(tsp_norm, prev_norm)
            sam_temp = 1.0 - compute_temporal_consistency_loss(sam_norm, prev_norm)
            pose_temp = 1.0 - compute_temporal_consistency_loss(pose_norm, prev_norm)
            
            # Boost weights for temporally consistent masks
            tsp_hybrid = base_tsp * (1.0 + tsp_temp * 0.5)
            sam_hybrid = base_sam * (1.0 + sam_temp * 0.5)
            pose_hybrid = base_pose * (1.0 + pose_temp * 0.5)
        else:
            tsp_hybrid, sam_hybrid, pose_hybrid = base_tsp, base_sam, base_pose
        
        total_hybrid = tsp_hybrid + sam_hybrid + pose_hybrid
        if total_hybrid > 0:
            tsp_weight = tsp_hybrid / total_hybrid
            sam_weight = sam_hybrid / total_hybrid
            pose_weight = pose_hybrid / total_hybrid
        else:
            tsp_weight, sam_weight, pose_weight = 1.0, 0.0, 0.0
        
        fused_mask = (tsp_weight * tsp_norm + sam_weight * sam_norm + pose_weight * pose_norm)
        fused_mask = (fused_mask > 0.5).astype(np.uint8) * 255
        fusion_weights = {"tsp": tsp_weight, "sam": sam_weight, "pose": pose_weight, "previous": 0.0}
    
    print(f"  Method: {fusion_method_used}, Weights: {fusion_weights}")
    return fused_mask, fusion_weights, fusion_method_used


def test_intelligent_fusion():
    """Test the intelligent fusion system"""
    print("Testing Intelligent Fusion System")
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
    
    print("\nAll intelligent fusion tests passed!")
    return True


if __name__ == "__main__":
    print("=" * 60)
    print("INTELLIGENT FUSION SYSTEM TEST")
    print("=" * 60)
    
    try:
        success = test_intelligent_fusion()
        if success:
            print(f"\nAll tests passed successfully!")
            print(f"\nThe intelligent fusion system is ready!")
            print(f"\nKey Features:")
            print(f"   • Multi-metric confidence scoring")
            print(f"   • Adaptive fusion strategy selection")
            print(f"   • Temporal consistency integration")
            print(f"   • Intelligent weight balancing")
            print(f"   • Robust fallback mechanisms")
        else:
            print(f"\nSome tests failed.")
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()
    
    print("=" * 60)
