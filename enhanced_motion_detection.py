#!/usr/bin/env python3
"""
Enhanced Motion Detection System for Subtle Movements
Captures speaker gestures, head movements, and other subtle motions
even when frame-to-frame differences are minimal.
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

class EnhancedMotionDetector:
    """Advanced motion detection for subtle movements in speaker videos."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Motion detection parameters
        self.motion_threshold = 1.0  # Lowered from 10.0 to 1.0
        self.optical_flow_threshold = 0.5
        self.edge_motion_threshold = 2.0
        self.temporal_window = 5  # Look at multiple frames for motion
        
        # Motion history for temporal analysis
        self.motion_history = []
        self.frame_history = []
        self.max_history = 10
        
    def detect_motion_enhanced(self, current_frame: np.ndarray, previous_frame: np.ndarray) -> Dict:
        """Enhanced motion detection using multiple techniques."""
        try:
            # Method 1: Enhanced frame differencing with adaptive threshold
            frame_diff_motion = self._detect_frame_difference_motion(current_frame, previous_frame)
            
            # Method 2: Optical flow for subtle movement detection
            optical_flow_motion = self._detect_optical_flow_motion(current_frame, previous_frame)
            
            # Method 3: Edge-based motion detection
            edge_motion = self._detect_edge_based_motion(current_frame, previous_frame)
            
            # Method 4: Temporal motion analysis (multiple frame comparison)
            temporal_motion = self._detect_temporal_motion(current_frame)
            
            # Combine all motion detection results
            combined_motion = self._combine_motion_detections(
                frame_diff_motion, optical_flow_motion, edge_motion, temporal_motion
            )
            
            # Update motion history
            self._update_motion_history(combined_motion)
            
            return combined_motion
            
        except Exception as e:
            self.logger.error(f"Enhanced motion detection failed: {e}")
            return {
                'motion_detected': False,
                'confidence': 0.0,
                'motion_type': 'error',
                'error': str(e)
            }
    
    def _detect_frame_difference_motion(self, current: np.ndarray, previous: np.ndarray) -> Dict:
        """Enhanced frame differencing with adaptive thresholding."""
        try:
            # Convert to grayscale
            gray_current = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
            gray_previous = cv2.cvtColor(previous, cv2.COLOR_BGR2GRAY)
            
            # Calculate absolute difference
            diff = cv2.absdiff(gray_current, gray_previous)
            
            # Apply Gaussian blur to reduce noise
            diff_blurred = cv2.GaussianBlur(diff, (5, 5), 0)
            
            # Adaptive thresholding
            _, thresh = cv2.threshold(diff_blurred, 15, 255, cv2.THRESH_BINARY)
            
            # Calculate motion metrics
            motion_pixels = np.sum(thresh > 0)
            total_pixels = thresh.size
            motion_ratio = motion_pixels / total_pixels
            
            # Calculate mean difference magnitude
            diff_magnitude = np.mean(diff)
            
            # Detect motion based on adaptive threshold
            has_motion = motion_ratio > 0.001 or diff_magnitude > self.motion_threshold
            
            return {
                'motion_detected': has_motion,
                'confidence': min(motion_ratio * 1000, 1.0),  # Scale confidence
                'motion_ratio': motion_ratio,
                'diff_magnitude': diff_magnitude,
                'motion_pixels': motion_pixels,
                'method': 'enhanced_frame_diff'
            }
            
        except Exception as e:
            self.logger.error(f"Frame difference motion detection failed: {e}")
            return {'motion_detected': False, 'confidence': 0.0, 'method': 'frame_diff_error'}
    
    def _detect_optical_flow_motion(self, current: np.ndarray, previous: np.ndarray) -> Dict:
        """Optical flow-based motion detection for subtle movements."""
        try:
            # Convert to grayscale
            gray_current = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
            gray_previous = cv2.cvtColor(previous, cv2.COLOR_BGR2GRAY)
            
            # Calculate optical flow using Lucas-Kanade method
            # Detect good features to track
            corners = cv2.goodFeaturesToTrack(gray_previous, maxCorners=100, qualityLevel=0.01, minDistance=10)
            
            if corners is None or len(corners) < 5:
                return {'motion_detected': False, 'confidence': 0.0, 'method': 'optical_flow_no_corners'}
            
            # Calculate optical flow
            flow, status, _ = cv2.calcOpticalFlowPyrLK(gray_previous, gray_current, corners, None)
            
            if flow is None:
                return {'motion_detected': False, 'confidence': 0.0, 'method': 'optical_flow_calc_failed'}
            
            # Filter valid flow vectors
            valid_flow = flow[status.ravel() == 1]
            valid_corners = corners[status.ravel() == 1]
            
            if len(valid_flow) < 3:
                return {'motion_detected': False, 'confidence': 0.0, 'method': 'optical_flow_insufficient_vectors'}
            
            # Calculate motion magnitude from flow vectors
            flow_magnitudes = np.sqrt(valid_flow[:, 0]**2 + valid_flow[:, 1]**2)
            avg_flow_magnitude = np.mean(flow_magnitudes)
            max_flow_magnitude = np.max(flow_magnitudes)
            
            # Detect motion based on flow magnitude
            has_motion = avg_flow_magnitude > self.optical_flow_threshold
            
            return {
                'motion_detected': has_motion,
                'confidence': min(avg_flow_magnitude / 10.0, 1.0),  # Scale confidence
                'avg_flow_magnitude': avg_flow_magnitude,
                'max_flow_magnitude': max_flow_magnitude,
                'valid_vectors': len(valid_flow),
                'method': 'optical_flow'
            }
            
        except Exception as e:
            self.logger.error(f"Optical flow motion detection failed: {e}")
            return {'motion_detected': False, 'confidence': 0.0, 'method': 'optical_flow_error'}
    
    def _detect_edge_based_motion(self, current: np.ndarray, previous: np.ndarray) -> Dict:
        """Edge-based motion detection for structural changes."""
        try:
            # Convert to grayscale
            gray_current = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
            gray_previous = cv2.cvtColor(previous, cv2.COLOR_BGR2GRAY)
            
            # Detect edges using Canny
            edges_current = cv2.Canny(gray_current, 50, 150)
            edges_previous = cv2.Canny(gray_previous, 50, 150)
            
            # Calculate edge difference
            edge_diff = cv2.absdiff(edges_current, edges_previous)
            
            # Calculate edge motion metrics
            edge_motion_pixels = np.sum(edge_diff > 0)
            total_edge_pixels = edge_diff.size
            edge_motion_ratio = edge_motion_pixels / total_edge_pixels
            
            # Detect motion based on edge changes
            has_motion = edge_motion_ratio > 0.0005  # Very sensitive threshold
            
            return {
                'motion_detected': has_motion,
                'confidence': min(edge_motion_ratio * 2000, 1.0),  # Scale confidence
                'edge_motion_ratio': edge_motion_ratio,
                'edge_motion_pixels': edge_motion_pixels,
                'method': 'edge_based'
            }
            
        except Exception as e:
            self.logger.error(f"Edge-based motion detection failed: {e}")
            return {'motion_detected': False, 'confidence': 0.0, 'method': 'edge_based_error'}
    
    def _detect_temporal_motion(self, current_frame: np.ndarray) -> Dict:
        """Temporal motion analysis using multiple frame history."""
        try:
            if len(self.frame_history) < 3:
                return {'motion_detected': False, 'confidence': 0.0, 'method': 'temporal_insufficient_history'}
            
            # Convert current frame to grayscale
            gray_current = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
            
            # Compare with multiple previous frames
            temporal_diffs = []
            for prev_frame in self.frame_history[-3:]:  # Last 3 frames
                if prev_frame is not None:
                    gray_prev = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
                    diff = cv2.absdiff(gray_current, gray_prev)
                    temporal_diffs.append(np.mean(diff))
            
            if not temporal_diffs:
                return {'motion_detected': False, 'confidence': 0.0, 'method': 'temporal_no_diffs'}
            
            # Calculate temporal motion metrics
            avg_temporal_diff = np.mean(temporal_diffs)
            temporal_variance = np.var(temporal_diffs)
            
            # Detect motion based on temporal consistency
            has_motion = avg_temporal_diff > 1.0 or temporal_variance > 0.5
            
            return {
                'motion_detected': has_motion,
                'confidence': min(avg_temporal_diff / 5.0, 1.0),  # Scale confidence
                'avg_temporal_diff': avg_temporal_diff,
                'temporal_variance': temporal_variance,
                'method': 'temporal_analysis'
            }
            
        except Exception as e:
            self.logger.error(f"Temporal motion detection failed: {e}")
            return {'motion_detected': False, 'confidence': 0.0, 'method': 'temporal_error'}
    
    def _combine_motion_detections(self, *motion_results: Dict) -> Dict:
        """Combine multiple motion detection results intelligently."""
        try:
            # Count how many methods detected motion
            motion_detected_count = sum(1 for result in motion_results if result.get('motion_detected', False))
            total_methods = len(motion_results)
            
            # Calculate combined confidence
            confidences = [result.get('confidence', 0.0) for result in motion_results]
            avg_confidence = np.mean(confidences) if confidences else 0.0
            
            # Determine overall motion detection
            # If ANY method detects motion, consider it motion
            overall_motion = motion_detected_count > 0
            
            # Boost confidence if multiple methods agree
            if motion_detected_count > 1:
                avg_confidence = min(avg_confidence * 1.5, 1.0)
            
            # Determine motion type based on which methods detected motion
            motion_types = []
            for result in motion_results:
                if result.get('motion_detected', False):
                    method = result.get('method', 'unknown')
                    motion_types.append(method)
            
            motion_type = '+'.join(motion_types) if motion_types else 'none'
            
            return {
                'motion_detected': overall_motion,
                'confidence': avg_confidence,
                'motion_type': motion_type,
                'methods_agreeing': motion_detected_count,
                'total_methods': total_methods,
                'individual_results': motion_results,
                'method': 'enhanced_combined'
            }
            
        except Exception as e:
            self.logger.error(f"Motion detection combination failed: {e}")
            return {'motion_detected': False, 'confidence': 0.0, 'motion_type': 'combination_error'}
    
    def _update_motion_history(self, motion_result: Dict):
        """Update motion history for temporal analysis."""
        try:
            # Add current motion result to history
            self.motion_history.append(motion_result)
            
            # Keep only recent history
            if len(self.motion_history) > self.max_history:
                self.motion_history.pop(0)
                
        except Exception as e:
            self.logger.error(f"Failed to update motion history: {e}")
    
    def add_frame_to_history(self, frame: np.ndarray):
        """Add frame to history for temporal analysis."""
        try:
            self.frame_history.append(frame.copy())
            
            # Keep only recent frames
            if len(self.frame_history) > self.max_history:
                self.frame_history.pop(0)
                
        except Exception as e:
            self.logger.error(f"Failed to add frame to history: {e}")
    
    def get_motion_statistics(self) -> Dict:
        """Get motion statistics from history."""
        try:
            if not self.motion_history:
                return {'total_detections': 0, 'motion_rate': 0.0}
            
            total_detections = len(self.motion_history)
            motion_detected = sum(1 for result in self.motion_history if result.get('motion_detected', False))
            motion_rate = motion_detected / total_detections
            
            return {
                'total_detections': total_detections,
                'motion_detected': motion_detected,
                'motion_rate': motion_rate,
                'recent_motion_types': [result.get('motion_type', 'unknown') for result in self.motion_history[-5:]]
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get motion statistics: {e}")
            return {'total_detections': 0, 'motion_rate': 0.0}


def test_enhanced_motion_detection():
    """Test the enhanced motion detection system."""
    import cv2
    import os
    
    # Initialize detector
    detector = EnhancedMotionDetector()
    
    # Test video path
    video_path = "input/ted/video9.mp4"
    
    if not os.path.exists(video_path):
        print(f"❌ Video not found: {video_path}")
        return
    
    print("🔍 Testing Enhanced Motion Detection on video9...")
    print("This should detect subtle speaker movements that the basic system missed.")
    
    # Load video
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    # Load first 20 frames
    for i in range(20):
        ret, frame = cap.read()
        if ret:
            frames.append(frame.copy())
            detector.add_frame_to_history(frame)
        else:
            break
    
    cap.release()
    
    if len(frames) < 2:
        print("❌ Not enough frames loaded")
        return
    
    print(f"✅ Loaded {len(frames)} frames")
    print("\n🔍 Analyzing motion between consecutive frames...")
    
    # Test motion detection between consecutive frames
    motion_results = []
    
    for i in range(1, len(frames)):
        current_frame = frames[i]
        previous_frame = frames[i-1]
        
        # Detect motion using enhanced system
        motion_result = detector.detect_motion_enhanced(current_frame, previous_frame)
        motion_results.append(motion_result)
        
        print(f"Frame {i-1} → {i}:")
        print(f"  Motion: {'✅ YES' if motion_result['motion_detected'] else '❌ NO'}")
        print(f"  Confidence: {motion_result['confidence']:.3f}")
        print(f"  Type: {motion_result['motion_type']}")
        print(f"  Methods agreeing: {motion_result['methods_agreeing']}/{motion_result['total_methods']}")
        
        if motion_result['motion_detected']:
            print(f"  🎯 MOTION DETECTED! Method: {motion_result['motion_type']}")
        print()
    
    # Get overall statistics
    stats = detector.get_motion_statistics()
    print("📊 MOTION DETECTION STATISTICS:")
    print(f"  Total frame pairs analyzed: {stats['total_detections']}")
    print(f"  Motion detected in: {stats['motion_detected']} frame pairs")
    print(f"  Motion rate: {stats['motion_rate']:.1%}")
    print(f"  Recent motion types: {', '.join(stats['recent_motion_types'])}")
    
    # Summary
    motion_detected_count = sum(1 for result in motion_results if result['motion_detected'])
    print(f"\n🎯 SUMMARY: Enhanced system detected motion in {motion_detected_count}/{len(motion_results)} frame pairs")
    
    if motion_detected_count > 0:
        print("✅ SUCCESS: Enhanced motion detection is working! Subtle movements are now detectable.")
    else:
        print("⚠️  WARNING: Still no motion detected. Video may have extremely minimal movement.")


if __name__ == "__main__":
    test_enhanced_motion_detection()
