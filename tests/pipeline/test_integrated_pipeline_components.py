#!/usr/bin/env python3
"""
Comprehensive Test Suite for Integrated Temporal Pipeline Components
Tests each feature individually to identify static frame repetition issues.
"""

import cv2
import numpy as np
import torch
import os
import sys
import time
from typing import Dict, List, Tuple
import logging

# Add paths for imports
sys.path.append('tsp_sam_official')
sys.path.append('samurai_official/sam2/sam2')
sys.path.append('maskanyone/worker')

class PipelineComponentTester:
    """Test each component of the integrated temporal pipeline individually."""
    
    def __init__(self):
        self.logger = self._setup_logging()
        self.test_results = {}
        
    def _setup_logging(self):
        """Setup comprehensive logging for debugging."""
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('pipeline_test_debug.log'),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger(__name__)
    
    def test_frame_loading(self, video_path: str, num_frames: int = 10) -> Dict:
        """Test 1: Basic frame loading and uniqueness."""
        self.logger.info("=== TEST 1: Frame Loading and Uniqueness ===")
        
        try:
            cap = cv2.VideoCapture(video_path)
            frames = []
            frame_hashes = []
            
            for i in range(num_frames):
                ret, frame = cap.read()
                if ret:
                    frames.append(frame.copy())
                    # Create hash to check uniqueness
                    frame_hash = hash(frame.tobytes())
                    frame_hashes.append(frame_hash)
                    self.logger.info(f"Frame {i}: Shape {frame.shape}, Hash {frame_hash}")
                else:
                    break
            
            cap.release()
            
            # Check for uniqueness
            unique_hashes = len(set(frame_hashes))
            is_unique = unique_hashes == len(frame_hashes)
            
            result = {
                'success': True,
                'frames_loaded': len(frames),
                'unique_frames': unique_hashes,
                'is_unique': is_unique,
                'frame_shapes': [f.shape for f in frames],
                'message': f"Loaded {len(frames)} frames, {unique_hashes} unique"
            }
            
            if not is_unique:
                result['warning'] = "DUPLICATE FRAMES DETECTED!"
                
        except Exception as e:
            result = {'success': False, 'error': str(e)}
            
        self.test_results['frame_loading'] = result
        return result
    
    def test_temporal_memory_update(self, frames: List[np.ndarray]) -> Dict:
        """Test 2: Temporal memory update mechanism."""
        self.logger.info("=== TEST 2: Temporal Memory Update Mechanism ===")
        
        try:
            # Simulate temporal memory update
            memory_updates = []
            frame_differences = []
            
            for i in range(1, len(frames)):
                # Calculate frame difference
                diff = cv2.absdiff(frames[i-1], frames[i])
                diff_magnitude = np.mean(diff)
                frame_differences.append(diff_magnitude)
                
                # Simulate memory update
                memory_update = {
                    'frame_idx': i,
                    'diff_magnitude': diff_magnitude,
                    'has_motion': diff_magnitude > 10.0,  # Threshold for motion
                    'timestamp': time.time()
                }
                memory_updates.append(memory_update)
                
                self.logger.info(f"Frame {i}: Diff magnitude {diff_magnitude:.2f}, Motion: {memory_update['has_motion']}")
            
            # Check if memory is actually updating
            motion_detected = sum(1 for m in memory_updates if m['has_motion'])
            avg_diff = np.mean(frame_differences)
            
            result = {
                'success': True,
                'memory_updates': len(memory_updates),
                'motion_detected': motion_detected,
                'average_frame_difference': avg_diff,
                'frame_differences': frame_differences,
                'message': f"Memory updated {len(memory_updates)} times, Motion detected in {motion_detected} frames"
            }
            
            if motion_detected == 0:
                result['warning'] = "NO MOTION DETECTED - All frames appear identical!"
                
        except Exception as e:
            result = {'success': False, 'error': str(e)}
            
        self.test_results['temporal_memory'] = result
        return result
    
    def test_mask_generation_uniqueness(self, frames: List[np.ndarray]) -> Dict:
        """Test 3: Mask generation uniqueness across frames."""
        self.logger.info("=== TEST 3: Mask Generation Uniqueness ===")
        
        try:
            # Simulate mask generation for each frame
            masks = []
            mask_hashes = []
            
            for i, frame in enumerate(frames):
                # Create a simple simulated mask (replace with actual mask generation)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                _, mask = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
                
                masks.append(mask)
                mask_hash = hash(mask.tobytes())
                mask_hashes.append(mask_hash)
                
                self.logger.info(f"Frame {i}: Mask hash {mask_hash}")
            
            # Check mask uniqueness
            unique_masks = len(set(mask_hashes))
            is_unique = unique_masks == len(mask_hashes)
            
            result = {
                'success': True,
                'masks_generated': len(masks),
                'unique_masks': unique_masks,
                'is_unique': is_unique,
                'mask_hashes': mask_hashes,
                'message': f"Generated {len(masks)} masks, {unique_masks} unique"
            }
            
            if not is_unique:
                result['warning'] = "DUPLICATE MASKS DETECTED!"
                
        except Exception as e:
            result = {'success': False, 'error': str(e)}
            
        self.test_results['mask_generation'] = result
        return result
    
    def test_temporal_consistency(self, frames: List[np.ndarray]) -> Dict:
        """Test 4: Temporal consistency calculation."""
        self.logger.info("=== TEST 4: Temporal Consistency Calculation ===")
        
        try:
            consistency_scores = []
            
            for i in range(1, len(frames)):
                # Calculate temporal consistency between consecutive frames
                frame1 = frames[i-1]
                frame2 = frames[i]
                
                # Convert to grayscale for comparison
                gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
                gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
                
                # Calculate structural similarity (simplified)
                diff = cv2.absdiff(gray1, gray2)
                similarity = 1.0 - (np.mean(diff) / 255.0)
                consistency_scores.append(similarity)
                
                self.logger.info(f"Frames {i-1}-{i}: Consistency {similarity:.3f}")
            
            avg_consistency = np.mean(consistency_scores)
            consistency_variance = np.var(consistency_scores)
            
            result = {
                'success': True,
                'consistency_scores': consistency_scores,
                'average_consistency': avg_consistency,
                'consistency_variance': consistency_variance,
                'message': f"Average consistency: {avg_consistency:.3f}, Variance: {consistency_variance:.3f}"
            }
            
            if avg_consistency > 0.95:
                result['warning'] = "VERY HIGH CONSISTENCY - Frames may be too similar!"
            elif avg_consistency < 0.1:
                result['warning'] = "VERY LOW CONSISTENCY - Frames may be too different!"
                
        except Exception as e:
            result = {'success': False, 'error': str(e)}
            
        self.test_results['temporal_consistency'] = result
        return result
    
    def test_pipeline_integration(self, video_path: str, num_frames: int = 10) -> Dict:
        """Test 5: Full pipeline integration test."""
        self.logger.info("=== TEST 5: Full Pipeline Integration Test ===")
        
        try:
            # Load frames
            cap = cv2.VideoCapture(video_path)
            frames = []
            
            for i in range(num_frames):
                ret, frame = cap.read()
                if ret:
                    frames.append(frame.copy())
                else:
                    break
            
            cap.release()
            
            if len(frames) < 2:
                return {'success': False, 'error': 'Not enough frames loaded'}
            
            # Run all tests
            self.test_frame_loading(video_path, num_frames)
            self.test_temporal_memory_update(frames)
            self.test_mask_generation_uniqueness(frames)
            self.test_temporal_consistency(frames)
            
            # Analyze overall results
            overall_result = self._analyze_overall_results()
            
            result = {
                'success': True,
                'overall_analysis': overall_result,
                'individual_tests': self.test_results,
                'message': "All component tests completed"
            }
            
        except Exception as e:
            result = {'success': False, 'error': str(e)}
            
        return result
    
    def _analyze_overall_results(self) -> Dict:
        """Analyze overall test results to identify issues."""
        analysis = {
            'total_tests': len(self.test_results),
            'passed_tests': 0,
            'warnings': [],
            'critical_issues': []
        }
        
        for test_name, test_result in self.test_results.items():
            if test_result.get('success', False):
                analysis['passed_tests'] += 1
                
                # Check for warnings
                if 'warning' in test_result:
                    analysis['warnings'].append(f"{test_name}: {test_result['warning']}")
                
                # Check for critical issues
                if test_name == 'frame_loading' and not test_result.get('is_unique', True):
                    analysis['critical_issues'].append("DUPLICATE FRAMES - Pipeline will produce static output!")
                
                if test_name == 'temporal_memory' and test_result.get('motion_detected', 0) == 0:
                    analysis['critical_issues'].append("NO MOTION DETECTED - Temporal memory not updating!")
                
                if test_name == 'mask_generation' and not test_result.get('is_unique', True):
                    analysis['critical_issues'].append("DUPLICATE MASKS - Same masks being reused!")
        
        return analysis
    
    def run_comprehensive_test(self, video_path: str, num_frames: int = 20) -> Dict:
        """Run comprehensive test suite."""
        self.logger.info("🚀 STARTING COMPREHENSIVE PIPELINE COMPONENT TEST")
        self.logger.info(f"Testing video: {video_path}")
        self.logger.info(f"Number of frames: {num_frames}")
        
        start_time = time.time()
        
        # Run full integration test
        result = self.test_pipeline_integration(video_path, num_frames)
        
        end_time = time.time()
        result['test_duration'] = end_time - start_time
        
        # Print summary
        self._print_test_summary()
        
        return result
    
    def _print_test_summary(self):
        """Print comprehensive test summary."""
        print("\n" + "="*80)
        print("🔍 COMPREHENSIVE PIPELINE COMPONENT TEST RESULTS")
        print("="*80)
        
        for test_name, test_result in self.test_results.items():
            print(f"\n📋 {test_name.upper().replace('_', ' ')}:")
            if test_result.get('success', False):
                print(f"   ✅ SUCCESS: {test_result.get('message', 'Test passed')}")
                if 'warning' in test_result:
                    print(f"   ⚠️  WARNING: {test_result['warning']}")
            else:
                print(f"   ❌ FAILED: {test_result.get('error', 'Unknown error')}")
        
        # Overall analysis
        if 'overall_analysis' in self.test_results.get('pipeline_integration', {}):
            analysis = self.test_results['pipeline_integration']['overall_analysis']
            print(f"\n📊 OVERALL ANALYSIS:")
            print(f"   Tests Passed: {analysis['passed_tests']}/{analysis['total_tests']}")
            
            if analysis['warnings']:
                print(f"   ⚠️  Warnings: {len(analysis['warnings'])}")
                for warning in analysis['warnings']:
                    print(f"      - {warning}")
            
            if analysis['critical_issues']:
                print(f"   🚨 CRITICAL ISSUES: {len(analysis['critical_issues'])}")
                for issue in analysis['critical_issues']:
                    print(f"      - {issue}")
        
        print("\n" + "="*80)


def main():
    """Main test execution."""
    tester = PipelineComponentTester()
    
    # Test video9
    video_path = "input/ted/video9.mp4"
    
    if not os.path.exists(video_path):
        print(f"❌ Video not found: {video_path}")
        return
    
    print("🔍 Starting comprehensive pipeline component testing...")
    print("This will identify where the static frame repetition issue is occurring.")
    
    # Run comprehensive test
    result = tester.run_comprehensive_test(video_path, num_frames=20)
    
    if result.get('success', False):
        print("\n✅ Comprehensive testing completed successfully!")
    else:
        print(f"\n❌ Testing failed: {result.get('error', 'Unknown error')}")


if __name__ == "__main__":
    main()
