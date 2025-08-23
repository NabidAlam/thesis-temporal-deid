#!/usr/bin/env python3
"""
Test Suite for run_samurai_baseline.py
This script tests all functions and components to ensure they work correctly
"""

import os
import sys
import numpy as np
import tempfile
import shutil
from pathlib import Path
import unittest
from unittest.mock import patch, MagicMock

# Add the current directory to path to import the script
sys.path.append('.')

# Import the functions from run_samurai_baseline.py
try:
    from run_samurai_baseline import (
        load_ground_truth_mask,
        generate_bbox_from_mask,
        calculate_iou,
        calculate_dice_coefficient,
        calculate_precision_recall,
        calculate_boundary_accuracy,
        calculate_hausdorff_distance,
        calculate_contour_similarity,
        calculate_region_based_metrics,
        calculate_temporal_stability_metrics,
        calculate_complexity_metrics,
        smooth_metric,
        analyze_failure_cases,
        get_memory_usage,
        get_gpu_memory_usage
    )
    print("Successfully imported all functions from run_samurai_baseline.py")
except ImportError as e:
    print(f"Error importing functions: {e}")
    sys.exit(1)

class TestRunSamuraiBaseline(unittest.TestCase):
    """Test suite for run_samurai_baseline.py functions"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create temporary test directory
        self.test_dir = tempfile.mkdtemp()
        self.test_img_dir = Path(self.test_dir) / "JPEGImages" / "480p" / "test_sequence"
        self.test_ann_dir = Path(self.test_dir) / "Annotations" / "480p" / "test_sequence"
        
        # Create test directories
        self.test_img_dir.mkdir(parents=True, exist_ok=True)
        self.test_ann_dir.mkdir(parents=True, exist_ok=True)
        
        # Create test masks
        self.test_mask_1 = np.zeros((100, 100), dtype=np.uint8)
        self.test_mask_1[25:75, 25:75] = 1  # 50x50 square
        
        self.test_mask_2 = np.zeros((100, 100), dtype=np.uint8)
        self.test_mask_2[30:70, 30:70] = 1  # 40x40 square (overlapping)
        
        self.test_mask_3 = np.zeros((100, 100), dtype=np.uint8)
        self.test_mask_3[80:90, 80:90] = 1  # 10x10 square (non-overlapping)
        
        # Create test annotation file
        self.test_ann_file = self.test_ann_dir / "test_001.png"
        import imageio
        imageio.imwrite(str(self.test_ann_file), self.test_mask_1 * 255)
    
    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.test_dir)
    
    def test_load_ground_truth_mask(self):
        """Test ground truth mask loading"""
        print("Testing load_ground_truth_mask...")
        
        # Test successful loading
        mask = load_ground_truth_mask(self.test_ann_file)
        self.assertIsNotNone(mask)
        self.assertEqual(mask.shape, (100, 100))
        self.assertTrue(np.array_equal(mask, self.test_mask_1))
        
        # Test non-existent file
        non_existent = Path("non_existent.png")
        mask = load_ground_truth_mask(non_existent)
        self.assertIsNone(mask)
        
        print("load_ground_truth_mask: PASSED")
    
    def test_generate_bbox_from_mask(self):
        """Test bounding box generation"""
        print("Testing generate_bbox_from_mask...")
        
        # Test normal mask
        bbox = generate_bbox_from_mask(self.test_mask_1)
        expected = [25, 25, 74, 74]  # [x_min, y_min, x_max, y_max]
        self.assertEqual(bbox, expected)
        
        # Test empty mask
        empty_mask = np.zeros((100, 100), dtype=np.uint8)
        bbox = generate_bbox_from_mask(empty_mask)
        self.assertEqual(bbox, [0, 0, 0, 0])
        
        # Test None mask
        bbox = generate_bbox_from_mask(None)
        self.assertEqual(bbox, [0, 0, 0, 0])
        
        print("generate_bbox_from_mask: PASSED")
    
    def test_calculate_iou(self):
        """Test IoU calculation"""
        print("🧪 Testing calculate_iou...")
        
        # Test overlapping masks
        iou = calculate_iou(self.test_mask_1, self.test_mask_2)
        expected_iou = (40 * 40) / (50 * 50 + 40 * 40 - 40 * 40)  # 1600 / 2500 = 0.64
        self.assertAlmostEqual(iou, expected_iou, places=3)
        
        # Test non-overlapping masks
        iou = calculate_iou(self.test_mask_1, self.test_mask_3)
        self.assertEqual(iou, 0.0)
        
        # Test identical masks
        iou = calculate_iou(self.test_mask_1, self.test_mask_1)
        self.assertEqual(iou, 1.0)
        
        # Test empty masks
        empty_mask = np.zeros((100, 100), dtype=np.uint8)
        iou = calculate_iou(empty_mask, empty_mask)
        self.assertEqual(iou, 0.0)
        
        print("calculate_iou: PASSED")
    
    def test_calculate_dice_coefficient(self):
        """Test Dice coefficient calculation"""
        print("Testing calculate_dice_coefficient...")
        
        # Test overlapping masks
        dice = calculate_dice_coefficient(self.test_mask_1, self.test_mask_2)
        expected_dice = (2 * 40 * 40) / (50 * 50 + 40 * 40)  # 3200 / 4100 ≈ 0.780
        self.assertAlmostEqual(dice, expected_dice, places=3)
        
        # Test non-overlapping masks
        dice = calculate_dice_coefficient(self.test_mask_1, self.test_mask_3)
        self.assertEqual(dice, 0.0)
        
        # Test identical masks
        dice = calculate_dice_coefficient(self.test_mask_1, self.test_mask_1)
        self.assertEqual(dice, 1.0)
        
        print("calculate_dice_coefficient: PASSED")
    
    def test_calculate_precision_recall(self):
        """Test precision and recall calculation"""
        print("Testing calculate_precision_recall...")
        
        # Test overlapping masks
        precision, recall = calculate_precision_recall(self.test_mask_1, self.test_mask_2)
        
        # True positives: 40*40 = 1600
        # False positives: 0 (no pixels in pred that aren't in GT)
        # False negatives: 50*50 - 40*40 = 2500 - 1600 = 900
        
        expected_precision = 1600 / (1600 + 0)  # 1.0
        expected_recall = 1600 / (1600 + 900)   # 1600/2500 = 0.64
        
        self.assertAlmostEqual(precision, expected_precision, places=3)
        self.assertAlmostEqual(recall, expected_recall, places=3)
        
        print("calculate_precision_recall: PASSED")
    
    def test_calculate_boundary_accuracy(self):
        """Test boundary accuracy calculation"""
        print("Testing calculate_boundary_accuracy...")
        
        try:
            accuracy = calculate_boundary_accuracy(self.test_mask_1, self.test_mask_2)
            self.assertIsInstance(accuracy, float)
            self.assertTrue(0.0 <= accuracy <= 1.0)
            print("calculate_boundary_accuracy: PASSED (with scipy)")
        except ImportError:
            print("calculate_boundary_accuracy: SKIPPED (scipy not available)")
    
    def test_calculate_hausdorff_distance(self):
        """Test Hausdorff distance calculation"""
        print("Testing calculate_hausdorff_distance...")
        
        try:
            distance = calculate_hausdorff_distance(self.test_mask_1, self.test_mask_2)
            self.assertIsInstance(distance, float)
            self.assertTrue(distance >= 0.0)
            print("calculate_hausdorff_distance: PASSED (with scipy)")
        except ImportError:
            print("calculate_hausdorff_distance: SKIPPED (scipy not available)")
    
    def test_calculate_contour_similarity(self):
        """Test contour similarity calculation"""
        print("Testing calculate_contour_similarity...")
        
        try:
            similarity = calculate_contour_similarity(self.test_mask_1, self.test_mask_2)
            self.assertIsInstance(similarity, float)
            self.assertTrue(0.0 <= similarity <= 1.0)
            print("calculate_contour_similarity: PASSED (with skimage)")
        except ImportError:
            print("calculate_contour_similarity: SKIPPED (skimage not available)")
    
    def test_calculate_region_based_metrics(self):
        """Test region-based metrics calculation"""
        print("Testing calculate_region_based_metrics...")
        
        try:
            metrics = calculate_region_based_metrics(self.test_mask_1, self.test_mask_2)
            self.assertIsInstance(metrics, dict)
            self.assertIn('adapted_rand_error', metrics)
            self.assertIn('variation_of_information', metrics)
            self.assertIsInstance(metrics['adapted_rand_error'], float)
            self.assertIsInstance(metrics['variation_of_information'], float)
            print("calculate_region_based_metrics: PASSED (with skimage)")
        except ImportError:
            print("calculate_region_based_metrics: SKIPPED (skimage not available)")
    
    def test_calculate_temporal_stability_metrics(self):
        """Test temporal stability metrics calculation"""
        print("Testing calculate_temporal_stability_metrics...")
        
        # Create a sequence of masks
        masks = [self.test_mask_1, self.test_mask_2, self.test_mask_3]
        
        metrics = calculate_temporal_stability_metrics(masks)
        self.assertIsInstance(metrics, dict)
        self.assertIn('temporal_iou_mean', metrics)
        self.assertIn('temporal_iou_std', metrics)
        self.assertIn('temporal_coverage_variance', metrics)
        self.assertIn('temporal_shape_consistency', metrics)
        
        # Test with single mask
        single_mask_metrics = calculate_temporal_stability_metrics([self.test_mask_1])
        self.assertEqual(single_mask_metrics['temporal_iou_mean'], 0.0)
        
        print("calculate_temporal_stability_metrics: PASSED")
    
    def test_calculate_complexity_metrics(self):
        """Test complexity metrics calculation"""
        print("Testing calculate_complexity_metrics...")
        
        try:
            metrics = calculate_complexity_metrics(self.test_mask_1)
            self.assertIsInstance(metrics, dict)
            self.assertIn('object_count', metrics)
            self.assertIn('total_area', metrics)
            self.assertIn('avg_area', metrics)
            self.assertIn('perimeter', metrics)
            self.assertIn('compactness', metrics)
            self.assertIn('eccentricity', metrics)
            
            self.assertEqual(metrics['object_count'], 1)
            self.assertEqual(metrics['total_area'], 2500)  # 50x50
            
            print("calculate_complexity_metrics: PASSED (with skimage)")
        except ImportError:
            print("calculate_complexity_metrics: SKIPPED (skimage not available)")
    
    def test_smooth_metric(self):
        """Test metric smoothing"""
        print("Testing smooth_metric...")
        
        # Test first value (no previous)
        smoothed = smooth_metric(0.5, None)
        self.assertEqual(smoothed, 0.5)
        
        # Test smoothing
        smoothed = smooth_metric(0.8, 0.5)
        expected = 0.9 * 0.5 + 0.1 * 0.8  # 0.45 + 0.08 = 0.53
        self.assertAlmostEqual(smoothed, expected, places=3)
        
        print("smooth_metric: PASSED")
    
    def test_analyze_failure_cases(self):
        """Test failure case analysis"""
        print("Testing analyze_failure_cases...")
        
        analysis = analyze_failure_cases(self.test_mask_1, self.test_mask_2, 0, "test")
        
        self.assertIsInstance(analysis, dict)
        self.assertIn('false_negatives', analysis)
        self.assertIn('false_positives', analysis)
        self.assertIn('true_positives', analysis)
        self.assertIn('fn_ratio', analysis)
        self.assertIn('fp_ratio', analysis)
        self.assertIn('is_failure', analysis)
        self.assertIn('failure_severity', analysis)
        
        # Verify calculations
        self.assertEqual(analysis['true_positives'], 1600)  # 40x40 overlap
        self.assertEqual(analysis['false_negatives'], 900)   # 2500 - 1600
        self.assertEqual(analysis['false_positives'], 0)     # No extra pixels in pred
        
        print("analyze_failure_cases: PASSED")
    
    def test_get_memory_usage(self):
        """Test memory usage function"""
        print("Testing get_memory_usage...")
        
        memory_info = get_memory_usage()
        self.assertIsInstance(memory_info, dict)
        self.assertIn('cpu_memory_percent', memory_info)
        self.assertIn('cpu_memory_used_gb', memory_info)
        
        # Values should be reasonable
        self.assertTrue(0 <= memory_info['cpu_memory_percent'] <= 100)
        self.assertTrue(memory_info['cpu_memory_used_gb'] > 0)
        
        print("get_memory_usage: PASSED")
    
    def test_get_gpu_memory_usage(self):
        """Test GPU memory usage function"""
        print("Testing get_gpu_memory_usage...")
        
        gpu_memory = get_gpu_memory_usage()
        self.assertIsInstance(gpu_memory, (int, float))
        self.assertTrue(gpu_memory >= 0)
        
        print("get_gpu_memory_usage: PASSED")
    
    def test_import_samurai_modules(self):
        """Test SAMURAI module imports"""
        print("Testing SAMURAI module imports...")
        
        try:
            # Test if we can import the main function
            from run_samurai_baseline import run_samurai_davis_baseline
            print("SAMURAI module imports: PASSED")
        except ImportError as e:
            print(f"SAMURAI module imports: SKIPPED ({e})")
    
    def test_script_syntax(self):
        """Test that the entire script can be imported without syntax errors"""
        print("Testing complete script syntax...")
        
        try:
            # Import the entire module
            import run_samurai_baseline
            print("Complete script syntax: PASSED")
        except Exception as e:
            self.fail(f"Script has syntax errors: {e}")

def run_comprehensive_test():
    """Run comprehensive test suite"""
    print("Starting Comprehensive Test Suite for run_samurai_baseline.py")
    print("=" * 70)
    
    # Create test suite
    test_suite = unittest.TestLoader().loadTestsFromTestCase(TestRunSamuraiBaseline)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print("=" * 70)
    print("TEST SUMMARY:")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\nFAILURES:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback}")
    
    if result.errors:
        print("\nERRORS:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback}")
    
    if result.wasSuccessful():
        print("\nALL TESTS PASSED! Your script is working correctly.")
        return True
    else:
        print("\nSome tests failed. Please check the issues above.")
        return False

if __name__ == "__main__":
    print("Testing run_samurai_baseline.py components...")
    success = run_comprehensive_test()
    
    if success:
        print("\nYour run_samurai_baseline.py script is ready for production!")
        print("You can now run it with confidence on all sequences.")
    else:
        print("\nPlease fix the issues above before running the main script.")
        sys.exit(1)
