#!/usr/bin/env python3
"""
Test Behavioral Scenarios for Hybrid EnvisionObjectAnnotator Integration
Tests various behavioral research scenarios like "baby with ball" and "person on stage".
"""

import sys
import os
import cv2
import numpy as np
from pathlib import Path
import json
import time
from typing import Dict, List, Any

# Add parent directories to path
current_dir = Path(__file__).parent
integration_dir = current_dir.parent
base_dir = integration_dir.parent

sys.path.append(str(integration_dir / "integration"))

try:
    from hybrid_envisionbox_integration import HybridEnvisionBoxIntegration
except ImportError as e:
    print(f"Error importing integration: {e}")
    sys.exit(1)

class BehavioralScenarioTester:
    """Test various behavioral scenarios"""
    
    def __init__(self):
        self.integration = HybridEnvisionBoxIntegration()
        self.output_dir = integration_dir / "output"
        self.output_dir.mkdir(exist_ok=True)
        
        # Test scenarios configuration
        self.scenarios = {
            "baby_with_ball": {
                "description": "Baby playing with ball - track ball, de-identify people",
                "text_prompts": ["baby", "ball", "person", "face", "hand"],
                "deidentify_objects": ["person", "face"],
                "preserve_objects": ["ball", "baby", "hand"],
                "target_objects": ["ball"],
                "gaze_objects": ["baby", "face"],
                "expected_overlaps": ["baby_looking_at_ball", "face_looking_at_ball"]
            },
            "person_on_stage": {
                "description": "Person giving presentation - track speaker, de-identify audience",
                "text_prompts": ["person", "face", "audience", "stage", "podium"],
                "deidentify_objects": ["audience", "face"],
                "preserve_objects": ["person", "stage", "podium"],
                "target_objects": ["person", "podium"],
                "gaze_objects": ["audience", "face"],
                "expected_overlaps": ["audience_looking_at_person", "face_looking_at_person"]
            },
            "multiple_people": {
                "description": "Multiple people interaction - track interactions, selective de-identification",
                "text_prompts": ["person", "face", "group", "interaction"],
                "deidentify_objects": ["face"],
                "preserve_objects": ["person", "group", "interaction"],
                "target_objects": ["person"],
                "gaze_objects": ["face"],
                "expected_overlaps": ["face_looking_at_person"]
            }
        }
    
    def create_test_video(self, scenario_name: str, duration: int = 10) -> str:
        """Create a synthetic test video for the scenario"""
        print(f"🎬 Creating test video for scenario: {scenario_name}")
        
        # Video parameters
        width, height = 640, 480
        fps = 30
        total_frames = duration * fps
        
        # Create output path
        output_path = self.output_dir / f"test_{scenario_name}.mp4"
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        # Create synthetic frames based on scenario
        for frame_idx in range(total_frames):
            frame = self.create_synthetic_frame(scenario_name, frame_idx, width, height)
            out.write(frame)
        
        out.release()
        print(f"Test video created: {output_path}")
        return str(output_path)
    
    def create_synthetic_frame(self, scenario_name: str, frame_idx: int, width: int, height: int) -> np.ndarray:
        """Create a synthetic frame for testing"""
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        if scenario_name == "baby_with_ball":
            # Create baby and ball scenario
            # Baby (blue circle)
            baby_center = (width//3, height//2)
            baby_radius = 30
            cv2.circle(frame, baby_center, baby_radius, (255, 100, 100), -1)
            
            # Ball (red circle) - moving
            ball_x = width//2 + int(50 * np.sin(frame_idx * 0.1))
            ball_y = height//2 + int(30 * np.cos(frame_idx * 0.1))
            ball_center = (ball_x, ball_y)
            ball_radius = 20
            cv2.circle(frame, ball_center, ball_radius, (100, 100, 255), -1)
            
            # Person (green rectangle) - static
            person_rect = (width//2, height//4, 40, 80)
            cv2.rectangle(frame, person_rect[:2], (person_rect[0]+person_rect[2], person_rect[1]+person_rect[3]), (100, 255, 100), -1)
            
        elif scenario_name == "person_on_stage":
            # Create person on stage scenario
            # Stage (gray rectangle)
            stage_rect = (0, height//2, width, height//2)
            cv2.rectangle(frame, stage_rect[:2], (stage_rect[0]+stage_rect[2], stage_rect[1]+stage_rect[3]), (128, 128, 128), -1)
            
            # Speaker (blue rectangle) - center stage
            speaker_rect = (width//2-20, height//2+20, 40, 60)
            cv2.rectangle(frame, speaker_rect[:2], (speaker_rect[0]+speaker_rect[2], speaker_rect[1]+speaker_rect[3]), (255, 100, 100), -1)
            
            # Audience (small circles) - bottom
            for i in range(5):
                audience_x = 50 + i * 100
                audience_y = height - 50
                cv2.circle(frame, (audience_x, audience_y), 15, (100, 255, 100), -1)
            
        elif scenario_name == "multiple_people":
            # Create multiple people scenario
            # People (colored rectangles)
            people_positions = [
                (width//4, height//3, 30, 50),
                (width//2, height//3, 30, 50),
                (3*width//4, height//3, 30, 50)
            ]
            
            colors = [(255, 100, 100), (100, 255, 100), (100, 100, 255)]
            for i, (x, y, w, h) in enumerate(people_positions):
                cv2.rectangle(frame, (x, y), (x+w, y+h), colors[i], -1)
        
        return frame
    
    def test_scenario(self, scenario_name: str, video_path: str = None) -> Dict[str, Any]:
        """Test a specific behavioral scenario"""
        print(f"\nTesting scenario: {scenario_name}")
        print("=" * 50)
        
        scenario_config = self.scenarios[scenario_name]
        print(f"Description: {scenario_config['description']}")
        
        # Create test video if not provided
        if video_path is None:
            video_path = self.create_test_video(scenario_name)
        
        # Process video
        output_path = self.output_dir / f"processed_{scenario_name}.mp4"
        csv_path = self.output_dir / f"behavioral_data_{scenario_name}.csv"
        
        start_time = time.time()
        
        try:
            stats = self.integration.process_behavioral_video(
                video_path=video_path,
                output_path=str(output_path),
                text_prompts=scenario_config["text_prompts"],
                deidentify_objects=scenario_config["deidentify_objects"],
                preserve_objects=scenario_config["preserve_objects"],
                target_objects=scenario_config["target_objects"],
                gaze_objects=scenario_config["gaze_objects"]
            )
            
            # Export behavioral data
            self.integration.export_behavioral_data(stats, str(csv_path))
            
            processing_time = time.time() - start_time
            
            # Analyze results
            results = self.analyze_scenario_results(scenario_name, stats, scenario_config)
            
            print(f"Scenario {scenario_name} completed successfully")
            print(f"   Processing time: {processing_time:.2f}s")
            print(f"   Frames processed: {stats['processed_frames']}")
            print(f"   Output video: {output_path}")
            print(f"   Behavioral data: {csv_path}")
            
            return {
                "scenario": scenario_name,
                "success": True,
                "processing_time": processing_time,
                "stats": stats,
                "results": results,
                "output_path": str(output_path),
                "csv_path": str(csv_path)
            }
            
        except Exception as e:
            print(f"Scenario {scenario_name} failed: {e}")
            return {
                "scenario": scenario_name,
                "success": False,
                "error": str(e)
            }
    
    def analyze_scenario_results(self, scenario_name: str, stats: Dict[str, Any], 
                               scenario_config: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze results for a specific scenario"""
        results = {
            "overlap_detections": 0,
            "expected_overlaps_found": 0,
            "deidentification_applied": True,
            "objects_preserved": True
        }
        
        # Count overlap detections
        total_overlaps = 0
        expected_overlaps_found = 0
        
        for detection in stats["overlap_detections"]:
            for overlap_key, overlap_info in detection["overlaps"].items():
                total_overlaps += 1
                if overlap_key in scenario_config["expected_overlaps"]:
                    expected_overlaps_found += 1
        
        results["overlap_detections"] = total_overlaps
        results["expected_overlaps_found"] = expected_overlaps_found
        
        # Calculate success rate
        if len(stats["overlap_detections"]) > 0:
            results["overlap_success_rate"] = expected_overlaps_found / len(stats["overlap_detections"])
        else:
            results["overlap_success_rate"] = 0
        
        return results
    
    def run_all_scenarios(self) -> Dict[str, Any]:
        """Run all behavioral scenarios"""
        print("Running All Behavioral Scenarios")
        print("=" * 60)
        
        all_results = {}
        
        for scenario_name in self.scenarios.keys():
            result = self.test_scenario(scenario_name)
            all_results[scenario_name] = result
        
        # Generate summary report
        self.generate_summary_report(all_results)
        
        return all_results
    
    def generate_summary_report(self, all_results: Dict[str, Any]):
        """Generate a summary report of all scenarios"""
        print("\nBEHAVIORAL SCENARIOS SUMMARY REPORT")
        print("=" * 60)
        
        successful_scenarios = 0
        total_scenarios = len(all_results)
        
        for scenario_name, result in all_results.items():
            if result["success"]:
                successful_scenarios += 1
                print(f"{scenario_name}: SUCCESS")
                print(f"   Processing time: {result['processing_time']:.2f}s")
                print(f"   Frames processed: {result['stats']['processed_frames']}")
                
                if "results" in result:
                    results = result["results"]
                    print(f"   Overlap detections: {results['overlap_detections']}")
                    print(f"   Expected overlaps found: {results['expected_overlaps_found']}")
                    print(f"   Success rate: {results['overlap_success_rate']:.2%}")
            else:
                print(f"{scenario_name}: FAILED")
                print(f"   Error: {result['error']}")
        
        print(f"\nOverall Success Rate: {successful_scenarios}/{total_scenarios} ({successful_scenarios/total_scenarios:.1%})")
        
        # Save report to file
        report_path = self.output_dir / "behavioral_scenarios_report.json"
        with open(report_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        print(f"Detailed report saved to: {report_path}")

def main():
    """Main function"""
    if len(sys.argv) > 1:
        scenario_name = sys.argv[1]
        tester = BehavioralScenarioTester()
        
        if scenario_name == "all":
            tester.run_all_scenarios()
        elif scenario_name in tester.scenarios:
            tester.test_scenario(scenario_name)
        else:
            print(f"Unknown scenario: {scenario_name}")
            print(f"Available scenarios: {list(tester.scenarios.keys())}")
    else:
        print("Usage: python test_behavioral_scenarios.py <scenario_name>")
        print("Available scenarios:")
        print("  - baby_with_ball")
        print("  - person_on_stage") 
        print("  - multiple_people")
        print("  - all")

if __name__ == "__main__":
    main()
