#!/usr/bin/env python3
"""
Comprehensive analysis script to generate detailed processing reports
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
import cv2

def analyze_processing_results():
    """Analyze all processing results and generate comprehensive reports"""
    print("🔍 Analyzing Processing Results")
    print("=" * 50)
    
    output_dir = Path(__file__).parent / "output"
    
    # Find all CSV files
    csv_files = list(output_dir.glob("*behavioral_data.csv"))
    
    if not csv_files:
        print("❌ No behavioral data files found")
        return
    
    print(f"📊 Found {len(csv_files)} behavioral data files:")
    for csv_file in csv_files:
        print(f"  • {csv_file.name}")
    
    # Analyze each file
    all_analyses = {}
    
    for csv_file in csv_files:
        print(f"\n📈 Analyzing: {csv_file.name}")
        
        try:
            # Read CSV data
            df = pd.read_csv(csv_file)
            
            # Basic statistics
            total_frames = int(len(df))
            unique_overlap_types = int(df['overlap_type'].nunique())
            avg_overlap = float(df['overlap_percentage'].mean())
            max_overlap = float(df['overlap_percentage'].max())
            min_overlap = float(df['overlap_percentage'].min())
            
            # De-identification analysis
            deidentified_frames = int(len(df[df['overlap_percentage'] > 0]))
            deidentification_rate = float((deidentified_frames / total_frames) * 100) if total_frames > 0 else 0.0
            
            # Behavioral patterns
            looking_events = int(len(df[df['is_looking'] == True]))
            looking_rate = float((looking_events / total_frames) * 100) if total_frames > 0 else 0.0
            
            # Frame-by-frame analysis
            frame_analysis = []
            for frame_num in df['frame'].unique()[:10]:  # First 10 frames
                frame_data = df[df['frame'] == frame_num]
                frame_analysis.append({
                    "frame": int(frame_num),
                    "overlap_types": frame_data['overlap_type'].tolist(),
                    "overlap_percentages": [float(x) for x in frame_data['overlap_percentage'].tolist()],
                    "looking_events": int(frame_data['is_looking'].sum()),
                    "total_events": int(len(frame_data))
                })
            
            # Store analysis
            analysis = {
                "file_info": {
                    "filename": csv_file.name,
                    "file_size_kb": round(csv_file.stat().st_size / 1024, 2),
                    "total_frames": total_frames
                },
                "processing_stats": {
                    "unique_overlap_types": unique_overlap_types,
                    "average_overlap_percentage": round(avg_overlap, 2),
                    "max_overlap_percentage": round(max_overlap, 2),
                    "min_overlap_percentage": round(min_overlap, 2),
                    "deidentified_frames": deidentified_frames,
                    "deidentification_rate": round(deidentification_rate, 2),
                    "looking_events": looking_events,
                    "looking_rate": round(looking_rate, 2)
                },
                "behavioral_patterns": {
                    "overlap_type_distribution": {k: int(v) for k, v in df['overlap_type'].value_counts().to_dict().items()},
                    "overlap_percentage_distribution": {
                        "0-25%": int(len(df[(df['overlap_percentage'] >= 0) & (df['overlap_percentage'] <= 25)])),
                        "25-50%": int(len(df[(df['overlap_percentage'] > 25) & (df['overlap_percentage'] <= 50)])),
                        "50-75%": int(len(df[(df['overlap_percentage'] > 50) & (df['overlap_percentage'] <= 75)])),
                        "75-100%": int(len(df[(df['overlap_percentage'] > 75) & (df['overlap_percentage'] <= 100)]))
                    }
                },
                "frame_by_frame_sample": frame_analysis,
                "raw_data_sample": df.head(10).to_dict('records')
            }
            
            all_analyses[csv_file.stem] = analysis
            
            print(f"  ✅ Processed {total_frames} frames")
            print(f"  📊 Average overlap: {avg_overlap:.1f}%")
            print(f"  👁️ Looking events: {looking_events} ({looking_rate:.1f}%)")
            print(f"  🔒 De-identification rate: {deidentification_rate:.1f}%")
            
        except Exception as e:
            print(f"  ❌ Error analyzing {csv_file.name}: {e}")
    
    # Generate comprehensive report
    comprehensive_report = {
        "analysis_metadata": {
            "total_files_analyzed": len(all_analyses),
            "analysis_timestamp": pd.Timestamp.now().isoformat(),
            "output_directory": str(output_dir)
        },
        "summary_statistics": {
            "total_frames_processed": int(sum(analysis["file_info"]["total_frames"] for analysis in all_analyses.values())),
            "average_deidentification_rate": float(np.mean([analysis["processing_stats"]["deidentification_rate"] for analysis in all_analyses.values()])),
            "average_looking_rate": float(np.mean([analysis["processing_stats"]["looking_rate"] for analysis in all_analyses.values()])),
            "most_common_overlap_type": max(
                [item for analysis in all_analyses.values() for item in analysis["behavioral_patterns"]["overlap_type_distribution"].items()],
                key=lambda x: x[1]
            )[0] if all_analyses else "N/A"
        },
        "detailed_analyses": all_analyses,
        "system_insights": {
            "deidentification_effectiveness": "High" if np.mean([analysis["processing_stats"]["deidentification_rate"] for analysis in all_analyses.values()]) > 50 else "Moderate",
            "behavioral_detection_quality": "Good" if np.mean([analysis["processing_stats"]["looking_rate"] for analysis in all_analyses.values()]) > 20 else "Limited",
            "processing_consistency": "Consistent" if len(set(analysis["processing_stats"]["unique_overlap_types"] for analysis in all_analyses.values())) == 1 else "Variable"
        }
    }
    
    # Save comprehensive report
    report_path = output_dir / "comprehensive_processing_analysis.json"
    with open(report_path, 'w') as f:
        json.dump(comprehensive_report, f, indent=2)
    
    print(f"\n📋 Comprehensive report saved to: {report_path}")
    
    # Print summary
    print(f"\n📊 SUMMARY:")
    print(f"  • Total files analyzed: {comprehensive_report['analysis_metadata']['total_files_analyzed']}")
    print(f"  • Total frames processed: {comprehensive_report['summary_statistics']['total_frames_processed']}")
    print(f"  • Average de-identification rate: {comprehensive_report['summary_statistics']['average_deidentification_rate']:.1f}%")
    print(f"  • Average looking rate: {comprehensive_report['summary_statistics']['average_looking_rate']:.1f}%")
    print(f"  • Most common interaction: {comprehensive_report['summary_statistics']['most_common_overlap_type']}")
    
    print(f"\n🎯 SYSTEM INSIGHTS:")
    print(f"  • De-identification effectiveness: {comprehensive_report['system_insights']['deidentification_effectiveness']}")
    print(f"  • Behavioral detection quality: {comprehensive_report['system_insights']['behavioral_detection_quality']}")
    print(f"  • Processing consistency: {comprehensive_report['system_insights']['processing_consistency']}")
    
    return report_path

if __name__ == "__main__":
    try:
        report_path = analyze_processing_results()
        print(f"\n🎉 Analysis complete! Report available at: {report_path}")
        print("📤 You can now share this JSON report for detailed analysis!")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
