# EnvisionObjectAnnotator Integration with Hybrid Temporal Pipeline

This integration combines your hybrid temporal pipeline with EnvisionObjectAnnotator capabilities for multimodal object tracking and selective de-identification in behavioral research scenarios.

## 🎯 Purpose

This integration enables:
- **Text-guided object segmentation** using SAMURAI's multimodal capabilities
- **Selective de-identification** (blur people, preserve objects for analysis)
- **Object overlap detection** for behavioral research (e.g., "baby looking at ball")
- **Privacy-preserving behavioral analysis** for research applications

## 📁 Directory Structure

```
envisionbox_integration/
├── setup/                          # Setup and installation scripts
│   ├── setup_envisionbox_complete.py
│   └── check_environment.py
├── integration/                    # Core integration code
│   └── hybrid_envisionbox_integration.py
├── test_scenarios/                 # Behavioral scenario tests
│   └── test_behavioral_scenarios.py
├── utils/                          # Utility functions
├── configs/                        # Configuration files
│   └── main_config.json
├── output/                         # Output videos and data
├── logs/                          # Log files
└── README.md                      # This file
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Run complete setup
python setup/setup_envisionbox_complete.py

# Check environment
python setup/check_environment.py
```

### 2. Download SAM2 Checkpoints

Download the required SAM2 checkpoints to `checkpoints/` folder:
- `sam2_hiera_large.pt` (recommended)
- `sam2_hiera_base_plus.pt`
- `sam2_hiera_small.pt`

### 3. Test Behavioral Scenarios

```bash
# Test all scenarios
python test_scenarios/test_behavioral_scenarios.py all

# Test specific scenario
python test_scenarios/test_behavioral_scenarios.py baby_with_ball
```

### 4. Process Your Own Videos

```bash
# Process video with hybrid integration
python integration/hybrid_envisionbox_integration.py input_video.mp4 output_video.mp4
```

## 🧪 Behavioral Scenarios

### 1. Baby with Ball
- **Purpose**: Track ball movement, de-identify people, detect "baby looking at ball"
- **Text Prompts**: `["baby", "ball", "person", "face", "hand"]`
- **De-identify**: People, faces
- **Preserve**: Ball, baby, hands
- **Overlap Detection**: Baby gaze on ball

### 2. Person on Stage
- **Purpose**: Track speaker, de-identify audience, detect attention patterns
- **Text Prompts**: `["person", "face", "audience", "stage", "podium"]`
- **De-identify**: Audience, faces
- **Preserve**: Speaker, stage, podium
- **Overlap Detection**: Audience looking at speaker

### 3. Multiple People
- **Purpose**: Track interactions, selective de-identification
- **Text Prompts**: `["person", "face", "group", "interaction"]`
- **De-identify**: Faces only
- **Preserve**: People, groups, interactions
- **Overlap Detection**: Face-to-face interactions

## 🔧 Configuration

Edit `configs/main_config.json` to customize:

```json
{
  "sam2": {
    "model_type": "sam2_hiera_large",
    "device": "cuda"
  },
  "hybrid_integration": {
    "confidence_threshold": 0.5,
    "blur_strength": 15,
    "overlap_threshold": 10
  }
}
```

## 📊 Output Files

### Processed Videos
- `output/processed_[scenario].mp4` - Videos with selective de-identification

### Behavioral Data
- `output/behavioral_data_[scenario].csv` - Overlap detection data
- `output/behavioral_scenarios_report.json` - Summary report

### Logs
- `logs/hybrid_integration.log` - Processing logs

## 🔍 Key Features

### Text-Guided Segmentation
```python
# Use natural language to detect objects
text_prompts = ["person on stage", "ball", "baby"]
masks = integration.segment_with_text_prompts(frame, text_prompts)
```

### Selective De-identification
```python
# Blur people, preserve objects for analysis
processed_frame = integration.selective_deidentification(
    frame, masks, 
    deidentify_objects=["person", "face"],
    preserve_objects=["ball", "toy"]
)
```

### Object Overlap Detection
```python
# Detect "looking at" events
overlaps = integration.detect_object_overlaps(
    masks, 
    target_objects=["ball"],
    gaze_objects=["baby", "face"]
)
```

## 🎓 Research Applications

### Privacy-Preserving Behavioral Research
- **Infant Development**: Track object interactions while protecting privacy
- **Social Interaction Studies**: Analyze gaze patterns with selective de-identification
- **Educational Research**: Study learning behaviors in natural settings

### Multimodal Analysis
- **Eye-tracking Integration**: Combine with eye-tracking data
- **Gesture Recognition**: Track hand movements and object interactions
- **Attention Studies**: Analyze focus patterns in complex environments

## 🔗 Integration with Your Hybrid Pipeline

This integration extends your existing hybrid temporal pipeline with:

1. **SAMURAI Text Prompts**: Natural language object detection
2. **EnvisionObjectAnnotator Logic**: Object overlap detection
3. **Selective De-identification**: Privacy-preserving analysis
4. **Behavioral Data Export**: CSV and ELAN format support

## 📈 Performance Considerations

- **GPU Recommended**: SAM2 models benefit from CUDA acceleration
- **Memory Requirements**: 8GB+ RAM recommended for large videos
- **Processing Time**: ~2-5 seconds per frame depending on complexity
- **Optimization**: Frame skipping and lazy loading available

## 🐛 Troubleshooting

### Common Issues

1. **SAM2 Import Error**: Ensure SAM2 is properly installed
2. **CUDA Out of Memory**: Reduce batch size or use CPU
3. **FFmpeg Not Found**: Install FFmpeg system-wide
4. **Checkpoint Missing**: Download required SAM2 checkpoints

### Debug Mode

```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
python integration/hybrid_envisionbox_integration.py input.mp4 output.mp4
```

## 📚 References

- [EnvisionObjectAnnotator](https://github.com/DavAhm/EnvisionObjectAnnotator)
- [SAM2 Paper](https://arxiv.org/abs/2408.00714)
- [Your Hybrid Pipeline](../integrated_temporal_pipeline_hybrid.py)

## 🤝 Contributing

This integration is part of your thesis research. Key areas for extension:

1. **Additional Behavioral Scenarios**: Add more research use cases
2. **Performance Optimization**: Improve processing speed
3. **Integration Enhancements**: Better hybrid pipeline integration
4. **Evaluation Metrics**: Add more behavioral analysis metrics

## 📄 License

Part of your thesis research project. See main project license.
