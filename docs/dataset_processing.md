# Dataset Processing Guide

## Usage
```bash
# Use with ROCm profile
nix develop .#rocm --command dojo-manager process-dataset ./data/karate
```

## Directory Structure
```
data/karate/
├── raw/
│   └── technique1/
│       ├── video1.mp4
│       └── ...
├── processed/
├── poses/
└── annotations/
```

## Requirements
- 30+ videos per technique
- Annotation files in COCO JSON format
- Calibration data in `annotations/calibration.json`

## Output
- Processed videos in `processed/`
- Pose data in `poses/`
- Trained models in `~/.dojo/models/{coach_id}`

## GPU Recommendations
- 16GB VRAM for 1080p processing
- 32GB+ for 4K/batch processing

Run in nix shell with active profile:
```bash
nix develop .#cuda --command dojo-manager process-dataset ...
```