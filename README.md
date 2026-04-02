# Auto-Labeling Pipeline

Automated annotation pipeline combining SAM3, YOLOE, and Qwen3-SAM2 for industrial image datasets at scale. Reduces manual labeling effort by 80%+.

## Overview
Manual annotation is the biggest bottleneck in industrial vision projects. This pipeline automates bounding box and mask generation for new datasets by combining zero-shot detection with foundation segmentation models.

## Pipeline
```
Raw Images --> YOLOE (zero-shot detection) --> SAM3 (segmentation masks)
                                                    |
                                           Qwen3-SAM2 (refinement + verification)
                                                    |
                                           COCO/YOLO format annotations
```

## How It Works
1. **YOLOE** generates initial bounding box proposals using text prompts (e.g., "scratch", "hole", "bubble")
2. **SAM3** produces pixel-level segmentation masks from the bounding boxes
3. **Qwen3-SAM2** refines ambiguous cases using vision-language understanding
4. Annotations are exported in COCO JSON or YOLO TXT format

## Features
- Text-prompt driven: no training data needed to start
- Supports bounding box and instance segmentation output
- Batch processing for large image datasets
- Quality scoring per annotation for human review prioritization
- Export to COCO JSON, YOLO TXT, Pascal VOC XML

## Results
- 80%+ reduction in manual annotation time
- Tested across 10+ industrial domains (textile, glass, metal, automotive)
- Human review needed only for ~15-20% of generated annotations

## Tech Stack
- Python, PyTorch, HuggingFace Transformers
- SAM3, YOLOE, Qwen3-SAM2
- OpenCV for image I/O and visualization
