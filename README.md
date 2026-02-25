# PCB Defect Detection

An automated quality inspection system for printed circuit boards using deep learning. The model classifies six common defect types -- open circuits, shorts, mousebites, spurs, spurious copper, and pin-holes -- achieving a 91.2% F1-score on the DeepPCB benchmark dataset.

Built with PyTorch and ResNet-18 (transfer learning from ImageNet), the system runs inference at ~20 ms per image on a single GPU.

---

## Performance

| Metric         | Score      |
| -------------- | ---------- |
| F1 Score       | 91.2%      |
| Precision      | 86.2%      |
| Recall         | 98.3%      |
| Inference Time | 20ms/image |

Per-class F1 scores: Open 97.7%, Short 87.7%, Mousebite 90.5%, Spur 85.3%, Spurious Copper 95.7%, Pin-hole 93.2%.

---

## Architecture

ResNet-18 backbone with a regularized classifier head:

- Pretrained ImageNet weights (transfer learning)
- Dropout (0.5) before the final linear layer
- Multi-label sigmoid output (6 classes)
- Trained with Adam (lr=0.001, weight decay=1e-4) and BCE loss

Training used random flips, rotation, and color jitter for augmentation.

---

## Quick Start

```bash
# Clone and install
git clone https://github.com/VARUN3WARE/pcb-defect-detection.git
cd pcb-defect-detection
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Run inspection on sample images
cd src
python demo_inspection.py
```

To inspect a single image:

```bash
cd src
python defect_inspector.py \
  --image ../DeepPCB/PCBData/group20085/20085_not/20085291_test.jpg \
  --output ../results/annotated_result.jpg \
  --json ../results/results.json
```

To train from scratch:

```bash
cd src
python train_model.py --epochs 15
```

---

## Python API

```python
from src.defect_inspector import DefectInspector

inspector = DefectInspector(model_path="models/best_model_regularized.pth")
results = inspector.detect_defects("path/to/pcb_image.jpg")

print(results["quality_status"])       # PASS or FAIL
print(results["num_defects_detected"]) # int

for d in results["detected_defects"]:
    print(f"{d['type']}: {d['confidence']:.2%}")
```

---

## Project Structure

```
pcb-defect-detection/
  src/
    defect_inspector.py      # Main inspection engine
    train_model.py            # Training script
    demo_inspection.py        # Generates annotated sample outputs
    run_inspection.py         # Batch processing
    dataset_utils.py          # Data loading helpers
  models/
    best_model_regularized.pth
    training_results.json
  results/
    inspection_samples/       # 10 annotated samples with JSON reports
  DeepPCB/                    # Dataset (1,500 image pairs, 640x640)
```

---

## Dataset

This project uses the [DeepPCB](https://github.com/tangsanli5201/DeepPCB) dataset which contains 1,500 aligned template/test image pairs at 640x640 resolution, annotated with bounding boxes for six defect classes.

---

## Limitations

- **Dataset scope**: Trained on a single PCB type; generalization to other board types (flex, HDI, RF) is untested.
- **Classification only**: The model predicts defect presence per image but does not localize defects with bounding boxes.
- **Scale**: 1,500 training images is small compared to production-grade systems.

---

## License

MIT

---

## References

- DeepPCB dataset: [github.com/tangsanli5201/DeepPCB](https://github.com/tangsanli5201/DeepPCB)
- He et al., "Deep Residual Learning for Image Recognition", CVPR 2016
- PyTorch: [pytorch.org](https://pytorch.org)
