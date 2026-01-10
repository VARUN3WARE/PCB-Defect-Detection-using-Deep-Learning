# PCB Quality Inspection Samples

## Automated Defect Detection Results

This folder contains sample outputs from the automated PCB quality inspection system.

### System Capabilities
- ✅ Multi-label defect classification (6 types)
- ✅ Bounding box localization
- ✅ Confidence scoring
- ✅ Severity assessment
- ✅ (x, y) pixel coordinate output
- ✅ Quality control reports

### Defect Types Detected
1. **Open Circuit** - Electrical discontinuity in traces
2. **Short Circuit** - Unintended electrical connections
3. **Mousebite** - Notches in PCB edges
4. **Spur** - Unwanted copper protrusions
5. **Spurious Copper** - Excess copper residue
6. **Pin-hole** - Holes in copper layers

### Inspection Results Summary
- Total Samples: 10
- Pass Rate: 0.0%
- Failed: 10 samples

### Files
- `sample_XX_*.jpg` - Annotated images with bounding boxes
- `sample_XX_results.json` - Detailed JSON results with coordinates
- `inspection_summary.json` - Overall inspection statistics

### Model Performance
- F1 Score: 90.98%
- Precision: 85.85%
- Recall: 96.76%

### Usage
This system demonstrates automated quality control for PCB manufacturing:
1. Analyzes input images
2. Detects and localizes defects (bounding boxes)
3. Classifies defect types with confidence scores
4. Outputs (x,y) coordinates and severity assessment
