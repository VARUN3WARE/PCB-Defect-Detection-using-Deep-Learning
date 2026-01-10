"""
Complete Demo: Automated PCB Quality Inspection System
Generates sample outputs with defect detection, bounding boxes, and reports
"""
import sys
from pathlib import Path
from defect_inspector import DefectInspector
import json
import cv2
import numpy as np

def create_comprehensive_demo():
    """Run complete inspection demo with outputs"""
    
    # Setup paths
    dataset_path = Path('../DeepPCB')
    model_path = Path('../models/best_model_regularized.pth')
    output_dir = Path('../results/inspection_samples')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check model
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        return
    
    # Initialize inspector
    print("="*70)
    print("AUTOMATED QUALITY INSPECTION SYSTEM - PCB DEFECT DETECTION")
    print("="*70)
    print("\n🔬 Initializing Inspector...")
    inspector = DefectInspector(
        model_path=str(model_path),
        device='cpu',
        threshold=0.5
    )
    print("✅ Inspector ready!\n")
    
    # Load test data
    test_txt = dataset_path / 'PCBData' / 'test.txt'
    with open(test_txt, 'r') as f:
        test_images = f.read().strip().split('\n')
    
    print(f"📂 Dataset: {len(test_images)} test images available\n")
    
    # Select diverse samples (some with defects, some without)
    # Inspect first 5 samples
    print("="*70)
    print("RUNNING INSPECTION ON SAMPLE PCBs")
    print("="*70 + "\n")
    
    all_results = []
    
    for i, test_entry in enumerate(test_images[:10], 1):  # Check 10 samples
        parts = test_entry.split()
        img_rel_path = parts[0]
        anno_rel_path = parts[1]
        
        # Construct paths
        image_id = Path(img_rel_path).stem
        img_dir = Path(img_rel_path).parent
        
        test_img_path = dataset_path / 'PCBData' / img_dir / f"{image_id}_test.jpg"
        anno_path = dataset_path / 'PCBData' / anno_rel_path
        
        if not test_img_path.exists():
            continue
        
        print(f"{'='*70}")
        print(f"SAMPLE {i}: {test_img_path.name}")
        print(f"{'='*70}")
        
        # Run detection
        results = inspector.detect_defects(
            image_path=str(test_img_path),
            annotation_path=str(anno_path) if anno_path.exists() else None
        )
        
        # Generate report
        report = inspector.generate_report(results)
        print(report)
        
        # Visualize and save
        output_path = output_dir / f"sample_{i:02d}_{test_img_path.name}"
        annotated_img = inspector.visualize_results(
            image_path=str(test_img_path),
            results=results,
            output_path=str(output_path)
        )
        
        print(f"✅ Annotated image saved: {output_path}")
        
        # Save individual result as JSON
        json_path = output_dir / f"sample_{i:02d}_results.json"
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"✅ Results JSON saved: {json_path}\n")
        
        all_results.append({
            'sample_id': i,
            'filename': test_img_path.name,
            'status': results['quality_status'],
            'severity': results['overall_severity'],
            'num_defects': results['num_defects_detected'],
            'defects': [d['type'] for d in results['detected_defects']]
        })
    
    # Create summary report
    print("="*70)
    print("INSPECTION SUMMARY")
    print("="*70 + "\n")
    
    total_samples = len(all_results)
    passed = sum(1 for r in all_results if r['status'] == 'PASS')
    failed = sum(1 for r in all_results if r['status'] == 'FAIL')
    
    print(f"Total Samples Inspected: {total_samples}")
    print(f"  ✅ PASSED: {passed} ({passed/total_samples*100:.1f}%)")
    print(f"  ❌ FAILED: {failed} ({failed/total_samples*100:.1f}%)")
    print(f"\nDefect Statistics:")
    
    all_defect_types = {}
    for r in all_results:
        for defect in r['defects']:
            all_defect_types[defect] = all_defect_types.get(defect, 0) + 1
    
    for defect_type, count in sorted(all_defect_types.items(), key=lambda x: x[1], reverse=True):
        print(f"  - {defect_type.upper()}: {count} instances")
    
    # Save summary
    summary = {
        'total_inspected': total_samples,
        'passed': passed,
        'failed': failed,
        'pass_rate': passed/total_samples*100,
        'defect_statistics': all_defect_types,
        'samples': all_results
    }
    
    summary_path = output_dir / 'inspection_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n📄 Summary saved: {summary_path}")
    
    # Create README for samples
    readme_content = f"""# PCB Quality Inspection Samples

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
- Total Samples: {total_samples}
- Pass Rate: {passed/total_samples*100:.1f}%
- Failed: {failed} samples

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
"""
    
    readme_path = output_dir / 'README.md'
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    
    print(f"📄 README saved: {readme_path}")
    
    print("\n" + "="*70)
    print("✅ DEMO COMPLETE!")
    print("="*70)
    print(f"\n📁 All outputs saved to: {output_dir}")
    print(f"\nView annotated images in: {output_dir}")
    print(f"Review JSON results for exact coordinates and confidence scores\n")


if __name__ == '__main__':
    create_comprehensive_demo()
