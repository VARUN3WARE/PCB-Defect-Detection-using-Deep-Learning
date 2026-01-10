"""
Simple demo script to run PCB defect inspection
"""
import sys
from pathlib import Path
from defect_inspector import DefectInspector

def run_inspection_demo():
    """Run inspection on sample PCB images from dataset"""
    
    # Paths
    dataset_path = Path('../DeepPCB')
    model_path = Path('../models/best_model_regularized.pth')
    
    # Check if model exists
    if not model_path.exists():
        print(f"❌ Model not found at: {model_path}")
        print("Please train the model first using the notebook.")
        return
    
    # Initialize inspector
    print("🔬 Initializing PCB Defect Inspector...")
    inspector = DefectInspector(
        model_path=str(model_path),
        device='cpu',
        threshold=0.5
    )
    print("✅ Inspector ready!\n")
    
    # Load test file list
    test_txt = dataset_path / 'PCBData' / 'test.txt'
    if not test_txt.exists():
        print(f"❌ Test file not found: {test_txt}")
        return
    
    with open(test_txt, 'r') as f:
        test_images = f.read().strip().split('\n')
    
    # Inspect first 3 samples
    print("="*70)
    print("RUNNING INSPECTION ON SAMPLE PCBs")
    print("="*70 + "\n")
    
    for i, test_entry in enumerate(test_images[:3], 1):
        parts = test_entry.split()
        img_rel_path = parts[0]
        anno_rel_path = parts[1]
        
        # Construct paths (add _test suffix)
        image_id = Path(img_rel_path).stem
        img_dir = Path(img_rel_path).parent
        
        test_img_path = dataset_path / 'PCBData' / img_dir / f"{image_id}_test.jpg"
        anno_path = dataset_path / 'PCBData' / anno_rel_path
        
        if not test_img_path.exists():
            print(f"⚠️ Image not found: {test_img_path}")
            continue
        
        print(f"📋 SAMPLE {i}: {test_img_path.name}")
        print("-" * 70)
        
        # Run detection
        results = inspector.detect_defects(
            image_path=str(test_img_path),
            annotation_path=str(anno_path) if anno_path.exists() else None
        )
        
        # Generate report
        report = inspector.generate_report(results)
        print(report)
        
        # Save annotated image
        output_dir = Path('../results/inspections')
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"inspection_{i}_{test_img_path.name}"
        
        inspector.visualize_results(
            image_path=str(test_img_path),
            results=results,
            output_path=str(output_path)
        )
        print(f"📁 Annotated image saved: {output_path}\n")


if __name__ == '__main__':
    run_inspection_demo()
