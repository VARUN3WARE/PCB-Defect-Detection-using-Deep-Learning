"""
Automated Quality Inspection System for PCB Manufacturing
Detects and classifies defects with bounding boxes and confidence scores
"""
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
import json
from pathlib import Path

# Defect class mapping
DEFECT_CLASSES = {
    1: 'open',
    2: 'short',
    3: 'mousebite',
    4: 'spur',
    5: 'copper',
    6: 'pin-hole'
}

DEFECT_COLORS = {
    1: (255, 50, 50),     # open - RED
    2: (50, 255, 50),     # short - GREEN  
    3: (50, 150, 255),    # mousebite - BLUE
    4: (255, 220, 50),    # spur - YELLOW
    5: (255, 50, 255),    # copper - MAGENTA
    6: (50, 255, 255)     # pin-hole - CYAN
}

class PCBDefectClassifier(nn.Module):
    """ResNet-18 based multi-label classifier for PCB defects"""
    def __init__(self, num_classes=6):
        super(PCBDefectClassifier, self).__init__()
        # Load pretrained ResNet-18
        self.resnet = models.resnet18(pretrained=True)
        
        # Replace final FC layer for multi-label classification
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, num_classes),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.resnet(x)


class DefectInspector:
    """Main inspection class for PCB defect detection"""
    
    def __init__(self, model_path, device='cpu', threshold=0.5):
        """
        Initialize the inspector
        
        Args:
            model_path: Path to trained model weights
            device: 'cpu' or 'cuda'
            threshold: Detection threshold for classification
        """
        self.device = device
        self.threshold = threshold
        
        # Load model
        self.model = PCBDefectClassifier()
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.to(device)
        self.model.eval()
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def detect_defects(self, image_path, annotation_path=None):
        """
        Detect and classify defects in a PCB image
        
        Args:
            image_path: Path to input image
            annotation_path: Optional path to annotation file for ground truth
            
        Returns:
            dict: Detection results with defects, confidence scores, and locations
        """
        # Load image
        img_pil = Image.open(image_path).convert('RGB')
        img_cv = cv2.imread(str(image_path))
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        original_size = img_cv.shape[:2]
        
        # Preprocess for model
        img_tensor = self.transform(img_pil).unsqueeze(0).to(self.device)
        
        # Run inference
        with torch.no_grad():
            predictions = self.model(img_tensor).cpu().numpy()[0]
        
        # Classify defects
        detected_defects = []
        for class_idx, confidence in enumerate(predictions):
            if confidence > self.threshold:
                defect_type = DEFECT_CLASSES[class_idx + 1]
                detected_defects.append({
                    'type': defect_type,
                    'class_id': class_idx + 1,
                    'confidence': float(confidence)
                })
        
        # Load bounding boxes if annotation provided
        bounding_boxes = []
        if annotation_path and Path(annotation_path).exists():
            bounding_boxes = self._parse_annotation(annotation_path)
        
        # Calculate defect centers and severity
        defect_locations = []
        for bbox in bounding_boxes:
            center_x = (bbox['x1'] + bbox['x2']) // 2
            center_y = (bbox['y1'] + bbox['y2']) // 2
            area = (bbox['x2'] - bbox['x1']) * (bbox['y2'] - bbox['y1'])
            
            # Severity based on defect size
            severity = self._assess_severity(area, original_size)
            
            defect_locations.append({
                'defect_type': DEFECT_CLASSES.get(bbox['type'], 'unknown'),
                'center': (center_x, center_y),
                'bounding_box': (bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']),
                'area': area,
                'severity': severity
            })
        
        # Overall quality assessment
        num_defects = len(detected_defects)
        quality_status = 'PASS' if num_defects == 0 else 'FAIL'
        overall_severity = 'CRITICAL' if num_defects >= 4 else 'MODERATE' if num_defects >= 2 else 'MINOR'
        
        return {
            'image_path': str(image_path),
            'image_size': original_size,
            'quality_status': quality_status,
            'overall_severity': overall_severity if quality_status == 'FAIL' else 'NONE',
            'num_defects_detected': num_defects,
            'detected_defects': detected_defects,
            'defect_locations': defect_locations,
            'all_confidence_scores': {
                DEFECT_CLASSES[i+1]: float(predictions[i]) 
                for i in range(len(predictions))
            }
        }
    
    def _parse_annotation(self, anno_file):
        """Parse annotation file with space-separated format"""
        defects = []
        try:
            with open(anno_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()  # Space-separated, not comma!
                    if len(parts) >= 5:
                        defects.append({
                            'x1': int(parts[0]),
                            'y1': int(parts[1]),
                            'x2': int(parts[2]),
                            'y2': int(parts[3]),
                            'type': int(parts[4])
                        })
        except Exception as e:
            print(f"Warning: Error parsing annotation {anno_file}: {e}")
        return defects
    
    def _assess_severity(self, defect_area, image_size):
        """Assess defect severity based on size"""
        image_area = image_size[0] * image_size[1]
        ratio = defect_area / image_area
        
        if ratio > 0.05:
            return 'HIGH'
        elif ratio > 0.02:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def visualize_results(self, image_path, results, output_path=None):
        """
        Visualize detection results with bounding boxes
        
        Args:
            image_path: Path to input image
            results: Detection results from detect_defects()
            output_path: Path to save annotated image (optional)
            
        Returns:
            np.array: Annotated image
        """
        # Load image
        img = cv2.imread(str(image_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Draw bounding boxes
        for location in results['defect_locations']:
            bbox = location['bounding_box']
            defect_type = location['defect_type']
            severity = location['severity']
            
            # Get color for this defect type
            color = (255, 0, 0)  # default red
            for class_id, name in DEFECT_CLASSES.items():
                if name == defect_type:
                    color = DEFECT_COLORS[class_id]
                    break
            
            # Draw rectangle
            cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 3)
            
            # Add label
            label = f"{defect_type} ({severity})"
            cv2.putText(img, label, (bbox[0], max(bbox[1]-10, 20)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Draw center point
            center = location['center']
            cv2.circle(img, center, 5, color, -1)
        
        # Add header with overall status
        status_text = f"Status: {results['quality_status']} | Defects: {results['num_defects_detected']}"
        if results['quality_status'] == 'FAIL':
            status_text += f" | Severity: {results['overall_severity']}"
        
        cv2.rectangle(img, (0, 0), (img.shape[1], 40), (0, 0, 0), -1)
        cv2.putText(img, status_text, (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Save if output path provided
        if output_path:
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(output_path), img_bgr)
        
        return img
    
    def generate_report(self, results):
        """
        Generate detailed inspection report
        
        Args:
            results: Detection results from detect_defects()
            
        Returns:
            str: Formatted inspection report
        """
        report = f"""
{'='*70}
PCB QUALITY INSPECTION REPORT
{'='*70}
Image: {Path(results['image_path']).name}
Image Size: {results['image_size'][1]} x {results['image_size'][0]} pixels

INSPECTION RESULT: {results['quality_status']}
Overall Severity: {results['overall_severity']}
Defects Detected: {results['num_defects_detected']}

DETECTED DEFECTS:
"""
        if results['detected_defects']:
            for i, defect in enumerate(results['detected_defects'], 1):
                report += f"  {i}. {defect['type'].upper()} (Confidence: {defect['confidence']*100:.1f}%)\n"
        else:
            report += "  None - PCB passes quality check\n"
        
        if results['defect_locations']:
            report += f"\nDEFECT LOCATIONS:\n"
            for i, loc in enumerate(results['defect_locations'], 1):
                report += f"  {i}. {loc['defect_type'].upper()}\n"
                report += f"     Center: ({loc['center'][0]}, {loc['center'][1]}) pixels\n"
                report += f"     Bounding Box: {loc['bounding_box']}\n"
                report += f"     Area: {loc['area']} px² | Severity: {loc['severity']}\n"
        
        report += f"\nCONFIDENCE SCORES (All Classes):\n"
        for defect_type, confidence in results['all_confidence_scores'].items():
            marker = "✓" if confidence > self.threshold else " "
            report += f"  {marker} {defect_type:12s}: {confidence*100:5.1f}%\n"
        
        report += f"{'='*70}\n"
        return report


def main():
    """Example usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='PCB Defect Inspector')
    parser.add_argument('--image', type=str, required=True, help='Path to input PCB image')
    parser.add_argument('--model', type=str, default='../models/best_model_regularized.pth', 
                       help='Path to trained model')
    parser.add_argument('--annotation', type=str, help='Path to annotation file (optional)')
    parser.add_argument('--output', type=str, help='Path to save annotated image')
    parser.add_argument('--threshold', type=float, default=0.5, help='Detection threshold')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'])
    parser.add_argument('--json', type=str, help='Save results as JSON')
    
    args = parser.parse_args()
    
    # Initialize inspector
    inspector = DefectInspector(args.model, device=args.device, threshold=args.threshold)
    
    # Run detection
    print("Running defect detection...")
    results = inspector.detect_defects(args.image, args.annotation)
    
    # Generate report
    report = inspector.generate_report(results)
    print(report)
    
    # Visualize if output path provided
    if args.output:
        inspector.visualize_results(args.image, results, args.output)
        print(f"Annotated image saved to: {args.output}")
    
    # Save JSON if requested
    if args.json:
        with open(args.json, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {args.json}")


if __name__ == '__main__':
    main()
