import argparse
import os
import sys
import codecs

if sys.stdout.encoding != 'utf-8':
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())

import torch
import torch.onnx
from torch.ao.quantization import quantize_dynamic

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from wheeleye.models.detector import WheelEyeDetector
from wheeleye.models.classifier import WheelEyeClassifier

def export_model(model, dummy_input, save_path, opset_version=12):
    print(f"Exporting model to {save_path}...")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Export the model
    torch.onnx.export(
        model,
        dummy_input,
        save_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output1', 'output2'] if isinstance(model, WheelEyeDetector) else ['output1', 'output2', 'output3'],
        dynamic_axes={'input': {0: 'batch_size'},
                      'output1': {0: 'batch_size'},
                      'output2': {0: 'batch_size'}}
    )
    print(f"ONNX export successful: {save_path}")

def main():
    parser = argparse.ArgumentParser(description="Export WheelEye PyTorch models to ONNX")
    parser.add_argument("--detector-weights", type=str, default="weights/best.pt")
    parser.add_argument("--classifier-weights", type=str, default="weights/classify_best.pt")
    parser.add_argument("--out-dir", type=str, default="exports")
    parser.add_argument("--quantize", action="store_true", help="Apply dynamic quantization before export")
    
    args = parser.parse_args()
    
    device = torch.device('cpu') # Exporting is usually done on CPU
    
    # --- DETECTOR ---
    print("\n--- Processing Detector ---")
    detector = WheelEyeDetector(num_classes=4, num_anchors=9).to(device)
    if os.path.exists(args.detector_weights):
        ckpt = torch.load(args.detector_weights, map_location=device, weights_only=True)
        detector.load_state_dict(ckpt.get('model_state_dict', ckpt))
        print(f"Loaded detector weights from {args.detector_weights}")
    else:
        print("WARNING: Detector weights not found. Exporting randomly initialized model.")
        
    detector.eval()
    
    if args.quantize:
        print("Applying dynamic quantization to Detector...")
        detector = quantize_dynamic(detector, {torch.nn.Linear, torch.nn.Conv2d}, dtype=torch.qint8)
        
    dummy_input_det = torch.randn(1, 3, 512, 512, device=device)
    det_save_path = os.path.join(args.out_dir, "wheeleye_detector.onnx")
    export_model(detector, dummy_input_det, det_save_path)
    
    # --- CLASSIFIER ---
    print("\n--- Processing Classifier ---")
    classifier = WheelEyeClassifier().to(device)
    if os.path.exists(args.classifier_weights):
        ckpt = torch.load(args.classifier_weights, map_location=device, weights_only=True)
        classifier.load_state_dict(ckpt.get('model_state_dict', ckpt))
        print(f"Loaded classifier weights from {args.classifier_weights}")
    else:
        print("WARNING: Classifier weights not found. Exporting randomly initialized model.")
        
    classifier.eval()
    
    if args.quantize:
        print("Applying dynamic quantization to Classifier...")
        classifier = quantize_dynamic(classifier, {torch.nn.Linear, torch.nn.Conv2d}, dtype=torch.qint8)
        
    dummy_input_cls = torch.randn(1, 3, 224, 224, device=device)
    cls_save_path = os.path.join(args.out_dir, "wheeleye_classifier.onnx")
    export_model(classifier, dummy_input_cls, cls_save_path)
    
    print("\nExport process completed.")

if __name__ == "__main__":
    main()
