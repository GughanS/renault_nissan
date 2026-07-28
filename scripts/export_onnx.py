import argparse
import os
import sys
import codecs

if sys.stdout.encoding != 'utf-8':
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())

import torch
import torch.onnx
import onnx
from torch.ao.quantization import quantize_dynamic

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from wheeleye.models.detector import WheelEyeDetector
from wheeleye.models.classifier import WheelEyeClassifier


def prepare_for_export(model):
    """Strip training-specific state before ONNX export.
    
    - Sets eval mode (fixes BN running stats)
    - Removes gradient hooks / autograd metadata
    """
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def export_model(model, dummy_input, save_path, opset_version=17,
                 simplify=False, fp16=False):
    """Export a PyTorch model to ONNX with optional simplification and FP16."""
    print(f"Exporting model to {save_path} (opset {opset_version})...")
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)

    if fp16:
        print("  → Casting to FP16 for half-precision export...")
        model = model.half()
        dummy_input = dummy_input.half()

    # Determine output names based on model type
    if isinstance(model, WheelEyeDetector):
        output_names = ['cls_scores', 'bbox_preds']
    else:
        output_names = ['material', 'tier', 'size']

    # Export
    torch.onnx.export(
        model,
        dummy_input,
        save_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['input'],
        output_names=output_names,
        dynamic_axes={
            'input': {0: 'batch_size'},
            **{name: {0: 'batch_size'} for name in output_names}
        }
    )
    print(f"  ✓ ONNX export successful: {save_path}")

    # Validate the exported model
    print("  → Validating ONNX graph...")
    onnx_model = onnx.load(save_path)
    onnx.checker.check_model(onnx_model)
    print("  ✓ ONNX validation passed")

    # Optional simplification (critical for TensorRT on Jetson)
    if simplify:
        try:
            import onnxsim
            print("  → Simplifying ONNX graph (constant folding, redundant op removal)...")
            simplified_model, ok = onnxsim.simplify(onnx_model)
            if ok:
                onnx.save(simplified_model, save_path)
                print("  ✓ Simplification successful")
            else:
                print("  ⚠ Simplification returned not-ok, keeping original")
        except ImportError:
            print("  ⚠ onnx-simplifier not installed — skipping simplification. "
                  "Install with: pip install onnxsim")

    # Print model size
    file_size_mb = os.path.getsize(save_path) / (1024 * 1024)
    print(f"  → Model size: {file_size_mb:.2f} MB")


def main():
    parser = argparse.ArgumentParser(
        description="Export WheelEye PyTorch models to ONNX (optimised for NVIDIA Jetson)")
    parser.add_argument("--detector-weights", type=str, default="weights/best.pt")
    parser.add_argument("--classifier-weights", type=str, default="weights/classify_best.pt")
    parser.add_argument("--out-dir", type=str, default="exports")
    parser.add_argument("--quantize", action="store_true",
                        help="Apply dynamic quantization before export")
    parser.add_argument("--simplify", action="store_true",
                        help="Run onnx-simplifier for Jetson TensorRT ingestion")
    parser.add_argument("--fp16", action="store_true",
                        help="Export half-precision (FP16) graph")
    parser.add_argument("--opset", type=int, default=17,
                        help="ONNX opset version (default: 17)")

    args = parser.parse_args()
    device = torch.device('cpu')  # Exporting is usually done on CPU

    # --- DETECTOR ---
    print("\n--- Processing Detector ---")
    detector = WheelEyeDetector(num_classes=4, num_anchors=9).to(device)
    if os.path.exists(args.detector_weights):
        ckpt = torch.load(args.detector_weights, map_location=device, weights_only=True)
        detector.load_state_dict(ckpt.get('model_state_dict', ckpt))
        print(f"Loaded detector weights from {args.detector_weights}")
    else:
        print("WARNING: Detector weights not found. Exporting randomly initialized model.")

    detector = prepare_for_export(detector)

    if args.quantize:
        print("Applying dynamic quantization to Detector...")
        detector = quantize_dynamic(detector, {torch.nn.Linear, torch.nn.Conv2d},
                                    dtype=torch.qint8)

    dummy_input_det = torch.randn(1, 3, 512, 512, device=device)
    det_save_path = os.path.join(args.out_dir, "wheeleye_detector.onnx")
    export_model(detector, dummy_input_det, det_save_path,
                 opset_version=args.opset, simplify=args.simplify, fp16=args.fp16)

    # --- CLASSIFIER ---
    print("\n--- Processing Classifier ---")
    classifier = WheelEyeClassifier().to(device)
    if os.path.exists(args.classifier_weights):
        ckpt = torch.load(args.classifier_weights, map_location=device, weights_only=True)
        classifier.load_state_dict(ckpt.get('model_state_dict', ckpt))
        print(f"Loaded classifier weights from {args.classifier_weights}")
    else:
        print("WARNING: Classifier weights not found. Exporting randomly initialized model.")

    classifier = prepare_for_export(classifier)

    if args.quantize:
        print("Applying dynamic quantization to Classifier...")
        classifier = quantize_dynamic(classifier, {torch.nn.Linear, torch.nn.Conv2d},
                                      dtype=torch.qint8)

    dummy_input_cls = torch.randn(1, 3, 224, 224, device=device)
    cls_save_path = os.path.join(args.out_dir, "wheeleye_classifier.onnx")
    export_model(classifier, dummy_input_cls, cls_save_path,
                 opset_version=args.opset, simplify=args.simplify, fp16=args.fp16)

    print("\n✓ Export process completed.")


if __name__ == "__main__":
    main()
