import argparse
import os
import sys
import time
import torch
import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    ort = None

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from wheeleye.models.detector import WheelEyeDetector
from wheeleye.models.classifier import WheelEyeClassifier

def benchmark_pytorch(model, dummy_input, iterations=100, warmup=10):
    print("Warming up PyTorch model...")
    with torch.no_grad():
        for _ in range(warmup):
            model(dummy_input)
            
    print(f"Benchmarking PyTorch model for {iterations} iterations...")
    start_time = time.perf_counter()
    with torch.no_grad():
        for _ in range(iterations):
            model(dummy_input)
    end_time = time.perf_counter()
    
    total_time = end_time - start_time
    avg_time_ms = (total_time / iterations) * 1000
    fps = iterations / total_time
    
    return avg_time_ms, fps

def benchmark_onnx(onnx_path, dummy_input_np, iterations=100, warmup=10):
    if ort is None:
        print("onnxruntime is not installed. Skipping ONNX benchmark.")
        return None, None
        
    print(f"Loading ONNX model from {onnx_path}...")
    session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    
    print("Warming up ONNX model...")
    for _ in range(warmup):
        session.run(None, {input_name: dummy_input_np})
        
    print(f"Benchmarking ONNX model for {iterations} iterations...")
    start_time = time.perf_counter()
    for _ in range(iterations):
        session.run(None, {input_name: dummy_input_np})
    end_time = time.perf_counter()
    
    total_time = end_time - start_time
    avg_time_ms = (total_time / iterations) * 1000
    fps = iterations / total_time
    
    return avg_time_ms, fps

def main():
    parser = argparse.ArgumentParser(description="Benchmark WheelEye Models (PyTorch vs ONNX)")
    parser.add_argument("--detector-weights", type=str, default="weights/best.pt")
    parser.add_argument("--classifier-weights", type=str, default="weights/classify_best.pt")
    parser.add_argument("--export-dir", type=str, default="exports")
    parser.add_argument("--iterations", type=int, default=100)
    
    args = parser.parse_args()
    device = torch.device('cpu') # Benchmark on CPU for edge-deployment simulation
    
    # --- DETECTOR BENCHMARK ---
    print("\n" + "="*50)
    print("DETECTOR BENCHMARK")
    print("="*50)
    
    detector = WheelEyeDetector(num_classes=4, num_anchors=9).to(device)
    if os.path.exists(args.detector_weights):
        ckpt = torch.load(args.detector_weights, map_location=device, weights_only=True)
        detector.load_state_dict(ckpt.get('model_state_dict', ckpt))
    detector.eval()
    
    dummy_input_det = torch.randn(1, 3, 512, 512, device=device)
    dummy_input_det_np = dummy_input_det.numpy()
    
    det_pt_ms, det_pt_fps = benchmark_pytorch(detector, dummy_input_det, iterations=args.iterations)
    print(f"[PyTorch] Detector - Latency: {det_pt_ms:.2f} ms | FPS: {det_pt_fps:.2f}")
    
    det_onnx_path = os.path.join(args.export_dir, "wheeleye_detector.onnx")
    if os.path.exists(det_onnx_path):
        det_onnx_ms, det_onnx_fps = benchmark_onnx(det_onnx_path, dummy_input_det_np, iterations=args.iterations)
        if det_onnx_ms:
            print(f"[ONNX] Detector - Latency: {det_onnx_ms:.2f} ms | FPS: {det_onnx_fps:.2f}")
            speedup = det_pt_ms / det_onnx_ms
            print(f"--> ONNX is {speedup:.2f}x faster")
    else:
        print(f"ONNX model not found at {det_onnx_path}")
        
    # --- CLASSIFIER BENCHMARK ---
    print("\n" + "="*50)
    print("CLASSIFIER BENCHMARK")
    print("="*50)
    
    classifier = WheelEyeClassifier().to(device)
    if os.path.exists(args.classifier_weights):
        ckpt = torch.load(args.classifier_weights, map_location=device, weights_only=True)
        classifier.load_state_dict(ckpt.get('model_state_dict', ckpt))
    classifier.eval()
    
    dummy_input_cls = torch.randn(1, 3, 224, 224, device=device)
    dummy_input_cls_np = dummy_input_cls.numpy()
    
    cls_pt_ms, cls_pt_fps = benchmark_pytorch(classifier, dummy_input_cls, iterations=args.iterations)
    print(f"[PyTorch] Classifier - Latency: {cls_pt_ms:.2f} ms | FPS: {cls_pt_fps:.2f}")
    
    cls_onnx_path = os.path.join(args.export_dir, "wheeleye_classifier.onnx")
    if os.path.exists(cls_onnx_path):
        cls_onnx_ms, cls_onnx_fps = benchmark_onnx(cls_onnx_path, dummy_input_cls_np, iterations=args.iterations)
        if cls_onnx_ms:
            print(f"[ONNX] Classifier - Latency: {cls_onnx_ms:.2f} ms | FPS: {cls_onnx_fps:.2f}")
            speedup = cls_pt_ms / cls_onnx_ms
            print(f"--> ONNX is {speedup:.2f}x faster")
    else:
        print(f"ONNX model not found at {cls_onnx_path}")

if __name__ == "__main__":
    main()
