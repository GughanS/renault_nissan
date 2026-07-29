import os
import torch
import torch.onnx
import numpy as np
import onnxruntime as ort
import unittest
import tempfile
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from wheeleye.models.detector import WheelEyeDetector
from wheeleye.models.classifier import WheelEyeClassifier
class TestONNXParity(unittest.TestCase):
    def test_pytorch_onnx_numerical_parity(self):
        """
        Verify that the exported ONNX model produces numerically identical
        outputs to the PyTorch model for the same input.
        """
        device = torch.device('cpu')
        
        # 1. Load PyTorch model
        pt_model = WheelEyeDetector(num_classes=4, num_anchors=9).to(device)
        pt_model.eval()
        
        weight_path = 'weights/best.pt'
        if os.path.exists(weight_path):
            checkpoint = torch.load(weight_path, map_location=device, weights_only=False)
            if 'model_state_dict' in checkpoint:
                pt_model.load_state_dict(checkpoint['model_state_dict'])
            else:
                pt_model.load_state_dict(checkpoint)
        
        # 2. Export ONNX model to temporary file
        with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as tmp:
            onnx_path = tmp.name
            
        dummy_input = torch.randn(1, 3, 512, 512, dtype=torch.float32)
        
        torch.onnx.export(
            pt_model, dummy_input, onnx_path,
            export_params=True, opset_version=17,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['cls_scores', 'bbox_preds'],
            dynamic_axes={'input': {0: 'batch_size'}, 'cls_scores': {0: 'batch_size'}, 'bbox_preds': {0: 'batch_size'}}
        )
            
        ort_session = ort.InferenceSession(onnx_path)
        
        # 3. PyTorch Inference
        with torch.no_grad():
            pt_cls, pt_reg = pt_model(dummy_input)
            
        # 4. ONNX Inference
        ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.numpy()}
        ort_outs = ort_session.run(['cls_scores', 'bbox_preds'], ort_inputs)
        
        onnx_cls = torch.from_numpy(ort_outs[0])
        onnx_reg = torch.from_numpy(ort_outs[1])
        
        # 5. Assert numerical parity
        cls_diff = torch.abs(pt_cls - onnx_cls).max().item()
        reg_diff = torch.abs(pt_reg - onnx_reg).max().item()
        
        # Clean up
        os.remove(onnx_path)
        
        self.assertLess(
            cls_diff, 1e-4, 
            f"ONNX PARITY MISMATCH: Classification outputs differ by {cls_diff}"
        )
        self.assertLess(
            reg_diff, 1e-4, 
            f"ONNX PARITY MISMATCH: Regression outputs differ by {reg_diff}"
        )

    def test_pytorch_onnx_numerical_parity_classifier(self):
        """
        Verify that the exported ONNX model produces numerically identical
        outputs to the PyTorch model for the same input.
        """
        device = torch.device('cpu')
        
        # 1. Load PyTorch model
        pt_model = WheelEyeClassifier().to(device)
        pt_model.eval()
        
        weight_path = 'weights/classify_best.pt'
        if os.path.exists(weight_path):
            checkpoint = torch.load(weight_path, map_location=device, weights_only=False)
            if 'model_state_dict' in checkpoint:
                pt_model.load_state_dict(checkpoint['model_state_dict'])
            else:
                pt_model.load_state_dict(checkpoint)
        
        # 2. Export ONNX model to temporary file
        with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as tmp:
            onnx_path = tmp.name
            
        dummy_input = torch.randn(1, 3, 224, 224, dtype=torch.float32)
        
        torch.onnx.export(
            pt_model, dummy_input, onnx_path,
            export_params=True, opset_version=17,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['material', 'tier', 'size'],
            dynamic_axes={'input': {0: 'batch_size'}, 'material': {0: 'batch_size'}, 'tier': {0: 'batch_size'}, 'size': {0: 'batch_size'}}
        )
            
        ort_session = ort.InferenceSession(onnx_path)
        
        # 3. PyTorch Inference
        with torch.no_grad():
            pt_mat, pt_tier, pt_size = pt_model(dummy_input)
            
        # 4. ONNX Inference
        ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.numpy()}
        ort_outs = ort_session.run(['material', 'tier', 'size'], ort_inputs)
        
        onnx_mat = torch.from_numpy(ort_outs[0])
        onnx_tier = torch.from_numpy(ort_outs[1])
        onnx_size = torch.from_numpy(ort_outs[2])
        
        # 5. Assert numerical parity
        mat_diff = torch.abs(pt_mat - onnx_mat).max().item()
        tier_diff = torch.abs(pt_tier - onnx_tier).max().item()
        size_diff = torch.abs(pt_size - onnx_size).max().item()
        
        # Clean up
        os.remove(onnx_path)
        
        self.assertLess(
            mat_diff, 1e-4, 
            f"ONNX PARITY MISMATCH: Material outputs differ by {mat_diff}"
        )
        self.assertLess(
            tier_diff, 1e-4, 
            f"ONNX PARITY MISMATCH: Tier outputs differ by {tier_diff}"
        )
        self.assertLess(
            size_diff, 1e-4, 
            f"ONNX PARITY MISMATCH: Size outputs differ by {size_diff}"
        )

if __name__ == '__main__':
    unittest.main()
