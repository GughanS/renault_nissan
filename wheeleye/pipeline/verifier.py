import os
import torch
import numpy as np
import torchvision.transforms as T
import torchvision.ops as ops
from PIL import Image
from wheeleye.models.detector import WheelEyeDetector
from wheeleye.models.classifier import WheelEyeClassifier
from wheeleye.utils.anchors import generate_anchors, decode_boxes
from wheeleye.preprocessing import get_inference_transforms

try:
    import onnxruntime as ort
    HAS_ORT = True
except ImportError:
    ort = None
    HAS_ORT = False


class WheelEyeVerifier:
    """
    End-to-end wheel assembly verification pipeline.

    Supports two inference backends:
      - **PyTorch** (.pt weights) — full accuracy, slower
      - **ONNX Runtime** (.onnx weights) — up to 3× faster, meets 1.5s Takt time

    The backend is auto-selected based on the weight file extension.
    """

    def __init__(self, detector_weights_path=None, classifier_weights_path=None,
                 device='cpu'):
        self.device = torch.device(device)

        # Determine backend for each model
        self._det_onnx = (detector_weights_path or '').endswith('.onnx')
        self._cls_onnx = (classifier_weights_path or '').endswith('.onnx')

        # --- Detector ---
        if self._det_onnx and HAS_ORT and detector_weights_path and os.path.exists(detector_weights_path):
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] \
                if device != 'cpu' else ['CPUExecutionProvider']
            self.det_session = ort.InferenceSession(detector_weights_path,
                                                    providers=providers)
            self.detector = None
            print(f"[Verifier] Detector: ONNX Runtime ({detector_weights_path})")
        else:
            self.det_session = None
            self.detector = WheelEyeDetector(num_classes=4, num_anchors=9).to(self.device)
            if detector_weights_path and os.path.exists(detector_weights_path):
                ckpt = torch.load(detector_weights_path, map_location=self.device,
                                  weights_only=True)
                self.detector.load_state_dict(ckpt.get('model_state_dict', ckpt))
            self.detector.eval()

        # --- Classifier ---
        if self._cls_onnx and HAS_ORT and classifier_weights_path and os.path.exists(classifier_weights_path):
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] \
                if device != 'cpu' else ['CPUExecutionProvider']
            self.cls_session = ort.InferenceSession(classifier_weights_path,
                                                    providers=providers)
            self.classifier = None
            print(f"[Verifier] Classifier: ONNX Runtime ({classifier_weights_path})")
        else:
            self.cls_session = None
            self.classifier = WheelEyeClassifier().to(self.device)
            if classifier_weights_path and os.path.exists(classifier_weights_path):
                ckpt = torch.load(classifier_weights_path, map_location=self.device,
                                  weights_only=True)
                self.classifier.load_state_dict(ckpt.get('model_state_dict', ckpt))
            self.classifier.eval()

        # Detector anchors
        self.image_size = (512, 512)
        strides = [8, 16, 32]
        base_sizes = [32, 64, 128]
        scales = [1, 2**(1/3), 2**(2/3)]
        aspect_ratios = [0.5, 1.0, 2.0]
        self.anchors = generate_anchors(self.image_size, strides, base_sizes,
                                        scales, aspect_ratios).to(self.device)

        # Transforms
        self.det_transform = get_inference_transforms(self.image_size)
        self.cls_transform = get_inference_transforms((224, 224))

        # Class mappings
        self.det_classes = ['wheel', 'fastener', 'valve_stem', 'center_cap']
        self.cls_material = ['Steel', 'Alloy']
        self.cls_tier = ['Standard', 'Premium', 'Luxury']
        self.cls_size = ['17_inch', '18_inch', '19_inch']

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------

    def _run_detector(self, input_tensor):
        """Run detector inference via PyTorch or ONNX Runtime."""
        if self.det_session is not None:
            input_np = input_tensor.cpu().numpy()
            input_name = self.det_session.get_inputs()[0].name
            outputs = self.det_session.run(['cls_scores', 'bbox_preds'], {input_name: input_np})
            cls_scores = torch.tensor(outputs[0], device=self.device)
            bbox_preds = torch.tensor(outputs[1], device=self.device)
            return cls_scores, bbox_preds
        else:
            with torch.no_grad():
                return self.detector(input_tensor)

    def _run_classifier(self, input_tensor):
        """Run classifier inference via PyTorch or ONNX Runtime."""
        if self.cls_session is not None:
            input_np = input_tensor.cpu().numpy()
            input_name = self.cls_session.get_inputs()[0].name
            outputs = self.cls_session.run(['material', 'tier', 'size'], {input_name: input_np})
            mat_out = torch.tensor(outputs[0], device=self.device)
            tier_out = torch.tensor(outputs[1], device=self.device)
            size_out = torch.tensor(outputs[2], device=self.device)
            return mat_out, tier_out, size_out
        else:
            with torch.no_grad():
                return self.classifier(input_tensor)

    # ------------------------------------------------------------------
    # Post-processing
    # ------------------------------------------------------------------

    def _postprocess_detections(self, cls_scores, bbox_preds,
                                conf_thresh=0.45, iou_thresh=0.4):
        """Decode bounding boxes and apply NMS."""
        # cls_scores: (B, N, C), bbox_preds: (B, N, 4)
        cls_scores = torch.sigmoid(cls_scores[0])  # single image
        bbox_preds = decode_boxes(bbox_preds[0], self.anchors)

        # Convert cx,cy,w,h to x1,y1,x2,y2
        boxes_x1 = bbox_preds[:, 0] - bbox_preds[:, 2] / 2
        boxes_y1 = bbox_preds[:, 1] - bbox_preds[:, 3] / 2
        boxes_x2 = bbox_preds[:, 0] + bbox_preds[:, 2] / 2
        boxes_y2 = bbox_preds[:, 1] + bbox_preds[:, 3] / 2
        boxes = torch.stack([boxes_x1, boxes_y1, boxes_x2, boxes_y2], dim=1)

        detections = []
        for class_idx in range(cls_scores.shape[1]):
            scores = cls_scores[:, class_idx]
            mask = scores > conf_thresh
            if not mask.any():
                continue

            class_boxes = boxes[mask]
            class_scores_filtered = scores[mask]

            keep = ops.nms(class_boxes, class_scores_filtered, iou_thresh)

            for k in keep:
                detections.append({
                    'class_idx': class_idx,
                    'class_name': self.det_classes[class_idx],
                    'score': class_scores_filtered[k].item(),
                    'box': class_boxes[k].tolist()  # [x1, y1, x2, y2]
                })

        return detections

    # ------------------------------------------------------------------
    # Main verification
    # ------------------------------------------------------------------

    def verify(self, image_path, expected_manifest, conf_thresh=0.45):
        """
        Verify the assembly logic.
        expected_manifest: {
            'material': 'Alloy',
            'tier': 'Premium',
            'size': '18_inch'
        }
        """
        report = {
            'status': 'PASS',
            'messages': [],
            'detections': [],
            'classification': {'material': '---', 'tier': '---', 'size': '---'}
        }

        original_image = Image.open(image_path).convert('RGB')
        orig_w, orig_h = original_image.size

        # 1. Run Detector
        det_input = self.det_transform(original_image).unsqueeze(0).to(self.device)
        cls_scores, bbox_preds = self._run_detector(det_input)

        detections = self._postprocess_detections(cls_scores, bbox_preds, conf_thresh=conf_thresh)

        # The frontend assumes coordinates are on a 640x640 scale and calculates percentages.
        # We scale from model size (512) to 640.
        scale_x = 640 / self.image_size[0]
        scale_y = 640 / self.image_size[1]

        wheel_box = None
        defects_found = []

        for det in detections:
            x1, y1, x2, y2 = det['box']
            # Save the original 512-scaled coordinates for cropping before modifying for the UI
            det['raw_box'] = [x1, y1, x2, y2]
            det['box'] = [x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y]
            report['detections'].append(det)

            if det['class_name'] == 'wheel':
                if wheel_box is None or det['score'] > wheel_box['score']:
                    wheel_box = det
            elif det['class_name'] in ['scratch', 'dent']:
                defects_found.append(det['class_name'])

        # Business Logic 1: Defects
        if defects_found:
            report['status'] = 'FAIL'
            report['messages'].append(f"Defects detected: {', '.join(defects_found)}")

        # 2. Run Classifier if a wheel is found
        if not wheel_box:
            report['status'] = 'FAIL'
            report['messages'].append("No wheel detected in the image.")
            return report

        rx1, ry1, rx2, ry2 = wheel_box['raw_box']

        # Scale to original image size
        crop_scale_x = orig_w / self.image_size[0]
        crop_scale_y = orig_h / self.image_size[1]

        bx1 = rx1 * crop_scale_x
        by1 = ry1 * crop_scale_y
        bx2 = rx2 * crop_scale_x
        by2 = ry2 * crop_scale_y

        # Ensure correct ordering
        bx1, bx2 = min(bx1, bx2), max(bx1, bx2)
        by1, by2 = min(by1, by2), max(by1, by2)

        # Clamp to image boundaries
        bx1, by1 = max(0, int(bx1)), max(0, int(by1))
        bx2, by2 = min(orig_w, int(bx2)), min(orig_h, int(by2))
        
        if bx1 >= bx2 or by1 >= by2:
            report['status'] = 'FAIL'
            report['messages'].append("Invalid or zero-area bounding box detected.")
            return report

        wheel_crop = original_image.crop((bx1, by1, bx2, by2))
        cls_input = self.cls_transform(wheel_crop).unsqueeze(0).to(self.device)

        mat_out, tier_out, size_out = self._run_classifier(cls_input)

        pred_mat = self.cls_material[mat_out.argmax(1).item()]
        pred_tier = self.cls_tier[tier_out.argmax(1).item()]
        pred_size = self.cls_size[size_out.argmax(1).item()]

        report['classification'] = {
            'material': pred_mat,
            'tier': pred_tier,
            'size': pred_size
        }

        # Business Logic 3: Configuration match
        if pred_mat != expected_manifest.get('material'):
            report['status'] = 'FAIL'
            report['messages'].append(
                f"Wrong material! Expected {expected_manifest.get('material')}, got {pred_mat}")

        if pred_tier != expected_manifest.get('tier'):
            report['status'] = 'FAIL'
            report['messages'].append(
                f"Wrong tier! Expected {expected_manifest.get('tier')}, got {pred_tier}")

        if pred_size != expected_manifest.get('size'):
            report['status'] = 'FAIL'
            report['messages'].append(
                f"Wrong size! Expected {expected_manifest.get('size')}, got {pred_size}")

        if report['status'] == 'PASS':
            report['messages'].append("Assembly verified successfully.")

        return report
