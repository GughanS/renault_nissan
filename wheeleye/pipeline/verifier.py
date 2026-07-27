import torch
import torchvision.transforms as T
import torchvision.ops as ops
from PIL import Image
from wheeleye.models.detector import WheelEyeDetector
from wheeleye.models.classifier import WheelEyeClassifier
from wheeleye.utils.anchors import generate_anchors, decode_boxes

class WheelEyeVerifier:
    def __init__(self, detector_weights_path=None, classifier_weights_path=None, device='cpu'):
        self.device = torch.device(device)
        
        # Initialize models
        self.detector = WheelEyeDetector(num_classes=4, num_anchors=9).to(self.device)
        self.classifier = WheelEyeClassifier().to(self.device)
        
        # Load weights if provided
        if detector_weights_path:
            ckpt = torch.load(detector_weights_path, map_location=self.device, weights_only=True)
            self.detector.load_state_dict(ckpt.get('model_state_dict', ckpt))
        if classifier_weights_path:
            ckpt = torch.load(classifier_weights_path, map_location=self.device, weights_only=True)
            self.classifier.load_state_dict(ckpt.get('model_state_dict', ckpt))
            
        self.detector.eval()
        self.classifier.eval()
        
        # Detector anchors
        self.image_size = (512, 512)
        strides = [8, 16, 32]
        base_sizes = [32, 64, 128]
        scales = [1, 2, 4]
        aspect_ratios = [0.5, 1.0, 2.0]
        self.anchors = generate_anchors(self.image_size, strides, base_sizes, scales, aspect_ratios).to(self.device)
        
        # Transforms
        self.det_transform = T.Compose([
            T.Resize(self.image_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.cls_transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Class mappings
        self.det_classes = ['wheel', 'fastener', 'scratch', 'dent']
        self.cls_material = ['Steel', 'Alloy']
        self.cls_tier = ['Standard', 'Premium']
        self.cls_size = ['17_inch', '18_inch', '19_inch']

    def _postprocess_detections(self, cls_scores, bbox_preds, conf_thresh=0.5, iou_thresh=0.4):
        """Decode bounding boxes and apply NMS."""
        # cls_scores: (B, N, C), bbox_preds: (B, N, 4)
        cls_scores = torch.sigmoid(cls_scores[0]) # single image
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
                    'box': class_boxes[k].tolist() # [x1, y1, x2, y2]
                })
                
        return detections

    def verify(self, image_path, expected_manifest):
        """
        Verify the assembly logic.
        expected_manifest: {
            'material': 'Alloy',
            'tier': 'Premium',
            'size': '18_inch',
            'expected_fasteners': 5
        }
        """
        report = {
            'status': 'PASS',
            'messages': [],
            'detections': [],
            'classification': {}
        }
        
        original_image = Image.open(image_path).convert('RGB')
        orig_w, orig_h = original_image.size
        
        # 1. Run Detector
        det_input = self.det_transform(original_image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            cls_scores, bbox_preds = self.detector(det_input)
            
        detections = self._postprocess_detections(cls_scores, bbox_preds)
        
        # Scale boxes back to original image size
        scale_x = orig_w / self.image_size[0]
        scale_y = orig_h / self.image_size[1]
        
        wheel_box = None
        fastener_count = 0
        defects_found = []
        
        for det in detections:
            x1, y1, x2, y2 = det['box']
            det['box'] = [x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y]
            report['detections'].append(det)
            
            if det['class_name'] == 'wheel':
                if wheel_box is None or det['score'] > wheel_box['score']:
                    wheel_box = det
            elif det['class_name'] == 'fastener':
                fastener_count += 1
            elif det['class_name'] in ['scratch', 'dent']:
                defects_found.append(det['class_name'])
                
        # Business Logic 1: Defects
        if defects_found:
            report['status'] = 'FAIL'
            report['messages'].append(f"Defects detected: {', '.join(defects_found)}")
            
        # Business Logic 2: Fastener count
        expected_fasteners = expected_manifest.get('expected_fasteners', 5)
        if fastener_count != expected_fasteners:
            report['status'] = 'FAIL'
            report['messages'].append(f"Expected {expected_fasteners} fasteners, found {fastener_count}")
            
        # 2. Run Classifier if a wheel is found
        if not wheel_box:
            report['status'] = 'FAIL'
            report['messages'].append("No wheel detected in the image.")
            return report
            
        bx1, by1, bx2, by2 = wheel_box['box']
        # Clamp to image boundaries
        bx1, by1 = max(0, int(bx1)), max(0, int(by1))
        bx2, by2 = min(orig_w, int(bx2)), min(orig_h, int(by2))
        
        wheel_crop = original_image.crop((bx1, by1, bx2, by2))
        cls_input = self.cls_transform(wheel_crop).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            mat_out, tier_out, size_out = self.classifier(cls_input)
            
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
            report['messages'].append(f"Wrong material! Expected {expected_manifest.get('material')}, got {pred_mat}")
            
        if pred_tier != expected_manifest.get('tier'):
            report['status'] = 'FAIL'
            report['messages'].append(f"Wrong tier! Expected {expected_manifest.get('tier')}, got {pred_tier}")
            
        if pred_size != expected_manifest.get('size'):
            report['status'] = 'FAIL'
            report['messages'].append(f"Wrong size! Expected {expected_manifest.get('size')}, got {pred_size}")
            
        if report['status'] == 'PASS':
            report['messages'].append("Assembly verified successfully.")
            
        return report
