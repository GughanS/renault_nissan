import os
import torch
import unittest
from PIL import Image
from wheeleye.datasets.yolo_dataset import YOLODataset
from wheeleye.preprocessing import get_inference_transforms

class TestPreprocessingParity(unittest.TestCase):
    def test_train_inference_parity(self):
        """
        Verify that feeding an image through the training dataset loader
        produces the EXACT same tensor as feeding it through the inference
        preprocessing pipeline.
        """
        # Ensure we have a synthetic image to test with
        img_dir = 'data/images'
        label_dir = 'data/labels'
        
        if not os.path.exists(img_dir) or len(os.listdir(img_dir)) == 0:
            self.skipTest("No synthetic images found. Run generate_synthetic_data.py first.")
            
        # Get first image path
        img_file = [f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.jpeg', '.png'))][0]
        img_path = os.path.join(img_dir, img_file)
        
        # 1. Inference Pipeline
        inference_transforms = get_inference_transforms((512, 512))
        raw_img = Image.open(img_path).convert('RGB')
        inf_tensor = inference_transforms(raw_img)
        
        # 2. Training Pipeline (no augmentation, no mosaic)
        dataset = YOLODataset(img_dir, label_dir, img_size=(512, 512), augment=False, mosaic=False)
        
        # Find the index of the image we picked
        idx = dataset.img_paths.index(img_path)
        train_tensor, _, _ = dataset[idx]
        
        # 3. Assert parity
        diff = torch.abs(inf_tensor - train_tensor).max().item()
        
        self.assertLess(
            diff, 1e-6,
            f"PREPROCESSING MISMATCH: The training and inference tensors differ by {diff}. "
            "This causes a silent distribution shift during deployment."
        )

if __name__ == '__main__':
    unittest.main()
