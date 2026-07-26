import os
import sys
import shutil
import torch
from PIL import Image
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.train_detect import train

class DummyArgs:
    def __init__(self, img_dir, label_dir, weights_dir):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.epochs = 5
        self.batch_size = 2
        self.lr = 1e-3
        self.weights_dir = weights_dir
        self.resume = False
        self.device = 'cpu' # Use CPU for local smoke test

def test_overfitting():
    # Setup dummy data directories
    test_dir = 'dummy_data'
    img_dir = os.path.join(test_dir, 'images')
    label_dir = os.path.join(test_dir, 'labels')
    weights_dir = os.path.join(test_dir, 'weights')
    
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)
    
    # Generate 4 dummy images and labels
    for i in range(4):
        # Create a random RGB image
        img_array = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        img.save(os.path.join(img_dir, f'dummy_{i}.jpg'))
        
        # Create dummy label: class cx cy w h
        # e.g., class 0, center (0.5, 0.5), size 0.2 0.2
        with open(os.path.join(label_dir, f'dummy_{i}.txt'), 'w') as f:
            f.write("0 0.5 0.5 0.2 0.2\n")
            f.write("1 0.2 0.2 0.1 0.1\n")
            
    args = DummyArgs(img_dir, label_dir, weights_dir)
    
    print("Starting smoke test training loop...")
    try:
        train(args)
        print("Smoke test passed! Training loop completed without errors.")
    except Exception as e:
        print(f"Smoke test failed: {e}")
        raise e
    finally:
        # Cleanup
        shutil.rmtree(test_dir)

if __name__ == '__main__':
    test_overfitting()
