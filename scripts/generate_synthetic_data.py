import os
import argparse
import random
from PIL import Image, ImageDraw
import numpy as np

# Classes:
# 0: wheel_assembly
# 1: lug_nut
# 2: valve_stem
# 3: center_cap

def generate_synthetic_image(img_size=(512, 512)):
    # Create random background (e.g., greyish/brownish representing a car body or factory floor)
    bg_color = (random.randint(100, 150), random.randint(100, 150), random.randint(100, 150))
    img = Image.new('RGB', img_size, bg_color)
    draw = ImageDraw.Draw(img)
    
    boxes = [] # (cls, cx, cy, w, h)
    
    # 1. Draw Wheel Assembly (Class 0)
    # Random position and size for the wheel
    wheel_r = random.randint(100, 200)
    wheel_cx = random.randint(wheel_r + 10, img_size[0] - wheel_r - 10)
    wheel_cy = random.randint(wheel_r + 10, img_size[1] - wheel_r - 10)
    
    # Dark circle for the tire
    draw.ellipse([wheel_cx - wheel_r, wheel_cy - wheel_r, wheel_cx + wheel_r, wheel_cy + wheel_r], fill=(30, 30, 30))
    # Inner circle for the rim
    rim_r = int(wheel_r * 0.7)
    draw.ellipse([wheel_cx - rim_r, wheel_cy - rim_r, wheel_cx + rim_r, wheel_cy + rim_r], fill=(180, 180, 180))
    
    # Wheel Assembly Box
    boxes.append((0, wheel_cx / img_size[0], wheel_cy / img_size[1], (wheel_r * 2) / img_size[0], (wheel_r * 2) / img_size[1]))
    
    # 2. Draw Center Cap (Class 3)
    cap_r = int(rim_r * 0.2)
    draw.ellipse([wheel_cx - cap_r, wheel_cy - cap_r, wheel_cx + cap_r, wheel_cy + cap_r], fill=(100, 100, 200))
    boxes.append((3, wheel_cx / img_size[0], wheel_cy / img_size[1], (cap_r * 2) / img_size[0], (cap_r * 2) / img_size[1]))
    
    # 3. Draw Lug Nuts (Class 1) - typically 5 around the center cap
    num_lugs = 5
    lug_r = int(cap_r * 0.4)
    lug_dist = cap_r * 2.0
    for i in range(num_lugs):
        angle = i * (2 * np.pi / num_lugs) + random.uniform(0, 0.5)
        lug_cx = wheel_cx + int(lug_dist * np.cos(angle))
        lug_cy = wheel_cy + int(lug_dist * np.sin(angle))
        draw.ellipse([lug_cx - lug_r, lug_cy - lug_r, lug_cx + lug_r, lug_cy + lug_r], fill=(200, 200, 200))
        boxes.append((1, lug_cx / img_size[0], lug_cy / img_size[1], (lug_r * 2) / img_size[0], (lug_r * 2) / img_size[1]))
        
    # 4. Draw Valve Stem (Class 2) - somewhere on the edge of the rim
    valve_r = int(lug_r * 0.8)
    valve_angle = random.uniform(0, 2 * np.pi)
    valve_dist = rim_r * 0.9
    valve_cx = wheel_cx + int(valve_dist * np.cos(valve_angle))
    valve_cy = wheel_cy + int(valve_dist * np.sin(valve_angle))
    draw.rectangle([valve_cx - valve_r, valve_cy - valve_r, valve_cx + valve_r, valve_cy + valve_r], fill=(50, 50, 50))
    boxes.append((2, valve_cx / img_size[0], valve_cy / img_size[1], (valve_r * 2) / img_size[0], (valve_r * 2) / img_size[1]))
    
    return img, boxes

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-images', type=int, default=100)
    parser.add_argument('--out-dir', type=str, default='data')
    args = parser.parse_args()
    
    img_dir = os.path.join(args.out_dir, 'images')
    label_dir = os.path.join(args.out_dir, 'labels')
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)
    
    print(f"Generating {args.num_images} synthetic images...")
    for i in range(args.num_images):
        img, boxes = generate_synthetic_image()
        
        # Save image
        img_path = os.path.join(img_dir, f'synth_{i:04d}.jpg')
        img.save(img_path)
        
        # Save labels (YOLO format)
        label_path = os.path.join(label_dir, f'synth_{i:04d}.txt')
        with open(label_path, 'w') as f:
            for b in boxes:
                f.write(f"{b[0]} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f} {b[4]:.6f}\n")
                
    print(f"Done! Dataset saved to {args.out_dir}")

if __name__ == '__main__':
    main()
