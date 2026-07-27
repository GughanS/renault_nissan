import os
import argparse
import random
import csv
from PIL import Image, ImageDraw
import numpy as np

# YOLO Classes:
# 0: wheel_assembly
# 1: lug_nut
# 2: valve_stem
# 3: center_cap

# Classification Targets:
# Material: 0 (steel), 1 (alloy)
# Tier: 0 (economy), 1 (mid), 2 (premium-sport)
# Size: 0 (<=15"), 1 (16-17"), 2 (18"+)

def generate_synthetic_image(img_size=(512, 512)):
    # Create random background
    bg_color = (random.randint(100, 150), random.randint(100, 150), random.randint(100, 150))
    img = Image.new('RGB', img_size, bg_color)
    draw = ImageDraw.Draw(img)
    
    boxes = [] # (cls, cx, cy, w, h)
    
    # Randomly assign classification labels
    material = random.randint(0, 1)
    tier = random.randint(0, 2)
    size = random.randint(0, 2)
    
    # 1. Draw Wheel Assembly (Class 0)
    wheel_r = random.randint(100, 200)
    wheel_cx = random.randint(wheel_r + 10, img_size[0] - wheel_r - 10)
    wheel_cy = random.randint(wheel_r + 10, img_size[1] - wheel_r - 10)
    
    # Tire color
    draw.ellipse([wheel_cx - wheel_r, wheel_cy - wheel_r, wheel_cx + wheel_r, wheel_cy + wheel_r], fill=(30, 30, 30))
    
    # Rim color depends somewhat on material (alloy=brighter, steel=darker/duller)
    if material == 1: # alloy
        rim_color = (200, 200, 200)
    else: # steel
        rim_color = (130, 130, 130)
    
    rim_r = int(wheel_r * 0.7)
    draw.ellipse([wheel_cx - rim_r, wheel_cy - rim_r, wheel_cx + rim_r, wheel_cy + rim_r], fill=rim_color)
    
    # Wheel Assembly Box
    boxes.append((0, wheel_cx / img_size[0], wheel_cy / img_size[1], (wheel_r * 2) / img_size[0], (wheel_r * 2) / img_size[1]))
    
    # 2. Draw Center Cap (Class 3)
    cap_r = int(rim_r * 0.2)
    cap_color = (100, 100, 200) if tier == 2 else (150, 150, 150) # premium has blueish cap
    draw.ellipse([wheel_cx - cap_r, wheel_cy - cap_r, wheel_cx + cap_r, wheel_cy + cap_r], fill=cap_color)
    boxes.append((3, wheel_cx / img_size[0], wheel_cy / img_size[1], (cap_r * 2) / img_size[0], (cap_r * 2) / img_size[1]))
    
    # 3. Draw Lug Nuts (Class 1)
    # Economy=4, Mid=5, Premium=6 lugs
    num_lugs = 4 if tier == 0 else (5 if tier == 1 else 6)
    lug_r = int(cap_r * 0.4)
    lug_dist = cap_r * 2.0
    for i in range(num_lugs):
        angle = i * (2 * np.pi / num_lugs) + random.uniform(0, 0.5)
        lug_cx = wheel_cx + int(lug_dist * np.cos(angle))
        lug_cy = wheel_cy + int(lug_dist * np.sin(angle))
        draw.ellipse([lug_cx - lug_r, lug_cy - lug_r, lug_cx + lug_r, lug_cy + lug_r], fill=(220, 220, 220))
        boxes.append((1, lug_cx / img_size[0], lug_cy / img_size[1], (lug_r * 2) / img_size[0], (lug_r * 2) / img_size[1]))
        
    # 4. Draw Valve Stem (Class 2)
    valve_r = int(lug_r * 0.8)
    valve_angle = random.uniform(0, 2 * np.pi)
    valve_dist = rim_r * 0.9
    valve_cx = wheel_cx + int(valve_dist * np.cos(valve_angle))
    valve_cy = wheel_cy + int(valve_dist * np.sin(valve_angle))
    draw.rectangle([valve_cx - valve_r, valve_cy - valve_r, valve_cx + valve_r, valve_cy + valve_r], fill=(50, 50, 50))
    boxes.append((2, valve_cx / img_size[0], valve_cy / img_size[1], (valve_r * 2) / img_size[0], (valve_r * 2) / img_size[1]))
    
    classify_labels = {'material': material, 'tier': tier, 'size': size}
    
    # Extra return: the absolute pixel coords of the wheel to crop it
    wheel_bbox = (wheel_cx - wheel_r, wheel_cy - wheel_r, wheel_cx + wheel_r, wheel_cy + wheel_r)
    
    return img, boxes, classify_labels, wheel_bbox

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-images', type=int, default=100)
    parser.add_argument('--out-dir', type=str, default='data')
    parser.add_argument('--task', type=str, default='detect', choices=['detect', 'classify', 'both'])
    args = parser.parse_args()
    
    # Detection folders
    if args.task in ['detect', 'both']:
        img_dir = os.path.join(args.out_dir, 'images')
        label_dir = os.path.join(args.out_dir, 'labels')
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(label_dir, exist_ok=True)
        
    # Classification folders
    if args.task in ['classify', 'both']:
        crop_dir = os.path.join(args.out_dir, 'crops')
        os.makedirs(crop_dir, exist_ok=True)
        csv_path = os.path.join(args.out_dir, 'crops_labels.csv')
        csv_file = open(csv_path, 'w', newline='')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['filename', 'material', 'tier', 'size'])
    
    print(f"Generating {args.num_images} synthetic images for task '{args.task}'...")
    for i in range(args.num_images):
        img, boxes, cls_labels, wheel_bbox = generate_synthetic_image()
        
        # Save detection data
        if args.task in ['detect', 'both']:
            img_path = os.path.join(img_dir, f'synth_{i:04d}.jpg')
            img.save(img_path)
            
            label_path = os.path.join(label_dir, f'synth_{i:04d}.txt')
            with open(label_path, 'w') as f:
                for b in boxes:
                    f.write(f"{b[0]} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f} {b[4]:.6f}\n")
                    
        # Save classification data (crops and labels)
        if args.task in ['classify', 'both']:
            crop_img = img.crop(wheel_bbox)
            crop_filename = f'crop_{i:04d}.jpg'
            crop_path = os.path.join(crop_dir, crop_filename)
            crop_img.save(crop_path)
            
            csv_writer.writerow([crop_filename, cls_labels['material'], cls_labels['tier'], cls_labels['size']])
            
    if args.task in ['classify', 'both']:
        csv_file.close()
                
    print(f"Done! Dataset saved to {args.out_dir}")

if __name__ == '__main__':
    main()
