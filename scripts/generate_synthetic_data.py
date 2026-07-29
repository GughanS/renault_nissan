import os
import argparse
import random
import csv
import math
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
import numpy as np
import cv2

# YOLO Classes:
# 0: wheel_assembly
# 1: lug_nut
# 2: valve_stem
# 3: center_cap

# Classification Targets:
# Material: 0 (steel), 1 (alloy)
# Tier: 0 (economy), 1 (mid), 2 (premium-sport)
# Size: 0 (<=15"), 1 (16-17"), 2 (18"+)

# ---------------------------------------------------------------------------
# Helper: radial gradient for metallic sheen
# ---------------------------------------------------------------------------

def _radial_gradient(size, center, radius, color_inner, color_outer):
    """Create a radial gradient image simulating a curved metallic surface."""
    arr = np.zeros((size[1], size[0], 3), dtype=np.float64)
    cx, cy = center
    for y in range(size[1]):
        for x in range(size[0]):
            d = math.hypot(x - cx, y - cy) / max(radius, 1)
            d = min(d, 1.0)
            for c in range(3):
                arr[y, x, c] = color_inner[c] * (1 - d) + color_outer[c] * d
    return arr

def _fast_radial_gradient(size, center, radius, color_inner, color_outer):
    """Vectorised radial gradient — much faster than the pixel loop version."""
    ys = np.arange(size[1]).reshape(-1, 1)
    xs = np.arange(size[0]).reshape(1, -1)
    d = np.sqrt((xs - center[0]) ** 2 + (ys - center[1]) ** 2) / max(radius, 1)
    d = np.clip(d, 0, 1)[..., np.newaxis]  # (H, W, 1)
    inner = np.array(color_inner, dtype=np.float64).reshape(1, 1, 3)
    outer = np.array(color_outer, dtype=np.float64).reshape(1, 1, 3)
    return inner * (1 - d) + outer * d


def _perlin_noise(size, scale=64):
    """Simple Perlin-like noise approximation for surface texture."""
    h, w = size
    noise = np.random.randn(h // scale + 2, w // scale + 2)
    # Bilinear upsample
    from PIL import Image as _Img
    noise_img = _Img.fromarray(((noise - noise.min()) / (noise.max() - noise.min() + 1e-8) * 255).astype(np.uint8))
    noise_img = noise_img.resize((w, h), Image.BILINEAR)
    return np.array(noise_img).astype(np.float64) / 255.0  # 0..1


# ---------------------------------------------------------------------------
# Data Augmentations
# ---------------------------------------------------------------------------

def _apply_motion_blur(img_arr, kernel_size=15):
    """Simulate motion blur from a fast-moving conveyor belt."""
    # Create horizontal motion blur kernel
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)
    kernel /= kernel_size
    blurred = cv2.filter2D(img_arr, -1, kernel)
    return blurred

def _draw_glare(draw, img_size):
    """Draw a harsh glare polygon simulating lighting glare."""
    w, h = img_size
    poly = [
        (random.randint(0, w // 2), 0),
        (random.randint(w // 2, w), 0),
        (random.randint(w // 2, w), random.randint(h // 4, h // 2)),
        (0, random.randint(h // 4, h // 2))
    ]
    # Draw a polygon with low opacity white
    draw.polygon(poly, fill=(255, 255, 255, random.randint(20, 60)))

# ---------------------------------------------------------------------------
# Factory background generator
# ---------------------------------------------------------------------------

def _generate_factory_background(img_size):
    """Generate a realistic factory floor / conveyor background."""
    w, h = img_size

    # Base concrete grey with slight colour cast
    base_r = random.randint(55, 85)
    base_g = base_r + random.randint(-5, 5)
    base_b = base_r + random.randint(-8, 3)
    bg = np.full((h, w, 3), [base_r, base_g, base_b], dtype=np.float64)

    # Add low-frequency noise for concrete patchiness
    noise_low = _perlin_noise((h, w), scale=128)
    bg += (noise_low[..., np.newaxis] - 0.5) * 40

    # Add high-frequency noise for grain
    grain = np.random.normal(0, 8, (h, w, 3))
    bg += grain

    # Occasional dark oil stain
    if random.random() > 0.6:
        stain_cx = random.randint(0, w)
        stain_cy = random.randint(0, h)
        stain_r = random.randint(40, 120)
        ys = np.arange(h).reshape(-1, 1)
        xs = np.arange(w).reshape(1, -1)
        dist = np.sqrt((xs - stain_cx) ** 2 + (ys - stain_cy) ** 2)
        mask = np.clip(1 - dist / stain_r, 0, 1) ** 2
        bg -= mask[..., np.newaxis] * random.randint(20, 50)

    # Conveyor belt lines
    if random.random() > 0.5:
        line_y = random.randint(h // 4, 3 * h // 4)
        bg[max(0, line_y - 2):min(h, line_y + 2), :, :] -= 25

    bg = np.clip(bg, 0, 255).astype(np.uint8)
    return Image.fromarray(bg)


# ---------------------------------------------------------------------------
# 3-D wheel renderer
# ---------------------------------------------------------------------------

def _draw_tire(draw, cx, cy, outer_r, inner_r, img_size):
    """Draw a realistic tire with sidewall text, tread texture, and depth."""
    # Outer tire shadow (ground contact shadow)
    shadow_offset = int(outer_r * 0.04)
    draw.ellipse(
        [cx - outer_r + shadow_offset, cy - outer_r + shadow_offset,
         cx + outer_r + shadow_offset, cy + outer_r + shadow_offset],
        fill=(10, 10, 10, 180)
    )

    # Main tire body — dark rubber with slight radial shading
    for r_step in range(outer_r, inner_r, -1):
        t = (r_step - inner_r) / max(outer_r - inner_r, 1)
        shade = int(18 + t * 12)  # darker at edge, lighter near rim
        draw.ellipse(
            [cx - r_step, cy - r_step, cx + r_step, cy + r_step],
            fill=(shade, shade, shade + 2, 255)
        )

    # Tread grooves — concentric dashed arcs
    num_grooves = random.randint(3, 5)
    for g in range(num_grooves):
        groove_r = inner_r + int((outer_r - inner_r) * (0.2 + 0.6 * g / max(num_grooves - 1, 1)))
        num_dashes = random.randint(40, 70)
        for d in range(num_dashes):
            angle = d * 2 * math.pi / num_dashes
            if d % 3 == 0:
                continue  # gap
            x1 = cx + int((groove_r - 2) * math.cos(angle))
            y1 = cy + int((groove_r - 2) * math.sin(angle))
            x2 = cx + int((groove_r + 2) * math.cos(angle))
            y2 = cy + int((groove_r + 2) * math.sin(angle))
            draw.line([(x1, y1), (x2, y2)], fill=(12, 12, 12, 255), width=1)

    # Sidewall bead — a thin bright ring at the rim-tire junction
    bead_r = inner_r + 3
    draw.ellipse(
        [cx - bead_r, cy - bead_r, cx + bead_r, cy + bead_r],
        outline=(45, 45, 45, 255), width=2
    )


def _draw_rim(draw, img_arr, cx, cy, rim_r, material, tier, img_size):
    """Draw a 3-D looking rim with specular highlights and depth."""
    # Rim base colours
    if material == 1:  # alloy
        base_inner = (210, 215, 220)
        base_outer = (140, 145, 155)
        spec_color = (250, 252, 255)
    else:  # steel
        base_inner = (120, 118, 115)
        base_outer = (65, 63, 60)
        spec_color = (180, 180, 180)

    # Radial gradient for 3-D curvature
    grad = _fast_radial_gradient(img_size, (cx, cy), rim_r, base_inner, base_outer)

    # Apply gradient only within rim circle
    ys = np.arange(img_size[1]).reshape(-1, 1)
    xs = np.arange(img_size[0]).reshape(1, -1)
    dist = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
    rim_mask = (dist <= rim_r).astype(np.float64)

    # Add metallic noise
    metal_noise = np.random.normal(0, 6 if material == 1 else 3, (img_size[1], img_size[0], 3))
    grad += metal_noise

    # Blend into image array
    for c in range(3):
        img_arr[:, :, c] = img_arr[:, :, c] * (1 - rim_mask) + np.clip(grad[:, :, c], 0, 255) * rim_mask

    # Outer rim lip — bright ring for depth
    lip_width = max(3, int(rim_r * 0.04))
    draw.ellipse(
        [cx - rim_r, cy - rim_r, cx + rim_r, cy + rim_r],
        outline=(*spec_color, 200), width=lip_width
    )

    # Inner shadow ring
    inner_shadow_r = int(rim_r * 0.92)
    draw.ellipse(
        [cx - inner_shadow_r, cy - inner_shadow_r, cx + inner_shadow_r, cy + inner_shadow_r],
        outline=(base_outer[0] - 30, base_outer[1] - 30, base_outer[2] - 30, 120), width=2
    )

    return img_arr


def _draw_spokes(draw, cx, cy, rim_r, hub_r, material, tier):
    """Draw 3-D spokes with shading and depth for alloy, or punched holes for steel."""
    if material == 1:  # alloy
        num_spokes = {0: 5, 1: 7, 2: 10}.get(tier, 6)
        spoke_width = max(8, int(rim_r * 0.12))
        spoke_inner_r = int(hub_r * 1.2)
        spoke_outer_r = int(rim_r * 0.88)

        for i in range(num_spokes):
            angle = i * 2 * math.pi / num_spokes + random.uniform(-0.02, 0.02)

            # Spoke shadow (offset slightly)
            for offset in [(3, 3)]:
                sx1 = cx + int(spoke_inner_r * math.cos(angle)) + offset[0]
                sy1 = cy + int(spoke_inner_r * math.sin(angle)) + offset[1]
                sx2 = cx + int(spoke_outer_r * math.cos(angle)) + offset[0]
                sy2 = cy + int(spoke_outer_r * math.sin(angle)) + offset[1]
                draw.line([(sx1, sy1), (sx2, sy2)], fill=(40, 40, 40, 100),
                          width=spoke_width + 4)

            # Main spoke body
            sx1 = cx + int(spoke_inner_r * math.cos(angle))
            sy1 = cy + int(spoke_inner_r * math.sin(angle))
            sx2 = cx + int(spoke_outer_r * math.cos(angle))
            sy2 = cy + int(spoke_outer_r * math.sin(angle))

            # Gradient along spoke: brighter at center, darker at edge
            steps = 8
            for s in range(steps):
                t = s / steps
                interp_x1 = int(sx1 + (sx2 - sx1) * t)
                interp_y1 = int(sy1 + (sy2 - sy1) * t)
                interp_x2 = int(sx1 + (sx2 - sx1) * (t + 1 / steps))
                interp_y2 = int(sy1 + (sy2 - sy1) * (t + 1 / steps))
                shade = int(200 - t * 60) if material == 1 else int(130 - t * 40)
                draw.line(
                    [(interp_x1, interp_y1), (interp_x2, interp_y2)],
                    fill=(shade, shade + 3, shade + 5, 255),
                    width=int(spoke_width * (1 - t * 0.3))
                )

            # Spoke highlight (specular edge)
            hx1 = cx + int(spoke_inner_r * math.cos(angle)) - 1
            hy1 = cy + int(spoke_inner_r * math.sin(angle)) - 1
            hx2 = cx + int(spoke_outer_r * 0.6 * math.cos(angle)) - 1
            hy2 = cy + int(spoke_outer_r * 0.6 * math.sin(angle)) - 1
            draw.line([(hx1, hy1), (hx2, hy2)], fill=(240, 245, 255, 80),
                      width=max(1, spoke_width // 4))

        # Between-spoke dark pockets (depth)
        for i in range(num_spokes):
            mid_angle = (i + 0.5) * 2 * math.pi / num_spokes
            pocket_r = int((spoke_inner_r + spoke_outer_r) * 0.45)
            pocket_cx = cx + int(pocket_r * math.cos(mid_angle))
            pocket_cy = cy + int(pocket_r * math.sin(mid_angle))
            pocket_size = int(rim_r * 0.12)
            draw.ellipse(
                [pocket_cx - pocket_size, pocket_cy - pocket_size,
                 pocket_cx + pocket_size, pocket_cy + pocket_size],
                fill=(25, 25, 30, 200)
            )
    else:
        # Steel wheel — punched circular holes with depth shading
        num_holes = {0: 6, 1: 8, 2: 10}.get(tier, 8)
        hole_dist = rim_r * 0.6
        hole_r = int(rim_r * 0.13)

        for i in range(num_holes):
            angle = i * 2 * math.pi / num_holes
            hx = cx + int(hole_dist * math.cos(angle))
            hy = cy + int(hole_dist * math.sin(angle))

            # Hole shadow (depth effect)
            draw.ellipse(
                [hx - hole_r + 2, hy - hole_r + 2, hx + hole_r + 2, hy + hole_r + 2],
                fill=(15, 15, 15, 200)
            )
            # Hole body
            draw.ellipse(
                [hx - hole_r, hy - hole_r, hx + hole_r, hy + hole_r],
                fill=(20, 20, 22, 255)
            )
            # Inner rim highlight (light catching the hole edge)
            draw.arc(
                [hx - hole_r + 1, hy - hole_r + 1, hx + hole_r - 1, hy + hole_r - 1],
                start=200, end=340,
                fill=(100, 100, 105, 150), width=1
            )


def _draw_specular_highlight(draw, cx, cy, rim_r):
    """Add a large specular highlight blob simulating overhead factory lighting."""
    # Off-center highlight to simulate directional light
    highlight_cx = cx + random.randint(-int(rim_r * 0.3), int(rim_r * 0.15))
    highlight_cy = cy + random.randint(-int(rim_r * 0.35), -int(rim_r * 0.1))
    highlight_r = int(rim_r * random.uniform(0.2, 0.35))

    # Multiple concentric ellipses with decreasing opacity
    for step in range(highlight_r, 0, -3):
        alpha = int(15 + (1 - step / highlight_r) * 40)
        draw.ellipse(
            [highlight_cx - step, highlight_cy - int(step * 0.6),
             highlight_cx + step, highlight_cy + int(step * 0.6)],
            fill=(255, 255, 255, min(alpha, 55))
        )


def _draw_hub_and_cap(draw, cx, cy, hub_r, tier):
    """Draw the central hub and centre cap with 3-D shading."""
    # Hub ring — dark metallic
    draw.ellipse(
        [cx - hub_r, cy - hub_r, cx + hub_r, cy + hub_r],
        fill=(90, 90, 92, 255), outline=(60, 60, 60, 255), width=3
    )
    # Hub inner shading
    inner_hub = int(hub_r * 0.85)
    draw.ellipse(
        [cx - inner_hub, cy - inner_hub, cx + inner_hub, cy + inner_hub],
        fill=(110, 112, 115, 255)
    )

    # Centre cap
    cap_r = int(hub_r * 0.55)
    if tier == 2:
        cap_base = (30, 50, 160)  # Premium blue
        cap_highlight = (60, 90, 210)
    elif tier == 1:
        cap_base = (50, 50, 52)
        cap_highlight = (90, 90, 95)
    else:
        cap_base = (80, 80, 82)
        cap_highlight = (130, 130, 135)

    # Cap shadow
    draw.ellipse(
        [cx - cap_r + 2, cy - cap_r + 2, cx + cap_r + 2, cy + cap_r + 2],
        fill=(30, 30, 30, 180)
    )
    # Cap body
    draw.ellipse(
        [cx - cap_r, cy - cap_r, cx + cap_r, cy + cap_r],
        fill=(*cap_base, 255), outline=(40, 40, 42, 255), width=2
    )
    # Cap highlight arc
    draw.arc(
        [cx - cap_r + 3, cy - cap_r + 3, cx + cap_r - 3, cy + int(cap_r * 0.2)],
        start=180, end=360,
        fill=(*cap_highlight, 100), width=2
    )
    # Logo dot
    draw.ellipse([cx - 3, cy - 3, cx + 3, cy + 3], fill=(255, 255, 255, 180))

    return cap_r


def _draw_lug_nuts(draw, cx, cy, lug_dist, num_lugs, lug_r):
    """Draw lug nuts with 3-D hexagonal shading."""
    lug_boxes = []
    for i in range(num_lugs):
        angle = i * (2 * math.pi / num_lugs) + random.uniform(-0.03, 0.03)
        lug_cx = cx + int(lug_dist * math.cos(angle))
        lug_cy = cy + int(lug_dist * math.sin(angle))

        # Socket shadow
        draw.ellipse(
            [lug_cx - lug_r - 3, lug_cy - lug_r - 3,
             lug_cx + lug_r + 3, lug_cy + lug_r + 3],
            fill=(35, 35, 35, 200)
        )
        # Lug body (outer ring)
        draw.ellipse(
            [lug_cx - lug_r, lug_cy - lug_r, lug_cx + lug_r, lug_cy + lug_r],
            fill=(195, 198, 200, 255), outline=(120, 120, 122, 255), width=1
        )
        # Hexagonal facet simulation — inner highlight
        inner_r = int(lug_r * 0.6)
        draw.ellipse(
            [lug_cx - inner_r, lug_cy - inner_r, lug_cx + inner_r, lug_cy + inner_r],
            fill=(220, 225, 230, 255)
        )
        # Specular dot
        draw.ellipse(
            [lug_cx - 2, lug_cy - 3, lug_cx + 2, lug_cy - 1],
            fill=(250, 252, 255, 200)
        )

        # Bounding box for YOLO (with padding)
        bb_size = lug_r * 2.5
        lug_boxes.append((lug_cx, lug_cy, bb_size, bb_size))

    return lug_boxes


def _draw_valve_stem(draw, cx, cy, rim_r, valve_r):
    """Draw a valve stem with metallic shading."""
    valve_angle = random.uniform(0, 2 * math.pi)
    valve_dist = rim_r * 0.85
    valve_cx = cx + int(valve_dist * math.cos(valve_angle))
    valve_cy = cy + int(valve_dist * math.sin(valve_angle))

    stem_h = int(valve_r * 2.5)
    stem_w = int(valve_r * 0.8)

    # Stem direction — points outward from center
    dx = math.cos(valve_angle)
    dy = math.sin(valve_angle)

    # Stem shadow
    draw.rectangle(
        [valve_cx - stem_w + 2, valve_cy - stem_w + 2,
         valve_cx + stem_w + 2, valve_cy + stem_w + 2],
        fill=(15, 15, 15, 200)
    )
    # Stem body (dark rubber base)
    draw.rectangle(
        [valve_cx - stem_w, valve_cy - stem_w,
         valve_cx + stem_w, valve_cy + stem_w],
        fill=(50, 50, 52, 255)
    )
    # Valve cap (metallic top)
    cap_r = int(valve_r * 0.5)
    draw.ellipse(
        [valve_cx - cap_r, valve_cy - cap_r, valve_cx + cap_r, valve_cy + cap_r],
        fill=(160, 162, 165, 255), outline=(90, 90, 92, 255), width=1
    )
    # Highlight
    draw.ellipse(
        [valve_cx - 2, valve_cy - 3, valve_cx + 1, valve_cy - 1],
        fill=(230, 235, 240, 180)
    )

    bb_size = valve_r * 2.5
    return valve_cx, valve_cy, bb_size, bb_size


# ---------------------------------------------------------------------------
# Main image generation pipeline
# ---------------------------------------------------------------------------

def generate_synthetic_image(img_size=(512, 512)):
    """Generate a photorealistic synthetic wheel assembly image with factory background."""

    # --- Factory background ---
    img = _generate_factory_background(img_size)
    draw = ImageDraw.Draw(img, 'RGBA')
    img_arr = np.array(img).astype(np.float64)

    boxes = []  # (cls, cx, cy, w, h) normalised

    material = random.randint(0, 1)
    tier = random.randint(0, 2)
    size = random.randint(0, 2)

    # Wheel geometry
    wheel_r = random.randint(130, 200)
    wheel_cx = random.randint(wheel_r + 30, img_size[0] - wheel_r - 30)
    wheel_cy = random.randint(wheel_r + 30, img_size[1] - wheel_r - 30)
    rim_r = int(wheel_r * 0.73)
    hub_r = int(rim_r * 0.32)

    # 1. Tire
    _draw_tire(draw, wheel_cx, wheel_cy, wheel_r, rim_r, img_size)

    # Flush draw to array, then do rim gradient blending on the array
    img = Image.alpha_composite(Image.new('RGBA', img_size, (0, 0, 0, 0)), img.convert('RGBA'))
    draw = ImageDraw.Draw(img, 'RGBA')
    img_arr = np.array(img.convert('RGB')).astype(np.float64)

    # 2. Rim (gradient + metallic texture applied to numpy array)
    img_arr = _draw_rim(draw, img_arr, wheel_cx, wheel_cy, rim_r, material, tier, img_size)
    img = Image.fromarray(np.clip(img_arr, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(img, 'RGBA')

    # 3. Spokes / holes
    _draw_spokes(draw, wheel_cx, wheel_cy, rim_r, hub_r, material, tier)

    # 4. Hub and centre cap
    cap_r = _draw_hub_and_cap(draw, wheel_cx, wheel_cy, hub_r, tier)

    # 5. Specular highlight (factory overhead light reflection)
    _draw_specular_highlight(draw, wheel_cx, wheel_cy, rim_r)

    # Wheel assembly bounding box
    boxes.append((0, wheel_cx / img_size[0], wheel_cy / img_size[1],
                  (wheel_r * 2) / img_size[0], (wheel_r * 2) / img_size[1]))

    # Centre cap bounding box
    boxes.append((3, wheel_cx / img_size[0], wheel_cy / img_size[1],
                  (cap_r * 2) / img_size[0], (cap_r * 2) / img_size[1]))

    # 6. Lug nuts
    num_lugs = {0: 4, 1: 5, 2: 6}.get(tier, 5)
    lug_r = max(5, int(hub_r * 0.28))
    lug_dist = hub_r * 1.5
    lug_boxes = _draw_lug_nuts(draw, wheel_cx, wheel_cy, lug_dist, num_lugs, lug_r)
    for lx, ly, lw, lh in lug_boxes:
        boxes.append((1, lx / img_size[0], ly / img_size[1],
                      lw / img_size[0], lh / img_size[1]))

    # 7. Valve stem
    vx, vy, vw, vh = _draw_valve_stem(draw, wheel_cx, wheel_cy, rim_r,
                                       max(4, int(lug_r * 0.8)))
    boxes.append((2, vx / img_size[0], vy / img_size[1],
                  vw / img_size[0], vh / img_size[1]))

    # --- Post-processing for photorealism ---
    img = img.convert('RGB')

    # Subtle Gaussian blur to soften harsh edges
    img = img.filter(ImageFilter.GaussianBlur(radius=0.6))
    
    # Optional Data Augmentations
    if random.random() > 0.5:
        # Re-convert to array for motion blur
        arr = np.array(img).astype(np.float32)
        arr = _apply_motion_blur(arr, kernel_size=random.choice([9, 15, 21]))
        img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
        
    if random.random() > 0.5:
        # Add glare
        img = img.convert('RGBA')
        draw_glare = ImageDraw.Draw(img, 'RGBA')
        _draw_glare(draw_glare, img_size)
        img = img.convert('RGB')

    # Random brightness / contrast jitter simulating factory lighting variation
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(random.uniform(0.85, 1.15))
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(random.uniform(0.9, 1.1))

    # Sensor noise
    final_arr = np.array(img).astype(np.float32)
    final_arr += np.random.normal(0, random.uniform(2, 6), final_arr.shape)
    final_arr = np.clip(final_arr, 0, 255).astype(np.uint8)
    img = Image.fromarray(final_arr)

    # Slight JPEG compression artefacts (simulating real camera pipeline)
    if random.random() > 0.5:
        img = img.filter(ImageFilter.SHARPEN)

    classify_labels = {'material': material, 'tier': tier, 'size': size}
    wheel_bbox = (
        max(0, wheel_cx - wheel_r), max(0, wheel_cy - wheel_r),
        min(img_size[0], wheel_cx + wheel_r), min(img_size[1], wheel_cy + wheel_r)
    )

    return img, boxes, classify_labels, wheel_bbox


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-images', type=int, default=100)
    parser.add_argument('--out-dir', type=str, default='data')
    parser.add_argument('--task', type=str, default='detect', choices=['detect', 'classify', 'both'])
    args = parser.parse_args()

    if args.task in ['detect', 'both']:
        img_dir = os.path.join(args.out_dir, 'images')
        label_dir = os.path.join(args.out_dir, 'labels')
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(label_dir, exist_ok=True)

    if args.task in ['classify', 'both']:
        crop_dir = os.path.join(args.out_dir, 'crops')
        os.makedirs(crop_dir, exist_ok=True)
        csv_path = os.path.join(args.out_dir, 'crops_labels.csv')
        csv_file = open(csv_path, 'w', newline='')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['filename', 'material', 'tier', 'size'])

    print(f"Generating {args.num_images} photorealistic synthetic images for task '{args.task}'...")
    for i in range(args.num_images):
        img, boxes, cls_labels, wheel_bbox = generate_synthetic_image()

        if args.task in ['detect', 'both']:
            img_path = os.path.join(img_dir, f'synth_{i:04d}.jpg')
            img.save(img_path, quality=95)

            label_path = os.path.join(label_dir, f'synth_{i:04d}.txt')
            with open(label_path, 'w') as f:
                for b in boxes:
                    f.write(f"{b[0]} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f} {b[4]:.6f}\n")

        if args.task in ['classify', 'both']:
            crop_img = img.crop(wheel_bbox)
            crop_filename = f'crop_{i:04d}.jpg'
            crop_path = os.path.join(crop_dir, crop_filename)
            crop_img.save(crop_path, quality=95)

            csv_writer.writerow([crop_filename, cls_labels['material'],
                                 cls_labels['tier'], cls_labels['size']])

        if (i + 1) % 50 == 0:
            print(f"  Generated {i + 1}/{args.num_images} images...")

    if args.task in ['classify', 'both']:
        csv_file.close()

    print(f"Done! Dataset saved to {args.out_dir}")


if __name__ == '__main__':
    main()
