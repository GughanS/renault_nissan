import argparse
import json
import os
import sys

# Ensure wheeleye module is available
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from wheeleye.pipeline.verifier import WheelEyeVerifier

def main():
    parser = argparse.ArgumentParser(description="Verify wheel assembly against manifest")
    parser.add_argument("--image", type=str, required=True, help="Path to the image to verify")
    parser.add_argument("--detector-weights", type=str, help="Path to detector weights", default="weights/best.pt")
    parser.add_argument("--classifier-weights", type=str, help="Path to classifier weights", default="weights/classify_best.pt")
    parser.add_argument("--material", type=str, required=True, choices=['Steel', 'Alloy'])
    parser.add_argument("--tier", type=str, required=True, choices=['Standard', 'Premium'])
    parser.add_argument("--size", type=str, required=True, choices=['17_inch', '18_inch', '19_inch'])
    parser.add_argument("--fasteners", type=int, default=5, help="Expected number of fasteners")
    
    args = parser.parse_args()
    
    manifest = {
        'material': args.material,
        'tier': args.tier,
        'size': args.size,
        'expected_fasteners': args.fasteners
    }
    
    print("Loading pipeline...")
    verifier = WheelEyeVerifier(
        detector_weights_path=args.detector_weights if os.path.exists(args.detector_weights) else None,
        classifier_weights_path=args.classifier_weights if os.path.exists(args.classifier_weights) else None,
    )
    
    print(f"Verifying {args.image}...")
    report = verifier.verify(args.image, manifest)
    
    print("\n" + "="*50)
    print(f"STATUS: {report['status']}")
    print("="*50)
    
    if report['messages']:
        print("MESSAGES:")
        for msg in report['messages']:
            print(f" - {msg}")
            
    print("\nDETECTIONS:")
    for d in report['detections']:
        print(f" - {d['class_name']} ({d['score']:.2f})")
        
    if report['classification']:
        print("\nCLASSIFICATION:")
        print(f" - Material: {report['classification']['material']}")
        print(f" - Tier: {report['classification']['tier']}")
        print(f" - Size: {report['classification']['size']}")

if __name__ == "__main__":
    main()
