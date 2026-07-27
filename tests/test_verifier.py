import os
import sys
import unittest
from PIL import Image

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from wheeleye.pipeline.verifier import WheelEyeVerifier

class TestWheelEyeVerifier(unittest.TestCase):
    def setUp(self):
        # Create a dummy image
        self.dummy_image_path = "dummy_test_image.jpg"
        img = Image.new('RGB', (800, 600), color='gray')
        img.save(self.dummy_image_path)
        
        # Initialize verifier without weights (random initialization)
        # This will just test that the pipeline can run end-to-end without crashing.
        self.verifier = WheelEyeVerifier()
        
    def tearDown(self):
        if os.path.exists(self.dummy_image_path):
            os.remove(self.dummy_image_path)
            
    def test_pipeline_end_to_end(self):
        manifest = {
            'material': 'Alloy',
            'tier': 'Premium',
            'size': '18_inch',
            'expected_fasteners': 5
        }
        
        # Since it's random weights, we might not detect a wheel.
        # But the code should handle "no wheel detected" gracefully.
        report = self.verifier.verify(self.dummy_image_path, manifest)
        
        self.assertIn('status', report)
        self.assertIn('messages', report)
        self.assertIn('detections', report)
        self.assertIn('classification', report)
        
        # We expect a FAIL because random weights likely won't predict exactly what we need,
        # or won't find a wheel at all.
        self.assertEqual(report['status'], 'FAIL')

if __name__ == '__main__':
    unittest.main()
