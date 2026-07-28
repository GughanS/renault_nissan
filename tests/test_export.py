import os
import sys
import unittest
import subprocess

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

class TestExportONNX(unittest.TestCase):
    def test_export_script_runs(self):
        """Test that the ONNX export script runs successfully."""
        # Run export_onnx.py as a subprocess
        script_path = os.path.join(os.path.dirname(__file__), '..', 'scripts', 'export_onnx.py')
        export_dir = os.path.join(os.path.dirname(__file__), '..', 'exports_test')
        
        # Ensure clean directory
        if not os.path.exists(export_dir):
            os.makedirs(export_dir)
            
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        
        result = subprocess.run(
            [sys.executable, script_path, '--out-dir', export_dir],
            capture_output=True,
            text=True,
            env=env
        )
        
        self.assertEqual(result.returncode, 0, f"Export script failed:\n{result.stderr}")
        
        # Check that files were created
        self.assertTrue(os.path.exists(os.path.join(export_dir, 'wheeleye_detector.onnx')))
        self.assertTrue(os.path.exists(os.path.join(export_dir, 'wheeleye_classifier.onnx')))

    def test_exported_onnx_is_valid(self):
        """Verify exported ONNX models pass onnx.checker validation."""
        export_dir = os.path.join(os.path.dirname(__file__), '..', 'exports_test')
        det_path = os.path.join(export_dir, 'wheeleye_detector.onnx')
        cls_path = os.path.join(export_dir, 'wheeleye_classifier.onnx')

        if not os.path.exists(det_path):
            self.skipTest("ONNX exports not found — run test_export_script_runs first")

        import onnx
        for path in [det_path, cls_path]:
            model = onnx.load(path)
            # check_model raises on failure
            onnx.checker.check_model(model)

if __name__ == '__main__':
    unittest.main()
