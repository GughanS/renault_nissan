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

if __name__ == '__main__':
    unittest.main()
