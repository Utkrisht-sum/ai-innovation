import unittest
from unittest.mock import patch, MagicMock
import sys
import os

# Create a mock for torch and other dependencies
mock_torch = MagicMock()
mock_torch.no_grad = MagicMock(return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock()))

mock_cv2 = MagicMock()
mock_np = MagicMock()
mock_plt = MagicMock()
mock_sklearn = MagicMock()
mock_sklearn_metrics = MagicMock()
mock_torchvision = MagicMock()
mock_pil = MagicMock()
mock_model = MagicMock()

sys.modules['torch'] = mock_torch
sys.modules['torch.utils'] = MagicMock()
sys.modules['torch.utils.data'] = MagicMock()
sys.modules['cv2'] = mock_cv2
sys.modules['numpy'] = mock_np
sys.modules['matplotlib'] = mock_plt
sys.modules['matplotlib.pyplot'] = mock_plt
sys.modules['sklearn'] = mock_sklearn
sys.modules['sklearn.metrics'] = mock_sklearn_metrics
sys.modules['torchvision'] = mock_torchvision
sys.modules['torchvision.transforms'] = MagicMock()
sys.modules['torchvision.transforms.functional'] = MagicMock()
sys.modules['PIL'] = mock_pil

# Mock project modules to isolate tests
sys.modules['model'] = mock_model
sys.modules['dataset'] = MagicMock()

import evaluate
import infer

class TestSecureDeserialization(unittest.TestCase):

    @patch('os.path.exists', return_value=True)
    def test_evaluate_secure_load(self, mock_exists):
        # Reset mock before test
        mock_torch.load.reset_mock()
        mock_model.FASModel.return_value.load_state_dict = MagicMock()
        mock_model.get_device.return_value = 'cpu'

        # We need to mock FASDataset to return a mock dataset with len > 0
        mock_dataset = MagicMock()
        mock_dataset.__len__.return_value = 1

        # Call evaluate function but mock DataLoader so it doesn't loop
        with patch('evaluate.FASDataset', return_value=mock_dataset), \
             patch('evaluate.DataLoader', return_value=[]), \
             patch('evaluate.accuracy_score', return_value=1.0), \
             patch('evaluate.confusion_matrix', return_value=MagicMock()), \
             patch('evaluate.ConfusionMatrixDisplay', return_value=MagicMock()), \
             patch('evaluate.roc_curve', return_value=(MagicMock(), MagicMock(), MagicMock())), \
             patch('evaluate.auc', return_value=1.0):
            evaluate.evaluate(model_path='dummy_model.pt', data_dir='dummy_data')

        # Verify torch.load was called with weights_only=True
        mock_torch.load.assert_called_with('dummy_model.pt', map_location='cpu', weights_only=True)
        print("evaluate.py torch.load verified successfully.")

    @patch('os.path.exists', return_value=True)
    @patch('infer.cv2.imread', return_value=MagicMock())
    def test_infer_single_image_secure_load(self, mock_imread, mock_exists):
        # Reset mock before test
        mock_torch.load.reset_mock()

        mock_model_instance = MagicMock()
        mock_model.FASModel.return_value = mock_model_instance
        mock_model.get_device.return_value = 'cpu'

        # Mock the model output so it returns a dummy tensor
        # which allows torch.softmax and other functions to run smoothly
        mock_output = MagicMock()
        mock_model_instance.return_value = mock_output
        mock_probs = MagicMock()
        mock_probs.__getitem__.return_value = MagicMock(__getitem__=MagicMock(return_value=MagicMock(item=MagicMock(return_value=0.9))))
        mock_torch.softmax.return_value = mock_probs

        # Prevent forward pass/gradcam to keep test simple and focused on deserialization
        with patch('infer.get_transforms', return_value=MagicMock()), \
             patch('infer.get_gradcam', return_value=MagicMock()), \
             patch('infer.overlay_gradcam', return_value=MagicMock()), \
             patch('infer.cv2.resize', return_value=MagicMock()), \
             patch('infer.cv2.putText', return_value=MagicMock()), \
             patch('infer.cv2.imwrite', return_value=MagicMock()):
            infer.infer_single_image(image_path='dummy_image.jpg', model_path='dummy_model.pt')

        # Verify torch.load was called with weights_only=True
        mock_torch.load.assert_called_with('dummy_model.pt', map_location='cpu', weights_only=True)
        print("infer.py (single image) torch.load verified successfully.")

    @patch('infer.cv2.VideoCapture')
    def test_run_webcam_secure_load(self, mock_videocapture):
        # Reset mock before test
        mock_torch.load.reset_mock()
        mock_model.FASModel.return_value.load_state_dict = MagicMock()
        mock_model.get_device.return_value = 'cpu'

        # Mock VideoCapture to return an object where isOpened() is False to exit early
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = False
        mock_videocapture.return_value = mock_cap

        with patch('infer.get_transforms', return_value=MagicMock()):
            infer.run_webcam(model_path='dummy_model.pt')

        # Verify torch.load was called with weights_only=True
        mock_torch.load.assert_called_with('dummy_model.pt', map_location='cpu', weights_only=True)
        print("infer.py (webcam) torch.load verified successfully.")

if __name__ == '__main__':
    unittest.main()
