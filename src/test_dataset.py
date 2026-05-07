import unittest
import sys
from unittest.mock import MagicMock, call

# Mock required scientific modules that are not installed in the environment
mock_cv2 = MagicMock()
mock_numpy = MagicMock()
mock_torch = MagicMock()
mock_torch.utils = MagicMock()
mock_torch.utils.data = MagicMock()
mock_torchvision = MagicMock()
mock_transforms = MagicMock()
mock_pil = MagicMock()

sys.modules['cv2'] = mock_cv2
sys.modules['numpy'] = mock_numpy
sys.modules['torch'] = mock_torch
sys.modules['torch.utils'] = mock_torch.utils
sys.modules['torch.utils.data'] = mock_torch.utils.data
sys.modules['torchvision'] = mock_torchvision
sys.modules['torchvision.transforms'] = mock_transforms
sys.modules['PIL'] = mock_pil

# IMPORTANT: we need mock_torchvision.transforms to point to the SAME mock object
# so that the dataset.py code imports the right thing when doing `from torchvision import transforms`
mock_torchvision.transforms = mock_transforms

# Now import the target module
from dataset import get_transforms

class TestDatasetTransforms(unittest.TestCase):
    def setUp(self):
        # We need to make sure Compose returns something predictable to verify it
        self.mock_compose_return = MagicMock()
        mock_transforms.Compose.return_value = self.mock_compose_return

        # Setup return values for individual transforms so we can track them in Compose list
        self.mock_resize = MagicMock()
        self.mock_flip = MagicMock()
        self.mock_jitter = MagicMock()
        self.mock_tensor = MagicMock()
        self.mock_normalize = MagicMock()

        mock_transforms.Resize.return_value = self.mock_resize
        mock_transforms.RandomHorizontalFlip.return_value = self.mock_flip
        mock_transforms.ColorJitter.return_value = self.mock_jitter
        mock_transforms.ToTensor.return_value = self.mock_tensor
        mock_transforms.Normalize.return_value = self.mock_normalize

        # Reset mock call counts before each test
        mock_transforms.reset_mock()

    def test_get_transforms_train(self):
        """Test get_transforms when is_train=True"""
        result = get_transforms(is_train=True)

        # Verify result is the returned composition
        self.assertEqual(result, self.mock_compose_return)

        # Verify individual transforms were instantiated with correct arguments
        mock_transforms.Resize.assert_called_once_with((224, 224))
        mock_transforms.RandomHorizontalFlip.assert_called_once_with()
        mock_transforms.ColorJitter.assert_called_once_with(brightness=0.2, contrast=0.2)
        mock_transforms.ToTensor.assert_called_once_with()
        mock_transforms.Normalize.assert_called_once_with(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

        # Verify Compose was called once with the correct ordered list of transforms
        mock_transforms.Compose.assert_called_once_with([
            self.mock_resize,
            self.mock_flip,
            self.mock_jitter,
            self.mock_tensor,
            self.mock_normalize
        ])

    def test_get_transforms_eval(self):
        """Test get_transforms when is_train=False"""
        result = get_transforms(is_train=False)

        # Verify result is the returned composition
        self.assertEqual(result, self.mock_compose_return)

        # Verify individual transforms were instantiated with correct arguments
        mock_transforms.Resize.assert_called_once_with((224, 224))
        mock_transforms.ToTensor.assert_called_once_with()
        mock_transforms.Normalize.assert_called_once_with(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

        # Verify these were NOT called
        mock_transforms.RandomHorizontalFlip.assert_not_called()
        mock_transforms.ColorJitter.assert_not_called()

        # Verify Compose was called once with the correct ordered list of transforms
        mock_transforms.Compose.assert_called_once_with([
            self.mock_resize,
            self.mock_tensor,
            self.mock_normalize
        ])

if __name__ == '__main__':
    unittest.main()
