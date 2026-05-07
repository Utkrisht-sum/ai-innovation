import sys
import unittest
from unittest.mock import MagicMock

# Create mock objects for the necessary libraries to allow import of dataset.py
mock_cv2 = MagicMock()
mock_np = MagicMock()
mock_torch = MagicMock()
mock_torch_utils_data = MagicMock()
mock_torch_utils_data.Dataset = object  # Needs to be a valid class for FASDataset to inherit from
mock_torchvision = MagicMock()
mock_torchvision_transforms = MagicMock()
mock_pil = MagicMock()

# We need to mock specific transforms so we can check if they are in the Compose
class MockTransform:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

class MockCompose(MockTransform): pass
class MockResize(MockTransform): pass
class MockRandomHorizontalFlip(MockTransform): pass
class MockColorJitter(MockTransform): pass
class MockToTensor(MockTransform): pass
class MockNormalize(MockTransform): pass

mock_torchvision_transforms.Compose = MockCompose
mock_torchvision_transforms.Resize = MockResize
mock_torchvision_transforms.RandomHorizontalFlip = MockRandomHorizontalFlip
mock_torchvision_transforms.ColorJitter = MockColorJitter
mock_torchvision_transforms.ToTensor = MockToTensor
mock_torchvision_transforms.Normalize = MockNormalize

# Inject the mocks into sys.modules
sys.modules['cv2'] = mock_cv2
sys.modules['numpy'] = mock_np
sys.modules['torch'] = mock_torch
sys.modules['torch.utils'] = MagicMock()
sys.modules['torch.utils.data'] = mock_torch_utils_data
sys.modules['torchvision'] = mock_torchvision
sys.modules['torchvision.transforms'] = mock_torchvision_transforms
sys.modules['PIL'] = mock_pil

# Now we can safely import dataset
# However, because dataset.py does `from torchvision import transforms`,
# and we mocked `torchvision`, it will get `mock_torchvision.transforms`
# We need to ensure that `mock_torchvision.transforms` IS `mock_torchvision_transforms`.
mock_torchvision.transforms = mock_torchvision_transforms

from dataset import get_transforms

class TestDatasetTransforms(unittest.TestCase):
    def test_get_transforms_train_includes_augmentations(self):
        # Call get_transforms with is_train=True
        compose_transform = get_transforms(is_train=True)

        # We expect a Compose object
        self.assertIsInstance(compose_transform, MockCompose)

        # We expect it to have a list of transforms as the first argument
        transforms_list = compose_transform.args[0]

        # Verify the exact order and types of transforms
        self.assertEqual(len(transforms_list), 5)
        self.assertIsInstance(transforms_list[0], MockResize)
        self.assertIsInstance(transforms_list[1], MockRandomHorizontalFlip)
        self.assertIsInstance(transforms_list[2], MockColorJitter)
        self.assertIsInstance(transforms_list[3], MockToTensor)
        self.assertIsInstance(transforms_list[4], MockNormalize)

    def test_get_transforms_eval_excludes_augmentations(self):
        # Call get_transforms with is_train=False
        compose_transform = get_transforms(is_train=False)

        # We expect a Compose object
        self.assertIsInstance(compose_transform, MockCompose)

        # We expect it to have a list of transforms as the first argument
        transforms_list = compose_transform.args[0]

        # Verify the exact order and types of transforms
        self.assertEqual(len(transforms_list), 3)
        self.assertIsInstance(transforms_list[0], MockResize)
        self.assertIsInstance(transforms_list[1], MockToTensor)
        self.assertIsInstance(transforms_list[2], MockNormalize)

if __name__ == '__main__':
    unittest.main()
