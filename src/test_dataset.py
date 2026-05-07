import sys
import unittest
from unittest.mock import MagicMock, patch

# Mock torchvision before importing dataset
mock_torchvision = MagicMock()
mock_transforms = MagicMock()

# Set up the Compose mock to just return the list of transforms passed to it
# so we can easily inspect them
def mock_compose(transforms_list):
    return transforms_list
mock_transforms.Compose = mock_compose

# Create specific mock classes/functions for the transforms so we can identify them
class MockResize:
    def __init__(self, *args, **kwargs):
        self.name = 'Resize'

class MockRandomHorizontalFlip:
    def __init__(self, *args, **kwargs):
        self.name = 'RandomHorizontalFlip'

class MockColorJitter:
    def __init__(self, *args, **kwargs):
        self.name = 'ColorJitter'

class MockToTensor:
    def __init__(self, *args, **kwargs):
        self.name = 'ToTensor'

class MockNormalize:
    def __init__(self, *args, **kwargs):
        self.name = 'Normalize'

mock_transforms.Resize = MockResize
mock_transforms.RandomHorizontalFlip = MockRandomHorizontalFlip
mock_transforms.ColorJitter = MockColorJitter
mock_transforms.ToTensor = MockToTensor
mock_transforms.Normalize = MockNormalize

mock_torchvision.transforms = mock_transforms

# Mock other modules that might be imported by dataset.py
sys.modules['torchvision'] = mock_torchvision
sys.modules['torchvision.transforms'] = mock_transforms
sys.modules['cv2'] = MagicMock()
sys.modules['numpy'] = MagicMock()
sys.modules['torch'] = MagicMock()
sys.modules['torch.utils'] = MagicMock()
sys.modules['torch.utils.data'] = MagicMock()
sys.modules['PIL'] = MagicMock()

# Now we can import the module to test
from dataset import get_transforms

class TestGetTransforms(unittest.TestCase):

    def test_get_transforms_train(self):
        """Test get_transforms with is_train=True includes augmentation."""
        transforms_list = get_transforms(is_train=True)

        # Extract names to make assertions easier
        transform_names = [t.name if hasattr(t, 'name') else str(type(t)) for t in transforms_list]

        # Verify length
        self.assertEqual(len(transform_names), 5)

        # Verify exact order and inclusion
        self.assertEqual(transform_names[0], 'Resize')
        self.assertEqual(transform_names[1], 'RandomHorizontalFlip')
        self.assertEqual(transform_names[2], 'ColorJitter')
        self.assertEqual(transform_names[3], 'ToTensor')
        self.assertEqual(transform_names[4], 'Normalize')

    def test_get_transforms_eval(self):
        """Test get_transforms with is_train=False excludes augmentation."""
        transforms_list = get_transforms(is_train=False)

        # Extract names to make assertions easier
        transform_names = [t.name if hasattr(t, 'name') else str(type(t)) for t in transforms_list]

        # Verify length
        self.assertEqual(len(transform_names), 3)

        # Verify exact order and inclusion
        self.assertEqual(transform_names[0], 'Resize')
        self.assertEqual(transform_names[1], 'ToTensor')
        self.assertEqual(transform_names[2], 'Normalize')

        # Verify exclusion
        self.assertNotIn('RandomHorizontalFlip', transform_names)
        self.assertNotIn('ColorJitter', transform_names)

if __name__ == '__main__':
    unittest.main()
