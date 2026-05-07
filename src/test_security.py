import unittest
import ast
import os
import glob

class TestSecurity(unittest.TestCase):
    def test_torch_load_secure(self):
        """
        Test that all calls to torch.load have weights_only=True.
        """
        src_dir = os.path.dirname(os.path.abspath(__file__))
        python_files = glob.glob(os.path.join(src_dir, '*.py'))

        for file_path in python_files:
            with open(file_path, 'r', encoding='utf-8') as f:
                tree = ast.parse(f.read(), filename=file_path)

            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    # Check if it's a call to torch.load
                    is_torch_load = False
                    if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
                        if node.func.value.id == 'torch' and node.func.attr == 'load':
                            is_torch_load = True

                    if is_torch_load:
                        # Check if weights_only=True is passed as a keyword argument
                        has_weights_only = False
                        for kw in node.keywords:
                            if kw.arg == 'weights_only':
                                if isinstance(kw.value, ast.Constant) and kw.value.value is True:
                                    has_weights_only = True
                                elif isinstance(kw.value, ast.NameConstant) and kw.value.value is True: # For older python versions
                                    has_weights_only = True

                        self.assertTrue(
                            has_weights_only,
                            f"Insecure torch.load call found in {file_path} at line {node.lineno}. Missing weights_only=True"
                        )

if __name__ == '__main__':
    unittest.main()
