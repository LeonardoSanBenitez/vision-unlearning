import os
import tempfile
from vision_unlearning.utils.model_management import save_model_card

def test_save_model_card():
    with tempfile.TemporaryDirectory() as temp_dir:
        save_model_card(
            repo_id = 'somewhere',
            base_model = 'somewhere',
            dataset_forget_name = 'somewhere',
            dataset_retain_name = 'somewhere',
            repo_folder = temp_dir,
        )
        assert os.path.exists(os.path.join(temp_dir, 'README.md'))
