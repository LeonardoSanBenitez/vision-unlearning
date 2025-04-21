from typing import List, Literal, Dict
from PIL import Image
import torch
import torch_fidelity
from torchvision import transforms
from vision_unlearning.metrics.base import Metric
import os


class FrechetInceptionDistance(Metric):
    metrics: Literal['FID'] = ['FID']
    real_imgs_path: str = None
    gen_imgs_path: str = None
    real_imgs: List[torch.Tensor] = None
    gen_imgs: List[torch.Tensor] = None

    def model_post_init(self, __context: dict = None) -> None:
        assert self.real_imgs_path is not None or self.real_imgs is not None,\
            "Could not find real images data!\r\nPlease define a path to a folder or a torch.Tensor with the images."
        assert self.gen_imgs_path is not None or self.gen_imgs is not None,\
            "Could not find generated images data!\r\nPlease define a path to a folder or a torch.Tensor with the images."

        if self.real_imgs_path:
            assert self.verify_images_in_path(self.real_imgs_path), \
                f"No valid images found in the folder '{self.real_imgs_path}'."
        else:
            assert self.real_imgs.dim() == 4, \
                "The real images tensor should have 4 dimensions (batch_size, channels, height, width)."
        if self.gen_imgs_path:
            assert self.verify_images_in_path(self.gen_imgs_path), \
                f"No valid images found in the folder '{self.gen_imgs_path}'."
        else:
            assert self.gen_imgs.dim() == 4, \
                "The generated images tensor should have 4 dimensions (batch_size, channels, height, width)."
        pass

    def verify_images_in_path(self, path: str) -> bool:
        """
        Verifies if the given path contains image files.
        :param path: Path to the folder to check.
        :return: True if images are found, False otherwise.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"The path '{path}' does not exist.")
        if not os.path.isdir(path):
            raise NotADirectoryError(f"The path '{path}' is not a directory.")

        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        for file in os.listdir(path):
            if os.path.splitext(file)[1].lower() in valid_extensions:
                return True
        return False

    def process_tensor_images(self, tensor_list: List[torch.Tensor]):
        transform_tensor = transforms.Compose([
            transforms.Resize((299, 299)),  # Resize for Inception compatibility
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize to [-1, 1]
        ])
        
        return [transform_tensor(img) for img in tensor_list]
    
    def load_images_from_folder(self, folder_path: str, transform: transforms.Compose) -> torch.Tensor:
        """
        Loads all images from a folder, applies transformations, and returns them as a torch.Tensor.
        :param folder_path: Path to the folder containing images.
        :param transform: Transformations to apply to each image.
        :return: A torch.Tensor containing all images in the folder.
        """

        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_tensors = []

        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            if os.path.splitext(file_name)[1].lower() in valid_extensions:
                try:
                    image = Image.open(file_path).convert('RGB')  # Ensure 3-channel RGB
                    image_tensor = transform(image)
                    image_tensors.append(image_tensor)
                except Exception as e:
                    print(f"Error loading image {file_name}: {e}")

        if not image_tensors:
            raise ValueError(f"No valid images found in the folder '{folder_path}'.")

        return torch.stack(image_tensors)  # Stack into a single tensor

    def score(self) -> Dict[str, float]:
        scores: Dict[str, float] = {}

        transform = transforms.Compose([
            transforms.Resize((299, 299)),  # Resize for Inception compatibility
            transforms.ToTensor(),  # Convert image to tensor [0,1]
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize to [-1, 1]
        ])

        if self.real_imgs_path:
            real_images = self.real_imgs_path #self.load_images_from_folder(self.real_imgs_path, transform)
        else:
            real_images = self.process_tensor_images(self.real_imgs)

        if self.gen_imgs_path:
            gen_images = self.gen_imgs_path #self.load_images_from_folder(self.gen_imgs_path, transform)
        else:
            gen_images = self.process_tensor_images(self.gen_imgs)

        fid_val = torch_fidelity.calculate_metrics(
            input1=real_images,
            input2=gen_images,
            fid=True,
            #metrics=['fid'],
            cuda=torch.cuda.is_available(),
        )

        assert fid_val is not None
        scores['FID'] = float(fid_val['frechet_inception_distance'])

        return scores
        
