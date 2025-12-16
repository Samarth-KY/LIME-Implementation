"""
LIME Explainer Module
---------------------
A custom implementation of Local Interpretable Model-agnostic Explanations (LIME)
for computer vision models.
"""

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import models
from torchvision.transforms import v2
from PIL import Image
from skimage.segmentation import quickshift, mark_boundaries
from sklearn.linear_model import LinearRegression
from sklearn.metrics.pairwise import cosine_similarity
import statsmodels.api as sm
import matplotlib.pyplot as plt
import requests
import urllib.request
from io import BytesIO
from typing import Tuple, List

class LimeExplainer:
    def __init__(self, model: torch.nn.Module, num_perturbations: int = 300, device: str = "cpu"):
        """
        Initializes the LIME Explainer with a PyTorch model.
        
        Args:
            model: Pretrained PyTorch model (e.g., InceptionV3)
            num_perturbations: Number of perturbed samples to generate
            device: 'cpu' or 'cuda'
        """
        self.model = model.to(device)
        self.model.eval()
        self.num_perturbations = num_perturbations
        self.device = device
        
        # Standard InceptionV3 normalization
        self.transform_input = v2.Compose([
            v2.Resize(299),
            v2.CenterCrop(299),
            v2.ToImage(), 
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Visualization transform (keep as numpy for segmentation)
        self.transform_viz = v2.Compose([
            v2.Resize(299),
            v2.CenterCrop(299)
        ])

        # Load ImageNet labels for readability
        self.labels = self._load_imagenet_labels()

    def _load_imagenet_labels(self) -> List[str]:
        url = 'https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json'
        try:
            return requests.get(url).json()
        except Exception as e:
            print(f"Warning: Could not load labels. {e}")
            return []

    def load_image_from_url(self, url: str) -> np.ndarray:
        """Fetches an image from a URL and converts it to a numpy array."""
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as response:
            img = Image.open(BytesIO(response.read())).convert('RGB')
            img = self.transform_viz(img)
            return np.array(img)

    def _generate_perturbations(self, img_np: np.ndarray, segments: np.ndarray) -> Tuple[torch.Tensor, np.ndarray]:
        """Generates perturbed samples by masking superpixels."""
        num_superpixels = len(np.unique(segments))
        perturbations = []
        masks = []
        
        # Generate random binary masks
        # 1 = keep superpixel, 0 = hide superpixel
        random_masks = np.random.binomial(1, 0.5, size=(self.num_perturbations, num_superpixels))

        for mask in random_masks:
            temp_img = img_np.copy()
            # Find indices where mask is 0 (hidden)
            zeros = np.where(mask == 0)[0]
            # Create a boolean mask for pixels to hide
            mask_pixels = np.isin(segments, zeros)
            temp_img[mask_pixels] = 0
            
            # Prepare for model
            pil_img = Image.fromarray(temp_img)
            tensor_img = self.transform_input(pil_img)
            perturbations.append(tensor_img)
            masks.append(mask)

        return torch.stack(perturbations).to(self.device), np.array(masks)

    def _compute_weights(self, original_img_np: np.ndarray, perturbed_tensors: torch.Tensor) -> np.ndarray:
        """Computes weights based on cosine similarity to the original image."""
        original_tensor = self.transform_input(Image.fromarray(original_img_np))
        original_flat = original_tensor.flatten().cpu().numpy().reshape(1, -1)
        
        perturbed_flat = perturbed_tensors.flatten(start_dim=1).cpu().numpy()
        
        distances = cosine_similarity(original_flat, perturbed_flat)[0]
        # Kernel width (sigma)
        sigma = 0.25
        weights = np.exp(-(1 - distances)**2 / sigma**2)
        return weights

    def explain(self, img_np: np.ndarray, top_k: int = 2) -> dict:
        """
        Main function to generate LIME explanation.
        
        Returns:
            dict containing:
            - 'top_classes': List of (class_name, probability)
            - 'explanations': List of weighted images (numpy arrays) for each top class
            - 'segments': Integer array of superpixel labels for the image, used for visualization.
        """
        # 1. Segment Image
        segments = quickshift(img_np, kernel_size=4, max_dist=200, ratio=0.2)
        
        # 2. Generate Perturbations
        perturbed_tensors, binary_masks = self._generate_perturbations(img_np, segments)
        
        # 3. Get Model Predictions
        with torch.no_grad():
            outputs = self.model(perturbed_tensors)
            probabilities = F.softmax(outputs, dim=1)
        
        # 4. Get Top Predictions for Original Image
        original_tensor = self.transform_input(Image.fromarray(img_np)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            orig_output = self.model(original_tensor)
            orig_probs = F.softmax(orig_output, dim=1)
        
        top_probs, top_indices = orig_probs.topk(top_k, dim=1)
        
        results = {
            "top_classes": [],
            "explanations": []
        }
        
        # 5. Calculate Weights (Similarity)
        weights = self._compute_weights(img_np, perturbed_tensors)

        # 6. Fit Linear Model for each Top Class
        feature_matrix = sm.add_constant(binary_masks)
        
        for i in range(top_k):
            class_idx = top_indices[0, i].item()
            class_prob = top_probs[0, i].item()
            class_name = self.labels[class_idx] if self.labels else str(class_idx)
            
            results["top_classes"].append((class_name, class_prob))
            
            # Get probabilities for this specific class across all perturbations
            target_probs = probabilities[:, class_idx].cpu().numpy()
            
            # Weighted Linear Regression
            model_reg = sm.WLS(target_probs, feature_matrix, weights=weights)
            fitted = model_reg.fit()
            
            # Get top superpixels
            coef = fitted.params[1:] # ignore intercept
            top_superpixels = np.argsort(-np.abs(coef))[:4] # Top 4 features
            
            # Create Explanation Mask
            mask = np.zeros_like(segments, dtype=np.uint8)
            for sp_idx in top_superpixels:
                mask[segments == sp_idx] = 1
                
            explanation_img = img_np * mask[:, :, np.newaxis]
            results["explanations"].append(explanation_img)

        return {
            "top_classes": results["top_classes"],
            "explanations": results["explanations"],
            "segments": segments
        }

    def visualize(self, img_np: np.ndarray, results: dict):
        segments = results["segments"]
        explanations = results["explanations"]
        top_classes = results["top_classes"]

        num_classes = len(explanations)

        fig, axes = plt.subplots(
            1,
            num_classes + 2,
            figsize=(5 * (num_classes + 2), 5),
            constrained_layout=True
        )

        # Original image
        axes[0].imshow(img_np)
        axes[0].set_title("Original")
        axes[0].axis("off")

        # All superpixels
        all_sp = mark_boundaries(img_np, segments)
        axes[1].imshow(all_sp)
        axes[1].set_title("All superpixels")
        axes[1].axis("off")

        # Top-k explanations
        for i, (expl_img, (cls_name, prob)) in enumerate(
            zip(explanations, top_classes)
        ):
            ax = axes[i + 2]
            ax.imshow(expl_img)
            ax.set_title(f"{cls_name}\nProb {prob:.2f}")
            ax.axis("off")

        plt.show()


if __name__ == "__main__":
    # Initialize Model
    inception = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT)
    
    # Initialize Explainer
    explainer = LimeExplainer(model=inception)
    
    # Run on an a few examples
    url = 'https://heavyequipmentcollege.edu/wp-content/uploads/2021/03/how-forklift-works-hec-scaled-1.jpg'
    
    print(f"Processing {url}...")
    
    img_np = explainer.load_image_from_url(url)
    results = explainer.explain(img_np, top_k=2)
    
    explainer.visualize(img_np, results)