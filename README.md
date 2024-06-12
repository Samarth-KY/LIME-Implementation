
# LIME Implementation

## Introduction

This repository contains an implementation of the LIME (Local Interpretable Model-agnostic Explanations) algorithm. LIME is used to explain predictions of any machine learning classifier by approximating it locally with an interpretable model.

## Algorithm Explanation

LIME aims to explain individual predictions of black-box machine learning models. It does so by:
1. Perturbing the input data and observing how the predictions change.
2. Fitting an interpretable model (like a linear model) to these perturbed samples and their predictions.
3. Using this interpretable model to explain the behavior of the black-box model locally.

### Key Steps in LIME:
1. **Input Perturbation**: Generate perturbed samples around the instance to be explained.
2. **Model Prediction**: Get predictions for these perturbed samples using the black-box model.
3. **Weight Calculation**: Assign weights to perturbed samples based on their similarity to the original instance.
4. **Interpretable Model**: Fit an interpretable model (e.g., linear regression) to the weighted perturbed samples.
5. **Explanation**: Use the interpretable model to explain the prediction of the original instance.

## Installation and Setup

You can install these libraries using `pip`:
```bash
pip install matplotlib scikit-image numpy torch torchvision scikit-learn statsmodels pillow
```

## Usage

### Preprocessing the Image

There are two preprocessing functions provided:
- `preprocess_image_inception(img)`: Preprocesses the image according to the requirements of InceptionV3.
- `preprocess_image_segment(img)`: Preprocesses the image for segmentation purposes.

### Loading and Displaying Images

Images are loaded from URLs and segmented using the `quickshift` algorithm. The segments are then displayed with their boundaries marked.

```python
# Example code to load and display segmented images
img1_url = 'https://example.com/image1.jpg'

# Load images
img1 = Image.open(BytesIO(urllib.request.urlopen(img1_url).read()))

# Preprocess and segment images
img1_np = preprocess_image_segment(img1)
segments1 = quickshift(img1_np, kernel_size=4, max_dist=200, ratio=0.2)

# Display segments
fig, axes = plt.subplots(1, 1, figsize=(15, 6))
axes[0].imshow(mark_boundaries(img1_np, segments1))
plt.show()
```


## Useful Links

- [LIME Paper](https://arxiv.org/abs/1602.04938): Original paper introducing the LIME algorithm.
- [skimage Quickshift](https://scikit-image.org/docs/dev/api/skimage.segmentation.html#skimage.segmentation.quickshift): Documentation for the `quickshift` segmentation algorithm.
- [Torchvision](https://pytorch.org/vision/stable/index.html): PyTorch library for image transformations and models.
