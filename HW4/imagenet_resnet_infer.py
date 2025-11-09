import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import json
import os 
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kendalltau, spearmanr
import torch.nn.functional as F


# Load the pre-trained ResNet18 model
model = models.resnet18(pretrained=True)
model.eval()  # Set model to evaluation mode

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Define the image preprocessing transformations
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]   
    )
])

# Load the ImageNet class index mapping
with open("HW4/imagenet_class_index.json") as f:
    class_idx = json.load(f)
idx2label = [class_idx[str(k)][1] for k in range(len(class_idx))]
idx2synset = [class_idx[str(k)][0] for k in range(len(class_idx))]
id2label = {v[0]: v[1] for v in class_idx.values()}


imagenet_path = 'HW4/imagenet_samples/'


def lime_explanation(image_tensor, model, predicted_class, num_samples=1000, num_features=100):
    _, _, H, W = image_tensor.shape
    
    segment_size = 16 
    segments = np.zeros((H, W), dtype=int)
    segment_id = 0
    for i in range(0, H, segment_size):
        for j in range(0, W, segment_size):
            segments[i:i+segment_size, j:j+segment_size] = segment_id
            segment_id += 1
    
    num_segments = segment_id
    
    samples = []
    predictions = []
    
    for _ in range(num_samples):
        mask = np.random.randint(0, 2, size=num_segments)
        
        binary_mask = np.zeros((H, W))
        for seg_id in range(num_segments):
            binary_mask[segments == seg_id] = mask[seg_id]
        
        perturbed_image = image_tensor.clone()
        for c in range(3):
            perturbed_image[0, c] = perturbed_image[0, c] * torch.tensor(binary_mask).float().to(device)
        
        with torch.no_grad():
            output = model(perturbed_image)
            prob = F.softmax(output, dim=1)[0, predicted_class].item()
        
        samples.append(mask)
        predictions.append(prob)
    
    X = np.array(samples)
    y = np.array(predictions)
    
    lambda_reg = 1.0
    weights = np.linalg.solve(X.T @ X + lambda_reg * np.eye(num_segments), X.T @ y)
    
    importance_map = np.zeros((H, W))
    for seg_id in range(num_segments):
        importance_map[segments == seg_id] = weights[seg_id]
    
    return importance_map

def smoothgrad_explanation(image_tensor, model, predicted_class, num_samples=50, noise_level=0.15):
    gradients_sum = torch.zeros_like(image_tensor)
    
    for _ in range(num_samples):
        noise = torch.randn_like(image_tensor) * noise_level
        noisy_image = image_tensor + noise
        noisy_image.requires_grad = True
        
        output = model(noisy_image)
        
        score = output[0, predicted_class]
        
        model.zero_grad()
        score.backward()
        
        gradients_sum += noisy_image.grad.data
    
    smoothed_gradients = gradients_sum / num_samples
    
    attribution = torch.abs(smoothed_gradients).sum(dim=1).squeeze().cpu().numpy()
    
    return attribution

def visualize_explanations(original_image, lime_map, smoothgrad_map, predicted_label, img_name):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(original_image)
    axes[0].set_title(f'Original\nPrediction: {predicted_label}')
    axes[0].axis('off')
    
    im1 = axes[1].imshow(lime_map, cmap='RdBu_r', vmin=-np.abs(lime_map).max(), vmax=np.abs(lime_map).max())
    axes[1].set_title('LIME Explanation')
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1])
    
    im2 = axes[2].imshow(smoothgrad_map, cmap='hot')
    axes[2].set_title('SmoothGrad Explanation')
    axes[2].axis('off')
    plt.colorbar(im2, ax=axes[2])
    
    plt.tight_layout()
    plt.savefig(f'explanations_{img_name}.png', dpi=150, bbox_inches='tight')
    plt.show()

def compute_feature_rankings(lime_map, smoothgrad_map):
    lime_flat = lime_map.flatten()
    smoothgrad_flat = smoothgrad_map.flatten()
    
    lime_ranking = np.argsort(np.abs(lime_flat))[::-1]
    smoothgrad_ranking = np.argsort(np.abs(smoothgrad_flat))[::-1]
    
    return lime_ranking, smoothgrad_ranking

kendall_correlations = []
spearman_correlations = []

# List of image file paths
image_paths = os.listdir(imagenet_path)

for img_path in image_paths:
    # Open and preprocess the image
    my_img = os.path.join(imagenet_path, img_path)
    input_image = Image.open(my_img).convert('RGB')
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0).to(device)  # Add batch dimension and move to device

    # Perform inference
    with torch.no_grad():
        output = model(input_batch)

    # Get the predicted class index
    _, predicted_idx = torch.max(output, 1)
    predicted_idx = predicted_idx.item()
    predicted_synset = idx2synset[predicted_idx]
    predicted_label = idx2label[predicted_idx]

    print(f"Processing: {img_path}")
    print(f"Predicted label: {predicted_synset} ({predicted_label})")
    
    # Pass input_batch (which has shape [1, 3, 224, 224]) to explanation functions
    lime_map = lime_explanation(input_batch, model, predicted_idx, num_samples=500)
    smoothgrad_map = smoothgrad_explanation(input_batch, model, predicted_idx, num_samples=50)
    
    visualize_explanations(input_image, lime_map, smoothgrad_map, predicted_label, os.path.basename(img_path).split('.')[0])
    
    lime_ranking, smoothgrad_ranking = compute_feature_rankings(lime_map, smoothgrad_map)
    kendall_corr, _ = kendalltau(lime_ranking, smoothgrad_ranking)
    spearman_corr, _ = spearmanr(lime_ranking, smoothgrad_ranking)
    kendall_correlations.append(kendall_corr)
    spearman_correlations.append(spearman_corr)
    
    print(f"Kendall's Tau: {kendall_corr:.4f}")
    print(f"Spearman's Rho: {spearman_corr:.4f}\n")

avg_kendall = np.nanmean(kendall_correlations)
avg_spearman = np.nanmean(spearman_correlations)
print(f"Average Kendall's Tau: {avg_kendall:.4f}")
print(f"Average Spearman's Rho: {avg_spearman:.4f}")