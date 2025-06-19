import os
import sys
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import colorsys
import random
import torch
import torchvision.transforms as T
from typing import List, Dict, Tuple, Any, Optional
import requests
from io import BytesIO




class EnhancedInstagramGridLayoutGenerator:
    def __init__(self, image_folder: str):

        self.image_folder = image_folder
        self.images = []
        self.original_images = []
        self.image_paths = []
        self.color_profiles = []
        self.layout_options = []
        self.image_features = []
        
        # Initialize DINOv2 model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        self.model = None
        self.transform = None
        
    def _load_dinov2_model(self):
        """Load the DINOv2 model for image feature extraction."""
        try:
            # Try to import the required libraries
            try:
                print("Attempting to import transformers...")
                from transformers import AutoImageProcessor, AutoModel
                print("Successfully imported transformers!")
                
                # Define transforms for preprocessing
                self.transform = T.Compose([
                    T.Resize((224, 224)),
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
                
                # Try to load the model
                try:
                    print("Loading DINOv2 model (this may take a moment)...")
                    self.processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
                    self.model = AutoModel.from_pretrained("facebook/dinov2-base").to(self.device)
                    print("DINOv2 model loaded successfully!")
                    return True
                except Exception as e:
                    print(f"Error loading DINOv2 model: {e}")
                    print("Will fall back to color analysis only")
                    return False
                    
            except ImportError as e:
                print(f"Error importing transformers: {e}")
                print("Please install required packages with: pip install transformers torch torchvision")
                print("Falling back to color analysis only.")
                return False
        except Exception as e:
            print(f"Unexpected error during DINOv2 setup: {e}")
            return False
        
    def load_images(self):
        """Load and crop images from the specified folder."""
        valid_extensions = ['.jpg', '.jpeg', '.png']
        
        try:
            # Debugging info
            print(f"Loading images from folder: {self.image_folder}")
            files_in_dir = os.listdir(self.image_folder)
            print(f"Files in directory: {files_in_dir}")
            
            # Clear existing data
            self.images = []
            self.original_images = []
            self.image_paths = []
            self.color_profiles = []
            self.layout_options = []
            self.image_features = []
            
            # Try to load DINOv2 model but don't require it
            if self.model is None:
                self._load_dinov2_model()  # Ignore the result
            
            # Load images from folder
            for filename in os.listdir(self.image_folder):
                print(f"Checking file: {filename}")
                if any(filename.lower().endswith(ext) for ext in valid_extensions):
                    image_path = os.path.join(self.image_folder, filename)
                    try:
                        print(f"Loading image: {image_path}")
                        # Load original image
                        img = Image.open(image_path)
                        
                        # Store original image
                        self.original_images.append(img.copy())
                        
                        # Create square crop of the image
                        img_square = self._crop_square(img)
                        
                        self.images.append(img_square)
                        self.image_paths.append(image_path)
                        print(f"Successfully loaded image: {filename}")
                    except Exception as e:
                        print(f"Error loading image {filename}: {e}")
                        
            print(f"Loaded {len(self.images)} images")
            
            # Return success if we have any images
            return len(self.images) > 0
                
        except Exception as e:
            print(f"Error accessing folder {self.image_folder}: {e}")
            return False
            
    def _crop_square(self, img: Image.Image) -> Image.Image:
        """
        Crop an image to a square, preserving the center.
        
        Args:
            img (PIL.Image): The input image
            
        Returns:
            PIL.Image: Square cropped image
        """
        width, height = img.size
        
        # Determine the size of the square
        size = min(width, height)
        
        # Calculate coordinates for center crop
        left = (width - size) // 2
        top = (height - size) // 2
        right = left + size
        bottom = top + size
        
        # Crop and return
        return img.crop((left, top, right, bottom))
    
    def _extract_image_features(self):
        """Extract image features using DINOv2 model."""
        if not self.images:
            print("No images loaded. Please load images first.")
            return []
        
        print("Extracting image features...")
        features = []
        
        # Check if model is loaded
        if self.model is None:
            print("DINOv2 model not loaded. Returning empty features.")
            # Create dummy features
            for _ in self.images:
                features.append(np.random.rand(768))  # DINOv2 base has 768-dim embeddings
            self.image_features = features
            return features
        
        try:
            with torch.no_grad():
                for img in self.images:
                    # Convert PIL image to tensor
                    if self.transform:
                        img_tensor = self.transform(img).unsqueeze(0).to(self.device)
                        
                        # Get DINOv2 features
                        outputs = self.model(img_tensor)
                        
                        # Get CLS token as image embedding
                        img_embedding = outputs.last_hidden_state[:, 0].cpu().numpy()[0]
                        features.append(img_embedding)
            
            self.image_features = features
            print(f"Extracted {len(features)} feature vectors")
            return features
        except Exception as e:
            print(f"Error extracting features: {e}")
            # Fallback to color analysis
            print("Falling back to color-based features...")
            # Create dummy features
            for _ in self.images:
                features.append(np.random.rand(768))  # DINOv2 base has 768-dim embeddings
            self.image_features = features
            return features
    
    def extract_dominant_colors(self, num_colors=3):
        """
        Extract dominant colors from each image.
        
        Args:
            num_colors (int): Number of dominant colors to extract per image
        """
        self.color_profiles = []
        
        for img in self.images:
            # Resize image to speed up processing
            img_small = img.resize((100, 100))
            img_array = np.array(img_small)
            
            # Reshape the array for KMeans
            reshaped_array = img_array.reshape(-1, 3)
            
            # Apply KMeans to find dominant colors
            kmeans = KMeans(n_clusters=num_colors)
            kmeans.fit(reshaped_array)
            
            # Get the RGB values of the centroids
            colors = kmeans.cluster_centers_.astype(int)
            
            # Convert to hex for better visualization
            hex_colors = ['#%02x%02x%02x' % (r, g, b) for r, g, b in colors]
            
            # Store both RGB and hex values
            self.color_profiles.append({
                'rgb': colors,
                'hex': hex_colors
            })
            
        return self.color_profiles
    
    def select_best_images(self, target_count=9):
        """
        Select the best images using DINOv2 features and other quality metrics
        
        Args:
            target_count (int): Number of images to select (default 9)
        
        Returns:
            list: Indices of selected images
        """
        if len(self.images) <= target_count:
            return list(range(len(self.images)))
        
        # Ensure features are extracted
        if not self.image_features:
            self._extract_image_features()
        
        image_scores = []
        
        for i, img in enumerate(self.images):
            score = 0
            
            # 1. Basic image quality (30% weight)
            width, height = img.size
            resolution_score = min((width * height) / 1000000, 2)  # Cap at 2MP
            
            # Aspect ratio score (prefer square or close to square)
            aspect_ratio = max(width, height) / min(width, height)
            aspect_score = max(0, 2 - aspect_ratio)  # Perfect square = 2, very wide = 0
            
            # Brightness and contrast
            gray_img = img.convert('L')
            brightness = np.mean(np.array(gray_img))
            contrast = np.std(np.array(gray_img))
            
            brightness_score = 1 - abs(brightness - 128) / 128  # Prefer moderate brightness
            contrast_score = min(contrast / 64, 1)  # Prefer good contrast
            
            quality_score = (resolution_score * 0.4 + aspect_score * 0.2 + 
                            brightness_score * 0.2 + contrast_score * 0.2)
            score += quality_score * 0.3
            
            # 2. DINOv2-based content quality (40% weight)
            if self.image_features and i < len(self.image_features):
                feature_vector = np.array(self.image_features[i])
                
                # Feature diversity score - measure how "interesting" the content is
                feature_variance = np.var(feature_vector)
                feature_diversity = min(feature_variance / 0.1, 1)  # Normalize
                
                # Feature magnitude - indicates content richness
                feature_magnitude = np.linalg.norm(feature_vector)
                magnitude_score = min(feature_magnitude / 100, 1)  # Normalize
                
                # Semantic content score (higher values often indicate more complex/interesting content)
                semantic_score = min(np.mean(np.abs(feature_vector)), 1)
                
                dinov2_score = (feature_diversity * 0.4 + magnitude_score * 0.3 + semantic_score * 0.3)
                score += dinov2_score * 0.4
            
            # 3. Color diversity and harmony (20% weight)
            if i < len(self.color_profiles):
                colors = self.color_profiles[i]['rgb']
                color_variance = np.var(colors.flatten())
                color_diversity = min(color_variance / 5000, 1)
                score += color_diversity * 0.2
            
            # 4. Uniqueness within the set (10% weight)
            uniqueness_score = 0
            if self.image_features and i < len(self.image_features):
                similarities = []
                for j, other_feature in enumerate(self.image_features):
                    if i != j:
                        similarity = self.calculate_feature_similarity(i, j)
                        similarities.append(similarity)
                
                if similarities:
                    # Prefer images that are neither too similar nor too different
                    avg_similarity = np.mean(similarities)
                    uniqueness_score = 1 - min(abs(avg_similarity - 0.5) * 2, 1)
            
            score += uniqueness_score * 0.1
            
            image_scores.append((i, score, {
                'quality': quality_score,
                'dinov2': dinov2_score if 'dinov2_score' in locals() else 0,
                'color': color_diversity if 'color_diversity' in locals() else 0,
                'uniqueness': uniqueness_score
            }))
        
        # Sort by score and select top images
        image_scores.sort(key=lambda x: x[1], reverse=True)
        selected_indices = [idx for idx, score, details in image_scores[:target_count]]
        
        # Print selection details for debugging
        print(f"\nSelected {len(selected_indices)} best images from {len(self.images)}:")
        for i, (idx, score, details) in enumerate(image_scores[:target_count]):
            print(f"  #{i+1}: Image {idx+1} (Score: {score:.3f}) - Quality: {details['quality']:.2f}, "
                f"Content: {details['dinov2']:.2f}, Color: {details['color']:.2f}, Unique: {details['uniqueness']:.2f}")
        
        return selected_indices
    
    def calculate_color_harmony(self, img1_colors, img2_colors):
        """
        Calculate color harmony between two images based on their color profiles.
        
        Args:
            img1_colors (dict): Color profile of first image
            img2_colors (dict): Color profile of second image
            
        Returns:
            float: Harmony score (lower is more harmonious)
        """
        # Convert RGB to HSV for better color comparison
        def rgb_to_hsv_list(rgb_list):
            hsv_list = []
            for rgb in rgb_list:
                r, g, b = rgb
                h, s, v = colorsys.rgb_to_hsv(r/255, g/255, b/255)
                hsv_list.append((h, s, v))
            return hsv_list
        
        hsv1 = rgb_to_hsv_list(img1_colors['rgb'])
        hsv2 = rgb_to_hsv_list(img2_colors['rgb'])
        
        # Calculate minimum color differences across all color combinations
        min_diff = float('inf')
        for color1 in hsv1:
            for color2 in hsv2:
                # Hue is circular, so calculate the shortest distance
                h1, s1, v1 = color1
                h2, s2, v2 = color2
                
                # Hue distance (circular)
                h_diff = min(abs(h1 - h2), 1 - abs(h1 - h2))
                
                # Saturation and value distances
                s_diff = abs(s1 - s2)
                v_diff = abs(v1 - v2)
                
                # Weighted difference (hue matters most for visual harmony)
                diff = (h_diff * 5) + s_diff + v_diff
                
                min_diff = min(min_diff, diff)
                
        return min_diff
    
    def calculate_feature_similarity(self, idx1, idx2):
        """
        Calculate similarity between two images based on their DINOv2 features.
        
        Args:
            idx1 (int): Index of first image
            idx2 (int): Index of second image
            
        Returns:
            float: Similarity score (higher is more similar)
        """
        if not self.image_features or idx1 >= len(self.image_features) or idx2 >= len(self.image_features):
            # Fall back to color harmony if features are not available
            if self.color_profiles:
                return -self.calculate_color_harmony(self.color_profiles[idx1], self.color_profiles[idx2])
            return 0
        
        # Compute cosine similarity between feature vectors
        feat1 = self.image_features[idx1]
        feat2 = self.image_features[idx2]
        
        # Normalize vectors
        norm1 = np.linalg.norm(feat1)
        norm2 = np.linalg.norm(feat2)
        
        if norm1 == 0 or norm2 == 0:
            return 0
            
        # Compute cosine similarity
        cosine_similarity = np.dot(feat1, feat2) / (norm1 * norm2)
        
        return cosine_similarity
    
    def generate_layouts(self, num_layouts=3):
        """
        Generate different layout options based on image features and color harmony.
        
        Args:
            num_layouts (int): Number of layout options to generate
            
        Returns:
            list: Different layout options
        """
        # We need at least 9 images for a standard Instagram grid
        if len(self.images) < 9:
            print(f"Warning: Only {len(self.images)} images found. A standard Instagram grid needs 9 images.")
            needed = 9 - len(self.images)
            print(f"Please add {needed} more images for optimal results.")
            
        # Smart selection of best images if we have more than 9
        if len(self.images) > 9:
            print(f"Selecting best 9 images from {len(self.images)} uploaded images...")
            indices = self.select_best_images(target_count=9)
            images_to_use = len(indices)
            print(f"Selected images at positions: {indices}")
        else:
            images_to_use = len(self.images)
            indices = list(range(images_to_use))
        
        # Calculate similarity scores between all image pairs
        similarity_scores = {}
        for i in range(images_to_use):
            for j in range(i+1, images_to_use):
                score = self.calculate_feature_similarity(i, j)
                similarity_scores[(i, j)] = score
                similarity_scores[(j, i)] = score
        
        # Generate different layout options
        self.layout_options = []
        
        # Option 1: Visual Similarity Chain - arrange by maximizing similarity between adjacent images
        current_layout = [0]  # Start with the first image
        remaining = set(indices) - {0}
        
        while remaining:
            last_idx = current_layout[-1]
            # Choose the most similar image from remaining ones
            next_idx = max(remaining, key=lambda x: similarity_scores.get((last_idx, x), -float('inf')))
            current_layout.append(next_idx)
            remaining.remove(next_idx)
            
        self.layout_options.append({
            'name': 'Visual Similarity Flow',
            'layout': current_layout,
            'description': 'Images arranged to create a smooth visual flow based on image content similarity'
        })
        
        # Option 2: Group similar images together using KMeans clustering
        if self.image_features:
            # Create feature matrix
            feature_matrix = np.array(self.image_features[:images_to_use])
            
            # Apply KMeans to group similar images
            n_clusters = min(3, images_to_use)  # Group into 3 clusters or less if fewer images
            kmeans = KMeans(n_clusters=n_clusters)
            clusters = kmeans.fit_predict(feature_matrix)
            
            # Sort indices by cluster
            clustered_layout = []
            for cluster_id in range(n_clusters):
                cluster_images = [i for i, c in enumerate(clusters) if c == cluster_id and i < images_to_use]
                clustered_layout.extend(cluster_images)
                
            self.layout_options.append({
                'name': 'Content Grouping',
                'layout': clustered_layout,
                'description': 'Images grouped by similar visual content and style'
            })
        else:
            # Fallback to color-based grouping
            # Extract dominant colors if not already done
            if not self.color_profiles:
                self.extract_dominant_colors()
                
            # Create a feature vector for each image using its dominant colors
            color_features = []
            for profile in self.color_profiles[:images_to_use]:
                # Flatten RGB values into a feature vector
                features = profile['rgb'].flatten()
                color_features.append(features)
                
            # Cluster images by color
            kmeans = KMeans(n_clusters=min(3, images_to_use))
            clusters = kmeans.fit_predict(color_features)
            
            # Sort indices by cluster
            clustered_layout = []
            for cluster_id in range(min(3, images_to_use)):
                cluster_images = [i for i, c in enumerate(clusters) if c == cluster_id and i < images_to_use]
                clustered_layout.extend(cluster_images)
                
            self.layout_options.append({
                'name': 'Color Grouping',
                'layout': clustered_layout,
                'description': 'Images grouped by similar color profiles'
            })
        
        # Option 3: Create an aesthetic contrast layout
        # Sort images by brightness and contrast
        brightness_scores = []
        for img in self.images[:images_to_use]:
            # Convert to grayscale and calculate average pixel value (brightness)
            gray_img = img.convert('L')
            brightness = np.mean(np.array(gray_img))
            
            # Calculate contrast (standard deviation of pixel values)
            contrast = np.std(np.array(gray_img))
            
            # Combined score (normalize both factors)
            score = brightness / 255 + contrast / 128
            brightness_scores.append(score)
            
        contrast_layout = [i for i, _ in sorted(enumerate(brightness_scores), key=lambda x: x[1])]
        
        self.layout_options.append({
            'name': 'Brightness Flow',
            'layout': contrast_layout,
            'description': 'Images arranged to create a flow from darker to brighter images'
        })
        
        # Generate additional random layouts if requested
        while len(self.layout_options) < num_layouts:
            random_layout = list(indices)
            random.shuffle(random_layout)
            
            self.layout_options.append({
                'name': f'Creative Mix {len(self.layout_options) - 2}',
                'layout': random_layout,
                'description': 'Creatively mixed arrangement for unexpected visual impact'
            })
            
        return self.layout_options
    
    def visualize_layout(self, layout_index=0):
        """
        Visualize a specific layout option.
        
        Args:
            layout_index (int): Index of the layout to visualize
        """
        if not self.layout_options:
            print("No layouts generated yet. Call generate_layouts() first.")
            return
        
        if layout_index >= len(self.layout_options):
            print(f"Layout index {layout_index} out of range. Only {len(self.layout_options)} layouts available.")
            return
        
        layout = self.layout_options[layout_index]
        layout_indices = layout['layout']
        
        # Create a 3x3 grid for Instagram
        fig, axes = plt.subplots(3, 3, figsize=(10, 10))
        fig.suptitle(f"Layout: {layout['name']}", fontsize=16)
        plt.figtext(0.5, 0.01, layout['description'], ha='center', fontsize=12)
        
        # Disable axes
        for ax in axes.flatten():
            ax.axis('off')
        
        # Place images according to layout
        for i, idx in enumerate(layout_indices):
            if i >= 9:  # Only show first 9 images
                break
                
            row, col = i // 3, i % 3
            if idx < len(self.images):
                axes[row, col].imshow(self.images[idx])
                axes[row, col].set_title(f"Image {idx+1}")
            
        plt.tight_layout()
        plt.subplots_adjust(top=0.9, bottom=0.1)
        plt.show()
        
    def save_layout_visualization(self, output_folder, layout_index=0):
        """
        Save visualization of a specific layout option.
        
        Args:
            output_folder (str): Folder to save visualization
            layout_index (int): Index of the layout to visualize
        """
        if not self.layout_options:
            print("No layouts generated yet. Call generate_layouts() first.")
            return
        
        if layout_index >= len(self.layout_options):
            print(f"Layout index {layout_index} out of range. Only {len(self.layout_options)} layouts available.")
            return
        
        # Create output folder if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)
        
        layout = self.layout_options[layout_index]
        layout_indices = layout['layout']
        
        # Create a 3x3 grid for Instagram
        fig, axes = plt.subplots(3, 3, figsize=(10, 10))
        fig.suptitle(f"Layout: {layout['name']}", fontsize=16)
        plt.figtext(0.5, 0.01, layout['description'], ha='center', fontsize=12)
        
        # Disable axes
        for ax in axes.flatten():
            ax.axis('off')
        
        # Place images according to layout
        for i, idx in enumerate(layout_indices):
            if i >= 9:  # Only show first 9 images
                break
                
            row, col = i // 3, i % 3
            if idx < len(self.images):
                axes[row, col].imshow(self.images[idx])
                axes[row, col].set_title(f"Image {idx+1}")
            
        plt.tight_layout()
        plt.subplots_adjust(top=0.9, bottom=0.1)
        
        # Save the figure
        output_path = os.path.join(output_folder, f"layout_{layout_index}_{layout['name'].replace(' ', '_')}.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Layout visualization saved to {output_path}")
        
        return output_path
        
    def generate_layout_preview(self, layout_index=0):
        """
        Generate a preview image of the layout for display in Streamlit.
        
        Args:
            layout_index (int): Index of the layout to visualize
            
        Returns:
            PIL.Image: Combined grid image
        """
        if not self.layout_options or layout_index >= len(self.layout_options):
            return None
        
        layout = self.layout_options[layout_index]
        layout_indices = layout['layout']
        
        # Calculate the size of the grid
        img_size = 300  # Size of each image in the grid
        grid_size = img_size * 3  # 3x3 grid
        
        # Create a new blank image for the grid
        grid_img = Image.new('RGB', (grid_size, grid_size), color=(255, 255, 255))
        
        # Place images according to layout
        for i, idx in enumerate(layout_indices):
            if i >= 9:  # Only use first 9 images
                break
                
            if idx < len(self.images):
                # Resize image to fit in the grid
                img = self.images[idx].resize((img_size, img_size))
                
                # Calculate position in grid
                row, col = i // 3, i % 3
                pos_x = col * img_size
                pos_y = row * img_size
                
                # Paste image into grid
                grid_img.paste(img, (pos_x, pos_y))
        
        return grid_img


# Example usage
if __name__ == "__main__":
    # Initialize the generator with a folder containing images
    generator = EnhancedInstagramGridLayoutGenerator("sample_images")
    
    # Load images
    if generator.load_images():
        # Generate layout options
        layouts = generator.generate_layouts(num_layouts=5)
        
        # Visualize the first layout
        generator.visualize_layout(0)
        
        # Save all layout visualizations
        for i in range(len(layouts)):
            generator.save_layout_visualization("output_layouts", i)
    else:
        print("Failed to load images. Please check the image folder path.")
