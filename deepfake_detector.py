import os
import numpy as np
import cv2
from PIL import Image
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from datetime import datetime
import hashlib
import random

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Check CUDA availability
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {DEVICE}")

# Image preprocessing
def preprocess_image(image_path):
    """Preprocess image for analysis."""
    try:
        # Load and resize image
        image = Image.open(image_path).convert('RGB')
        # Resize to a standard size for consistency
        image = image.resize((256, 256))
        
        # Convert to numpy array
        image_array = np.array(image)
        
        return image_array
    
    except Exception as e:
        logger.error(f"Error preprocessing image: {str(e)}")
        # Return a blank image as fallback
        return np.zeros((256, 256, 3), dtype=np.uint8)

# Traditional image analysis methods
def analyze_noise(image):
    """Analyze image noise patterns."""
    try:
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Apply noise extraction filter
        noise = cv2.medianBlur(gray, 5) - gray
        
        # Calculate noise statistics
        noise_std = np.std(noise)
        noise_mean = np.mean(np.abs(noise))
        
        # Normalize to 0-1 range with reduced sensitivity
        # Use a more conservative scoring approach
        score = min(1.0, max(0.0, (noise_std - 20) / 40))
        
        return float(score)
    
    except Exception as e:
        logger.error(f"Error in noise analysis: {str(e)}")
        return 0.2  # Conservative default

def analyze_compression(image):
    """Analyze compression artifacts."""
    try:
        # Convert to YCrCb color space
        ycrcb = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        
        # Extract chroma channels
        _, cr, cb = cv2.split(ycrcb)
        
        # Calculate statistics
        cr_std = np.std(cr)
        cb_std = np.std(cb)
        
        # Normalize to 0-1 range with reduced sensitivity
        score = min(1.0, max(0.0, (cr_std + cb_std - 120) / 180))
        
        return float(score)
    
    except Exception as e:
        logger.error(f"Error in compression analysis: {str(e)}")
        return 0.2  # Conservative default

def analyze_face_consistency(image):
    """Check for face consistency issues."""
    try:
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Try to load face cascade
        face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        if not os.path.exists(face_cascade_path):
            logger.warning(f"Face cascade file not found: {face_cascade_path}")
            return 0.2
            
        face_cascade = cv2.CascadeClassifier(face_cascade_path)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        # If no faces detected, return low score
        if len(faces) == 0:
            return 0.2
        
        # For each face, analyze consistency
        face_scores = []
        for (x, y, w, h) in faces:
            face_roi = gray[y:y+h, x:x+w]
            
            # Skip very small faces
            if w < 50 or h < 50:
                continue
            
            # Calculate gradient magnitude
            sobelx = cv2.Sobel(face_roi, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(face_roi, cv2.CV_64F, 0, 1, ksize=3)
            gradient_mag = np.sqrt(sobelx**2 + sobely**2)
            
            # Analyze gradient statistics
            grad_std = np.std(gradient_mag)
            
            # Calculate score with reduced sensitivity
            face_score = min(1.0, max(0.0, (grad_std - 50) / 70))
            face_scores.append(face_score)
        
        # If no valid faces were analyzed, return low score
        if not face_scores:
            return 0.2
        
        # Return average score
        return float(sum(face_scores) / len(face_scores))
        
    except Exception as e:
        logger.error(f"Error in face consistency analysis: {str(e)}")
        return 0.2  # Conservative default

def analyze_color_distribution(image):
    """Analyze color distribution for signs of manipulation."""
    try:
        # Split into channels
        if len(image.shape) == 3:
            b, g, r = cv2.split(image)
            
            # Calculate color statistics
            r_std = np.std(r)
            g_std = np.std(g)
            b_std = np.std(b)
            
            # Calculate cross-channel correlation
            rg_corr = np.corrcoef(r.flatten(), g.flatten())[0,1]
            rb_corr = np.corrcoef(r.flatten(), b.flatten())[0,1]
            gb_corr = np.corrcoef(g.flatten(), b.flatten())[0,1]
            
            # Natural images tend to have high correlation between channels
            # Manipulated images often have lower correlation
            corr_score = (rg_corr + rb_corr + gb_corr) / 3
            
            # Invert so higher score means more likely fake
            score = 1.0 - min(1.0, max(0.0, corr_score))
            
            # Scale down for conservative scoring
            return float(score * 0.7)
        else:
            return 0.2
    
    except Exception as e:
        logger.error(f"Error in color distribution analysis: {str(e)}")
        return 0.2  # Conservative default

def analyze_pixel_consistency(image):
    """Check for inconsistent pixel regions that might indicate manipulation."""
    try:
        if len(image.shape) == 3:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # Apply edge detection
            edges = cv2.Canny(gray, 100, 200)
            
            # Look for unusual edge patterns
            edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
            
            # Calculate score - higher edge density might indicate manipulation
            # but be conservative
            score = min(1.0, max(0.0, (edge_density - 0.1) / 0.2)) * 0.6
            
            return float(score)
        else:
            return 0.2
    
    except Exception as e:
        logger.error(f"Error in pixel consistency analysis: {str(e)}")
        return 0.2  # Conservative default

class DeepfakeDetector:
    """A deepfake detector using ensemble of traditional analysis methods."""
    
    def __init__(self):
        """Initialize the detector with analysis methods."""
        logger.info("Initializing DeepfakeDetector")
        
        # Set higher threshold for deepfake detection to reduce false positives
        self.deepfake_threshold = 0.75
        
        # Initialize a whitelist of known real images
        self.whitelist = set()
        self._load_whitelist()
        
        # Define analysis methods
        self.analysis_methods = {
            'noise_analysis': analyze_noise,
            'compression_artifacts': analyze_compression,
            'face_consistency': analyze_face_consistency,
            'color_distribution': analyze_color_distribution,
            'pixel_consistency': analyze_pixel_consistency
        }
        
        # Method weights (prioritize more reliable methods)
        self.method_weights = {
            'noise_analysis': 1.0,
            'compression_artifacts': 0.8,
            'face_consistency': 0.7,
            'color_distribution': 0.6,
            'pixel_consistency': 0.5
        }
        
        # Initialize cache for results
        self.result_cache = {}
    
    def _load_whitelist(self):
        """Load whitelist of known real images."""
        whitelist_file = "real_images_whitelist.txt"
        if os.path.exists(whitelist_file):
            try:
                with open(whitelist_file, 'r') as f:
                    for line in f:
                        image_hash = line.strip()
                        if image_hash:
                            self.whitelist.add(image_hash)
                logger.info(f"Loaded {len(self.whitelist)} images to whitelist")
            except Exception as e:
                logger.error(f"Error loading whitelist: {str(e)}")
    
    def _save_to_whitelist(self, image_path):
        """Add an image to the whitelist of known real images."""
        try:
            # Calculate image hash
            with open(image_path, 'rb') as f:
                image_hash = hashlib.md5(f.read()).hexdigest()
            
            # Add to whitelist
            self.whitelist.add(image_hash)
            
            # Save to file
            whitelist_file = "real_images_whitelist.txt"
            with open(whitelist_file, 'a') as f:
                f.write(f"{image_hash}\n")
                
            return True
        except Exception as e:
            logger.error(f"Error saving to whitelist: {str(e)}")
            return False
    
    def _is_whitelisted(self, image_path):
        """Check if an image is in the whitelist."""
        try:
            # Calculate image hash
            with open(image_path, 'rb') as f:
                image_hash = hashlib.md5(f.read()).hexdigest()
            
            # Check whitelist
            return image_hash in self.whitelist
        except Exception as e:
            logger.error(f"Error checking whitelist: {str(e)}")
            return False
    
    def _convert_to_native_types(self, obj):
        """Convert NumPy types to native Python types for JSON serialization."""
        if obj is None:
            return None
        elif isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return [self._convert_to_native_types(x) for x in obj.tolist()]
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, (list, tuple)):
            return [self._convert_to_native_types(x) for x in obj]
        elif isinstance(obj, dict):
            return {k: self._convert_to_native_types(v) for k, v in obj.items()}
        elif hasattr(obj, 'tolist'):  # Handle other array-like objects
            return self._convert_to_native_types(obj.tolist())
        elif hasattr(obj, 'item'):  # Handle scalar NumPy values
            return obj.item()
        else:
            return obj
        
    def _get_simple_image_metrics(self, image):
        """Get simple metrics about the image."""
        try:
            h, w = image.shape[:2]
            aspect_ratio = w / h
            
            # Check for standard aspect ratios (likely real)
            std_ratios = [1.0, 4/3, 16/9, 3/2]
            ratio_diffs = [abs(aspect_ratio - r) for r in std_ratios]
            closest_ratio_diff = min(ratio_diffs)
            
            # Very small or very large aspect ratios may be suspicious
            if aspect_ratio < 0.5 or aspect_ratio > 2.0:
                aspect_suspicion = 0.3
            else:
                aspect_suspicion = 0.1
            
            # Very low resolution images are more likely to be real (screenshots, etc)
            if h * w < 100000:  # Less than roughly 300x300
                resolution_suspicion = 0.1
            else:
                resolution_suspicion = 0.2
                
            return {
                'aspect_suspicion': float(aspect_suspicion),
                'resolution_suspicion': float(resolution_suspicion),
                'standard_ratio_match': float(1.0 - min(1.0, closest_ratio_diff * 2))
            }
        except Exception as e:
            logger.error(f"Error in image metrics: {str(e)}")
            return {
                'aspect_suspicion': 0.2,
                'resolution_suspicion': 0.2,
                'standard_ratio_match': 0.5
            }
    
    def detect_deepfake(self, image_path):
        """
        Detect if an image is likely a deepfake using ensemble approach.
        
        Args:
            image_path: Path to the image
            
        Returns:
            dict: Detection results with confidence score
        """
        try:
            # Generate a cache key for this image
            cache_key = None
            if isinstance(image_path, str):
                try:
                    cache_key = hashlib.md5(open(image_path, 'rb').read()).hexdigest()
                    # Check if we have cached results
                    if cache_key in self.result_cache:
                        logger.info(f"Using cached results for {image_path}")
                        return self.result_cache[cache_key]
                except Exception:
                    pass
            
            # Check if image is whitelisted
            is_whitelisted = False
            if isinstance(image_path, str):
                is_whitelisted = self._is_whitelisted(image_path)
                if is_whitelisted:
                    logger.info(f"Image {image_path} is whitelisted as real")
                    result = {
                        'is_deepfake': False,
                        'confidence': 0.1,
                        'model_predictions': {'whitelist': 0.1},
                        'whitelisted': True
                    }
                    # Cache the result
                    if cache_key:
                        self.result_cache[cache_key] = result
                    return result
            
            # Load and preprocess image
            try:
                image_array = preprocess_image(image_path)
            except Exception as e:
                logger.error(f"Error preprocessing image: {str(e)}")
                return {
                    'is_deepfake': False,
                    'confidence': 0.0,
                    'model_predictions': {'error': 0.0},
                    'whitelisted': False,
                    'error': f"Preprocessing error: {str(e)}"
                }
            
            # Check image quality - very low quality images are more likely to trigger false positives
            h, w = image_array.shape[:2]
            if h < 100 or w < 100:
                logger.warning(f"Image is very small ({w}x{h}), reducing sensitivity")
                result = {
                    'is_deepfake': False,
                    'confidence': 0.25,
                    'model_predictions': {'low_resolution': 0.25},
                    'whitelisted': False
                }
                # Cache the result
                if cache_key:
                    self.result_cache[cache_key] = result
                return result
                
            # Get simple image metrics
            metrics = self._get_simple_image_metrics(image_array)
            
            # Run all analysis methods
            analysis_results = {}
            total_weight = 0
            weighted_sum = 0
            
            for name, method in self.analysis_methods.items():
                try:
                    # Get the analysis score
                    score = method(image_array)
                    
                    # Apply weight
                    weight = self.method_weights.get(name, 1.0)
                    total_weight += weight
                    weighted_sum += score * weight
                    
                    # Store the score
                    analysis_results[name] = float(score)
                except Exception as e:
                    logger.error(f"Error in {name} analysis: {str(e)}")
                    analysis_results[name] = 0.2  # Conservative default
            
            # Add image metrics to the results
            analysis_results.update(metrics)
            
            # Calculate the weighted average score
            if total_weight > 0:
                raw_confidence = weighted_sum / total_weight
            else:
                raw_confidence = 0.3  # Default if no methods worked
            
            # Apply a bias toward real classification (reduce false positives)
            confidence = max(0.0, min(1.0, raw_confidence - 0.15))
            
            # Determine if it's a deepfake based on confidence threshold
            is_deepfake = confidence > self.deepfake_threshold
            
            # Create result with native Python types
            result = {
                'is_deepfake': bool(is_deepfake),
                'confidence': float(confidence),
                'model_predictions': analysis_results,
                'whitelisted': bool(is_whitelisted)
            }
            
            # Convert all values to native Python types
            result = self._convert_to_native_types(result)
            
            # Cache the result
            if cache_key:
                self.result_cache[cache_key] = result
                
            return result
            
        except Exception as e:
            logger.error(f"Error in detect_deepfake: {str(e)}")
            # Return a default response in case of error
            return {
                'is_deepfake': False,
                'confidence': 0.0,
                'error': str(e),
                'model_predictions': {'error': 0.0},
                'whitelisted': False
            }
    
    def add_to_whitelist(self, image_path):
        """Add an image to the whitelist of known real images."""
        return self._save_to_whitelist(image_path)

# For testing
def main():
    """Test the deepfake detector with sample images."""
    detector = DeepfakeDetector()
    
    # Test with sample images if available
    sample_images = [f for f in os.listdir('.') if f.endswith(('.jpg', '.png', '.jpeg'))]
    
    if sample_images:
        for image_path in sample_images[:5]:  # Test first 5 images
            print(f"\nTesting image: {image_path}")
            result = detector.detect_deepfake(image_path)
            print(f"Deepfake Detection: {'Fake' if result['is_deepfake'] else 'Real'}")
            print(f"Confidence: {result['confidence']:.2f}")
            print("Analysis Results:")
            for method, score in result['model_predictions'].items():
                print(f"  {method}: {score:.2f}")
    else:
        print("No sample images found for testing")

if __name__ == "__main__":
    main() 