import cv2
import numpy as np
import time
import subprocess
import os

class CameraCalibrator:
    def __init__(self, nir_port=0, vis_port=1, width=2028, height=1520):
        self.nir_port = nir_port
        self.vis_port = vis_port
        self.width = width
        self.height = height

    def capture_single_image(self, port, filename):
        """Capture a single image from specified camera"""
        try:
            cmd = [
                'libcamera-still',
                '--camera', str(port),
                '--width', str(self.width),
                '--height', str(self.height),
                '--output', filename,
                '--immediate',
                '--nopreview'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            
            if result.returncode != 0:
                print(f"Error capturing from camera {port}: {result.stderr}")
                return None
                
            return filename
            
        except Exception as e:
            print(f"Exception capturing from camera {port}: {e}")
            return None

    def analyze_image(self, image_path, camera_name):
        """Analyze image statistics"""
        if not os.path.exists(image_path):
            print(f"Image file {image_path} not found")
            return None
            
        img = cv2.imread(image_path)
        if img is None:
            print(f"Failed to read image {image_path}")
            return None
            
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Calculate statistics
        stats = {
            'mean': np.mean(gray),
            'std': np.std(gray),
            'min': np.min(gray),
            'max': np.max(gray),
            'median': np.median(gray),
            'pixels_above_200': np.sum(gray > 200),
            'pixels_above_100': np.sum(gray > 100),
            'pixels_above_50': np.sum(gray > 50),
            'total_pixels': gray.size
        }
        
        print(f"\n{camera_name} Camera Statistics:")
        print(f"  Mean brightness: {stats['mean']:.1f}")
        print(f"  Standard deviation: {stats['std']:.1f}")
        print(f"  Min/Max: {stats['min']}/{stats['max']}")
        print(f"  Median: {stats['median']:.1f}")
        print(f"  Pixels > 200: {stats['pixels_above_200']} ({stats['pixels_above_200']/stats['total_pixels']*100:.1f}%)")
        print(f"  Pixels > 100: {stats['pixels_above_100']} ({stats['pixels_above_100']/stats['total_pixels']*100:.1f}%)")
        print(f"  Pixels > 50: {stats['pixels_above_50']} ({stats['pixels_above_50']/stats['total_pixels']*100:.1f}%)")
        
        return stats

    def capture_and_analyze(self, description):
        """Capture images from both cameras and analyze them"""
        print(f"\n=== {description} ===")
        
        # Capture images
        nir_file = f"/tmp/nir_cal_{int(time.time())}.jpg"
        vis_file = f"/tmp/vis_cal_{int(time.time())}.jpg"
        
        print("Capturing NIR image...")
        nir_result = self.capture_single_image(self.nir_port, nir_file)
        print("Capturing VIS image...")
        vis_result = self.capture_single_image(self.vis_port, vis_file)
        
        if nir_result and vis_result:
            # Analyze images
            nir_stats = self.analyze_image(nir_file, "NIR")
            vis_stats = self.analyze_image(vis_file, "VIS")
            
            # Calculate NDVI if both images are valid
            if nir_stats and vis_stats:
                nir_img = cv2.imread(nir_file)
                vis_img = cv2.imread(vis_file)
                
                nir_gray = cv2.cvtColor(nir_img, cv2.COLOR_BGR2GRAY).astype(np.float32)
                vis_gray = cv2.cvtColor(vis_img, cv2.COLOR_BGR2GRAY).astype(np.float32)
                
                # Calculate NDVI
                denom = nir_gray + vis_gray
                denom[denom == 0] = 0.01
                ndvi = (nir_gray - vis_gray) / denom
                
                print(f"\nNDVI Statistics:")
                print(f"  Mean NDVI: {np.mean(ndvi):.3f}")
                print(f"  Std NDVI: {np.std(ndvi):.3f}")
                print(f"  Min/Max NDVI: {np.min(ndvi):.3f}/{np.max(ndvi):.3f}")
                print(f"  Pixels with NDVI > 0: {np.sum(ndvi > 0)} ({np.sum(ndvi > 0)/ndvi.size*100:.1f}%)")
                print(f"  Pixels with NDVI > 0.1: {np.sum(ndvi > 0.1)} ({np.sum(ndvi > 0.1)/ndvi.size*100:.1f}%)")
                print(f"  Pixels with NDVI > 0.2: {np.sum(ndvi > 0.2)} ({np.sum(ndvi > 0.2)/ndvi.size*100:.1f}%)")
            
            # Clean up
            try:
                os.remove(nir_file)
                os.remove(vis_file)
            except:
                pass
        else:
            print("Failed to capture one or both images")

def main():
    calibrator = CameraCalibrator()
    
    print("Camera Calibration Tool")
    print("=======================")
    print("This tool will help you understand your camera sensitivity.")
    print("Follow the prompts to capture images under different conditions.")
    
    while True:
        print("\nOptions:")
        print("1. Capture with halogen light (NIR source)")
        print("2. Capture with natural sunlight")
        print("3. Capture with indoor lighting")
        print("4. Capture with plants in sunlight")
        print("5. Capture with plants under halogen")
        print("6. Exit")
        
        choice = input("\nEnter your choice (1-6): ").strip()
        
        if choice == '1':
            input("Point cameras at halogen light, then press Enter...")
            calibrator.capture_and_analyze("Halogen Light (NIR Source)")
        elif choice == '2':
            input("Point cameras at natural sunlight, then press Enter...")
            calibrator.capture_and_analyze("Natural Sunlight")
        elif choice == '3':
            input("Point cameras at indoor lighting, then press Enter...")
            calibrator.capture_and_analyze("Indoor Lighting")
        elif choice == '4':
            input("Point cameras at plants in sunlight, then press Enter...")
            calibrator.capture_and_analyze("Plants in Sunlight")
        elif choice == '5':
            input("Point cameras at plants under halogen light, then press Enter...")
            calibrator.capture_and_analyze("Plants under Halogen")
        elif choice == '6':
            print("Exiting calibration tool.")
            break
        else:
            print("Invalid choice. Please enter 1-6.")

if __name__ == "__main__":
    main() 