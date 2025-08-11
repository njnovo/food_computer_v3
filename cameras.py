import cv2
import numpy as np
import time
import subprocess
import os

class CameraPair:
    def __init__(self, nir_port, vis_port, width=640, height=480, plant_count=4):
        self.nir_port = nir_port
        self.vis_port = vis_port
        self.width = width
        self.height = height
        self.plant_count = plant_count

        # Load calibration settings
        self.calibration_settings = self._load_calibration_settings()

        # Initialize cameras using libcamera
        self.nir_cam = self._init_libcamera_camera(nir_port, "NIR")
        self.vis_cam = self._init_libcamera_camera(vis_port, "VIS")

    def _init_libcamera_camera(self, port, camera_type):
        """Initialize camera using libcamera"""
        print(f"Initializing {camera_type} camera at port {port}")
        
        # Test if camera is available
        try:
            result = subprocess.run(['libcamera-hello', '--list-cameras'], 
                                  capture_output=True, text=True, timeout=5)
            if str(port) not in result.stdout:
                raise RuntimeError(f"Camera {port} not found in libcamera list")
        except subprocess.TimeoutExpired:
            raise RuntimeError("libcamera-hello command timed out")
        except FileNotFoundError:
            raise RuntimeError("libcamera-hello not found")
        
        return port  # Return port number for libcamera

    def _load_calibration_settings(self):
        """Load calibration settings from file"""
        calibration_file = "camera_calibration.json"
        if os.path.exists(calibration_file):
            try:
                import json
                with open(calibration_file, 'r') as f:
                    settings = json.load(f)
                print(f"Loaded calibration settings from {calibration_file}")
                return settings
            except Exception as e:
                print(f"Failed to load calibration settings: {e}")
        return {}

    def _capture_with_libcamera(self, port, output_file):
        """Capture a frame using libcamera-still"""
        try:
            # Use libcamera-still to capture a frame
            cmd = [
                'libcamera-still',
                '--camera', str(port),
                '--width', str(self.width),
                '--height', str(self.height),
                '--output', output_file,
                '--immediate',
                '--nopreview'
            ]
            
            # Use calibration settings if available, otherwise default settings
            if hasattr(self, 'calibration_settings') and port in self.calibration_settings:
                settings = self.calibration_settings[port]
                cmd.extend([
                    '--shutter', str(settings['shutter']),
                    '--gain', str(settings['gain']),
                    '--ev', str(settings['ev'])
                ])
            else:
                # Default settings - same for both cameras
                cmd.extend([
                    '--shutter', '50000',  # 0.05 second exposure
                    '--gain', '2.0',  # Moderate gain
                    '--ev', '0.0'  # No exposure compensation
                ])
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
            
            if result.returncode != 0:
                print(f"libcamera-still error for camera {port}: {result.stderr}")
                return None
            
            print(f"Successfully captured from camera {port}")
            return output_file
            
        except subprocess.TimeoutExpired:
            print(f"libcamera-still timed out for camera {port}")
            return None
        except Exception as e:
            print(f"Error capturing from camera {port}: {e}")
            return None

    def capture_images(self):
        """Capture images from both cameras"""
        # Create temporary files for captures
        nir_file = f"/tmp/nir_capture_{int(time.time())}.jpg"
        vis_file = f"/tmp/vis_capture_{int(time.time())}.jpg"
        
        # Capture from both cameras
        nir_result = self._capture_with_libcamera(self.nir_cam, nir_file)
        vis_result = self._capture_with_libcamera(self.vis_cam, vis_file)
        
        if nir_result is None or vis_result is None:
            raise RuntimeError("Failed to capture from one or both cameras")
        
        # Read the captured images
        nir_img = cv2.imread(nir_file)
        vis_img = cv2.imread(vis_file)
        
        # Clean up temporary files
        try:
            os.remove(nir_file)
            os.remove(vis_file)
        except:
            pass
        
        if nir_img is None or vis_img is None:
            raise RuntimeError("Failed to read captured images")
        
        # Convert to grayscale
        nir_gray = cv2.cvtColor(nir_img, cv2.COLOR_BGR2GRAY)
        vis_gray = cv2.cvtColor(vis_img, cv2.COLOR_BGR2GRAY)
        
        return nir_gray, vis_gray

    def calculate_ndvi(self, nir, vis):
        nir = nir.astype(np.float32)
        vis = vis.astype(np.float32)

        denom = nir + vis
        denom[denom == 0] = 0.01  # prevent division by 0

        ndvi = (nir - vis) / denom
        return ndvi

    def get_ndvi_per_plant(self, ndvi_image):
        """Find plant hotspots using NDVI thresholding and clustering"""
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(ndvi_image, (5, 5), 0)
        
        # Look for regions with higher NDVI values (closer to 0 or positive)
        # Since we're getting negative values, look for less negative regions
        ndvi_normalized = ((blurred + 1) * 127.5).astype(np.uint8)  # Convert to 0-255 range
        
        # Use simple thresholding to find brighter regions
        # Since NDVI values are negative, we want to find regions that are less negative
        # This means they should be brighter in the normalized image
        _, binary = cv2.threshold(ndvi_normalized, 127, 255, cv2.THRESH_BINARY)
        
        # Apply morphological operations to clean up the binary image
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # Find contours of plant regions
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        plant_ndvis = []
        min_contour_area = 20  # Very small minimum area
        max_contour_area = 100000  # Large maximum area
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if min_contour_area < area < max_contour_area:
                # Create mask for this plant
                mask = np.zeros_like(ndvi_image)
                cv2.fillPoly(mask, [contour], 1)
                
                # Calculate mean NDVI for this plant region
                plant_ndvi = np.mean(ndvi_image[mask == 1])
                
                # Include all regions
                plant_ndvis.append(plant_ndvi)
        
        # If no plants found, return empty list
        if not plant_ndvis:
            print("No plants detected in image")
            return []
        
        return plant_ndvis

    def get_display_ndvi_image(self, ndvi):
        ndvi_normalized = ((ndvi + 1) / 2 * 255).astype(np.uint8)
        return cv2.applyColorMap(ndvi_normalized, cv2.COLORMAP_JET)
    
    def get_plant_detection_image(self, ndvi_image):
        """Create an image showing detected plant regions"""
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(ndvi_image, (5, 5), 0)
        
        # Find high NDVI regions
        ndvi_normalized = ((blurred + 1) * 127.5).astype(np.uint8)
        
        # Use simple thresholding to find brighter regions
        _, binary = cv2.threshold(ndvi_normalized, 127, 255, cv2.THRESH_BINARY)
        
        # Apply morphological operations to clean up the binary image
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Create visualization image
        vis_image = cv2.cvtColor(ndvi_normalized, cv2.COLOR_GRAY2BGR)
        min_contour_area = 20
        max_contour_area = 100000
        
        plant_count = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            if min_contour_area < area < max_contour_area:
                # Calculate mean NDVI for this region
                mask = np.zeros_like(ndvi_image)
                cv2.fillPoly(mask, [contour], 1)
                plant_ndvi = np.mean(ndvi_image[mask == 1])
                
                # Draw contour with color based on NDVI value
                if plant_ndvi > -0.8:
                    color = (0, 255, 0)  # Green for higher NDVI
                elif plant_ndvi > -0.9:
                    color = (0, 255, 255)  # Yellow for medium NDVI
                else:
                    color = (0, 0, 255)  # Red for lower NDVI
                
                cv2.drawContours(vis_image, [contour], -1, color, 2)
                
                # Add plant number and NDVI value
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    cv2.putText(vis_image, f"{plant_count}:{plant_ndvi:.2f}", 
                              (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    plant_count += 1
        
        return vis_image

    def release(self):
        # No need to release libcamera resources
        cv2.destroyAllWindows()

def main():
    camera_pair = CameraPair(nir_port=0, vis_port=1, width=2028, height=1520)

    try:
        while True:
            print("Capturing images...")
            nir_img, vis_img = camera_pair.capture_images()
            ndvi = camera_pair.calculate_ndvi(nir_img, vis_img)
            ndvi_colored = camera_pair.get_display_ndvi_image(ndvi)
            plant_ndvis = camera_pair.get_ndvi_per_plant(ndvi)

            print(f"Found {len(plant_ndvis)} plants:")
            for i, val in enumerate(plant_ndvis):
                print(f"Plant {i}: NDVI = {val:.3f}")
            if plant_ndvis:
                print(f"Average NDVI: {np.mean(plant_ndvis):.3f}")
            print("-" * 40)

            # Show all images
            cv2.imshow("NIR Image", nir_img)
            cv2.imshow("Visible Image", vis_img)
            cv2.imshow("NDVI", ndvi_colored)
            
            # Show plant detection visualization
            plant_detection_img = camera_pair.get_plant_detection_image(ndvi)
            cv2.imshow("Plant Detection", plant_detection_img)

            # Press Q to exit
            if cv2.waitKey(2000) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("Stopping capture...")
    finally:
        camera_pair.release()

if __name__ == "__main__":
    main()

