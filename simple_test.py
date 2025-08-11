import cv2
import time

print("Testing simple camera capture...")

# Try multiple camera ports
for port in [0, 8, 1, 2, 3, 4, 5]:
    print(f"\nTrying camera port {port}...")
    
    cap = cv2.VideoCapture(port, cv2.CAP_V4L2)
    
    if not cap.isOpened():
        print(f"  Failed to open camera {port}")
        continue
    
    print(f"  Camera {port} opened successfully")
    
    # Try to read a frame
    ret, frame = cap.read()
    
    if ret:
        print(f"  SUCCESS! Frame shape: {frame.shape}")
        print(f"  Frame type: {frame.dtype}")
        cap.release()
        break
    else:
        print(f"  Failed to read frame from camera {port}")
        cap.release()

print("\nTest complete") 