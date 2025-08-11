import cv2
import time

def test_camera(port):
    print(f"Testing camera at port {port}")
    
    # Try different pixel formats
    formats = [
        ('YUYV', cv2.VideoWriter_fourcc('Y', 'U', 'Y', 'V')),
        ('MJPG', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')),
        ('RGB3', cv2.VideoWriter_fourcc('R', 'G', 'B', '3')),
    ]
    
    for format_name, fourcc in formats:
        print(f"  Trying format: {format_name}")
        cam = cv2.VideoCapture(port, cv2.CAP_V4L2)
        
        if not cam.isOpened():
            print(f"    Failed to open camera")
            continue
            
        # Set format
        cam.set(cv2.CAP_PROP_FOURCC, fourcc)
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Try to read a frame
        time.sleep(0.5)  # Give camera time to initialize
        
        for attempt in range(5):
            ret, frame = cam.read()
            if ret:
                print(f"    SUCCESS! Captured frame with {format_name}")
                print(f"    Frame shape: {frame.shape}")
                cam.release()
                return True
            time.sleep(0.1)
        
        print(f"    Failed to capture with {format_name}")
        cam.release()
    
    return False

if __name__ == "__main__":
    print("Testing camera 0 (NIR):")
    test_camera(0)
    print("\nTesting camera 8 (VIS):")
    test_camera(8) 