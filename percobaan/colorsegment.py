import cv2
import numpy as np
import pyrealsense2 as rs

# Define callback function for trackbars
def on_trackbar(val):
    pass

# Create a window for trackbars
cv2.namedWindow('Trackbars')

# Define initial values for trackbars
h_min, s_min, v_min = 0, 0, 0
h_max, s_max, v_max = 255, 255, 255

# Create trackbars for HSV range
cv2.createTrackbar('H Min', 'Trackbars', h_min, 255, on_trackbar)
cv2.createTrackbar('S Min', 'Trackbars', s_min, 255, on_trackbar)
cv2.createTrackbar('V Min', 'Trackbars', v_min, 255, on_trackbar)
cv2.createTrackbar('H Max', 'Trackbars', h_max, 255, on_trackbar)
cv2.createTrackbar('S Max', 'Trackbars', s_max, 255, on_trackbar)
cv2.createTrackbar('V Max', 'Trackbars', v_max, 255, on_trackbar)

# Create a pipeline for the RealSense camera
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start the pipeline
pipeline.start(config)

# Loop through frames from the camera
while True:
    # Wait for a new frame from the camera
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()

    # Convert the frame to a numpy array
    color_image = np.asanyarray(color_frame.get_data())

    # Convert to HSV color space
    hsv_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)

    # Get current trackbar values
    h_min = cv2.getTrackbarPos('H Min', 'Trackbars')
    s_min = cv2.getTrackbarPos('S Min', 'Trackbars')
    v_min = cv2.getTrackbarPos('V Min', 'Trackbars')
    h_max = cv2.getTrackbarPos('H Max', 'Trackbars')
    s_max = cv2.getTrackbarPos('S Max', 'Trackbars')
    v_max = cv2.getTrackbarPos('V Max', 'Trackbars')

    # Create lower and upper boundaries for color segmentation
    lower_color = np.array([h_min, s_min, v_min])
    upper_color = np.array([h_max, s_max, v_max])

    # Perform color segmentation
    mask = cv2.inRange(hsv_image, lower_color, upper_color)
    result = cv2.bitwise_and(color_image, color_image, mask=mask)

    # Display the resulting image
    cv2.imshow('Result', result)

    # Check for key press
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

# Stop the pipeline and close all windows
pipeline.stop()


cv2.destroyAllWindows()
