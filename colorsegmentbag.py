import pyrealsense2 as rs
import numpy as np
import cv2
import argparse
import os.path
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

parser = argparse.ArgumentParser(description="Read recorded bag file and display depth stream in jet colormap.\
                                Remember to change the stream fps and format to match the recorded.")
parser.add_argument("-i", "--input", type=str, help="Path to the bag file")
args = parser.parse_args()
args.input = "/home/hafizh/Documents/rekamann.bag"
try:
    pipeline = rs.pipeline()
    config = rs.config()

    rs.config.enable_device_from_file(config, args.input)

    config.enable_stream(rs.stream.depth, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, rs.format.rgb8, 30) # enable color stream
    
    pipeline.start(config)
    
    colorizer = rs.colorizer()
    template = cv2.imread("/home/hafizh/Pictures/pic6.png")
    max_distance = 5000
    #cap = cv2.VideoCapture('/home/hafizh/Videos/lapangan.mkv')
    
    while True:
        #_, template = cap.read()
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame() # get color frame
        
        template = cv2.resize(template, (depth_frame.width, depth_frame.height))
        
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image_bgr = np.asanyarray(color_frame.get_data())
        color_image_8bit = cv2.cvtColor(color_image_bgr, cv2.COLOR_RGB2BGR)
        color_image = cv2.convertScaleAbs(color_image_8bit)

        depth_image[depth_image > max_distance] = 0
       
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET
        )
        hsving = cv2.cvtColor(color_image,cv2.COLOR_BGR2HSV)
        h_min = cv2.getTrackbarPos('H Min', 'Trackbars')
        s_min = cv2.getTrackbarPos('S Min', 'Trackbars')
        v_min = cv2.getTrackbarPos('V Min', 'Trackbars')
        h_max = cv2.getTrackbarPos('H Max', 'Trackbars')
        s_max = cv2.getTrackbarPos('S Max', 'Trackbars')
        v_max = cv2.getTrackbarPos('V Max', 'Trackbars')

    # Create lower and upper boundaries for color segmentation
        lower_color = np.array([h_min, s_min, v_min])
        upper_color = np.array([h_max, s_max, v_max])

        graying = cv2.cvtColor(hsving,cv2.COLOR_BGR2GRAY)

        mask = cv2.inRange(hsving , lower_color, upper_color)
        
        print(h_min,s_min,v_min,h_max,s_max,v_max)
        cv2.imshow('real', color_image)
        cv2.imshow('reall', depth_colormap)
        cv2.imshow("window",mask)
        if cv2.waitKey(1) == 27:
            break

finally:
    
    pipeline.stop()


cv2.destroyAllWindows()