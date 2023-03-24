import pyrealsense2 as rs
import numpy as np
import cv2
import argparse
import os.path

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

        graying = cv2.cvtColor(hsving,cv2.COLOR_BGR2GRAY)

        lower_white = np.array([100, 33 ,176], dtype=np.uint8)
        upper_white = np.array([129, 157 ,255], dtype=np.uint8)

        mask = cv2.inRange(hsving , lower_white, upper_white)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for i, contour in enumerate(contours):
         area = cv2.contourArea(contour)
         cv2.drawContours(mask, contours, i, (0, 0, 255), 2)
         #print("Area of object {}: {:.2f} pixels".format(i+1, area))
         moments = cv2.moments(contour)
        if (moments['m00'] != 0 and moments['m01'] != 0 ):
            center_x = int(moments['m10'] / moments['m00'])
            center_y = int(moments['m01'] / moments['m00'])
        else:
            center_x, center_y, distance = 0, 0, 0
        if center_x < depth_frame.width and center_y <  depth_frame.height:
            distance = depth_frame.get_distance(center_x, center_y)
            cv2.circle(mask, (center_x,center_y), 8, (0,0,0), cv2.FILLED)
            print("jarak gawang  : {:.2f} meters".format(distance))
           
        
        # print(template_size)
        cv2.imshow("window2", color_image)
        cv2.imshow("window23", mask)
        cv2.imshow("window3", depth_colormap)
      
        if cv2.waitKey(1) == 27:
            break
finally:
    
    pipeline.stop()


cv2.destroyAllWindows()