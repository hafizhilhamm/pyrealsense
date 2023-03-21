import pyrealsense2 as rs
import numpy as np
import cv2
import argparse
import os.path

parser = argparse.ArgumentParser(description="Read recorded bag file and display depth and color streams.")
parser.add_argument("-i", "--input", type=str, help="Path to the bag file")
args = parser.parse_args()
args.input = "/home/hafizh/Documents/rekamann.bag"
try:
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    rs.config.enable_device_from_file(config , args.input)
    
    pipeline.start(config)

    
    cv2.namedWindow("Depth Stream", cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow("color stream", cv2.WINDOW_AUTOSIZE)
    
    colorizer = rs.colorizer()
    template = cv2.imread("/home/hafizh/Pictures/pic6.png")
    max_distance = 4000
    #cap = cv2.VideoCapture('/home/hafizh/Videos/lapangan.mkv')
    
    while True:
        #_, template = cap.read()
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        template = cv2.resize(template, (depth_frame.width, depth_frame.height))
        
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        depth_image[depth_image > max_distance] = 0
       
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET
        )

        graydepth = cv2.cvtColor(depth_colormap,cv2.COLOR_BGR2GRAY)
        graylap = cv2.cvtColor(template,cv2.COLOR_BGR2GRAY)

        blurgdepth = cv2.GaussianBlur(graydepth,(7,7),0)
        blurglap = cv2.GaussianBlur(graylap,(7,7),0)

        foreground = cv2.subtract(blurgdepth,blurglap)

        binary =  cv2.threshold(foreground , 25 , 255 , cv2.THRESH_BINARY)[1]

        result = np.zeros_like(depth_colormap)
        result = cv2.bitwise_and(depth_colormap, depth_colormap, mask=binary)
        
        keni = cv2.Canny(binary,100,200)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dil = cv2.dilate(keni, kernel)
        contours, hierarchy = cv2.findContours(dil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if( 7000 < area < 15000):
             cv2.drawContours(result, contours, i, (0, 0, 255), 2)
             print("Area of object {}: {:.2f} pixels".format(i+1, area))
             moments = cv2.moments(contour)
             center_x = int(moments['m10'] / moments['m00'])
             center_y = int(moments['m01'] / moments['m00'])
             distance = depth_frame.get_distance(center_x, center_y)
             cv2.circle(result, (center_x,center_y), 5, (0,255,255), cv2.FILLED)
             print("jarak : {:.2f} meters".format(distance))

             
        images = np.hstack((color_image, depth_colormap))
        # print(template_size)
        cv2.imshow("window2", images)
        cv2.imshow("depth", depth_colormap)
        cv2.imshow("cut", template)
        cv2.imshow("hasil",result)
      
        if cv2.waitKey(1) == 27:
            break
finally:
    
    pipeline.stop()


cv2.destroyAllWindows()