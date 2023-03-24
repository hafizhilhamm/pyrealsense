import cv2
import pyrealsense2 as rs
import numpy as np

template = cv2.imread("/home/hafizh/Pictures/pic9.png")
template = cv2.resize(template, (640,480))

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)


pipeline.start(config)

max_distance = 10000

try:
    while True:
        
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        
        
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

        result = np.zeros_like(color_image)
        result = cv2.bitwise_and(color_image, color_image, mask=binary)

        hsving = cv2.cvtColor(result,cv2.COLOR_BGR2HSV)

        graying = cv2.cvtColor(hsving,cv2.COLOR_BGR2GRAY)

        lower_white = np.array([105 , 39 , 117], dtype=np.uint8)
        upper_white = np.array([141 , 255 , 255], dtype=np.uint8)

        mask = cv2.inRange(hsving , lower_white, upper_white)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            cv2.drawContours(mask, contours, i, (0, 0, 255), 2)
            moments = cv2.moments(contour)
             #print("Area of object {}: {:.2f} pixels".format(i+1, area))
            if (moments['m00'] != 0 and moments['m01'] != 0 and  200 < area < 500 ):
             center_x = int(moments['m10'] / moments['m00'])
             center_y = int(moments['m01'] / moments['m00'])
             distance = depth_frame.get_distance(center_x, center_y)
             cv2.circle(mask, (center_x,center_y), 6, (0,0,0), cv2.FILLED)
             print("jarak gawang  : {:.2f} meters".format(distance))

        lower = np.array([0,61,28], dtype=np.uint8)
        upper = np.array([43,129,115], dtype=np.uint8)

        mask2 = cv2.inRange(hsving , lower, upper)
        contours, _ = cv2.findContours(mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for i, contour in enumerate(contours):
         area = cv2.contourArea(contour)
         cv2.drawContours(mask, contours, i, (0, 0, 255), 2)
         moments2 = cv2.moments(mask2)
         #print(area)
        if (moments2['m00'] != 0 and moments2['m01'] != 0 and  2000 < area ):
         center_x2 = int(moments2['m10'] / moments2['m00'])
         center_y2 = int(moments2['m01'] / moments2['m00'])
         cv2.circle(mask2, (center_x2,center_y2), 8, (0,0,255), cv2.FILLED)
         distance2 = depth_frame.get_distance(center_x2, center_y2)
         print("jarak robot : {:.2f} meters".format(distance2))

        #print(center_y)
        images = np.hstack((color_image, depth_colormap))
        images2 = np.hstack((mask, mask2))
        # print(template_size)
        cv2.imshow("window", images)
        cv2.imshow("window2", images2)
        cv2.imshow("window3", result)
        cv2.imshow("windoww", mask)
        if cv2.waitKey(1) == 27:
            break
finally:
    
    pipeline.stop()


cv2.destroyAllWindows()
