import cv2
import pyrealsense2 as rs
import numpy as np

template = cv2.imread("/home/hafizh/Pictures/pic9.png")
template = cv2.resize(template, (640,480))

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

def on_trackbar(val):
    pass

# Create a window for trackbars
cv2.namedWindow('Trackbars')

# Define initial values for trackbars
min_distance = 0
max_distance = 10000

# Create trackbars for HSV range
cv2.createTrackbar('Min distance', 'Trackbars', min_distance, 10000, on_trackbar)
cv2.createTrackbar('Max distance', 'Trackbars', max_distance, 10000 , on_trackbar)


pipeline.start(config)

try:
    while True:
        
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        
        
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        max_distance = cv2.getTrackbarPos('Max distance', 'Trackbars')
        min_distance = cv2.getTrackbarPos('Min distance', 'Trackbars')
        
        depth_image[ depth_image > max_distance] = 0
        depth_image[ depth_image < min_distance] = 0
       
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
            if( 18000 <  area < 30000 ):
             cv2.drawContours(result, contours, i, (0, 0, 255), 2)
             print("Area of object {}: {:.2f} pixels".format(i+1, area))
             moments = cv2.moments(contour)
             center_x = int(moments['m10'] / moments['m00'])
             center_y = int(moments['m01'] / moments['m00'])
             distance = depth_frame.get_distance(center_x, center_y)
             cv2.circle(result, (center_x,center_y), 5, (0,255,255), cv2.FILLED)
             print("jarak gawang  : {:.2f} meters".format(distance))

        hsving = cv2.cvtColor(color_image,cv2.COLOR_BGR2HSV)

        graying = cv2.cvtColor(color_image,cv2.COLOR_BGR2GRAY)

        lower_white = np.array([0,74,3], dtype=np.uint8)
        upper_white = np.array([45,136,81], dtype=np.uint8)

        mask = cv2.inRange(hsving , lower_white, upper_white)
        moments2 = cv2.moments(mask)
        center_x = int(moments2['m10'] / moments2['m00'])
        center_y = int(moments2['m01'] / moments2['m00'])
        cv2.circle(mask, (center_x,center_y), 5, (0,255,255), cv2.FILLED)
        distance = depth_frame.get_distance(center_x, center_y)
        print("jarak : {:.2f} meters".format(distance))
            
        #print(center_y)
        images = np.hstack((color_image, depth_colormap))

        # print(template_size)
        cv2.imshow("window", images)
        #cv2.imshow("window2", dil)
        #cv2.imshow("depth", depth_colormap)
        #cv2.imshow("cut", template)
        cv2.imshow("mask",mask)
        cv2.imshow("hasil",result)
      
        if cv2.waitKey(1) == 27:
            break
finally:
    
    pipeline.stop()


cv2.destroyAllWindows()
