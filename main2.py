#percobaan
import cv2
import pyrealsense2 as rs
import numpy as np

template = cv2.imread("/home/hafizh/Pictures/pic5.png") # read the template image in grayscale
template = cv2.resize(template, (640,480)) # resize the template to match its size in the depth image # convert the template to uint8
bag = cv2.VideoCapture("/home/hafizh/Videos/bla.bag")
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)


pipeline.start(config)

max_distance = 3700

try:
    while True:
        rs.config.enable_device_from_file(config, bag)
        ret, frame = bag.read()
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        
        
        depth_image = np.asanyarray(depth_frame.get_data()) # convert depth image to uint8
        color_image = np.asanyarray(color_frame.get_data())

        
        depth_image[depth_image > max_distance] = 0

       
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET
        )
        graydepth = cv2.cvtColor(depth_colormap,cv2.COLOR_BGR2GRAY)
        graylap = cv2.cvtColor(template,cv2.COLOR_BGR2GRAY)

        blurgdepth = cv2.GaussianBlur(graydepth,(7,7),0)
        blurglap = cv2.GaussianBlur(graylap,(7,7),0)
        
        # perform template matching in the depth image
       
        # create a mask that is black everywhere except where the template is located
        mask = np.zeros_like(depth_colormap)

        # apply the mask to the depth image
        depth_image = cv2.bitwise_and(mask, template)

        images = np.hstack((color_image, depth_colormap))
        bag.release()
        cv2.imshow("window", images)
        cv2.imshow("dep",depth_image)
        cv2.imshow("ress",template)
        cv2.imshow("resss",frame)
        if cv2.waitKey(1) == 27:
            break
finally:
    
    pipeline.stop()


cv2.destroyAllWindows()
