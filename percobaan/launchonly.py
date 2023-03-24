import cv2
import pyrealsense2 as rs
import numpy as np


pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)


pipeline.start(config)


try:
    while True:
        
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        
        
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

       
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET
        )

        #print(center_y)
        images = np.hstack((color_image, depth_colormap))
        
        # print(template_size)
        cv2.imshow("window", images)
        if cv2.waitKey(1) == 27:
            break
finally:
    
    pipeline.stop()


cv2.destroyAllWindows()
