import pyrealsense2 as rs
import numpy as np
import cv2


try:

    pipeline = rs.pipeline()

    config = rs.config()

    rs.config.enable_device_from_file(config, './data/0000210.bag')

    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    pipeline.start(config)

    colorizer = rs.colorizer(color_scheme=0)

    frames = pipeline.wait_for_frames()

    color_frame = frames.get_color_frame()

    color_img = np.asanyarray(color_frame.get_data())

    depth_frame = frames.get_depth_frame()

    depth_color_frame = colorizer.colorize(depth_frame)

    depth_color_image = np.asanyarray(depth_color_frame.get_data())

    cv2.imshow("Colr Stream", color_img)

    cv2.imshow("Depth Stream", depth_color_image)
    key = cv2.waitKey(0)

finally:
    pipeline.stop()
