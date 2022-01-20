import pyrealsense2 as rs
import numpy as np
import cv2
import os
import json
import time


def find_device_that_supports_advanced_mode():
    ctx = rs.context()
    ds5_dev = rs.device()
    devices = ctx.query_devices()
    for dev in devices:
        if dev.supports(rs.camera_info.product_id) and str(dev.get_info(rs.camera_info.product_id)) in DS5_product_ids:
            if dev.supports(rs.camera_info.name):
                print("Found device that supports advanced mode:", dev.get_info(rs.camera_info.name))
            return dev
    raise Exception("No D400 product line device that supports advanced mode was found")


def set_advanced_mode_from_json(jsonfilepath):
    try:
        dev = find_device_that_supports_advanced_mode()
        advnc_mode = rs.rs400_advanced_mode(dev)
        print("Advanced mode is", "enabled" if advnc_mode.is_enabled() else "disabled")

        # Loop until we successfully enable advanced mode
        while not advnc_mode.is_enabled():
            print("Trying to enable advanced mode...")
            advnc_mode.toggle_advanced_mode(True)
            # At this point the device will disconnect and re-connect.
            print("Sleeping for 5 seconds...")
            time.sleep(5)
            # The 'dev' object will become invalid and we need to initialize it again
            dev = find_device_that_supports_advanced_mode()
            advnc_mode = rs.rs400_advanced_mode(dev)
            print("Advanced mode is", "enabled" if advnc_mode.is_enabled() else "disabled")

        #根据json文件对摄像机进行设置
        with open(jsonfilepath, "r") as f:
            as_json_object = json.load(f)
            if type(next(iter(as_json_object))) != str:
                as_json_object = {k.encode('utf-8'): v.encode("utf-8") for k, v in as_json_object.items()}
            json_string = str(as_json_object).replace("'", '\"')
            advnc_mode.load_json(json_string)
            print('setting over')
        # serialized_string = advnc_mode.serialize_json()
        # print("Controls as JSON: \n", serialized_string)
    except Exception as e:
        print(e)
        pass


def stream_config(read_path):
    pipeline = rs.pipeline()  # 定义流程pipeline
    config = rs.config()  # 定义配置config

    # 如果从bag文件中读取，加上文件路径
    rs.config.enable_device_from_file(config, read_path)

    # 配置depth流、color流
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # 开始传输
    profile = pipeline.start(config)

    # 创建对齐对象
    alight_to = rs.stream.color  # 向rgb图像对齐
    align = rs.align(alight_to)

    return pipeline, profile, align


def obtain_frames_from_stream(pipeline, align):

    # 获得depth、color帧
    frames = pipeline.wait_for_frames()

    # frames.get_depth_frame()是640x360的深度图像
    # 进行对齐处理
    aligned_frames = align.process(frames)

    # 获得对齐后的深度帧和色彩帧
    depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()

    # 验证两张图像都得到了
    if not depth_frame or not color_frame:
        return None

    return depth_frame, color_frame


# 定义过滤器
def filters_config():

    spatial = rs.spatial_filter()
    spatial.set_option(rs.option.filter_magnitude, 5)
    spatial.set_option(rs.option.filter_smooth_alpha, 1)
    spatial.set_option(rs.option.filter_smooth_delta, 50)
    spatial.set_option(rs.option.holes_fill, 3)  # 5 = fill all the zero pixels

    temporal = rs.temporal_filter()

    hole_filling = rs.hole_filling_filter()

    depth_to_disparity = rs.disparity_transform(True)
    disparity_to_depth = rs.disparity_transform(False)

    return spatial, temporal, hole_filling, depth_to_disparity, disparity_to_depth


def depth_processing(depth_frame, spatial, temporal, hole_filling, depth_to_disparity, disparity_to_depth):

    disparity_depth = depth_to_disparity.process(depth_frame)
    # colorized_depth = np.asanyarray(colorizer.colorize(disparity_depth).get_data())
    # print('disparity depth', np.shape(colorized_depth))
    # cv2.imshow('Disparity Depth', colorized_depth)

    # Spatial Filter is a fast implementation of Domain-Transform Edge Preserving Smoothing
    filtered_depth = spatial.process(disparity_depth)
    # colorized_depth = np.asanyarray(colorizer.colorize(filtered_depth).get_data())
    # print('spatial filtered depth', np.shape(colorized_depth))
    # cv2.imshow('Spatial Filtered Depth', colorized_depth)

    # Our implementation of Temporal Filter does basic temporal smoothing and hole-filling.
    temp_filtered = temporal.process(filtered_depth)
    # colorized_depth = np.asanyarray(colorizer.colorize(temp_filtered).get_data())
    # print('temporal filtered depth', np.shape(colorized_depth))
    # cv2.imshow('Temp Filtered Depth', colorized_depth)

    to_depth = disparity_to_depth.process(temp_filtered)

    filled_depth = hole_filling.process(to_depth)

    # 显示处理后的深度图像
    # colorized_depth = np.asanyarray(colorizer.colorize(filled_depth).get_data())
    # print('filled depth', np.shape(colorized_depth))
    # cv2.imshow('Hole Filled Depth', colorized_depth)

    return filled_depth


def getData_align2rgb(save_path, NO):

    try:

        global dir_path

        # 配置图像流
        pipeline, profile, align = stream_config(save_path + os.sep + dir_path + '.bag')

        # 定义过滤器
        spatial, temporal, hole_filling, depth_to_disparity, disparity_to_depth = filters_config()

        # 定义着色器
        colorizer = rs.colorizer(color_scheme=0)

        # 设置高密度模式
        jsonfilepath = './data/HighDensityPreset.json'
        set_advanced_mode_from_json(jsonfilepath)

        intr_profile = profile.get_stream(rs.stream.color)
        intr = intr_profile.as_video_stream_profile().get_intrinsics()
        intrisic_matrix = np.eye(3)
        intrisic_matrix[0, 0] = intr.fx
        intrisic_matrix[0, 2] = intr.ppx
        intrisic_matrix[1, 1] = intr.fy
        intrisic_matrix[1, 2] = intr.ppy
        print(intrisic_matrix)

        # 获得RGB和深度帧
        index = 0
        while index < 30:
            index = index + 1
            depth_frame, color_frame = obtain_frames_from_stream(pipeline, align)

        processed_frame = depth_processing(depth_frame, spatial, temporal, hole_filling, depth_to_disparity, disparity_to_depth)

        # processed_frame是frame类，需要转换成depth_frame类计算距离
        processed_depth = processed_frame.as_depth_frame()

        colorized_depth = np.asanyarray(colorizer.colorize(processed_depth).get_data())

        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(processed_depth.get_data())

        # 保存为png格式图片
        cv2.imwrite(save_path + os.sep + '0000%d-color.png' % NO, color_image)
        cv2.imwrite(save_path + os.sep + '0000%d-depth.png' % NO, depth_image)

        # 保存frame文件
        sv_color = rs.save_single_frameset(save_path + os.sep + '0000%d-color' % NO)
        sv_depth = rs.save_single_frameset(save_path + os.sep + '0000%d-depth' % NO)
        sv_color.process(color_frame)
        sv_depth.process(processed_depth)

        cv2.namedWindow('Processed Depth', cv2.WINDOW_NORMAL)
        cv2.imshow('Processed Depth', colorized_depth)

        cv2.namedWindow('RGB Image', cv2.WINDOW_NORMAL)
        cv2.imshow('RGB Image', color_image)

        print('************************')

        key = cv2.waitKey(0)

        # 按esc退出
        if key == 27:
            cv2.destroyAllWindows()

    finally:
        pipeline.stop()


if __name__ == '__main__':

    dir_path = '0000210'
    getData_align2rgb('./data/' + dir_path, 0)
