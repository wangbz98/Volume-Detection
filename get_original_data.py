import pyrealsense2 as rs
import numpy as np
import cv2
import os
import json


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


def get_ori_data(save_path, NO):

    pipeline = rs.pipeline()  # 定义流程pipeline
    config = rs.config()  # 定义配置config

    # 配置depth流、color流
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # 开始传输
    profile = pipeline.start(config)

    try:

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
        while 1:

            while index < 30:
                index = index + 1
                frames = pipeline.wait_for_frames()

            # 获得对齐后的深度帧和色彩帧
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()

            # 验证两张图像都得到了
            if depth_frame and color_frame:
                break

        sv = rs.save_single_frameset(save_path + os.sep + '0000%d' % NO)
        sv.process(frames)

    finally:
        pipeline.stop()


if __name__ == '__main__':

    save_path = './data'
    NO = 2

    get_ori_data(save_path, NO)




