import pyrealsense2 as rs
import numpy as np
import cv2
import os
import random
import math


def read_frames(read_path):

    try:

        pipeline_color = rs.pipeline()
        pipeline_depth = rs.pipeline()

        config_color = rs.config()
        config_depth = rs.config()

        rs.config.enable_device_from_file(config_color, read_path + os.sep + '00000-color34.bag')
        rs.config.enable_device_from_file(config_depth, read_path + os.sep + '00000-depth10.bag')

        config_color.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config_depth.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

        pipeline_color.start(config_color)
        pipeline_depth.start(config_depth)

        color_frame = pipeline_color.wait_for_frames().get_color_frame()
        depth_frame = pipeline_depth.wait_for_frames().get_depth_frame()

        colorizer = rs.colorizer(color_scheme=0)

        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(colorizer.colorize(depth_frame).get_data())

        cv2.imshow('color', color_image)
        cv2.imshow('depth', depth_image)

        cv2.waitKey(0)

    finally:
        pipeline_color.stop()
        pipeline_depth.stop()

    return color_frame, depth_frame


# pixel（宽，高）
def pixel_to_point(pixel, depth_frame):

    # 获取相机深度内参，用于获得真实空间坐标
    intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics

    # 获取深度图像各点的距离
    dist = depth_frame.get_distance(pixel[0], pixel[1])

    # 映射各点到空间坐标系下
    point = rs.rs2_deproject_pixel_to_point(intrinsics, pixel, dist)

    return point


def distance_3d(ori_upixel, ori_vpixel, depth_frame):

    upoint = pixel_to_point(ori_upixel, depth_frame)
    vpoint = pixel_to_point(ori_vpixel, depth_frame)

    # 计算两点之间的距离
    dist_3d = math.sqrt(math.pow(upoint[0] - vpoint[0], 2) +
                        math.pow(upoint[1] - vpoint[1], 2) +
                        math.pow(upoint[2] - vpoint[2], 2))

    return dist_3d


# 随机获取桌面上三点
def plane_random_pixels(np_label):
    plane_x, plane_y = np.where(np_label == 0)  # 高，宽
    num_plane = len(plane_x)
    plane_pixels = []
    while True:

        while True:
            if len(plane_pixels) == 3:
                break
            rand = int(random.uniform(0, num_plane))
            if 160 < plane_x[rand] < 320 or 220 < plane_y[rand] < 420:
                continue
            else:
                plane_pixels.append([plane_y[rand], plane_x[rand]])  # 宽，高

        if (plane_pixels[2][0] - plane_pixels[0][0]) / (plane_pixels[1][0] - plane_pixels[0][0]) == \
                (plane_pixels[2][1] - plane_pixels[0][1]) / (plane_pixels[1][1] - plane_pixels[0][1]):
            plane_pixels.clear()
            continue
        else:
            break

    return plane_pixels


# 构建桌面平面方程
def construct_plane(pointA, pointB, pointC):
    """
    法向量    ：n=(Nx,Ny,Nz)
    平面上某点：A=(Ax,Ay,Az)
    点法式方程：Nx(X-Ax)+Ny(Y-Ay)+Nz(Z-Az)
    :param pointA:
    :param pointB:
    :param pointC:
    :return:（Nx, Ny, Nz, D）代表：Nx X + Ny Y + Nz Z + D = 0
    """
    pointA = np.asarray(pointA)
    pointB = np.asarray(pointB)
    pointC = np.asarray(pointC)

    vector_AB = np.asmatrix(pointB - pointA)
    vector_AC = np.asmatrix(pointC - pointA)

    n = np.cross(vector_AB, vector_AC)  # 向量叉乘，求法向量

    Nx = n[0, 0]
    Ny = n[0, 1]
    Nz = n[0, 2]

    D = -(Nx * pointA[0] + Ny * pointA[1] + Nz * pointA[2])

    return Nx, Ny, Nz, D


# 计算物体顶面上一点到桌面的距离为高
def distance_point_to_plane(plane_points, point_top):
    """
    距离计算公式：|AD·n| / |n|
    :param plane_points:(pointA, pointB, pointC)平面内三点
    :param point_top:平面外一点
    :return:点到面的距离
    """
    Nx, Ny, Nz, D = construct_plane(plane_points[0], plane_points[1], plane_points[2])

    AD_n = Nx * point_top[0] + Ny * point_top[1] + Nz * point_top[2] + D

    mod_n = np.sqrt(np.sum(np.square([Nx, Ny, Nz])))

    distance = abs(AD_n) / mod_n

    return distance


# 获取关键点
def key_pixel_get(np_label, NO_object):

    # 若输入为tensor
    # np_label = np.array(np_label.cpu())

    index_x, index_y = np.where(np_label == NO_object)  # 高，宽

    pixel_first = [index_y[0], index_x[0]]  # 最上面一行最左的点（宽，高）

    pixel_last = [index_y[-1], index_x[-1]]  # 最下面一行最右的点（宽，高）

    return pixel_first, pixel_last


def volume_calculate(object_shape, l, h):

    if object_shape == 'Column':
        S = math.pi * ((0.5 * l) ** 2)

    elif object_shape == 'Five':
        S = 5/8 * (l ** 2) * (math.tan(0.2 * math.pi) + math.tan(0.1 * math.pi))

    elif object_shape == 'Four':
        S = 0.5 * (l ** 2)

    elif object_shape == 'Tri':
        S = 0.25 * (l ** 2) * math.tan(math.pi / 3)

    return h * S


def main(read_path, object_shapes):

    color_frame, depth_frame = read_frames(read_path)

    label = np.loadtxt(read_path + os.sep + '0_mask.txt')  # 高，宽

    cv2.imshow('Label', label)

    object_num = int(np.max(label))
    print(object_num)

    volumes = []

    for object_order in range(object_num):

        l_upixel, l_vpixel = key_pixel_get(label, object_order + 1)  # 宽，高

        divide = 4
        u = [int(1/divide * (l_upixel[0] - l_vpixel[0]) + l_vpixel[0]), int(1/divide * (l_upixel[1] - l_vpixel[1]) + l_vpixel[1])]
        v = [int((divide - 1)/divide * (l_upixel[0] - l_vpixel[0]) + l_vpixel[0]), int((divide - 1)/divide * (l_upixel[1] - l_vpixel[1]) + l_vpixel[1])]

        # u = [int(0.25 * (l_upixel[0] - l_vpixel[0]) + l_vpixel[0]), int(0.25 * (l_upixel[1] - l_vpixel[1]) + l_vpixel[1])]
        # v = [int(0.75 * (l_upixel[0] - l_vpixel[0]) + l_vpixel[0]), int(0.75 * (l_upixel[1] - l_vpixel[1]) + l_vpixel[1])]

        color_img = np.asanyarray(color_frame.get_data())
        cv2.circle(color_img, (l_upixel[0], l_upixel[1]), 3, (0, 0, 255))
        cv2.circle(color_img, (l_vpixel[0], l_vpixel[1]), 3, (0, 0, 255))
        cv2.circle(color_img, (u[0], u[1]), 3, (0, 0, 255))
        cv2.circle(color_img, (v[0], v[1]), 3, (0, 0, 255))

        # color_img = np.asanyarray(color_frame.get_data())
        # cv2.circle(color_img, (l_upixel[0], l_upixel[1]), 3, (0, 0, 255))
        # cv2.circle(color_img, (l_vpixel[0], l_vpixel[1]), 3, (0, 0, 255))
        # cv2.imshow('color1', color_img)

        # plane_pixels = plane_random_pixels(label)  # 宽，高

        # 指定桌面像素
        plane_pixels = [[int(l_upixel[0] / 2), int(l_upixel[1] / 2)], [int((640 + l_upixel[0]) / 2), int(l_upixel[1] / 2)],
                    [int((640 + l_vpixel[0]) / 2), int((480 + l_vpixel[1]) / 2)]]  # 宽，高

        cv2.circle(color_img, (plane_pixels[0][0], plane_pixels[0][1]), 3, (0, 0, 255))
        cv2.circle(color_img, (plane_pixels[1][0], plane_pixels[1][1]), 3, (0, 0, 255))
        cv2.circle(color_img, (plane_pixels[2][0], plane_pixels[2][1]), 3, (0, 0, 255))

        cv2.imshow('color1', color_img)
        cv2.waitKey(0)

        # 获得顶面点三维坐标和桌面点三维坐标
        plane_points = []
        for i in range(len(plane_pixels)):
            plane_points.append(pixel_to_point(plane_pixels[i], depth_frame))

        pixel_top = [int((l_upixel[0] + l_vpixel[0]) / 2), int((l_upixel[1] + l_vpixel[1]) / 2)]  # 宽，高

        point_top = pixel_to_point(pixel_top, depth_frame)

        # l = distance_3d(ori_upixel=l_upixel, ori_vpixel=l_vpixel, depth_frame=depth_frame)

        print('Align depth to RGB results: ')

        l = divide/(divide - 2) * distance_3d(ori_upixel=u, ori_vpixel=v, depth_frame=depth_frame)

        print('Side length is %.3f cm' % (l * 100))

        h = distance_point_to_plane(plane_points, point_top)

        print('Height is %.3f cm' % (h * 100))

        volumes.append(volume_calculate(object_shape=object_shapes[object_order], l=l, h=h))

    return volumes


if __name__ == '__main__':

    data_path = './data/0000210'
    object_shapes = ['Five', 'Four']

    volumes = main(data_path, object_shapes)

    for volume in volumes:

        print('\nVolume is %.3f cm³' % (volume * 1000000))


