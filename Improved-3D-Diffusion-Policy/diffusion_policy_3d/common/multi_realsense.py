#!/usr/bin/env python3
import cv2
import numpy as np
from collections import deque
import pyrealsense2 as rs
from multiprocessing import Process, Pipe, Queue, Event
import time
import multiprocessing
from diffusion_policy_3d.common.pcd_downsampling import grid_sample_pcd, random_uniform_downsample, color_weighted_downsample

multiprocessing.set_start_method("fork")

np.printoptions(3, suppress=True)


def get_realsense_id():
    ctx = rs.context()
    devices = ctx.query_devices()
    devices = [devices[i].get_info(rs.camera_info.serial_number) for i in range(len(devices))]
    devices.sort()  # Make sure the order is correct
    print("Found {} devices: {}".format(len(devices), devices))
    return devices


def init_given_realsense_L515(
    device,
    enable_rgb=True,
    enable_depth=False,
    sync_mode=0,
):
    # use `rs-enumerate-devices` to check available resolutions
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(device)
    print("Initializing camera {}".format(device))

    if enable_depth:
        #     Depth         1024x768      @ 30Hz     Z16
        # Depth         640x480       @ 30Hz     Z16
        # Depth         320x240       @ 30Hz     Z16
        # L515
        h, w = 768, 1024
        config.enable_stream(rs.stream.depth, w, h, rs.format.z16, 30)
    if enable_rgb:
        # L515
        h, w = 540, 960
        config.enable_stream(rs.stream.color, w, h, rs.format.rgb8, 30)

    config.resolve(pipeline)
    profile = pipeline.start(config)

    if enable_depth:

        # Get the depth sensor (or any other sensor you want to configure)
        device = profile.get_device()
        depth_sensor = device.query_sensors()[0]

        # Set the inter-camera sync mode
        # Use 1 for master, 2 for slave, 0 for default (no sync)
        # for L515
        depth_sensor.set_option(rs.option.inter_cam_sync_mode, sync_mode)

        # set min distance
        # for L515
        # depth_sensor.set_option(rs.option.min_distance, 0.05)
        depth_sensor.set_option(rs.option.visual_preset, rs.l500_visual_preset.short_range)

        # get depth scale
        depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
        align = rs.align(rs.stream.color)

        depth_profile = profile.get_stream(rs.stream.depth)
        intrinsics = depth_profile.as_video_stream_profile().get_intrinsics()
        camera_info = CameraInfo(intrinsics.width, intrinsics.height, intrinsics.fx, intrinsics.fy, intrinsics.ppx, intrinsics.ppy)

        print("camera {} init.".format(device))
        return pipeline, align, depth_scale, camera_info
    else:
        print("camera {} init.".format(device))
        return pipeline, None, None, None


def init_given_realsense_D455(
    device,
    enable_rgb=True,
    enable_depth=False,
    sync_mode=0,
):
    # use `rs-enumerate-devices` to check available resolutions
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(device)
    print("Initializing camera {}".format(device))

    if enable_depth:
        #     Depth         1024x768      @ 30Hz     Z16
        # Depth         640x480       @ 30Hz     Z16
        # Depth         320x240       @ 30Hz     Z16

        # D455
        # h, w = 720, 1280
        h, w = 480, 640
        config.enable_stream(rs.stream.depth, w, h, rs.format.z16, 30)
    if enable_rgb:

        # h, w = 720, 1280
        h, w = 480, 640
        config.enable_stream(rs.stream.color, w, h, rs.format.rgb8, 30)

    config.resolve(pipeline)
    profile = pipeline.start(config)

    if enable_depth:

        # Get the depth sensor (or any other sensor you want to configure)
        device = profile.get_device()
        depth_sensor = device.query_sensors()[0]

        # get depth scale
        depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
        align = rs.align(rs.stream.color)

        depth_profile = profile.get_stream(rs.stream.depth)
        intrinsics = depth_profile.as_video_stream_profile().get_intrinsics()
        camera_info = CameraInfo(intrinsics.width, intrinsics.height, intrinsics.fx, intrinsics.fy, intrinsics.ppx, intrinsics.ppy)

        print("camera {} init.".format(device))
        return pipeline, align, depth_scale, camera_info
    else:
        print("camera {} init.".format(device))
        return pipeline, None, None, None


def init_given_realsense_D435(
    device,
    enable_rgb=True,
    enable_depth=False,
    sync_mode=0,
):
    # use `rs-enumerate-devices` to check available resolutions
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(device)
    print("Initializing camera {}".format(device))

    if enable_depth:
        #     Depth         1024x768      @ 30Hz     Z16
        # Depth         640x480       @ 30Hz     Z16
        # Depth         320x240       @ 30Hz     Z16

        # D455
        h, w = 720, 1280
        config.enable_stream(rs.stream.depth, w, h, rs.format.z16, 30)
    if enable_rgb:

        h, w = 720, 1280
        config.enable_stream(rs.stream.color, w, h, rs.format.rgb8, 30)

    config.resolve(pipeline)
    profile = pipeline.start(config)

    if enable_depth:

        # Get the depth sensor (or any other sensor you want to configure)
        device = profile.get_device()
        depth_sensor = device.query_sensors()[0]

        # get depth scale
        depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
        align = rs.align(rs.stream.color)

        depth_profile = profile.get_stream(rs.stream.depth)
        intrinsics = depth_profile.as_video_stream_profile().get_intrinsics()
        camera_info = CameraInfo(intrinsics.width, intrinsics.height, intrinsics.fx, intrinsics.fy, intrinsics.ppx, intrinsics.ppy)

        print("camera {} init.".format(device))
        return pipeline, align, depth_scale, camera_info
    else:
        print("camera {} init.".format(device))
        return pipeline, None, None, None


class CameraInfo:
    """Camera intrisics for point cloud creation."""

    def __init__(self, width, height, fx, fy, cx, cy, scale=1):
        self.width = width
        self.height = height
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.scale = scale


class SingleVisionProcess(Process):
    def __init__(
        self,
        device,
        queue,
        enable_rgb=True,
        enable_depth=False,
        enable_pointcloud=False,
        sync_mode=0,
        num_points=2048,
        z_far=1.0,
        z_near=0.1,
        use_grid_sampling=True,
        img_size=384,
    ) -> None:
        super(SingleVisionProcess, self).__init__()
        self.queue = queue
        self.device = device

        self.enable_rgb = enable_rgb
        self.enable_depth = enable_depth
        self.enable_pointcloud = enable_pointcloud
        self.sync_mode = sync_mode

        self.use_grid_sampling = use_grid_sampling

        self.resize = True
        # self.height, self.width = 512, 512
        self.height, self.width = img_size, img_size

        # point cloud params
        self.z_far = z_far
        self.z_near = z_near
        self.num_points = num_points

    def get_vision(self):
        frame = self.pipeline.wait_for_frames()

        if self.enable_depth:
            aligned_frames = self.align.process(frame)
            # Get aligned frames
            color_frame = aligned_frames.get_color_frame()
            color_frame = np.asanyarray(color_frame.get_data())

            depth_frame = aligned_frames.get_depth_frame()
            depth_frame = np.asanyarray(depth_frame.get_data())

            clip_lower = 0.01
            clip_high = 1.0
            depth_frame = depth_frame.astype(np.float32)
            depth_frame *= self.depth_scale
            depth_frame[depth_frame < clip_lower] = clip_lower
            depth_frame[depth_frame > clip_high] = clip_high

            if self.enable_pointcloud:
                # Nx6 raw point cloud
                point_cloud_frame = self.create_colored_point_cloud(color_frame, depth_frame, far=self.z_far, near=self.z_near)
            else:
                point_cloud_frame = None
        else:
            color_frame = frame.get_color_frame()
            color_frame = np.asanyarray(color_frame.get_data())
            depth_frame = None
            point_cloud_frame = None

        return color_frame, depth_frame, point_cloud_frame

    def run(self):
        device_name = "L515"
        if device_name == "L515":
            init_given_realsense = init_given_realsense_L515
        elif device_name == "D435":
            init_given_realsense = init_given_realsense_D435
        elif device_name == "D455":
            init_given_realsense = init_given_realsense_D455
        else:
            raise NotImplementedError("Unknown device name {}".format(device_name))

        self.pipeline, self.align, self.depth_scale, self.camera_info = init_given_realsense(
            self.device, enable_rgb=self.enable_rgb, enable_depth=self.enable_depth, sync_mode=self.sync_mode
        )

        debug = False
        while True:
            # 获取原始图像和点云数据
            color_frame, depth_frame, point_cloud_frame = self.get_vision()

            # 图像预处理
            if self.resize:
                if self.enable_rgb and color_frame is not None:
                    color_frame = cv2.resize(color_frame, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
                if self.enable_depth and depth_frame is not None:
                    depth_frame = cv2.resize(depth_frame, (self.width, self.height), interpolation=cv2.INTER_LINEAR)

            # 点云预处理
            if self.enable_pointcloud and point_cloud_frame is not None:
                # 体素下采样
                if self.use_grid_sampling:
                    point_cloud_frame = grid_sample_pcd(point_cloud_frame, grid_size=0.002)
                # 随机均匀采样
                point_cloud_frame = color_weighted_downsample(point_cloud_frame, self.num_points, target_color=(255, 255, 0), temperature=20.0)
                np.random.shuffle(point_cloud_frame)

            self.queue.put([color_frame, depth_frame, point_cloud_frame])
            time.sleep(0.05)

    def terminate(self) -> None:
        # self.pipeline.stop()
        return super().terminate()

    def create_colored_point_cloud(self, color, depth, far=1.0, near=0.1):
        assert depth.shape[0] == color.shape[0] and depth.shape[1] == color.shape[1]

        # Create meshgrid for pixel coordinates
        xmap = np.arange(color.shape[1])
        ymap = np.arange(color.shape[0])
        xmap, ymap = np.meshgrid(xmap, ymap)

        # Calculate 3D coordinates
        points_z = depth / self.camera_info.scale
        points_x = (xmap - self.camera_info.cx) * points_z / self.camera_info.fx
        points_y = (ymap - self.camera_info.cy) * points_z / self.camera_info.fy
        cloud = np.stack([points_x, points_y, points_z], axis=-1)
        cloud = cloud.reshape([-1, 3])

        # Clip points based on depth
        mask = (cloud[:, 2] < far) & (cloud[:, 2] > near)
        cloud = cloud[mask]
        color = color.reshape([-1, 3])
        color = color[mask]

        colored_cloud = np.hstack([cloud, color.astype(np.float32)])
        return colored_cloud


class MultiRealSense(object):
    def __init__(
        self,
        use_front_cam=True,
        use_right_cam=False,
        front_cam_idx=0,
        right_cam_idx=1,
        front_num_points=4096,
        right_num_points=1024,
        front_z_far=1.0,
        front_z_near=0.1,
        right_z_far=0.5,
        right_z_near=0.01,
        use_grid_sampling=True,
        img_size=512,
    ):

        self.devices = get_realsense_id()

        self.front_queue = Queue(maxsize=3)
        self.right_queue = Queue(maxsize=3)

        # 0: f1380328, 1: f1422212

        # sync_mode: Use 1 for master, 2 for slave, 0 for default (no sync)

        if use_front_cam:
            self.front_process = SingleVisionProcess(
                self.devices[front_cam_idx],
                self.front_queue,
                enable_rgb=True,
                enable_depth=True,
                enable_pointcloud=True,
                sync_mode=1,
                num_points=front_num_points,
                z_far=front_z_far,
                z_near=front_z_near,
                use_grid_sampling=use_grid_sampling,
                img_size=img_size,
            )
        if use_right_cam:
            self.right_process = SingleVisionProcess(
                self.devices[right_cam_idx],
                self.right_queue,
                enable_rgb=True,
                enable_depth=True,
                enable_pointcloud=True,
                sync_mode=1,
                num_points=right_num_points,
                z_far=right_z_far,
                z_near=right_z_near,
                use_grid_sampling=use_grid_sampling,
                img_size=img_size,
            )

        if use_front_cam:
            self.front_process.start()
            print("front camera start.")

        if use_right_cam:
            self.right_process.start()
            print("right camera start.")

        self.use_front_cam = use_front_cam
        self.use_right_cam = use_right_cam

    def __call__(self):
        cam_dict = {}
        if self.use_front_cam:
            front_color, front_depth, front_point_cloud = self.front_queue.get()
            cam_dict.update({"color": front_color, "depth": front_depth, "point_cloud": front_point_cloud})

        if self.use_right_cam:
            right_color, right_depth, right_point_cloud = self.right_queue.get()
            cam_dict.update({"right_color": right_color, "right_depth": right_depth, "right_point_cloud": right_point_cloud})
        return cam_dict

    def finalize(self):
        if self.use_front_cam:
            self.front_process.terminate()
        if self.use_right_cam:
            self.right_process.terminate()

    def __del__(self):
        self.finalize()


if __name__ == "__main__":
    cam = MultiRealSense(use_right_cam=False, front_num_points=4096, use_grid_sampling=False, img_size=1024, front_z_far=1.0, front_z_near=0.2)

    from open3d_viz import AsyncPointCloudViewer

    viewer = AsyncPointCloudViewer(
        width=960,
        height=720,
        point_size=2.0,
        queue_size=1,
        window_name="RealSense Live Point Cloud",
    )

    try:
        while True:
            out = cam()
            pc = out.get("point_cloud", None)
            if pc is not None and pc.size:
                viewer.update(pc)
            else:
                time.sleep(0.01)
    except KeyboardInterrupt:
        pass
    finally:
        viewer.close()
        cam.finalize()
