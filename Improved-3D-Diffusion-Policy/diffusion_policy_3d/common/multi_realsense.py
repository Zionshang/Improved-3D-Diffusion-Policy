#!/usr/bin/env python3
import cv2
import numpy as np
import pyrealsense2 as rs
from multiprocessing import Process, Queue
import time
import multiprocessing
from diffusion_policy_3d.common.pcd_downsampling import grid_sample_pcd, color_weighted_downsample, random_uniform_downsample

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
        enable_rgb=False,
        enable_depth=False,
        enable_pointcloud=False,
        sync_mode=0,
        num_points=2048,
        z_far=1.0,
        z_near=0.1,
        use_grid_sampling=False,
        grid_size=0.002,
        img_size=384,
        resize_image=False,
        use_color_sampling=False,
        target_color=(255, 255, 0),
        color_temperature=20.0,
        use_uniform_sampling=False,
    ) -> None:
        super(SingleVisionProcess, self).__init__()
        self.queue = queue
        self.device = device

        self.enable_rgb = enable_rgb
        self.enable_depth = enable_depth
        self.enable_pointcloud = enable_pointcloud
        self.sync_mode = sync_mode

        self.use_grid_sampling = use_grid_sampling
        self.grid_size = grid_size

        self.resize_image = resize_image
        # self.height, self.width = 512, 512
        self.height, self.width = img_size, img_size

        # point cloud params
        self.z_far = z_far
        self.z_near = z_near
        self.num_points = num_points
        self.use_color_sampling = use_color_sampling
        self.target_color = target_color
        self.color_temperature = color_temperature
        self.use_uniform_sampling = use_uniform_sampling

    def get_vision(self):
        frame = self.pipeline.wait_for_frames()

        if self.enable_depth:
            aligned_frames = self.align.process(frame)
            # Get aligned frames
            color_frame = aligned_frames.get_color_frame()
            color_frame = np.asanyarray(color_frame.get_data())

            depth_frame = aligned_frames.get_depth_frame()
            depth_frame = np.asanyarray(depth_frame.get_data())

            depth_frame = depth_frame.astype(np.float32)
            depth_frame *= self.depth_scale

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
        self.pipeline, self.align, self.depth_scale, self.camera_info = init_given_realsense_L515(
            self.device, enable_rgb=self.enable_rgb, enable_depth=self.enable_depth, sync_mode=self.sync_mode
        )

        while True:
            # 获取原始图像和点云数据
            color_frame, depth_frame, point_cloud_frame = self.get_vision()

            # 图像预处理
            if self.resize_image:
                if self.enable_rgb and color_frame is not None:
                    color_frame = cv2.resize(color_frame, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
                if self.enable_depth and depth_frame is not None:
                    depth_frame = cv2.resize(depth_frame, (self.width, self.height), interpolation=cv2.INTER_LINEAR)

            # 点云预处理
            if self.enable_pointcloud and point_cloud_frame is not None:
                # 体素下采样
                if self.use_grid_sampling:
                    point_cloud_frame = grid_sample_pcd(point_cloud_frame, grid_size=self.grid_size)
                if self.use_color_sampling:
                    # 颜色引导采样
                    # Handle shared array for target_color
                    if hasattr(self.target_color, 'get_obj'): # Check if it is a shared array
                         current_target_color = tuple(self.target_color[:])
                    else:
                         current_target_color = self.target_color

                    point_cloud_frame = color_weighted_downsample(
                        point_cloud_frame, self.num_points, target_color=current_target_color, temperature=self.color_temperature
                    )
                if self.use_uniform_sampling:
                    point_cloud_frame = random_uniform_downsample(point_cloud_frame, self.num_points)
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


class L515Camera:
    def __init__(
        self,
        device_idx=0,
        num_points=4096,
        z_far=1.0,
        z_near=0.1,
        use_grid_sampling=False,
        grid_size=0.002,
        img_size=512,
        enable_rgb=False,
        enable_depth=False,
        enable_pointcloud=False,
        resize_image=False,
        use_color_sampling=False,
        use_uniform_sampling=False,
        target_color=(255, 255, 0),
        color_temperature=20.0,
    ):
        self.devices = get_realsense_id()
        self.device = self.devices[device_idx]

        self.queue = Queue(maxsize=3)
        
        # Create shared array for target_color
        self.shared_target_color = multiprocessing.Array('i', 3)
        self.shared_target_color[:] = target_color

        self.process = SingleVisionProcess(
            self.device,
            self.queue,
            enable_rgb=enable_rgb,
            enable_depth=enable_depth,
            enable_pointcloud=enable_pointcloud,
            sync_mode=0,
            num_points=num_points,
            z_far=z_far,
            z_near=z_near,
            use_grid_sampling=use_grid_sampling,
            grid_size=grid_size,
            img_size=img_size,
            resize_image=resize_image,
            use_color_sampling=use_color_sampling,
            use_uniform_sampling=use_uniform_sampling,
            target_color=self.shared_target_color,
            color_temperature=color_temperature,
        )

        self.process.start()
        print("L515 camera process started.")

    def set_target_color(self, color):
        self.shared_target_color[:] = color

    def __call__(self):
        color, depth, point_cloud = self.queue.get()
        return {"color": color, "depth": depth, "point_cloud": point_cloud}

    def finalize(self):
        self.process.terminate()

    def __del__(self):
        self.finalize()


if __name__ == "__main__":
    cam = L515Camera(
            num_points=4096,
            enable_depth=True,
            enable_rgb=True,  # 为了使得深度图与rgb对齐，必须开启
            enable_pointcloud=True,
            # use_uniform_sampling=True,
            use_grid_sampling=True,
            grid_size=0.002,
            use_color_sampling=True,
            target_color=(100, 100, 0),
            color_temperature=10.0,
            z_far=1.2,
            z_near=0.2,
        )

    from open3d_viz import AsyncPointCloudViewer

    viewer = AsyncPointCloudViewer(
        width=960,
        height=720,
        point_size=5.0,
        queue_size=1,
        window_name="RealSense Live Point Cloud",
    )

    try:
        while True:
            out = cam()
            pc = out.get("point_cloud", None)
            if pc is not None and pc.size:
                viewer.update(pc)
                print("Point cloud size:", pc.shape)
            else:
                time.sleep(0.01)
    except KeyboardInterrupt:
        pass
    finally:
        viewer.close()
        cam.finalize()
