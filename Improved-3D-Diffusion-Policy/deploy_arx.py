import sys

# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode="w", buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode="w", buffering=1)

import hydra
import time
from omegaconf import OmegaConf
import pathlib
from diffusion_policy_3d.workspace.base_workspace import BaseWorkspace

# import tqdm
import torch
import os

os.environ["WANDB_SILENT"] = "True"
# allows arbitrary python code execution in configs using the ${eval:''} resolver
OmegaConf.register_new_resolver("eval", eval, replace=True)


from diffusion_policy_3d.common.multi_realsense import L515Camera

import numpy as np
import torch
from termcolor import cprint
from collections import deque

# import arx5 python interface
ARX_PY_DIR = pathlib.Path(__file__).resolve().parents[1].joinpath("arx5-sdk", "python")
sys.path.append(str(ARX_PY_DIR))
import arx5_interface as arx5
from open3d_viz import AsyncPointCloudViewer


class ArxX5EnvInference:
    """
    使用 ARX5 关节控制接口进行推理部署：
    - 观测包含 agent_pos(6关节+抓手=7) 与可选 point_cloud
    - 执行策略输出的连续关节指令
    """

    def __init__(
        self,
        obs_horizon=2,
        action_horizon=8,
        device="cpu",
        use_point_cloud=True,
        use_image=False,
        model="X5",
        interface="can0",
        visualize_point_cloud=True,
    ):

        # obs/action
        self.use_point_cloud = use_point_cloud
        self.use_image = use_image
        self.visualize_point_cloud = visualize_point_cloud and use_point_cloud

        # camera
        self.camera = L515Camera(
            num_points=4096,
            enable_depth=True,
            enable_rgb=True,  # 为了使得深度图与rgb对齐，必须开启
            enable_pointcloud=True,
            use_grid_sampling=True,
            grid_size=0.002,
            use_color_sampling=True,
            target_color=(255, 0, 255),
            color_temperature=10.0,
            z_far=1.2,
            z_near=0.2,
        )

        # horizon
        self.obs_horizon = obs_horizon
        self.action_horizon = action_horizon

        # inference device
        if device == "gpu":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cpu")
        print(f"Using device: {self.device}")

        # ARX5 控制器
        self.robot_config = arx5.RobotConfigFactory.get_instance().get_config(model)
        self.controller = arx5.Arx5JointController(model, interface)

        # 点云可视化（独立进程，非阻塞）
        self._pcd_viewer = AsyncPointCloudViewer(window_name="ARX5 Live Point Cloud", point_size=5) if self.visualize_point_cloud else None

    def set_target_color(self, color):
        if hasattr(self.camera, 'set_target_color'):
            self.camera.set_target_color(color)
            print(f"Target color updated to {color}")
        else:
            print("Camera does not support setting target color.")

    def step(self, action_list):

        for action_id in range(self.action_horizon):
            act = np.asarray(action_list[action_id]).reshape(-1)
            self.action_buf.append(act)
            # print(f"Action ID: {action_id}, Act: {act}")

            # 从action中构造关节命令
            js_cmd = arx5.JointState(self.robot_config.joint_dof)
            js_cmd.pos()[:] = act[: self.robot_config.joint_dof]
            js_cmd.gripper_pos = float(act[self.robot_config.joint_dof])
            js_cmd.timestamp = self.controller.get_timestamp() + (action_id + 1) * 0.08
            self.controller.set_joint_cmd(js_cmd)

            # 读取当前状态以构造observations
            js_current = self.controller.get_joint_state()
            arm_pos = js_current.pos().copy()
            grip_pos = js_current.gripper_pos
            env_qpos = np.concatenate([arm_pos, np.array([grip_pos])])
            self.env_qpos_buf.append(env_qpos)

            # 读取相机数据以构造observations
            cam_dict = self.camera()
            if self.use_point_cloud:
                self.cloud_buf.append(cam_dict["point_cloud"])
                if self._pcd_viewer is not None:
                    self._pcd_viewer.update(cam_dict["point_cloud"])
            if self.use_image:
                self.color_buf.append(cam_dict.get("color"))
                self.depth_buf.append(cam_dict.get("depth"))

        # 构造观测
        agent_pos = np.stack(list(self.env_qpos_buf), axis=0)
        obs_dict = {"agent_pos": torch.from_numpy(agent_pos).unsqueeze(0).to(self.device)}
        if self.use_point_cloud:
            obs_cloud = np.stack(list(self.cloud_buf), axis=0)
            obs_dict["point_cloud"] = torch.from_numpy(obs_cloud).unsqueeze(0).to(self.device)
        if self.use_image:
            obs_img = np.stack(list(self.color_buf), axis=0)
            obs_dict["image"] = torch.from_numpy(obs_img).permute(0, 3, 1, 2).unsqueeze(0)

        return obs_dict

    def reset(self, first_init=True):
        # init sliding window buffers (keep only last obs_horizon frames)
        self.color_buf = deque(maxlen=self.obs_horizon)
        self.depth_buf = deque(maxlen=self.obs_horizon)
        self.cloud_buf = deque(maxlen=self.obs_horizon)
        self.env_qpos_buf = deque(maxlen=self.obs_horizon)

        # keep only recent actions if needed
        self.action_buf = deque(maxlen=self.action_horizon)

        # 复位到 home
        if first_init:
            self.controller.reset_to_home()
            time.sleep(2.0)
            js_current = self.controller.get_joint_state()
            js_current.gripper_pos = self.robot_config.gripper_width
            js_current.timestamp = self.controller.get_timestamp() + 1.0
            self.controller.set_joint_cmd(js_current)
            time.sleep(2.0)
        print("ARX5 ready!")

        # 初始化一次相机
        cam_dict = self.camera()
        if self.use_point_cloud:
            self.cloud_buf.append(cam_dict["point_cloud"])
            if self._pcd_viewer is not None:
                self._pcd_viewer.update(cam_dict["point_cloud"])
        if self.use_image:
            self.color_buf.append(cam_dict.get("color"))
            self.depth_buf.append(cam_dict.get("depth"))

        # 读取一次关节状态
        js_current = self.controller.get_joint_state()
        arm_pos = js_current.pos().copy()
        grip_pos = js_current.gripper_pos
        env_qpos = np.concatenate([arm_pos, np.array([grip_pos])])
        self.env_qpos_buf.append(env_qpos)

        # 在重置环境后还没有历史帧的情况下，用当前最新一帧复制出一段长度为 obs_horizon 的观测序列
        agent_pos = np.stack([self.env_qpos_buf[-1]] * self.obs_horizon, axis=0)
        obs_dict = {"agent_pos": torch.from_numpy(agent_pos).unsqueeze(0).to(self.device)}
        if self.use_point_cloud:
            obs_cloud = np.stack([self.cloud_buf[-1]] * self.obs_horizon, axis=0)
            obs_dict["point_cloud"] = torch.from_numpy(obs_cloud).unsqueeze(0).to(self.device)
        if self.use_image:
            obs_img = np.stack([self.color_buf[-1]] * self.obs_horizon, axis=0)
            obs_dict["image"] = torch.from_numpy(obs_img).permute(0, 3, 1, 2).unsqueeze(0)

        return obs_dict


@hydra.main(config_path=str(pathlib.Path(__file__).parent.joinpath("diffusion_policy_3d", "config")))
def main(cfg: OmegaConf):
    torch.manual_seed(42)
    # resolve immediately so all the ${now:} resolvers
    # will use the same time.
    OmegaConf.resolve(cfg)
    cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = cls(cfg)

    if workspace.__class__.__name__ == "DPWorkspace":
        use_image = True
        use_point_cloud = False
    else:
        use_image = False
        use_point_cloud = True

    # fetch policy model
    policy = workspace.get_model()
    action_horizon = policy.horizon - policy.n_obs_steps + 1
    roll_out_length = 400

    first_init = True

    # 读取 ARX model / interface
    arx_model = "X5"
    arx_interface = "can0"

    env = ArxX5EnvInference(
        obs_horizon=policy.n_obs_steps,
        action_horizon=action_horizon,
        device="cpu",
        use_point_cloud=use_point_cloud,
        use_image=use_image,
        model=arx_model,
        interface=arx_interface,
        visualize_point_cloud=True,
    )

    if not torch.cuda.is_available():
        cprint("CUDA is not available! Inference will be slow on CPU.", "red", attrs=["bold"])
    else:
        cprint(f"CUDA is available. Using device: {env.device}", "green", attrs=["bold"])

    # Move policy to the inference device
    policy.to(env.device)

    obs_dict = env.reset(first_init=first_init)

    step_count = 0

    while step_count < roll_out_length:
        with torch.no_grad():
            t_start = time.time()
            action = policy(obs_dict)[0]
            t_end = time.time()
            print(f"inference time: {(t_end - t_start) * 1000.0:.2f} ms")
            action_list = [act.cpu().numpy() for act in action]

        obs_dict = env.step(action_list)
        step_count += action_horizon
        print(f"step: {step_count}")

    env.controller.reset_to_home()
    # 关闭可视化进程
    _viewer = getattr(env, "_pcd_viewer", None)
    if _viewer is not None:
        _viewer.close()


if __name__ == "__main__":
    main()
