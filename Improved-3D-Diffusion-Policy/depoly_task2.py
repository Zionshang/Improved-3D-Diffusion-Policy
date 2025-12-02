import sys

# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode="w", buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode="w", buffering=1)

import os
import pathlib
import time
from dataclasses import dataclass
from typing import Dict, List, Optional
from datetime import datetime

import hydra
import torch
from omegaconf import OmegaConf
from termcolor import cprint

from deploy_arx import ArxX5EnvInference
from diffusion_policy_3d.workspace.base_workspace import BaseWorkspace

os.environ["WANDB_SILENT"] = "True"
OmegaConf.register_new_resolver("eval", eval, replace=True)
OmegaConf.register_new_resolver("now", lambda pattern: datetime.now().strftime(pattern), replace=True)


@dataclass
class TaskConfig:
    name: str
    run_dir: pathlib.Path
    step_length: int
    end_pose: List[float]  # [joint1...joint6, gripper]
    ckpt_tag: Optional[str] = "latest"  # can be "latest" or filename


class TaskEnv(ArxX5EnvInference):
    """Allow changing horizons between tasks."""

    def reconfigure(self, obs_horizon: int, action_horizon: int):
        self.obs_horizon = obs_horizon
        self.action_horizon = action_horizon


def resolve_checkpoint(workspace: BaseWorkspace, run_dir: pathlib.Path, ckpt_tag: Optional[str]) -> pathlib.Path:
    if ckpt_tag is None or ckpt_tag == "latest":
        return workspace.get_checkpoint_path(tag="latest")
    if ckpt_tag == "best":
        return workspace.get_checkpoint_path(tag="best")
    path = pathlib.Path(ckpt_tag)
    if path.suffix == ".ckpt" and path.is_file():
        return path
    ckpt_dir = run_dir.joinpath("checkpoints")
    candidate = ckpt_dir.joinpath(ckpt_tag)
    if candidate.suffix != ".ckpt":
        candidate = candidate.with_suffix(".ckpt")
    return candidate


def load_task(task: TaskConfig, device: torch.device):
    cfg_path = task.run_dir.joinpath(".hydra", "config.yaml")
    if not cfg_path.is_file():
        raise FileNotFoundError(f"未找到配置文件：{cfg_path}")

    cfg = OmegaConf.load(cfg_path)
    OmegaConf.resolve(cfg)
    cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = cls(cfg, output_dir=str(task.run_dir))

    ckpt_path = resolve_checkpoint(workspace, task.run_dir, task.ckpt_tag)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"未找到 checkpoint：{ckpt_path}")

    workspace.load_checkpoint(path=ckpt_path)
    policy = workspace.ema_model if getattr(cfg.training, "use_ema", False) else workspace.model
    policy = policy.eval().to(device)

    obs_horizon = policy.n_obs_steps
    action_horizon = policy.horizon - policy.n_obs_steps + 1
    use_image = workspace.__class__.__name__ == "DPWorkspace"
    use_point_cloud = not use_image

    cprint(
        f"[Task {task.name}] run_dir={task.run_dir}, ckpt={ckpt_path.name}, "
        f"obs_horizon={obs_horizon}, action_horizon={action_horizon}, step_len={task.step_length}",
        "cyan",
    )

    return {
        "task": task,
        "workspace": workspace,
        "policy": policy,
        "obs_horizon": obs_horizon,
        "action_horizon": action_horizon,
        "use_point_cloud": use_point_cloud,
        "use_image": use_image,
    }


def run_task(env: TaskEnv, policy: torch.nn.Module, step_len: int, obs_dict: Dict[str, torch.Tensor], name: str):
    step_count = 0
    while step_count < step_len:
        with torch.no_grad():
            t_start = time.time()
            action = policy(obs_dict)[0]
            t_end = time.time()
            print(f"[{name}] inference time: {(t_end - t_start) * 1000.0:.2f} ms")
            action_list = [act.detach().cpu().numpy() for act in action]
        obs_dict = env.step(action_list)
        step_count += len(action_list)
        print(f"[{name}] step: {step_count}/{step_len}")
    return obs_dict


def move_to_pose(env: TaskEnv, target: List[float], name: str):
    if len(target) < env.robot_config.joint_dof + 1:
        raise ValueError(f"{name} 的 end_pose 长度不足，需要 {env.robot_config.joint_dof + 1} 个值")
    js_cmd = env.controller.get_joint_state()
    js_cmd.pos()[:] = target[: env.robot_config.joint_dof]
    js_cmd.gripper_pos = float(target[env.robot_config.joint_dof])
    js_cmd.timestamp = env.controller.get_timestamp() + 1.0
    env.controller.set_joint_cmd(js_cmd)
    print(f"[{name}] 已发送结束位姿。")


def main():
    # 固定配置：如需调整路径/步长/结束姿态，请直接改这里
    ROOT = pathlib.Path(__file__).resolve().parent
    RUN_ROOT = ROOT.joinpath("data", "outputs")

    pick_task = TaskConfig(
        name="pick_kettle",
        run_dir=RUN_ROOT.joinpath("x5-3d-idp3-pick_kettle-153_seed0"),
        step_length=200,
        # TODO: 更新为实际的关节+夹爪目标值
        end_pose=[0, 0, 0, 0, 0, 0, 0.04],
    )
    place_task = TaskConfig(
        name="place_kettle",
        run_dir=RUN_ROOT.joinpath("x5-3d-idp3-place_kettle_seed0"),
        step_length=200,
        # TODO: 更新为实际的关节+夹爪目标值
        end_pose=[0, 0, 0, 0, 0, 0, 0.04],
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tasks = [pick_task, place_task]
    loaded = [load_task(t, device) for t in tasks]
    if not loaded:
        raise RuntimeError("未加载到任何任务。")

    first_use_pc = loaded[0]["use_point_cloud"]
    first_use_img = loaded[0]["use_image"]
    for ld in loaded[1:]:
        if (ld["use_point_cloud"] != first_use_pc) or (ld["use_image"] != first_use_img):
            raise ValueError("两个任务的观测类型不一致（必须都用点云或都用图像）。")

    env = TaskEnv(
        obs_horizon=loaded[0]["obs_horizon"],
        action_horizon=loaded[0]["action_horizon"],
        device="gpu" if device.type == "cuda" else "cpu",
        use_point_cloud=first_use_pc,
        use_image=first_use_img,
        model="X5",
        interface="can0",
        visualize_point_cloud=True,
    )

    menu = (
        "1: pick_kettle\n"
        "2: place_kettle\n"
        "r: reset 相机/状态\n"
        "q: 退出"
    )
    cprint("选择任务：\n" + menu, "yellow")

    obs_dict = None
    first_init = True
    try:
        while True:
            choice = input("请输入选项 (1/2/r/q): ").strip().lower()
            if choice in {"q", "quit", "exit"}:
                break
            if choice == "r":
                obs_dict = env.reset(first_init=False)
                first_init = False
                continue
            if choice not in {"1", "2"}:
                print("无效选项，请输入 1 / 2 / r / q")
                continue

            idx = 0 if choice == "1" else 1
            info = loaded[idx]
            task = tasks[idx]

            env.reconfigure(info["obs_horizon"], info["action_horizon"])
            obs_dict = env.reset(first_init=first_init)
            first_init = False
            obs_dict = run_task(env, info["policy"], task.step_length, obs_dict, task.name)
            move_to_pose(env, task.end_pose, task.name)
    except KeyboardInterrupt:
        cprint("收到中断信号，正在停止并复位...", "red")
    finally:
        env.controller.reset_to_home()
        viewer = getattr(env, "_pcd_viewer", None)
        if viewer is not None:
            viewer.close()


if __name__ == "__main__":
    main()
