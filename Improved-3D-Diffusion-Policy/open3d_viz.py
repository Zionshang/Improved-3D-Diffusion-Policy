import atexit
import contextlib
import multiprocessing as mp
import time

import numpy as np
import open3d as o3d


def _viewer_loop(queue, width=960, height=720, point_size=2.0, window_name="Live Point Cloud"):
    """子进程渲染循环：接收 (points, colors) 或 "__quit__"，展示最新点云。"""
    vis = o3d.visualization.Visualizer()  # type: ignore[attr-defined]
    vis.create_window(window_name=window_name, width=width, height=height, visible=True)
    pcd = o3d.geometry.PointCloud()

    added = False
    last_pts = None
    last_cols = None

    try:
        while True:
            try:
                item = queue.get(timeout=0.01)
            except Exception:
                item = None

            if item == "__quit__":
                break
            if isinstance(item, tuple):
                last_pts, last_cols = item

            if last_pts is not None and last_pts.size:
                pcd.points = o3d.utility.Vector3dVector(last_pts)
                if last_cols is not None and last_cols.shape == last_pts.shape:
                    pcd.colors = o3d.utility.Vector3dVector(last_cols)
                else:
                    z = last_pts[:, 2]
                    zmin = float(np.min(z))
                    zptp = float(np.ptp(z)) + 1e-9
                    c = ((z - zmin) / zptp).clip(0.0, 1.0)
                    pcd.colors = o3d.utility.Vector3dVector(np.stack([c, 1.0 - c, 0.5 * np.ones_like(c)], axis=1))

                if not added:
                    vis.add_geometry(pcd)
                    vis.get_render_option().point_size = point_size
                    added = True
                else:
                    vis.update_geometry(pcd)

            vis.poll_events()
            vis.update_renderer()
            time.sleep(0.01)
    finally:
        with contextlib.suppress(Exception):
            vis.destroy_window()


class AsyncPointCloudViewer:
    """轻量级异步点云可视化器（独立进程，保留最新帧）。"""

    def __init__(
        self,
        width: int = 960,
        height: int = 720,
        point_size: float = 2.0,
        queue_size: int = 1,
        window_name: str = "Live Point Cloud",
    ) -> None:
        method = "fork" if "fork" in mp.get_all_start_methods() else "spawn"
        ctx = mp.get_context(method) if hasattr(mp, "get_context") else mp
        self._queue = ctx.Queue(maxsize=max(1, queue_size))  # type: ignore[attr-defined]
        self._proc = ctx.Process(  # type: ignore[attr-defined]
            target=_viewer_loop,
            args=(self._queue, width, height, point_size, window_name),
            daemon=True,
        )
        self._proc.start()
        atexit.register(self.close)

    def update(self, points: np.ndarray) -> None:
        """
        发送一帧点云到可视化进程（非阻塞，丢弃旧帧）。
        - points: (N, 3) 或 (N, 6)，前3列为XYZ，若为6列则后3列视作RGB。
        """
        if points is None:
            return
        pts_all = np.asarray(points)
        if pts_all.ndim != 2 or pts_all.shape[1] < 3 or pts_all.size == 0:
            return

        pts = pts_all[:, :3].astype(np.float64, copy=False)

        cols = None
        if pts_all.shape[1] >= 6:
            col = pts_all[:, 3:6].astype(np.float64, copy=False)
            if np.nanmax(col) > 1.0:
                col = col / 255.0
            cols = np.clip(col, 0.0, 1.0)

        # 保留最新帧：若队列满，先丢弃旧帧后再尝试一次
        try:
            self._queue.put_nowait((pts, cols))
        except Exception:
            with contextlib.suppress(Exception):
                self._queue.get_nowait()
            with contextlib.suppress(Exception):
                self._queue.put_nowait((pts, cols))

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self._queue.put_nowait("__quit__")
        with contextlib.suppress(Exception):
            if self._proc.is_alive():
                self._proc.join(timeout=1.0)
        with contextlib.suppress(Exception):
            if self._proc.is_alive():
                self._proc.kill()
