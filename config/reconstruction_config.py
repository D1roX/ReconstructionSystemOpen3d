import open3d as o3d
import os
from datetime import datetime


class ReconstructionConfig:
    """
    Класс, представляющий набор конфигураций для класса реконструкции поверхности
    (Reconstruction из файла reconstruction.py).
    """
    def __init__(self):
        self.device = o3d.core.Device('CPU:0')  # CPU:0 / CUDA:0
        self.voxel_size = 0.0058  # 0.0058
        self.block_resolution = 16
        self.depth_scale = 1000.0  # 1000.0
        self.block_count = 40000
        self.intrinsic = o3d.io.read_pinhole_camera_intrinsic('config/intrinsic.json')
        self.intrinsic_tensor = o3d.core.Tensor(self.intrinsic.intrinsic_matrix)
        self.depth_max = 3.0
        self.odometry_distance_thr = 0.07
        self.trunc_voxel_multiplier = 8.0
        self.depth_min = 0.1
        self.surface_weight_thr = 3.0

        if not os.path.exists('point clouds'):
            os.mkdir('point clouds')

        t = datetime.now()
        output_folder = f'point clouds\\{t.day}.{t.month}.{t.year} {t.hour}-{t.minute}-{t.second}'
        os.mkdir(output_folder)
        self.output_path = f'{output_folder}\\result_pcd.ply'
