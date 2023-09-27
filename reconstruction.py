import numpy as np
import open3d as o3d
from logger.logger import Logger

log = Logger('reconstruction log')


class Reconstruction:
    def __init__(self, config, depth):
        self.config = config
        self.T_frame_to_model = o3d.core.Tensor(np.eye(4))
        try:
            self.model = o3d.t.pipelines.slam.Model(self.config.voxel_size,
                                                    self.config.block_resolution,
                                                    self.config.block_count,
                                                    self.T_frame_to_model,
                                                    self.config.device)
            self.input_frame = o3d.t.pipelines.slam.Frame(depth.rows,
                                                          depth.columns,
                                                          self.config.intrinsic_tensor,
                                                          self.config.device)
            self.raycast_frame = o3d.t.pipelines.slam.Frame(depth.rows,
                                                            depth.columns,
                                                            self.config.intrinsic_tensor,
                                                            self.config.device)
            log.init('Успешная инициализация reconstruction.')
        except Exception as e:
            log.init(f'Ошибка инициализации reconstruction.\n{e}', True)

        self.image_index = 0

    def visualize_and_save_pcd(self):
        pcd = self.model.extract_pointcloud(self.config.surface_weight_thr)
        pcd = pcd.to_legacy()

        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis.add_geometry(pcd)
        vis.run()

        try:
            o3d.io.write_point_cloud(self.config.output_path, pcd)
            log.info(f'Итоговое облако точек успешно сохранено в {self.config.output_path}.')
        except Exception as e:
            log.error(f'Облако точек не сохранено.\n{e}')


    def set_new_frames(self, rgb, depth):
        rgb = o3d.t.geometry.Image(np.asarray(rgb)).to(self.config.device)
        depth = o3d.t.geometry.Image(np.asarray(depth)).to(self.config.device)
        self.input_frame.set_data_from_image('depth', depth)
        self.input_frame.set_data_from_image('color', rgb)

    def add_frames_to_model(self):
        if self.image_index > 0:
            result = self.model.track_frame_to_model(self.input_frame, self.raycast_frame,
                                                self.config.depth_scale,
                                                self.config.depth_max,
                                                self.config.odometry_distance_thr)
            self.T_frame_to_model = self.T_frame_to_model @ result.transformation

        self.model.update_frame_pose(self.image_index, self.T_frame_to_model)
        self.model.integrate(self.input_frame,
                             self.config.depth_scale,
                             self.config.depth_max,
                             self.config.trunc_voxel_multiplier)
        self.model.synthesize_model_frame(self.raycast_frame, self.config.depth_scale,
                                     self.config.depth_min, self.config.depth_max,
                                     self.config.trunc_voxel_multiplier, False)

    def launch(self, rgb, depth):
        self.set_new_frames(rgb, depth)
        self.add_frames_to_model()
        self.image_index += 1
        return True
