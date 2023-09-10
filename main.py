import time
import keyboard
import numpy as np
import open3d as o3d
from reconstruction import Reconstruction
from config.reconstruction_config import ReconstructionConfig
from video_pre_processing import VideoPreProcessing
from logger.logger import Logger

log = Logger('main log')


def get_depth_and_color_frames_from_dir(dir, index):
    image_index = f'{index:05}'
    try:
        rgb = o3d.t.io.read_image(f'{dir}\\rgb\\{image_index}.jpg')
        depth = o3d.t.io.read_image(f'{dir}\\depth\\{image_index}.png')
        return rgb, depth
    except FileNotFoundError as e:
        log.error(f'Ошибка чтения кадра из файла.\n{e}')
        return None, None


def get_t_depth_and_color_frames(video_pre_processing, priority=False):
    color, depth = video_pre_processing.get_frame(priority)
    if color:
        video_pre_processing.write_video(np.asarray(color.get_data()))
        color = o3d.t.geometry.Image(np.asarray(color.get_data()))
        depth = o3d.t.geometry.Image(np.asarray(depth.get_data()))
        return color, depth

    return None, None


def setup_video_pre_processing():
    video_pre_processing = VideoPreProcessing()
    video_pre_processing.set_config_realsense()
    video_pre_processing.set_video_capture()
    video_pre_processing.set_video_writer()
    return video_pre_processing


def setup_reconstruction(video_pre_processing):
    reconstruction_config = ReconstructionConfig()
    rgb, depth = get_t_depth_and_color_frames(video_pre_processing)
    # rgb, depth = get_depth_and_color_frames_from_dir('dataset', 0)
    iteration_time = time.time()
    reconstruction = Reconstruction(reconstruction_config, depth)
    log.info(f'{0} :  {time.time() - iteration_time}')
    return reconstruction


def run_system():
    image_index = 1
    priority = False
    video_pre_processing = setup_video_pre_processing()
    reconstruction = setup_reconstruction(video_pre_processing)
    total_time = time.time()
    while image_index <= 100:
        rgb, depth = get_t_depth_and_color_frames(video_pre_processing, priority)
        # rgb, depth = get_depth_and_color_frames_from_dir('dataset', image_index)
        if priority:
            priority = False

        if rgb is not None:
            iteration_time = time.time()
            success = reconstruction.launch(rgb, depth)
            log.info(f'{image_index} :  {time.time() - iteration_time}')
            image_index += 1

            if not success:
                priority = True

        # if keyboard.is_pressed('esc'):
        #     break

    log.info(f'FINAL TIME : {time.time() - total_time}')
    video_pre_processing.video_writer_release()
    reconstruction.visualize_and_save_pcd()


if __name__ == '__main__':
    run_system()
