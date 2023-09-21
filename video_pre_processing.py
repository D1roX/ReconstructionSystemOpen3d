import cv2
import pyrealsense2 as rs
import numpy as np
import time
from datetime import datetime
import os
from config.video_pre_processing_config import VideoPreProcessingConfig
from logger.logger import Logger

config = VideoPreProcessingConfig()
log = Logger('vpp log')


class VideoPreProcessing:
    """Класс пред обработчик видео."""
    def __init__(self):
        self.frames_for_analysis = None
        self.frame_interval = config.frame_interval
        self.cap = None
        self.device = None
        self.writer = None
        self.pipeline = rs.pipeline()
        self.config_realsense = rs.config()
        self.last_frame_time = 0
        self.video_fragments_count = 0
        self.frames_count = 0

        if not os.path.exists('records'):
            os.mkdir('records')

        t = datetime.now()
        self.output_folder = f'records\\{t.day}.{t.month}.{t.year} {t.hour}-{t.minute}-{t.second}'
        os.mkdir(self.output_folder)

    def set_config_realsense(self, stream_type=config.stream_type, weight=config.input_weight,
                             height=config.input_height, stream_format=config.stream_format, fps=config.fps):
        """
        Настройки конфигурации камеры realsense.
        :param stream_type: тип запущенного видопотока.
        :param weight: ширина видеопотока.
        :param height: высота видеопотока.
        :param stream_format: формат запущенного видеопотока.
        :param fps: количество кадров в секунду у видеопотока.
        """
        try:
            self.config_realsense.enable_stream(stream_type[0], weight, height, stream_format[0], fps)
            self.config_realsense.enable_stream(stream_type[1], weight, height, stream_format[1], fps)
            log.init('Камера realsense успешно настроена.')
        except Exception as e:
            log.init(f'Камера realsense не настроена.\n{e}', True)

    @staticmethod
    def color_palette(img, color_mode=cv2.COLOR_BAYER_RG2BGR):
        """
        Выбор цветовой палитры.
        :param img: передаваемое изображение прочитанное библиотекой cv2 (cv.imread).
        :param color_mode: передоваемая цветовая палитра cv2.
        :return: измененное изображение.
        """

        img = cv2.cvtColor(img, color_mode)
        # cv2.imshow("Image", img)
        # cv2.waitKey(0)
        return img

    @staticmethod
    def is_blur(img, blur_threshold=config.blur_threshold):
        """
        Проверка на заблюренность изображения.
        blur_threshold можно подобрать более оптимальный.
        :param img: передаваемое изображение прочитанное библиотекой cv2 (cv.imread).
        :param blur_threshold: порог размытости изображения.
        :return: возвращает True, если изображение размыто и False, если не размыто.
        """

        grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        var = cv2.Laplacian(grey, cv2.CV_64F).var()
        if var < blur_threshold:
            # cv2.imshow("Image", img)
            # cv2.waitKey(0)
            return True
        else:
            # cv2.imshow("Image", img)
            # cv2.waitKey(0)
            return False

    @staticmethod
    def set_image_size(img, weight, height):
        """
        Изменение разрешения изображения.
        :param img: передаваемое изображение прочитанное библиотекой cv2 (cv.imread).
        :param weight: ширина нового изображения.
        :param height: высота нового изображения.
        :return: возвращает полученное изображение.
        """

        new_img = cv2.resize(img, dsize=(weight, height), interpolation=cv2.INTER_AREA)
        return new_img

    def merge_images(self, image1, image2, image_weight=config.merge_iamge_weight,
                     image_height=config.merge_iamge_height):
        """
        Склейка вдух изображений.
        :param image1: первое передаваемое изображение прочитанное библиотекой cv2 (cv.imread).
        :param image2: второе передаваемое изображение прочитанное библиотекой cv2 (cv.imread).
        :param image_weight: ширина склеиного  изображения.
        :param image_height: высота склеиного изображения.
        :return: полученное изображение.
        """

        stitcher = cv2.Stitcher.create()
        status, stitched_image = stitcher.stitch([image1, image2])
        if image_height is not None or image_weight is not None:
            stitched_image = self.set_image_size(stitched_image, image_weight, image_height)
        if status == cv2.Stitcher_OK:
            # cv2.imshow('st', stitched_image)
            # cv2.waitKey(0)
            log.info('Успешное склеивание изображений.')
            return stitched_image
        else:
            log.info('Склеивание изображений не удалось.')

    @staticmethod
    def histogram_alignment(img):
        """
        Выравнивание гистограммы.
        :param img: передаваемое изображение прочитанное библиотекой cv2 (cv.imread).
        :return: полученное изображение.
        """

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        equ = cv2.equalizeHist(gray)
        equ = cv2.cvtColor(equ, cv2.COLOR_GRAY2BGR)
        result = np.hstack((img, equ))
        # cv2.imshow("res", result)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        return result

    def set_video_capture(self, path=config.input_path):
        """
        Использование видеопотока или видеофайла.
        :param device: номер используемой камеры.
        :param path: путь к видеофайлу.
        """

        if path is None:
            self.set_config_realsense()
            try:
                self.pipeline.start(self.config_realsense)
                log.init('Видеопоток успешно запущен.')
            except Exception as e:
                log.init(f'Ошибка запуска видеопотока.\n{e}', True)
        else:
            try:
                self.config_realsense.enable_device_from_file(path)
                self.pipeline.start(self.config_realsense)
                log.init('Видеопоток успешно запущен.')
            except FileNotFoundError as e:
                log.init(f'Ошибка чтения видеофайла.\n{e}', True)
            except Exception as e:
                log.init(f'Ошибка запуска видеопотока.\n{e}', True)

    def set_video_writer(self, fps=config.output_fps, weight=config.output_weight,
                         height=config.output_height, codec=config.codec):
        """
        Инициализация потока для записи видео.
        :param path: путь куда записывать файл и в каком формате.
        :param fps: количество кадорв в секунду у записываемого видео.
        :param weight: ширина кадра у записываемого видео.
        :param height: высота кадра у записываемого видео.
        :param codec: кодек у записываемого видео.
        """
        path = f'{self.output_folder}\\fragment_{self.video_fragments_count}.mp4'
        self.video_fragments_count += 1
        fourcc = cv2.VideoWriter_fourcc(*codec)
        try:
            self.writer = cv2.VideoWriter(path, fourcc, fps, (weight, height))
            log.init('cv2.VideoWriter успешно создан.')
        except Exception as e:
            log.init(f'Ошибка создания cv2.VideoWriter.\n{e}', True)

    def write_video(self, frame):
        """
        Запись изображения в видео.
        :param frame: передаваемое изображение.
        """
        if not self.writer:
            self.set_video_writer()

        self.writer.write(self.color_palette(frame, cv2.COLOR_RGB2BGR))
        self.frames_count += 1
        if self.frames_count / config.output_fps >= config.video_file_length:
            log.init('cv2.VideoWriter достиг маквсимальной длительности фрагмента. '
                     'Запуск создания cv2.VideoWriter для нового фрагмента.')
            self.frames_count = 0
            self.video_writer_release()
            self.set_video_writer()

    def video_writer_release(self):
        """
        Окончание записи видео.
        """
        if self.writer:
            self.writer.release()
            log.init('cv2.VideoWriter завершил работу.')

    def set_video_capture_size(self, weight=config.input_weight, height=config.input_height):
        """
        Задает разрешение видео.
        :param weight: ширина видео.
        :param height: высота видео.
        """

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, weight)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def set_video_capture_fps(self, fps=30):
        """
        Количество кадров у видео.
        :param fps: количество кадров в секунду.
        """
        self.cap.set(cv2.CAP_PROP_FPS, config.fps)

    def get_depth_frame(self, priority=False, left_img=None, right_img=None):
        """
        Получение кадра глубины.
        :param priority:
        :return: кадр глубины.
        """
        if time.time() - self.last_frame_time >= self.frame_interval or priority:
            if self.is_the_function_initialized_set_config_realsense:
                try:
                    frame = self.pipeline.wait_for_frames()
                    depth_frame = frame.get_depth_frame()
                    return depth_frame
                except Exception as e:
                    log.error(f'Ошибка получения кадра с realsense камеры.\n{e}')
            else:
                if left_img != None and right_img != None:
                    if config.StereoSGBM_or_StereoBM == 0:
                        return self.StereoSGBM_func_frame(left_img, right_img)
                    else:
                        return self.StereoBM_func_frame(left_img, right_img)
        return None, None

    def set_frame_interval(self, new_time=config.frame_interval):
        """
        Изменение интервала получения кадра.
        :param new_time: новое время для получения изображения.
        """

        self.frame_interval = new_time
        def StereoSGBM_func_video(self, skip_frame=config.skip_frame):
        """
        Функция получения стерео SGBM кадра
        :param skip_frame: текущий кадр для чтения по переданному костылю минимальноек значение равняется 200
        :return: стерео SGBM кадр
        """
        if skip_frame < 200:
            skip_frame = 200
            config.skip_frame = 200
        video = cv2.VideoCapture("video.avi")
        video.set(cv2.CAP_PROP_POS_FRAMES, skip_frame)
        config.skip_frame = +1
        # Вариант исполнения получения кадра глубины
        h = 0
        w = 0
        # Ширина и высота кадра
        x = 0
        x1 = 0
        y = 0
        # Отступы для первого кадра x y и для второго x1 y

        # Matched block size. It must be an odd number >=1 . Normally, it should be somewhere in the 3..11 range.
        block_size = 3
        min_disp = -176
        max_disp = 176
        # Maximum disparity minus minimum disparity. The value is always greater than zero.
        # In the current implementation, this parameter must be divisible by 16.
        num_disp = max_disp - min_disp
        # Margin in percentage by which the best (minimum) computed cost function value should "win" the second best value to consider the found match correct.
        # Normally, a value within the 5-15 range is good enough
        uniquenessRatio = 15
        # Maximum size of smooth disparity regions to consider their noise speckles and invalidate.
        # Set it to 0 to disable speckle filtering. Otherwise, set it somewhere in the 50-200 range.
        speckleWindowSize = 16
        # Maximum disparity variation within each connected component.
        # If you do speckle filtering, set the parameter to a positive value, it will be implicitly multiplied by 16.
        # Normally, 1 or 2 is good enough.
        speckleRange = 2
        disp12MaxDiff = 2

        stereo = cv2.StereoSGBM_create(
            minDisparity=min_disp,
            numDisparities=num_disp,
            blockSize=block_size,
            uniquenessRatio=uniquenessRatio,
            speckleWindowSize=speckleWindowSize,
            speckleRange=speckleRange,
            disp12MaxDiff=disp12MaxDiff,
            P1=8 * 1 * block_size * block_size,
            P2=32 * 1 * block_size * block_size,
        )
        # инициализации класса Стерео описание параметров с оф документации
        ret, frame = video.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape[:2]
        # print("H:" + str(height) + "  W:" + str(width))
        h = int(height)
        w = int(width / 2)  # делем на пополам потому что изображение сдвоенное
        x1 = w
        left_img = gray[y:y + h, x:x + w]  # получаем левое изображение из сдвоенного
        right_img = gray[y:y + h, x1:x1 + w]  # получаем правое изображение из сдвоенного
        disparity_SGBM = stereo.compute(left_img, right_img)
        # Normalize the values to a range from 0..255 for a grayscale image
        disparity_SGBM = cv2.normalize(disparity_SGBM, disparity_SGBM, alpha=125,
                                       beta=0, norm_type=cv2.NORM_MINMAX)  # увеличиваем контраст
        disparity_SGBM = np.uint8(disparity_SGBM)
        res = left_img + disparity_SGBM  # комплексированное изображение
        # cv.imshow('left', res)
        video.release()
        cv2.destroyAllWindows()
        return res

    def StereoSGBM_func_frame(self, left_img, right_img):
        """
        Получение стерео кадра ищ двух изображений
        :param left_img: левое изображение прочитанное библиотекой cv2 (cv.imread)
        :param right_img: правое изображение прочитанное библиотекой cv2 (cv.imread)
        :return: стерео SGBM кадр
        """
        # Вариант исполнения получения кадра глубины
        h = 0
        w = 0
        # Ширина и высота кадра
        x = 0
        x1 = 0
        y = 0
        # Отступы для первого кадра x y и для второго x1 y

        # Matched block size. It must be an odd number >=1 . Normally, it should be somewhere in the 3..11 range.
        block_size = 3
        min_disp = -176
        max_disp = 176
        # Maximum disparity minus minimum disparity. The value is always greater than zero.
        # In the current implementation, this parameter must be divisible by 16.
        num_disp = max_disp - min_disp
        # Margin in percentage by which the best (minimum) computed cost function value should "win" the second best value to consider the found match correct.
        # Normally, a value within the 5-15 range is good enough
        uniquenessRatio = 15
        # Maximum size of smooth disparity regions to consider their noise speckles and invalidate.
        # Set it to 0 to disable speckle filtering. Otherwise, set it somewhere in the 50-200 range.
        speckleWindowSize = 16
        # Maximum disparity variation within each connected component.
        # If you do speckle filtering, set the parameter to a positive value, it will be implicitly multiplied by 16.
        # Normally, 1 or 2 is good enough.
        speckleRange = 2
        disp12MaxDiff = 2

        stereo = cv2.StereoSGBM_create(
            minDisparity=min_disp,
            numDisparities=num_disp,
            blockSize=block_size,
            uniquenessRatio=uniquenessRatio,
            speckleWindowSize=speckleWindowSize,
            speckleRange=speckleRange,
            disp12MaxDiff=disp12MaxDiff,
            P1=8 * 1 * block_size * block_size,
            P2=32 * 1 * block_size * block_size,
        )
        # инициализации класса Стерео описание параметров с оф документации
        left_img = self.color_palette(left_img, cv2.COLOR_BGR2GRAY)
        right_img = self.color_palette(right_img, cv2.COLOR_BGR2GRAY)
        disparity_SGBM = stereo.compute(left_img, right_img)
        # Normalize the values to a range from 0..255 for a grayscale image
        disparity_SGBM = cv2.normalize(disparity_SGBM, disparity_SGBM, alpha=125,
                                       beta=0, norm_type=cv2.NORM_MINMAX)  # увеличиваем контраст
        disparity_SGBM = np.uint8(disparity_SGBM)
        res = left_img + disparity_SGBM  # комплексированное изображение
        # cv.imshow('left', res)
        return res

    def StereoBM_func(self, skip_frame=config.skip_frame):
        """
        Функция получения стерео BM кадра
        :param skip_frame: текущий кадр для чтения по переданному костылю минимальноек значение равняется 200
        :return: стерео BM кадр
        """
        if skip_frame < 200:
            skip_frame = 200
            config.skip_frame = 200
        video = cv2.VideoCapture("video.avi")
        video.set(cv2.CAP_PROP_POS_FRAMES, skip_frame)
        config.skip_frame = +1
        # Вариант исполнения получения кадра глубины
        h = 0
        w = 0
        # Ширина и высота кадра
        x = 0
        x1 = 0
        y = 0
        # Отступы для первого кадра x y и для второго x1 y

        stereo = cv2.StereoBM.create(numDisparities=16, blockSize=15)

        ret, frame = video.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape[:2]
        # print("H:" + str(height) + "  W:" + str(width))
        h = int(height)
        w = int(width / 2)
        x1 = w
        left_img = gray[y:y + h, x:x + w]
        right_img = gray[y:y + h, x1:x1 + w]
        disparity = stereo.compute(left_img, right_img)
        # Normalize the values to a range from 0..255 for a grayscale image
        disparity = cv2.normalize(disparity, disparity, alpha=125,
                                  beta=0, norm_type=cv2.NORM_MINMAX)
        disparity = np.uint8(disparity)
        res = disparity
        # cv.imshow('left', res)
        video.release()
        cv2.destroyAllWindows()
        return res

    def StereoBM_func_frame(self, left_img, right_img):
        """
        Получение стерео кадра ищ двух изображений
        :param left_img: левое изображение прочитанное библиотекой cv2 (cv.imread)
        :param right_img: правое изображение прочитанное библиотекой cv2 (cv.imread)
        :return: стерео BM кадр
        """
        # Вариант исполнения получения кадра глубины
        h = 0
        w = 0
        # Ширина и высота кадра
        x = 0
        x1 = 0
        y = 0
        # Отступы для первого кадра x y и для второго x1 y

        stereo = cv2.StereoBM.create(numDisparities=16, blockSize=15)

        left_img = self.color_palette(left_img, cv2.COLOR_BGR2GRAY)
        right_img = self.color_palette(right_img, cv2.COLOR_BGR2GRAY)
        disparity = stereo.compute(left_img, right_img)
        # Normalize the values to a range from 0..255 for a grayscale image
        disparity = cv2.normalize(disparity, disparity, alpha=125,
                                  beta=0, norm_type=cv2.NORM_MINMAX)
        disparity = np.uint8(disparity)
        res = disparity
        return res

    def get_color_frame(self):
        """
        Получение цветного изображения
        :return: цветное изображение
        """
        if self.is_the_function_initialized_set_config_realsense:
            frame = self.pipeline.wait_for_frames()
            color_frame = frame.get_color_frame()
            return color_frame
        else:
            ret, color_frame = self.cap.read()
            return color_frame

    def cut_frame(self, img):
        """
        Обрезка изображений пополам
        :param img: передаваемое изображение прочитанное библиотекой cv2 (cv.imread)
        :return: цветные изображения в двух экземплярах
        """
        height, width, _ = img.shape
        half_width = width // 2
        left_image = img[:, :half_width]
        right_image = img[:, half_width:]
        return left_image, right_image
