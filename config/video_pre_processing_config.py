import pyrealsense2 as rs


class VideoPreProcessingConfig:
    """
    Класс, представляющий набор конфигураций для класса препроцессинга видео
    (VideoPreProcessing из файла video_pre_processing.py).
    """
    def __init__(self):
        self.device = 0                                             # Номер используемой камеры
        self.input_path = ''                                        # Путь до читаемого видеофайла
        self.fps = 30                                               # Количество кадров в секунду при чтении в видео
        self.output_fps = 15                                        # Количество кадров в секунду для записи видео
        self.blur_threshold = 120                                   # Порог размытости изображения
        self.frame_interval = 0                                     # Интервал времени в который берется кадр 5 = 0.005
        self.input_weight = 640                                     # Входная ширина видео
        self.input_height = 480                                     # Входная высота видео
        self.output_weight = 640                                    # Ширина записываемого видеофайла
        self.output_height = 480                                    # Высота записываемого видеофайла
        self.video_file_length = 120                                # Длина записываемого видеофайла в секундах
        self.codec = 'mp4v'                                         # Кодек записываемого видеофайла
        self.stream_type = (rs.stream.depth, rs.stream.color)       # Тип запущенного видопотока
        self.stream_format = (rs.format.z16, rs.format.rgb8)        # Формат запущенного видеопотока
        self.merge_iamge_weight = 1500                              # Ширина слекинного изображения
        self.merge_iamge_height = 400                               # Высота слекинного изображения