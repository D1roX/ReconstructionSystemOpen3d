import logging
from logging.handlers import RotatingFileHandler
import sys
import threading
import os

# уровни логирования
LOGGING_LEVELS = {
    10: 'DEBUG',
    20: 'INFO',
    30: 'WARNING',
    40: 'ERROR',
    50: 'CRITICAL',
    60: 'SYSTEM',
    70: 'INIT',
    80: 'DIAGNOST',
}

# допустимые уровни логирования для handler'ов
HANDLER_LEVELS = {
    'CONSOLE_LEVEL': 10,
    'FILE_LEVEL': 60,
    'CUSTOM_CONSOLE_LEVELS': [10, 20, 30, 40, 50, 60, 70, 80],  # уровни логирования, которые будут записаны в консоль
    'CUSTOM_FILE_LEVELS': [20, 40, 60, 70, 80],  # уровни логирования, которые будут записаны в файл
}

# формат сообщения
MESSAGE_FORMAT = '%(asctime)s | %(name)-20s | %(message)s'


def set_custom_handler_levels(console_levels, file_levels):
    HANDLER_LEVELS['CUSTOM_CONSOLE_LEVELS'] = console_levels
    HANDLER_LEVELS['CUSTOM_FILE_LEVELS'] = file_levels


def make_msg(msg, is_error, level_index=10, level_name=None):
    """
    Функция добавления информации об уровне логирования и наличию ошибки к сообщению.
    :param msg: сообщение
    :param is_error: наличие ошибки
    :param level_index: индекс уровня логирования
    :param level_name: имя уровня логирования
    :return: итоговое сообщение
    """
    error = '[ERROR]' if is_error else ''
    if level_name is None:
        level_name = f'[{LOGGING_LEVELS[level_index]}]'
    msg = f'{error: <7} {level_name: <10} | {msg}'
    return msg


def isEnabledFor(level_index):
    """
    Функция для проверки наличия переданного уровня логирования в словаре существующих.
    :param level_index: индекс уровня логирования.
    :return:
    True, если переданный индекс существует в словаре;
    False, если отсутствует.
    """
    if level_index in LOGGING_LEVELS.keys():
        return True
    else:
        return False


def set_level_filter(levels=None):
    """
    Функция создания кастомного фильтра уровней логирования.
    :param levels: допустимые уровни логирования
    :return: фильтр
    """
    if levels is None:
        levels = HANDLER_LEVELS['CUSTOM_CONSOLE_LEVELS']
    return LevelFilter(levels)


def set_error_filter():
    """
    Функция создания кастомного фильтра наличия ошибки.
    :return: фильтр
    """
    return ErrorFilter()


class LevelFilter(logging.Filter):
    """
    Класс кастомного фильтра для регистрации указанных уровней логирования и отсеивания остальных.
    """
    def __init__(self, levels):
        """
        Инициализация допустимых уровней.
        :param levels: список допустимых уровней
        """
        self.__levels = levels

    def filter(self, log_record):
        """
        Функция, проверяющая наличие уровня из log_record в списке допустимых уровней __levels.
        :param log_record: запись лога
        :return:
        True, если уровень существует в списке допустимых уровней;
        False, если отсутствует.
        """
        if log_record.levelno in self.__levels:
            return True
        return False


class ErrorFilter(logging.Filter):
    """
    Класс кастомного фильтра для регистрации сообщений логера с наличием ошибки и отсеивания остальных.
    """
    def filter(self, log_record):
        """
        Функция, проверяющая наличие ошибки в log_record.
        :param log_record: запись лога
        :return:
        True, если ошибка есть;
        False, если отсутствует.
        """
        if '[ERROR]' in log_record.msg:
            return True
        return False


class Logger(logging.getLoggerClass()):
    """
    Класс логирования.
    """

    def __init__(self, name='logger', console_level=None, file_level=None, only_error=False):
        """
        Инициализация логера.
        :param name: имя логера
        :param console_level: минимальный уровень логирования для вывода в консоль
        :param file_level: минимальный уровень логирования для записи в файл
        :param only_error: флаг, позволяющий записывать в файл только ошибки, если True
        """
        do_rollover = True if len(logging.root.manager.loggerDict) == 0 else False
        super().__init__(name)

        logging.addLevelName(60, 'SYSTEM')
        logging.addLevelName(70, 'INIT')
        logging.addLevelName(80, 'DIAGNOST')

        file_filter = None
        console_filter = None
        if file_level is None:
            file_filter = set_level_filter(HANDLER_LEVELS['CUSTOM_FILE_LEVELS'])

        if console_level is None:
            console_filter = set_level_filter(HANDLER_LEVELS['CUSTOM_CONSOLE_LEVELS'])

        self._logger = logging.getLogger(name)
        self._logger.setLevel(logging.DEBUG)

        # добавление RotatingFileHandler для записи логов в файл
        if not os.path.exists('logs'):
            os.mkdir('logs')
        path = 'logs\\logfile.log'
        file_handler = RotatingFileHandler(path, backupCount=1)
        if file_filter:
            file_handler.addFilter(file_filter)
        else:
            file_handler.setLevel(file_level)

        if only_error:
            file_handler.addFilter(set_error_filter())

        formatter = logging.Formatter(MESSAGE_FORMAT)
        file_handler.setFormatter(formatter)
        if do_rollover:
            file_handler.doRollover()

        self._logger.addHandler(file_handler)

        # добавление StreamHandler для записи логов в консоль
        console_handler = logging.StreamHandler(stream=sys.stdout)
        if console_filter:
            console_handler.addFilter(console_filter)
        else:
            console_handler.setLevel(console_level)

        console_handler.setFormatter(formatter)
        self._logger.addHandler(console_handler)

    def make_log(self, level_index=10, msg='', is_error=False, *args):
        """
        Функция для записи сообщения с указанием уровня логирования и информацией о наличии ошибки.
        :param level_index: индекс уровеня логирования
        :param msg: сообщение
        :param is_error: наличие ошибки
        """
        level_name = '[NOTSET]'
        if isEnabledFor(level_index):
            level_name = f'[{LOGGING_LEVELS[level_index]}]'

        msg = make_msg(msg, is_error, level_index, level_name)
        self._logger.log(level_index, msg, *args)

    def logger(self, level_index=0, msg='', is_error=False):
        """
        Вызов функции для записи сообщения с указанием уровня логирования и информацией о наличии ошибки в отдельном
        потоке.
        :param level_index: индекс уровеня логирования
        :param msg: сообщение
        :param is_error: наличие ошибки
        """
        log_thread = threading.Thread(target=self.make_log, args=(level_index, msg, is_error))
        log_thread.start()

    def system(self, msg, is_error=False, *args):
        """
        Фуцнкция для регистрации лога с уровнем SYSTEM.
        :param msg: сообщение
        :param is_error: наличие ошибки
        :param args: дополнительные параметры
        """
        msg = make_msg(msg, is_error, 60)
        log_thread = threading.Thread(target=self._logger.log, args=(60, msg, *args))
        log_thread.start()

    def init(self, msg, is_error=False, *args):
        """
        Фуцнкция для регистрации лога с уровнем INIT.
        :param msg: сообщение
        :param is_error: наличие ошибки
        :param args: дополнительные параметры
        """
        msg = make_msg(msg, is_error, 70)
        log_thread = threading.Thread(target=self._logger.log, args=(70, msg, *args))
        log_thread.start()

    def diagnost(self, msg, is_error=False, *args):
        """
        Фуцнкция для регистрации лога с уровнем DIAGNOST.
        :param msg: сообщение
        :param is_error: наличие ошибки
        :param args: дополнительные параметры
        """
        msg = make_msg(msg, is_error, 80)
        log_thread = threading.Thread(target=self._logger.log, args=(80, msg, *args))
        log_thread.start()

    def info(self, msg, is_error=False, *args):
        """
        Фуцнкция для регистрации лога с уровнем INFO.
        :param msg: сообщение
        :param is_error: наличие ошибки
        :param args: дополнительные параметры
        """
        msg = make_msg(msg, is_error, 20)
        log_thread = threading.Thread(target=self._logger.log, args=(20, msg, *args))
        log_thread.start()

    def debug(self, msg, is_error=False, *args):
        """
        Фуцнкция для регистрации лога с уровнем DEBUG.
        :param msg: сообщение
        :param is_error: наличие ошибки
        :param args: дополнительные параметры
        """
        msg = make_msg(msg, is_error, 10)
        log_thread = threading.Thread(target=self._logger.log, args=(10, msg, *args))
        log_thread.start()

    def warning(self, msg, is_error=False, *args):
        """
        Фуцнкция для регистрации лога с уровнем WARNING.
        :param msg: сообщение
        :param is_error: наличие ошибки
        :param args: дополнительные параметры
        """
        msg = make_msg(msg, is_error, 30)
        log_thread = threading.Thread(target=self._logger.log, args=(30, msg, *args))
        log_thread.start()

    def error(self, msg, *args):
        """
        Фуцнкция для регистрации лога с уровнем ERROR.
        :param msg: сообщение
        :param args: дополнительные параметры
        """
        msg = make_msg(msg, False, 40)
        log_thread = threading.Thread(target=self._logger.log, args=(40, msg, *args))
        log_thread.start()

    def critical(self, msg, is_error=True, *args):
        """
        Фуцнкция для регистрации лога с уровнем CRITICAL.
        :param msg: сообщение
        :param is_error: наличие ошибки
        :param args: дополнительные параметры
        """
        msg = make_msg(msg, is_error, 50)
        log_thread = threading.Thread(target=self._logger.log, args=(50, msg, *args))
        log_thread.start()
