from config import ROOT_DIR


def get_logging_config():
    """
    This generate logging configuration w.r.t given params which will be use in
     code base to log something as per need.

    :return: A dict containing logging conf
    """
    return {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'verbose': {
                'format': '%(levelname)s %(asctime)s %(process)d %(thread)d %(filename)s %(module)s %(funcName)s %(lineno)d %(message)s'
            },
            'simple': {
                'format': '%(levelname)s: %(message)s'
            },
        },
        'handlers': {
            'console': {
                'level': 'DEBUG',
                'class': 'logging.StreamHandler',
                'formatter': 'verbose'
            },
            'file': {
                'level': 'INFO',
                'class': 'custom_file_handler.CustomTimedRotatingFileHandler',
                'filename': str(ROOT_DIR.path('logs/contify.log')),
                'formatter': 'verbose',
                'when': 'midnight',
                'backupCount': 10,
                'encoding': 'utf-8',
            },
            'access_logs': {
                'level': 'INFO',
                'class': 'custom_file_handler.CustomTimedRotatingFileHandler',
                'filename': str(ROOT_DIR.path('logs/access.log')),
                'formatter': 'simple',
                'when': 'midnight',
                'backupCount': 7,
                'encoding': 'utf-8',
            },

        },
        'loggers': {
            '': {
                'handlers': ['file'],
                'level': 'INFO',
            },
        }
    }
