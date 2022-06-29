import logging

_log_levels = {'critical': logging.CRITICAL,
               'error': logging.ERROR,
               'warning': logging.WARNING,
               'info': logging.INFO,
               'debug': logging.DEBUG}

logging.basicConfig(level=_log_levels['debug'])
logger = logging.getLogger(__name__)
