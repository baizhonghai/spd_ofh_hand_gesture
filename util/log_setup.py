import logging
import logging.config

logging.config.fileConfig('./config/logging_config.ini')

logger = logging.getLogger('my_app')

if __name__=='__main__':
    logger.info(f'd:df:')