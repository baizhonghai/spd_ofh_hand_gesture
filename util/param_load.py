import configparser
import os


def read_config_file(filename):
    config = configparser.ConfigParser()
    config.read(filename)
    return config


current_dir = os.path.dirname(os.path.abspath(__file__))
config_file_path = os.path.join(current_dir, '../config/param_config.ini')
config = read_config_file(config_file_path)
