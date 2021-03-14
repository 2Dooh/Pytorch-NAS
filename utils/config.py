from typing import Dict, Tuple

from easydict import EasyDict as edict

import json

import os

import logging
from logging.handlers import RotatingFileHandler

from pprint import pformat

import datetime

def process_config(json_file: str) -> Tuple[edict, Dict]:
    config, _ = load_json_config(json_file)
    try:
        getattr(config, 'exp_name')
    except AttributeError:
        now = datetime.datetime.now()
        setattr(config, 'exp_name', 'exp-{}'.format(now.strftime("%Y%m%d-%H%M")))

    config.summary_dir = os.path.join('experiments', config.exp_name, 'summaries/')
    config.checkpoint_dir = os.path.join('experiments', config.exp_name, 'checkpoints/')
    config.out_dir = os.path.join('experiments', config.exp_name, 'out/')
    config.log_dir = os.path.join('experiments', config.exp_name, 'logs/')

    makedirs(
        config.summary_dir,
        config.checkpoint_dir,
        config.out_dir,
        config.log_dir
    )

    setup_logging(config.log_dir)
    logging.getLogger().info(' {} '.format('*' * 40))
    logging.getLogger().info('Exp name: {}'.format(config.exp_name))
    logging.getLogger().info(' {} '.format('*' * 40))
    logging.getLogger().info('Configuration: \n')
    logging.getLogger().info(pformat(config))

    return config

def makedirs(*args):
    for arg in args:
        os.makedirs(arg, exist_ok=True)
    

def load_json_config(path: str) -> Tuple[edict, dict]:
    try:
        with open(path, 'r') as handle:
            config_dict = json.load(handle)
        config = dict_to_attributes(config_dict)
        return config, config_dict
    except ValueError:
        exit(-1)

def dict_to_attributes(dict: Dict) -> edict:
    return edict(dict)

def setup_logging(log_dir: str) -> None:
    file_format = \
        '[%(levelname)s] - %(asctime)s - %(name)s - : %(message)s in %(pathname)s:%(lineno)d'
    console_format = '[%(levelname)s]: %(message)s'

    # Main logger
    main_logger = logging.getLogger()
    main_logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(console_format))

    exp_file_handler = RotatingFileHandler(filename=os.path.join(log_dir, 'exp_debug.log'), 
                                        maxBytes=1e6, 
                                        backupCount=5)
    exp_file_handler.setLevel(logging.DEBUG)
    exp_file_handler.setFormatter(logging.Formatter(file_format))

    exp_errors_file_handler = RotatingFileHandler(filename=os.path.join(log_dir, 'exp_error.log'),
                                                maxBytes=1e6,
                                                backupCount=5)
    exp_errors_file_handler.setLevel(logging.WARNING)
    exp_errors_file_handler.setFormatter(logging.Formatter(file_format))

    main_logger.addHandler(console_handler)
    main_logger.addHandler(exp_file_handler)
    main_logger.addHandler(exp_errors_file_handler)
