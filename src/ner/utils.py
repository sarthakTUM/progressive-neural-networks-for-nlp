import json
import logging
import os
import shutil
import numpy as np
import torch
import pickle
import matplotlib.pyplot as plt
from src.ner import utils


class Params:
    """Class that loads hyperparameters from a json file.

    Example:
    ```
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    ```
    """

    def __init__(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    def save(self, json_path):
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']"""
        return self.__dict__


class RunningAverage():
    """A simple class that maintains the running average of a quantity

    Example:
    ```
    loss_avg = RunningAverage()
    loss_avg.update(2)
    loss_avg.update(4)
    loss_avg() = 3
    ```
    """

    def __init__(self):
        self.steps = 0
        self.total = 0

    def update(self, val):
        self.total += val
        self.steps += 1

    def __call__(self):
        return self.total / float(self.steps)


class SummaryWriter:
    def __init__(self, metrics, name):
        self.metrics = metrics
        self.summary = {key:[] for key in metrics}
        self.name = name
        self.labels = dict.fromkeys(metrics)

    def __call__(self, metric):
        if metric in self.summary:
            return self.summary[metric]
        else:
            raise Exception('{} not in summary'.format(metric))

    def label(self, metric):
        if metric in self.labels:
            return self.labels[metric]
        else: raise Exception('{} not in summary'.format(metric))

    def update(self, summary):
        for metric, value in summary.items():
            if metric in self.summary:
                self.summary[metric].append(value)


class Label:
    def __init__(self, anchor_metric, anchor_writer='val'):
        self.anchor_writer = anchor_writer
        self.anchor_metric = anchor_metric

    def update(self, writers):
        metrics = None
        arg_refs = dict()

        for writer in writers:
            # for anchor writer, set the labels and get the indices of values
            if writer.name == self.anchor_writer:
                metrics = writer.metrics
                # collect the indices of values
                arg_refs = dict.fromkeys(writer.metrics)
                # get value of anchor metric in anchor writer
                writer.labels[self.anchor_metric] = Label._get_func(self.anchor_metric)(writer.summary[self.anchor_metric])
                # get index (epoch) of anchor metric in anchor writer
                arg_refs[self.anchor_metric] = writer.summary[self.anchor_metric].index(writer.labels[self.anchor_metric])
                for metric in metrics:
                    if metric != self.anchor_metric:
                        # for all other metrics, get the value at index of anchor metric
                        writer.labels[metric] = writer.summary[metric][arg_refs[self.anchor_metric]]
                        arg_refs[metric] = writer.summary[metric].index(writer.labels[metric])
                break
        # for all other writers, get the value from the index obtained from anchor ref.
        for writer in writers:
            if writer.name != self.anchor_writer:
                for metric in metrics:
                    writer.labels[metric] = writer.summary[metric][arg_refs[metric]]

    @staticmethod
    def _get_func(metric):
        return max if metric in ['f1_score', 'accuracy', 'precision_score', 'recall_score'] else min


def plot(writers, plot_dir, save=False):
    metrics = writers[0].metrics
    for writer in writers:
        assert writer.metrics == metrics

    for metric in metrics:
        for writer in writers:
            y = writer(metric)
            x = range(0, len(y))
            label = '{}_{}: {}'.format(writer.name, metric, writer.label(metric))
            plt.plot(x, y, label=label)
            plt.xlabel('epoch')
            plt.ylabel(str(metric))
            plt.legend()
            plt.savefig(os.path.join(plot_dir, str(metric)+'.png'))
        plt.close()

    if save:
        for writer in writers:
            utils.save_obj(writer, os.path.join(plot_dir, writer.name+'.pkl'))


def set_logger(log_path):
    """Set the logger to log info in terminal and file `log_path`.

    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.

    Example:
    ```
    logging.info("Starting training...")
    ```

    Args:
        log_path: (string) where to log
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)


def save_map(map, map_name, path):
    if type(map) is dict:
        save_dict_to_json(map, os.path.join(path, map_name+'.json'))
    if type(map) is np.ndarray:
        save_array(map, os.path.join(path, map_name+'.npy'))


def save_array(d, path):
    np.save(path, d)


def save_dict_to_json(d, json_path):
    """Saves dict of floats in json file

    Args:
        d: (dict) of float-castable values (np.float, int, float, etc.)
        json_path: (string) path to json file
    """
    with open(json_path, 'w') as f:
        # We need to convert the values to float for json (it doesn't accept np.array, np.float, )
        d = {k: v for k, v in d.items()}
        json.dump(d, f, indent=4)


def save_checkpoint(state, is_best, checkpoint):
    """Saves model and training parameters at checkpoint + 'last.pth.tar'. If is_best==True, also saves
    checkpoint + 'best.pth.tar'

    Args:
        state: (dict) contains model's state_dict, may contain other keys such as epoch, optimizer state_dict
        is_best: (bool) True if it is the best model seen till now
        checkpoint: (string) folder where parameters are to be saved
    """
    filepath = os.path.join(checkpoint, 'last.pth')
    if not os.path.exists(checkpoint):
        print("Checkpoint Directory does not exist! Making directory {}".format(checkpoint))
        os.mkdir(checkpoint)
    else:
        print("Checkpoint Directory exists! ")
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'best.pth'))


def load_checkpoint(checkpoint, model, optimizer=None):
    """Loads model parameters (state_dict) from file_path. If optimizer is provided, loads state_dict of
    optimizer assuming it is present in checkpoint.

    Args:
        checkpoint: (string) filename which needs to be loaded
        model: (torch.nn.Module) model for which the parameters are loaded
        optimizer: (torch.optim) optional: resume optimizer from checkpoint
    """
    if not os.path.exists(checkpoint):
        print('file does not exist')
        raise ("File doesn't exist {}".format(checkpoint))
    checkpoint = torch.load(checkpoint)
    model.load_state_dict(checkpoint['state_dict'])

    if optimizer:
        optimizer.load_state_dict(checkpoint['optim_dict'])

    return checkpoint


def jsonKeys2int(x):
    if isinstance(x, dict):
        return {int(k):v for k,v in x.items()}
    return x


def json_to_dict(path, int_keys=False):
    with open(path) as f_in:
        if int_keys:
            return json.load(f_in, object_hook=jsonKeys2int)
        else:
            return json.load(f_in)


def save_obj(o, path):
    with open(path, 'wb') as output:
        pickle.dump(o, output, pickle.HIGHEST_PROTOCOL)


def load_obj(path):
    with open(path, 'rb') as input:
        return pickle.load(input)


