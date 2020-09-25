"""Evaluates the model"""

import argparse
import logging
import os

import numpy as np
import torch

from src.ner import utils
from src.booster.progressive_encoder import CharEncoder, EntityEncoder, WordEncoder
# from src.ner.model.net_pnn import LSTMCRF
from src.booster.progNN.net import LSTMCRF
from src.ner.model.data_loader import DataLoader


np.random.seed(0)


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data/ncbi_iobes_id', help="Directory containing the dataset")
parser.add_argument('--model_dir', default='experiments/ncbi_iobes_0.1_pnn', help="Directory containing params.json")
parser.add_argument('--restore_file', default='best', help="name of the file in --model_dir \
                     containing weights to load")


def evaluate(model, data_iterator, metrics, num_steps, label_encoder, mode='train'):
    """Evaluate the model on `num_steps` batches.

    Args:
        model: (torch.nn.Module) the neural network
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        data_iterator: (generator) a generator that generates batches of data and labels
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) hyperparameters
        num_steps: (int) number of batches to train on, each of size params.batch_size
    """

    # set model to evaluation mode
    model.eval()

    # compute metrics over the dataset
    preds_all = []
    true_all = []
    loss_all = []

    for _ in range(num_steps):

        # 1. fetch the next evaluation batch
        data_batch, labels_batch = next(data_iterator)

        loss = model(train_X=data_batch,
                     train_Y=labels_batch,
                     mode=mode)

        # 4. Get the model predictions
        preds = model.predict(train_X=data_batch, mode=mode)

        labels_batch = labels_batch[EntityEncoder.FEATURE_NAME].data.cpu().numpy()
        labels_batch_trunc = [labels_batch[o_idx][:len(o)] for o_idx, o in enumerate(preds)]

        # 5. decode the predictions and the ground_truths
        labels_batch_trunc = [label_encoder[EntityEncoder.FEATURE_NAME].decode(l) for l in labels_batch_trunc]
        preds = [label_encoder[EntityEncoder.FEATURE_NAME].decode(l) for l in preds]

        # 6. gather stats
        preds_all.extend(preds)
        true_all.extend(labels_batch_trunc)
        loss_all.append(loss.item())

    # compute mean of all metrics over all batches
    scores = {metric: metrics[metric](preds_all, true_all) for metric in metrics}
    scores['loss'] = np.mean(loss_all)
    metrics_string = " ; ".join("{}: {}".format(k, v) for k, v in scores.items())
    logging.info("- {} metrics : {}".format(mode, metrics_string))
    return scores


if __name__ == '__main__':

    # 1. set the device to train on
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 2. Load the parameters from json file
    args = parser.parse_args()
    network_params = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(network_params), "No json configuration file found at {}".format(network_params)
    params = utils.Params(network_params)
    # use GPU if available
    params.cuda = torch.cuda.is_available()

    # 3. Set the random seed for reproducible experiments
    torch.manual_seed(230)
    if params.cuda: torch.cuda.manual_seed(230)
    np.random.seed(0)

    # 4. Set the logger
    utils.set_logger(os.path.join(args.model_dir, 'train.log'))

    # 5. Create the input data pipeline
    logging.info("Loading the datasets...")
    # 5.1 specify features
    from collections import OrderedDict

    data_encoder = utils.load_obj(os.path.join(args.model_dir, 'data_encoder.pkl'))
    label_encoder = utils.load_obj(os.path.join(args.model_dir, 'label_encoder.pkl'))
    # 5.2 load data
    data_loader = DataLoader(params, args.data_dir, data_encoder, label_encoder)
    data = data_loader.load_data(['test'])
    test_data = data['test']
    # 5.3 specify the train and val dataset sizes
    params.test_size = test_data['size']
    test_data_iterator = data_loader.batch_iterator(test_data, params, shuffle=False)
    logging.info("- done.")

    # 6. Modeling
    # 6.1 Define the model
    model = LSTMCRF(params=params,
                    char_vocab_length=data_encoder[CharEncoder.FEATURE_NAME].vocab_length,
                    num_tags=label_encoder[EntityEncoder.FEATURE_NAME].num_tags,
                    pretrained_word_vecs=torch.from_numpy(data_encoder[WordEncoder.FEATURE_NAME].vectors),
                    dropout=params.dropout,
                    decoder_type=params.decoder,
                    bidirectional=True,
                    freeze_embeddings=False).to(device).float()

    # 6.2 fetch loss function and metrics
    from src.ner.evaluation import accuracy_score, f1_score, precision_score, recall_score

    metrics = {'accuracy': accuracy_score,
               'f1_score': f1_score,
               'precision_score': precision_score,
               'recall_score': recall_score}

    utils.load_checkpoint(os.path.join(args.model_dir, args.restore_file + '.pth'), model)

    # Evaluate
    num_steps = (params.test_size + 1) // params.batch_size
    test_metrics = evaluate(model,
                            test_data_iterator,
                            metrics,
                            num_steps,
                            label_encoder)
    save_path = os.path.join(args.model_dir, "metrics_test_{}.json".format(args.restore_file))
    utils.save_dict_to_json(test_metrics, save_path)
