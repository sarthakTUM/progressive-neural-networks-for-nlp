"""Evaluates the model"""

import argparse
import logging
import os
import numpy as np
import torch
from src.tc import utils
from src.booster.progressive_encoder import ClassEncoder, WordEncoder
from src.tc.model.net import CNNTC
from src.ner.model.data_loader import DataLoader


np.random.seed(0)


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data/health_personal_care', help="Directory containing the dataset")
parser.add_argument('--model_dir', default='experiments/health_personal_care', help="Directory containing params.json")
parser.add_argument('--restore_file', default='best', help="name of the file in --model_dir \
                     containing weights to load")


def evaluate(model,
             data_iterator,
             metrics,
             num_steps,
             data_encoder,
             label_encoder,
             mode='train'):

    # set model to evaluation mode
    model.eval()

    # compute metrics over the dataset
    preds_all = []
    true_all = []
    loss_all = []

    for _ in range(num_steps):

        # 1. fetch the next evaluation batch
        data_batch, labels_batch = next(data_iterator)

        # 2. compute model output
        loss, logits = model(input=data_batch,
                             labels=labels_batch,
                             label_pad_idx=label_encoder[ClassEncoder.FEATURE_NAME].pad_idx)

        # 3. get the predictions from model
        preds = model.predict(logits=logits,
                              mask=(data_batch[WordEncoder.FEATURE_NAME] != data_encoder[
                                  WordEncoder.FEATURE_NAME].pad_idx).float())

        labels_batch = labels_batch[ClassEncoder.FEATURE_NAME].data.cpu().numpy().squeeze()

        # 4. decode the predictions and the ground_truths
        labels = label_encoder[ClassEncoder.FEATURE_NAME].decode(labels_batch)
        preds = label_encoder[ClassEncoder.FEATURE_NAME].decode(preds)

        # 5. gather stats
        preds_all.extend(preds)
        true_all.extend(labels)
        loss_all.append(loss.item())

    # compute mean of all metrics over all batches
    scores = {metric: metrics[metric](true_all, preds_all) for metric in metrics}
    scores['loss'] = np.mean(loss_all)
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in scores.items())
    logging.info("- {} Eval metrics : {}".format(mode, metrics_string))
    return scores, preds_all


if __name__ == '__main__':

    # 1. set the device to train on
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data_to_use = 'test'

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

    data_encoder = utils.load_obj(os.path.join(args.model_dir, 'data_encoder.pkl'))
    label_encoder = utils.load_obj(os.path.join(args.model_dir, 'label_encoder.pkl'))

    # 5.2 load data
    data_loader = DataLoader(params,
                             args.data_dir,
                             data_encoder,
                             label_encoder)
    data = data_loader.load_data([data_to_use])
    test_data = data[data_to_use]
    # 5.3 specify the train and val dataset sizes
    params.test_size = test_data['size']
    test_data_iterator = data_loader.batch_iterator(test_data, params, shuffle=False, sort_by_legth=False)
    logging.info("- done.")

    # 6. Modeling
    # 6.1 Define the model
    from src.tc.model.net import CNNTC

    model = CNNTC(num_tags=label_encoder[ClassEncoder.FEATURE_NAME].num_tags,
                  pretrained_word_vecs=torch.from_numpy(data_encoder[WordEncoder.FEATURE_NAME].vectors),
                  dropout=params.dropout,
                  freeze_embeddings=params.freeze_wordembeddings).to(device).float()
    """optimizer = optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters()))"""

    # 6.2 define metrics
    from src.ner.evaluation import accuracy_score, binary_f1_score

    metrics = {'accuracy': accuracy_score,
               'binary_f1': binary_f1_score}

    utils.load_checkpoint(os.path.join(args.model_dir, args.restore_file + '.pth'), model)

    # Evaluate
    import math
    num_steps = math.ceil(params.test_size/params.batch_size)
    test_metrics, preds = evaluate(model,
                            test_data_iterator,
                            metrics,
                            num_steps,
                            data_encoder,
                            label_encoder)
    save_path = os.path.join(args.model_dir, "metrics_{}_{}.json".format(data_to_use, args.restore_file))
    utils.save_dict_to_json(test_metrics, save_path)

    preds_path = os.path.join(args.model_dir, "preds_"+data_to_use+".pkl")
    utils.save_obj(preds, preds_path)

