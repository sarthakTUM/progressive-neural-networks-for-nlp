"""Train the model"""

import argparse
import logging
import os
import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from tqdm import trange
from src.ner import utils
from src.ner.model.data_loader import DataLoader
from src.ner.evaluate import evaluate
from src.booster.progressive_encoder import CharEncoder, EntityEncoder, WordEncoder
from src.booster.progNN.net import LSTMCRF

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='../../ner/data/dj_ft_iobes_id', help="Directory containing the dataset")
parser.add_argument('--model_dir', default='../../ner/experiments/disease/ptft_fracs/dj/germevalft_alllayer_dj_100', help="Directory containing params.json")
parser.add_argument('--restore_file', default=None,
                    help="Optional, name of the file in --model_dir containing weights to reload before \
                    training")  # 'best' or 'train'


def train(model,
          optimizer,
          data_iterator,
          metrics,
          save_summary_steps,
          num_steps,
          label_encoder):


    loss_avg = utils.RunningAverage()
    preds_all = []
    true_all = []
    loss_all = []

    # Use tqdm for progress bar
    t = trange(num_steps)
    for i in t:

        model.train()

        # zero the gradients
        model.zero_grad()

        # fetch the next training batch
        train_batch, labels_batch = next(data_iterator)

        # compute model output and log_lik
        loss = model.loss(input=train_batch,
                          labels=labels_batch,
                          label_pad_idx=label_encoder[EntityEncoder.FEATURE_NAME].pad_idx)

        # clear previous gradients, compute gradients of all variables wrt loss
        loss.backward()

        # gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)

        # performs updates using calculated gradients
        optimizer.step()

        # Evaluate summaries only once in a while
        if i % save_summary_steps == 0:

            model.eval()

            # 1. get the predictions from model
            preds = model.predict(X=train_batch)

            labels_batch = labels_batch[EntityEncoder.FEATURE_NAME].data.cpu().numpy()
            labels_batch_trunc = [labels_batch[o_idx][:len(o)] for o_idx, o in enumerate(preds)]

            # 2. decode the predictions and the ground_truths
            labels_batch_trunc = [label_encoder[EntityEncoder.FEATURE_NAME].decode(l) for l in labels_batch_trunc]
            preds = [label_encoder[EntityEncoder.FEATURE_NAME].decode(l) for l in preds]

            # 3. gather stats
            preds_all.extend(preds)
            true_all.extend(labels_batch_trunc)
            loss_all.append(loss.item())

        # update the average loss
        loss_avg.update(loss.item())
        t.set_postfix(loss='{:05.3f}'.format(loss_avg()))

    # compute mean of all metrics in summary
    scores = {metric: metrics[metric](preds_all, true_all) for metric in metrics}
    scores['loss'] = np.mean(loss_all)
    metrics_string = " ; ".join("{}: {}".format(k, v) for k, v in scores.items())
    logging.info("- Train metrics: " + metrics_string)
    return scores


def train_and_evaluate(model,
                       data_loader,
                       train_data,
                       val_data,
                       test_data,
                       optimizer,
                       metrics,
                       params,
                       model_dir,
                       data_encoder,
                       label_encoder,
                       restore_file=None,
                       save_model=True,
                       eval=True):

    from src.ner.utils import SummaryWriter, Label, plot

    # plotting tools
    train_summary_writer = SummaryWriter([*metrics] + ['loss'], name='train')
    val_summary_writer = SummaryWriter([*metrics] + ['loss'], name='val')
    test_summary_writer = SummaryWriter([*metrics] + ['loss'], name='test')
    writers = [train_summary_writer, val_summary_writer, test_summary_writer]
    labeller = Label(anchor_metric='f1_score',
                     anchor_writer='val')
    plots_dir = os.path.join(model_dir, 'plots')
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    start_epoch = -1
    if restore_file is not None:
        logging.info("Restoring parameters from {}".format(restore_file))
        checkpoint = utils.load_checkpoint(restore_file, model, optimizer)
        start_epoch = checkpoint['epoch']

    # save the snapshot of parameters fro reproducibility
    utils.save_dict_to_json(params.dict, os.path.join(model_dir, 'train_snapshot.json'))

    # variable initialization
    best_val_score = 0.0
    patience = 0
    early_stopping_metric = 'f1_score'

    # set the Learning rate Scheduler
    lambda_lr = lambda epoch: 1 / (1 + (params.lr_decay_rate * epoch))
    lr_scheduler = LambdaLR(optimizer, lr_lambda=lambda_lr, last_epoch=start_epoch)

    # train over epochs
    for epoch in range(start_epoch + 1, params.num_epochs):
        lr_scheduler.step()
        # Run one epoch
        logging.info("Epoch {}/{}".format(epoch + 1, params.num_epochs))
        logging.info("Learning Rate : {}".format(lr_scheduler.get_lr()))

        # compute number of batches in one epoch (one full pass over the training set)
        num_steps = (params.train_size + 1) // params.batch_size
        train_data_iterator = data_loader.batch_iterator(train_data, params, shuffle=True)

        train_metrics = train(model,
                              optimizer,
                              train_data_iterator,
                              metrics,
                              params.save_summary_steps,
                              num_steps,
                              label_encoder)
        val_score = train_metrics[early_stopping_metric]
        is_best = val_score >= best_val_score
        train_summary_writer.update(train_metrics)
        if eval:
            # Evaluate for one epoch on validation set
            num_steps = (params.val_size + 1) // params.batch_size
            val_data_iterator = data_loader.batch_iterator(val_data, params, shuffle=False)
            val_metrics = evaluate(model,
                                   val_data_iterator,
                                   metrics,
                                   num_steps,
                                   label_encoder,
                                   mode='Val')

            val_score = val_metrics[early_stopping_metric]
            is_best = val_score >= best_val_score
            val_summary_writer.update(val_metrics)

            ### TEST
            num_steps = (params.test_size + 1) // params.batch_size
            test_data_iterator = data_loader.batch_iterator(test_data, params, shuffle=False)
            test_metrics = evaluate(model,
                                    test_data_iterator,
                                    metrics,
                                    num_steps,
                                    label_encoder,
                                    mode='Test')
            test_summary_writer.update(test_metrics)

        labeller.update(writers=writers)

        plot(writers=writers,
             plot_dir=plots_dir,
             save=True)

        # Save weights
        if save_model:
            utils.save_checkpoint({'epoch': epoch + 1,
                                   'state_dict': model.state_dict(),
                                   'optim_dict': optimizer.state_dict()},
                                  is_best=is_best,
                                  checkpoint=model_dir)

            # save encoders only if they do not exist yet
            if not os.path.exists(os.path.join(model_dir, 'data_encoder.pkl')):
                logging.info('saving encoders')
                utils.save_obj(data_encoder, os.path.join(model_dir, 'data_encoder.pkl'))
            if not os.path.exists(os.path.join(model_dir, 'label_encoder.pkl')):
                utils.save_obj(label_encoder, os.path.join(model_dir, 'label_encoder.pkl'))

        # If best_eval, best_save_path
        if is_best:
            patience = 0
            logging.info("- Found new best F1 score")
            best_val_score = val_score
            # Save best metrics in a json file in the model directory
            if eval:
                utils.save_dict_to_json(val_metrics, os.path.join(model_dir, 'plots', "metrics_val_best_weights.json"))
                utils.save_dict_to_json(test_metrics, os.path.join(model_dir, 'plots', "metrics_test_best_weights.json"))
            utils.save_dict_to_json(train_metrics, os.path.join(model_dir, 'plots', "metrics_train_best_weights.json"))
        else:
            if eval:
                patience += 1
                logging.info('current patience: {} ; max patience: {}'.format(patience, params.patience))
            if patience == params.patience:
                logging.info('patience reached. Exiting at epoch: {}'.format(epoch+1))
                # Save latest metrics in a json file in the model directory before exiting
                if eval:
                    utils.save_dict_to_json(val_metrics, os.path.join(model_dir, 'plots', "metrics_val_last_weights.json"))
                    utils.save_dict_to_json(test_metrics,
                                            os.path.join(model_dir, 'plots', "metrics_test_last_weights.json"))
                utils.save_dict_to_json(train_metrics, os.path.join(model_dir, 'plots', "metrics_train_last_weights.json"))
                epoch = epoch - patience
                break

        # Save latest metrics in a json file in the model directory at end of epoch
        if eval:
            utils.save_dict_to_json(val_metrics, os.path.join(model_dir, 'plots', "metrics_val_last_weights.json"))
            utils.save_dict_to_json(test_metrics, os.path.join(model_dir, 'plots', "metrics_test_last_weights.json"))
        utils.save_dict_to_json(train_metrics, os.path.join(model_dir, 'plots', "metrics_train_last_weights.json"))
    return epoch


if __name__ == '__main__':
    # TODO modularize the network : LATER
    # FINAL CHOICES:
        # 1. SGD with GC of 5.0
        # 2. CRF decoder?
        # 3. IOBES tagging
        # 4. Micro-F1 score
        # 5. LR scheduler as mentioned in Ma & Hovy 2016
        # 6. Glove 100D
        # 7. LSTM layers: 2
        # 8. Tune word embeddings: NO

    # 0. pretrained model dir
    pretrained_model_dir = '../../ner/experiments/disease/st_fracs/germeval_fasttext/st_germeval_100'
    all_layer = True

    # 1. set the device to train on
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 2. Load the parameters from json file
    args = parser.parse_args()
    network_params = os.path.join(pretrained_model_dir, 'params.json') # these should be loaded from the pretrained model
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

    # data_encoder = OrderedDict() # this should be loaded from the pretrained model
    data_encoder = utils.load_obj(os.path.join(pretrained_model_dir, 'data_encoder.pkl'))
    pretrained_label_encoder = utils.load_obj(os.path.join(pretrained_model_dir, 'label_encoder.pkl'))
    label_encoder = OrderedDict()
    label_encoder[EntityEncoder.FEATURE_NAME] = EntityEncoder(os.path.join(args.data_dir, 'feats'))

    params.data_feats = []
    params.label_feats = []
    for feat in data_encoder:
        params.data_feats.append(feat)
    for feat in label_encoder:
        params.label_feats.append(feat)

    # 5.2 load data
    data_loader = DataLoader(params, args.data_dir, data_encoder, label_encoder)
    data = data_loader.load_data(['train/frac_1.0', 'val', 'test'])
    train_data = data['train']
    val_data = data['val']
    test_data = data['test']

    # 5.3 specify the train and val dataset sizes
    params.train_size = train_data['size']
    # val_data = test_data
    params.val_size = val_data['size']
    params.test_size = test_data['size']
    logging.info("- done.")
    logging.info('train size: {}'.format(params.train_size))
    logging.info('val size: {}'.format(params.val_size))
    logging.info('test size: {}'.format(params.test_size))

    # 6. Modeling
    logging.info('pretrained model: {}'.format(pretrained_model_dir))
    logging.info('all layer: {}'.format(all_layer))
    logging.info('new model : {}'.format(args.model_dir))
    logging.info('new data: {}'.format(args.data_dir))
    # 6.1 Define the model
    model = LSTMCRF(params=params,
                    char_vocab_length=data_encoder[CharEncoder.FEATURE_NAME].vocab_length,
                    num_tags=pretrained_label_encoder[EntityEncoder.FEATURE_NAME].num_tags,
                    pretrained_word_vecs=torch.from_numpy(data_encoder[WordEncoder.FEATURE_NAME].vectors),
                    dropout=params.dropout,
                    decoder_type=params.decoder,
                    bidirectional=params.rnn_bidirectional,
                    freeze_embeddings=params.freeze_wordembeddings).to(device).float()
    model_total_params = sum(p.numel() for p in model.parameters())
    model_total_trainable_params = sum(p.numel() for p in filter(lambda p: p.requires_grad,
                                                                 model.parameters()))
    print('total params: ', model_total_params)
    print('total trainable params: ', model_total_trainable_params)

    # load the pre-trained model
    logging.info('loading pretrained model')
    utils.load_checkpoint(os.path.join(pretrained_model_dir, 'best.pth'), model)
    if not all_layer:
        for param in model.parameters():
            param.requires_grad = False
    model.reset_layers(label_encoder[EntityEncoder.FEATURE_NAME].num_tags)
    model.to(device).float()
    optimizer = optim.SGD(params=filter(lambda p: p.requires_grad, model.parameters()),
                          lr=params.learning_rate,
                          momentum=params.momentum)

    # 6.2 fetch loss function and metrics
    from src.ner.evaluation import accuracy_score, f1_score, precision_score, recall_score, classification_report

    metrics = {'accuracy': accuracy_score,
               'f1_score': f1_score,  # micro F1 score
               'precision_score': precision_score,
               'recall_score': recall_score,
               'classification_report': classification_report}

    # 7. Train the model
    logging.info("Starting training for {} epoch(s)".format(params.num_epochs))
    end_epoch = train_and_evaluate(model=model,
                                   data_loader=data_loader,
                                   train_data=train_data,
                                   val_data=val_data,
                                   test_data=test_data,
                                   optimizer=optimizer,
                                   metrics=metrics,
                                   params=params,
                                   model_dir=args.model_dir,
                                   data_encoder=data_encoder,
                                   label_encoder=label_encoder,
                                   restore_file=None,
                                   save_model=False,
                                   eval=True)


