"""Train the model"""

import argparse
import logging
import os
import numpy as np
import torch
import torch.optim as optim
from tqdm import trange
from src.ner import utils
from src.booster.progressive_data_loader import DataLoader
from src.tc.evaluate_pnn import evaluate
from src.booster.progressive_encoder import ClassEncoder, WordEncoder
from src.tc.model.net_pnn import CNNTC

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='../tc/data/kitchen_housewares', help="Directory containing the dataset")
parser.add_argument('--model_dir', default='../tc/experiments/pnn_fracs/kitchen_housewares/organic_pnn_kitchen_100', help="Directory containing params.json")
parser.add_argument('--restore_file', default=None,
                    help="Optional, name of the file in --model_dir containing weights to reload before \
                    training")  # 'best' or 'train'


def max_norm(model, max_val=3, eps=1e-8):
    for name, param in model.named_parameters():
        if 'bias' not in name and param.requires_grad:
            norm = param.norm(2, dim=0, keepdim=True)
            desired = torch.clamp(norm, 0, max_val)
            param = param * (desired / (eps + norm))
            model.state_dict()[name].data.copy_(param)


def train(model,
          optimizer,
          data_iterator,
          metrics,
          save_summary_steps,
          num_steps,
          label_encoder):

    # summary for current training loop and a running average object for loss
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
        loss = model(train_X=train_batch,
                     train_Y=labels_batch,
                     mode='train')

        # clear previous gradients, compute gradients of all variables wrt loss
        loss.backward()

        # performs updates using calculated gradients
        optimizer.step()

        max_norm(model)

        # Evaluate summaries only once in a while
        if i % save_summary_steps == 0:

            model.eval()
            # 1. get the predictions from model
            preds = model.predict(train_X=train_batch, mode='train')

            labels_batch = labels_batch[ClassEncoder.FEATURE_NAME].data.cpu().numpy()

            # 2. decode the predictions and the ground_truths
            labels = label_encoder[ClassEncoder.FEATURE_NAME].decode(np.concatenate(labels_batch).ravel().tolist())
            preds = label_encoder[ClassEncoder.FEATURE_NAME].decode(preds)

            # 3. gather stats
            preds_all.extend(preds)
            true_all.extend(labels)
            loss_all.append(loss.item())

        # update the average loss
        loss_avg.update(loss.item())
        t.set_postfix(loss='{:05.3f}'.format(loss_avg()))

        del loss

    # compute mean of all metrics in summary
    scores = {metric: metrics[metric](preds_all, true_all) for metric in metrics}
    scores['loss'] = np.mean(loss_all)
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in scores.items())
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
                       best_model='val',
                       save_model=True,
                       eval=True):
    from src.ner.utils import SummaryWriter, Label, plot

    # plotting tools
    train_summary_writer = SummaryWriter([*metrics] + ['loss'], name='train')
    val_summary_writer = SummaryWriter([*metrics] + ['loss'], name='val')
    test_summary_writer = SummaryWriter([*metrics] + ['loss'], name='test')
    writers = [train_summary_writer, val_summary_writer, test_summary_writer]
    labeller = Label(anchor_metric='accuracy',
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
    best_acc = 0.0
    patience = 0
    early_stopping_metric = 'accuracy'

    if not val_data and eval or not val_data and save_model == 'val':
        raise Exception('No validation data has been passed.')

    for epoch in range(start_epoch+1, params.num_epochs):

        # Run one epoch
        logging.info("Epoch {}/{}".format(epoch + 1, params.num_epochs))

        # compute number of batches in one epoch (one full pass over the training set)
        # num_steps = (params.train_size + 1) // params.batch_size
        num_steps = (train_data['size'] + 1) // params.batch_size
        train_data_iterator = data_loader.batch_iterator(train_data, batch_size=params.batch_size, shuffle=True)
        train_metrics = train(model,
                              optimizer,
                              train_data_iterator,
                              metrics,
                              params.save_summary_steps,
                              num_steps,
                              label_encoder)
        train_summary_writer.update(train_metrics)

        train_acc = train_metrics[early_stopping_metric]
        if best_model == 'train':
            is_best = train_acc >= best_acc
        if eval:
            # Evaluate for one epoch on validation set
            # num_steps = (params.val_size + 1) // params.batch_size
            num_steps = (val_data['size'] + 1) // params.batch_size
            val_data_iterator = data_loader.batch_iterator(val_data, batch_size=params.batch_size, shuffle=False)
            val_metrics = evaluate(model,
                                   val_data_iterator,
                                   metrics,
                                   num_steps,
                                   label_encoder,
                                   mode='val')
            val_summary_writer.update(val_metrics)

            val_acc = val_metrics[early_stopping_metric]
            if best_model == 'val':
                is_best = val_acc >= best_acc

            ### TEST
            # num_steps = (params.test_size + 1) // params.batch_size
            num_steps = (test_data['size'] + 1) // params.batch_size
            test_data_iterator = data_loader.batch_iterator(test_data, batch_size=params.batch_size, shuffle=False)
            test_metrics = evaluate(model,
                                    test_data_iterator,
                                    metrics,
                                    num_steps,
                                    label_encoder,
                                    mode='test')
            test_summary_writer.update(test_metrics)

        labeller.update(writers=writers)

        plot(writers=writers,
             plot_dir=plots_dir,
             save=True)

        # Save weights
        if save_model:
            utils.save_checkpoint({'epoch': epoch,
                                   'state_dict': model.state_dict(),
                                   'optim_dict': optimizer.state_dict()},
                                  is_best=is_best,
                                  checkpoint=model_dir)

            # save encoders only if they do not exist yet
            if not os.path.exists(os.path.join(model_dir, 'data_encoder.pkl')):
                utils.save_obj(data_encoder, os.path.join(model_dir, 'data_encoder.pkl'))
            if not os.path.exists(os.path.join(model_dir, 'label_encoder.pkl')):
                utils.save_obj(label_encoder, os.path.join(model_dir, 'label_encoder.pkl'))

        # If best_eval, best_save_path
        if is_best:
            patience = 0
            logging.info("- Found new best accuracy")
            best_acc = train_acc if best_model == 'train' else val_acc
            # Save best metrics in a json file in the model directory
            if eval:
                utils.save_dict_to_json(val_metrics, os.path.join(model_dir, "metrics_val_best_weights.json"))
            utils.save_dict_to_json(train_metrics, os.path.join(model_dir, "metrics_train_best_weights.json"))
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

    freeze_prev = True
    best_prev = True
    linear_adapter = True
    best_target = False

    # 0. pretrained model dir
    pretrained_model_dir = '../tc/experiments/st_fracs/organic/st_100_save'
    target_model_dir = '../ner/experiments/disease/st/st_bc5cdr_all'

    # 1. set the device to train on
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 2. Load the parameters from json file
    args = parser.parse_args()
    pretrained_network_params = os.path.join(pretrained_model_dir, 'params.json') # these should be loaded from the pretrained model
    assert os.path.isfile(pretrained_network_params), "No json configuration file found at {}".format(pretrained_network_params)
    new_network_params = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(new_network_params), "No json configuration file found at {}".format(new_network_params)

    pre_params = utils.Params(pretrained_network_params)
    new_params = utils.Params(new_network_params)

    # use GPU if available
    pre_params.cuda = torch.cuda.is_available()
    new_params.cuda = torch.cuda.is_available()

    # 3. Set the random seed for reproducible experiments
    torch.manual_seed(230)
    if new_params.cuda: torch.cuda.manual_seed(230)
    np.random.seed(0)

    # 4. Set the logger
    utils.set_logger(os.path.join(args.model_dir, 'train.log'))

    # 5. Create the input data pipeline
    logging.info("Loading the datasets...")
    # 5.1 specify features
    from collections import OrderedDict

    # 5.1.1 encoders for the new model
    logging.info('creating and loading data loaders')
    data_encoder = OrderedDict()
    data_encoder[WordEncoder.FEATURE_NAME] = WordEncoder(os.path.join(args.data_dir, 'feats'),
                                                         dim=new_params.embedding_dim)
    label_encoder = OrderedDict()
    label_encoder[ClassEncoder.FEATURE_NAME] = ClassEncoder(os.path.join(args.data_dir, 'feats'))
    new_params.data_feats = []
    new_params.label_feats = []
    for feat in data_encoder:
        new_params.data_feats.append(feat)
    for feat in label_encoder:
        new_params.label_feats.append(feat)

    # 5.1.2 encoders for the previous model
    pretrained_data_encoder = utils.load_obj(os.path.join(pretrained_model_dir, 'data_encoder.pkl'))
    pretrained_label_encoder = utils.load_obj(os.path.join(pretrained_model_dir, 'label_encoder.pkl'))
    pre_params.data_feats = []
    pre_params.label_feats = []
    for feat in pretrained_data_encoder:
        pre_params.data_feats.append(feat)
    for feat in label_encoder:
        pre_params.label_feats.append(feat)

    # 5.2 load data
    # 5.2.1 data loader for the new model
    data_loader = DataLoader(new_params,
                             args.data_dir,
                             data_encoder,
                             label_encoder,
                             device=device)

    pretrained_data_loader = DataLoader(pre_params,
                                        args.data_dir,# data_dir has to be the new data
                                        pretrained_data_encoder,
                                        device=device)

    # 6. Modeling
    logging.info('freeze_prev: {}'.format(str(freeze_prev)))
    logging.info('best_prev: {}'.format(str(best_prev)))
    logging.info('pretrained model: {}'.format(pretrained_model_dir))
    logging.info('new model : {}'.format(args.model_dir))
    logging.info('new data: {}'.format(args.data_dir))

    # 6.1.1 Define the model architecture for 1st column
    logging.info('loading previous models and creating new model')
    c1_model = CNNTC(num_tags=pretrained_label_encoder[ClassEncoder.FEATURE_NAME].num_tags,
                     pretrained_word_vecs=torch.from_numpy(data_encoder[WordEncoder.FEATURE_NAME].vectors),
                     dropout=pre_params.dropout,
                     freeze_embeddings=pre_params.freeze_wordembeddings).to(device).float()
    # 6.1.1.1. load the pre-trained model to fit the architecture
    if best_prev:
        utils.load_checkpoint(os.path.join(pretrained_model_dir, 'best.pth'), c1_model)

    # 6.1.2 Define the model for 2nd column
    c2_model = CNNTC(num_tags=label_encoder[ClassEncoder.FEATURE_NAME].num_tags,
                     pretrained_word_vecs=torch.from_numpy(data_encoder[WordEncoder.FEATURE_NAME].vectors),
                     dropout=new_params.dropout,
                     freeze_embeddings=new_params.freeze_wordembeddings).to(device).float()
    if best_target:
        utils.load_checkpoint(os.path.join(target_model_dir, 'best.pth'), c2_model)

    # 7. Convert to columns
    logging.info('creating columns')
    from src.booster.progNN.column_tc import Column

    """
    for cross-task experiments, use the following settings:
    1. column_1 = NER, column_2 = TC
    OR
    2. column_1 = TC, column_2 = NER
    """

    column_1 = Column(model=c1_model,
                      layers={'conv3': (100, 600),
                              'fc_1': (600, 128),
                              'fc_2':  (128, pretrained_label_encoder[ClassEncoder.FEATURE_NAME].num_tags)},
                      data_loader=pretrained_data_loader).to(device)
    column_2 = Column(model=c2_model,
                      layers={'conv3': (130, 200),
                              'fc_1': (600, 128),
                              'fc_2':  (128, label_encoder[ClassEncoder.FEATURE_NAME].num_tags)},
                      data_loader=data_loader).to(device)

    from src.booster.progNN.adapter import Adapter
    adapter = Adapter(prev_columns=[column_1], target_column=column_2, linear=linear_adapter).to(device)
    # 8. create progressive net
    logging.info('creating progressive net')
    from src.booster.progNN.prognet import ProgressiveNet

    progNet = ProgressiveNet(prev_columns=[column_1],
                             target_column=column_2,
                             adapter=adapter,
                             linear_adapter=linear_adapter,
                             freeze_prev=freeze_prev).to(device).float()

    progNet_total_params = sum(p.numel() for p in progNet.parameters())
    progNet_total_trainable_params = sum(p.numel() for p in filter(lambda p: p.requires_grad,
                                        progNet.parameters()))
    logging.info('total params: {}'.format(str(progNet_total_params)))
    logging.info('total trainable params: {}'.format(str(progNet_total_trainable_params)))

    # 9. define the metrics
    from src.ner.evaluation import accuracy_score
    metrics = {'accuracy': accuracy_score}

    # 10. Define the optimizer
    optimizer = optim.Adadelta(params=filter(lambda p: p.requires_grad, progNet.parameters()),
                               rho=0.95)

    # 11. Train ProgNet
    logging.info('Training progressive net')
    train_and_evaluate(progNet,
                       progNet.target_column.data_loader,
                       progNet.target_column.data['train'],
                       progNet.target_column.data['val'],
                       progNet.target_column.data['test'],
                       optimizer,
                       metrics,
                       new_params,
                       args.model_dir,
                       progNet.target_column.data_loader.data_encoder,
                       progNet.target_column.data_loader.label_encoder,
                       restore_file=None,
                       best_model='val',
                       save_model=False,
                       eval=True)
