"""Train the model"""

import argparse
import logging
import os
import numpy as np
import torch
import torch.optim as optim
from tqdm import trange
from src.ner import utils
from src.tc.evaluate import evaluate
from src.ner.model.data_loader import DataLoader
from src.booster.progressive_encoder import ClassEncoder, WordEncoder

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data/organic', help="Directory containing the dataset")
parser.add_argument('--model_dir', default='experiments/ptft_fracs/organic/kitchen_ptft_tla_100', help="Directory containing params.json")
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

    # set model to training mode
    model.train()

    # summary for current training loop and a running average object for loss
    summ = []
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
        loss, logits = model(input=train_batch,
                             labels=labels_batch,
                             label_pad_idx=label_encoder[ClassEncoder.FEATURE_NAME].pad_idx)

        # clear previous gradients, compute gradients of all variables wrt loss
        loss.backward()

        # performs updates using calculated gradients
        optimizer.step()

        max_norm(model)

        # Evaluate summaries only once in a while
        if i % save_summary_steps == 0:

            model.eval()
            # 1. get the predictions from model
            preds = model.predict(logits=logits,
                                  mask=(train_batch[WordEncoder.FEATURE_NAME] != data_encoder[WordEncoder.FEATURE_NAME].pad_idx).float())
            labels_batch = labels_batch[ClassEncoder.FEATURE_NAME].data.cpu().numpy().squeeze()

            # 2. decode the predictions and the ground_truths
            labels = label_encoder[ClassEncoder.FEATURE_NAME].decode(labels_batch)
            preds = label_encoder[ClassEncoder.FEATURE_NAME].decode(preds)

            # 3. gather stats
            preds_all.extend(preds)
            true_all.extend(labels)
            loss_all.append(loss.item())

        # update the average loss
        loss_avg.update(loss.item())
        t.set_postfix(loss='{:05.3f}'.format(loss_avg()))

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
        num_steps = (params.train_size + 1) // params.batch_size
        train_data_iterator = data_loader.batch_iterator(train_data, params, shuffle=True)

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
            num_steps = (params.val_size + 1) // params.batch_size
            val_data_iterator = data_loader.batch_iterator(val_data, params, shuffle=False)
            val_metrics = evaluate(model,
                                   val_data_iterator,
                                   metrics,
                                   num_steps,
                                   data_encoder,
                                   label_encoder,
                                   mode='Val')
            val_summary_writer.update(val_metrics)

            val_acc = val_metrics[early_stopping_metric]
            if best_model == 'val':
                is_best = val_acc >= best_acc

            ### TEST
            num_steps = (params.test_size + 1) // params.batch_size
            test_data_iterator = data_loader.batch_iterator(test_data, params, shuffle=False)
            test_metrics = evaluate(model,
                                    test_data_iterator,
                                    metrics,
                                    num_steps,
                                    data_encoder,
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
    # TODO modularize the network : LATER
    # TODO plotting

    # FINAL CHOICES:
        # 1. AdaDelta with MaxNorm Reg of 3.0 and Rho 0.95 with LR 1.0
        # 2. No Decoder (FC)
        # 3. SL tagging
        # 4. Accuracy score
        # 5. No Lr scheduling
        # 6. Word2Vec negative google news 300D
        # 7. Yoon Kim architecture
        # 8. Tune word embeddings: NO

    # 0. pretrained model dir
    pretrained_model_dir = 'experiments/st_fracs/kitchen_housewares/st_100_save'
    all_layer = True

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

    data_encoder = utils.load_obj(os.path.join(pretrained_model_dir, 'data_encoder.pkl'))
    pretrained_label_encoder = utils.load_obj(os.path.join(pretrained_model_dir, 'label_encoder.pkl'))
    label_encoder = OrderedDict()
    label_encoder[ClassEncoder.FEATURE_NAME] = ClassEncoder(os.path.join(args.data_dir, 'feats'))

    # 5.2 load data

    k_fold = None
    combine_train_dev = False
    train_on_dev = False

    data_loader = DataLoader(params, args.data_dir, data_encoder, label_encoder)
    if k_fold:
        logging.info('K-Fold turned on with folds: {}'.format(k_fold))
        splits_dir = [os.path.join(args.data_dir, 'split_'+str(split_num)) for split_num in range(1, k_fold+1)]
    else:
        splits_dir = [args.data_dir]

    for split_dir in splits_dir:
        logging.info('training for: {}'.format(split_dir))
        args.data_dir = split_dir
        if k_fold:
            split_model_dir = os.path.join(args.model_dir, os.path.basename(split_dir))
            if not os.path.exists(split_model_dir):
                os.makedirs(split_model_dir)
        else:
            split_model_dir = args.model_dir
        data = data_loader.load_data(['train/frac_1.0', 'val', 'test'], args.data_dir)
        train_data = data['train']
        val_data = data['val']
        test_data = data['test']

        # combine train and val data
        if combine_train_dev:
            logging.info('combining train and dev sets')
            for k, v in train_data.items():
                if isinstance(train_data[k], list):
                    train_data[k].extend(val_data[k])
                elif isinstance(train_data[k], int):
                    train_data[k] += val_data[k]

        # 5.3 specify the train and val dataset sizes
        params.train_size = train_data['size']
        params.val_size = val_data['size']
        params.test_size = test_data['size']
        logging.info("- done.")
        logging.info('train size: {}'.format(params.train_size))
        logging.info('val size: {}'.format(params.val_size))
        logging.info('test size: {}'.format(params.test_size))

        # 6. Modeling
        # 6.1 Define the model
        from src.tc.model.net import CNNTC
        model = CNNTC(num_tags=pretrained_label_encoder[ClassEncoder.FEATURE_NAME].num_tags,
                      pretrained_word_vecs=torch.from_numpy(data_encoder[WordEncoder.FEATURE_NAME].vectors),
                      dropout=params.dropout,
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
        model.reset_layers(label_encoder[ClassEncoder.FEATURE_NAME].num_tags)
        model.to(device).float()

        optimizer = optim.Adadelta(params=filter(lambda p: p.requires_grad, model.parameters()),
                                   rho=0.95)

        # 6.2 define metrics
        from src.ner.evaluation import accuracy_score
        metrics = {'accuracy': accuracy_score}

        # 7. Train the model
        logging.info("Starting training for {} epoch(s)".format(params.num_epochs))
        end_epoch = train_and_evaluate(model=model,
                                       data_loader=data_loader,
                                       train_data=train_data,
                                       val_data=test_data if combine_train_dev else val_data,
                                       test_data=test_data,
                                       optimizer=optimizer,
                                       metrics=metrics,
                                       params=params,
                                       model_dir=split_model_dir,
                                       data_encoder=data_encoder,
                                       label_encoder=label_encoder,
                                       restore_file=None,
                                       best_model='val',
                                       save_model=False,
                                       eval=True)

        if train_on_dev:
            logging.info('training on train and dev for {} epochs'.format(end_epoch+1))
            # 1. combine train and dev
            data = data_loader.load_data(['train', 'val', 'test'], args.data_dir)
            train_data = data['train']
            val_data = data['val']
            test_data = data['test']
            logging.info('combining train and dev sets')
            for k, v in train_data.items():
                if isinstance(train_data[k], list):
                    train_data[k].extend(val_data[k])
                elif isinstance(train_data[k], int):
                    train_data[k] += val_data[k]
            params.train_size = train_data['size']
            params.val_size = test_data['size']
            logging.info("- done.")
            logging.info('train size: {}'.format(params.train_size))
            logging.info('test size: {}'.format(params.val_size))

            # 2. delete old model and optimizer
            del model
            del optimizer

            # 3. construct a new model and optimizer
            model = CNNTC(num_tags=label_encoder[ClassEncoder.FEATURE_NAME].num_tags,
                          pretrained_word_vecs=torch.from_numpy(data_encoder[WordEncoder.FEATURE_NAME].vectors),
                          dropout=params.dropout,
                          freeze_embeddings=params.freeze_wordembeddings).to(device).float()
            optimizer = optim.Adadelta(params=filter(lambda p: p.requires_grad, model.parameters()),
                                       rho=0.95)

            # 4. train without evaluating with new number of epochs
            params.num_epochs = end_epoch+1
            final_model_dir = os.path.join(split_model_dir, 'final')
            if not os.path.exists(final_model_dir):
                os.makedirs(final_model_dir)
            train_and_evaluate(model=model,
                               train_data=train_data,
                               val_data=test_data,
                               test_data=test_data,
                               data_loader=data_loader,
                               optimizer=optimizer,
                               metrics=metrics,
                               params=params,
                               model_dir=final_model_dir,
                               data_encoder=data_encoder,
                               label_encoder=label_encoder,
                               restore_file=None,
                               best_model='train',
                               save_model=False,
                               eval=True)
        del model


