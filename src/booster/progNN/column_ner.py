import torch
import torch.nn as nn
from collections import OrderedDict
import random
from src.booster.progressive_encoder import EntityEncoder


class Column(nn.Module):
    def __init__(self, model, layers, data_loader):
        super(Column, self).__init__()
        self.__column_id = random.getrandbits(128)
        self.model = model
        self.layers = layers
        self.activation = []
        self.activation_size = OrderedDict()
        self.data_loader = data_loader
        self.data = self.data_loader.load_data(['train/frac_1.0', 'val', 'test'])
        self._convert_to_column()

    def freeze(self):
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, train_X, train_Y, activations):
        loss = self.model.loss_PNN(train_X, activations, train_Y)
        return loss

    """def activate(self, train_X_ID, mode='train'):
        self.activation = []
        train_X = self.data_loader.get_batch_X(self.data[mode], train_X_ID)
        self.model.activate(train_X)"""

    def activate(self, train_X_ID, mode='train'):
        self.activation = []
        train_X = self.data_loader.get_batch_X(self.data[mode], train_X_ID)
        return self.model.activate(train_X)

    def predict(self, pred_X, activations):
        preds = self.model.predict_PNN(pred_X, activations)
        return preds

    def _convert_to_column(self):

        for name, param in self.model.named_modules():
            if name in self.layers.keys():
                # param.register_forward_hook(self._get_activation(name))
                self.activation_size[name] = (self.layers[name][0], self.layers[name][1])
        """
        # for all the layers in the model, set hook
        for name, param in self.model.named_modules():
            if name in self.layers:
                # set forward hook
                param.register_forward_hook(self._get_activation(name))

                # get input and output sizes
                if isinstance(param, ConditionalRandomField):
                    self.activation_size[name] = (param.transitions.shape[1], param.transitions.shape[0])
                else:
                    self.activation_size[name] = (param.weight.shape[1], param.weight.shape[0])"""

    def _get_output(self, module, output):
        # RNN returns a tuple, so activation is the 0th element
        if isinstance(module, torch.nn.RNNBase):
            output = output[0]

        # activation should be the final hidden state, so outputs should be extracted from the PackedSequence object
        if isinstance(output, torch.nn.utils.rnn.PackedSequence):
            (unpacked_out, _) = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
            return unpacked_out.detach()
        else:
            return output.detach()

    def _get_activation(self, name):
        def hook(module, input, output):
            # self.activation[name] = self._get_output(module, output)
            self.activation.append(self._get_output(module, output))
        return hook

    def get_column_id(self):
        return self.__column_id

    @property
    def num_levels(self):
        return len(self.activation_size)

    @property
    def idx_layer_mapping(self):
        return {i: list(self.layers.keys())[i] for i in range(len(self.layers))}


if __name__ == '__main__':
    import torch.nn.functional as F

    class Model(nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            self.fc1 = nn.Linear(10, 20)
            self.fc2 = nn.Linear(20, 30)
            self.fc3 = nn.Linear(30, 5)

        def forward(self, input):
            x = self.fc1(input)
            x = self.fc2(x)
            x = self.fc3(x)
            return F.softmax(x)

    # create a test model
    test = Model()

    # convert model to column
    column = Column(test, layers={'fc1': (10, 20),
                                  'fc2': (20, 30),
                                  'fc3': (30, 5)})

    # pass a random input to model
    ip = torch.rand(10)
    column.model.forward(ip)

    # check the intermediate representations and sizes
    print('num activations: ', column.num_levels)
    print('layer mappings: ', column.idx_layer_mapping)
    print('activations: ', column.activation)
    print('activation sizes: ', column.activation_size)
