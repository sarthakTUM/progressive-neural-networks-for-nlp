import torch.nn as nn
import torch
import logging


class Adapter(nn.Module):

    def __init__(self,
                 prev_columns,
                 target_column,
                 linear=True):
        super(Adapter, self).__init__()
        self.linear = linear
        logging.info('linear adapter: {}'.format(str(linear)))
        self.lateral_connections = self._create_lateral_connections(prev_columns, target_column)
        self.dropout = nn.Dropout(0.5)
        if not self.linear:
            logging.info('creating non linear conenctions and scales..')
            self.non_linear_connections = self._create_non_linear_lateral_connections(target_column)
            self.scales = self._create_scales(prev_columns)

    def _create_scales(self, prev_columns):
        parameters = nn.ModuleList()
        for col_idx, col in enumerate(prev_columns):
            col_scales = nn.ParameterList()
            for _ in range(col.num_levels):
                col_scales.append(nn.Parameter(torch.rand(1)))
            parameters.append(col_scales)
        return parameters

    def _create_non_linear_lateral_connections(self, target_column):
        level_modules = nn.ModuleList()
        target_activation_sizes = list(target_column.activation_size.values())[1:]
        for activation_size in target_activation_sizes:
            op_size = activation_size[1]
            level_modules.append(nn.Sequential(
                nn.Linear(op_size, op_size),
                nn.Dropout(0.5),
                nn.ReLU(),
                nn.Linear(op_size, op_size),
                nn.Dropout(0.5)
            ))
        return level_modules

    def _create_lateral_connections(self, prev_columns, target_column):
        lateral_connections = nn.ModuleList()
        # for all activations in the target column except the first one
        for column_idx, column in enumerate(prev_columns):
            column_connections = nn.ModuleList()
            curr_column_activation_sizes = list(column.activation_size.values())
            target_column_activation_sizes = list(target_column.activation_size.values())
            for lateral_idx, (c_h, t_h) in enumerate(zip(curr_column_activation_sizes[:-1],
                                                         target_column_activation_sizes[1:])):
                # create Linear layer with size (c_h[1], t_h[1])
                column_connections.append(nn.Linear(c_h[1], t_h[1]).to(torch.device('cuda')))
            lateral_connections.append(column_connections)
        return lateral_connections

    def forward(self, col_activation, col_idx):
        lateral_activations = []
        column_laterals = self.lateral_connections[col_idx]
        for lateral_idx, _ in enumerate(column_laterals):
            column_activation_at_idx = col_activation[lateral_idx]
            lateral_activations.append(self.dropout(column_laterals[lateral_idx](column_activation_at_idx)))
        return lateral_activations

    def non_linear_forward(self, activation_at_level, level):
        op = self.non_linear_connections[level-1](activation_at_level)
        return op

    def scale(self, activation, col_idx, level):
        return torch.mul(activation, self.scales[col_idx][level-1])

    """
    def _create_lateral_connections(self, prev_columns, target_column):
        lateral_connections = dict()
        # for all activations in the target column except the first one
        for column_idx, column in enumerate(prev_columns):
            lateral_connections[column.get_column_id()] = dict()
            curr_column_activation_sizes = list(column.activation_size.values())
            target_column_activation_sizes = list(target_column.activation_size.values())
            for lateral_idx, (c_h, t_h) in enumerate(zip(curr_column_activation_sizes[:-1],
                                                         target_column_activation_sizes[1:])):
                # create Linear layer with size (c_h[1], t_h[1])
                lateral_connections[column.get_column_id()][lateral_idx] = nn.Linear(c_h[1], t_h[1]).to(torch.device('cuda'))
        return lateral_connections"""

    """def forward(self, prev_columns, target_column):
        column_ids = [col.get_column_id() for col in prev_columns]
        activation = dict()
        # 3. store the activations of previous columns
        for col_idx, column in enumerate(prev_columns):
            column_laterals = self.lateral_connections[col_idx]
            activation[column.get_column_id()] = dict()
            for lateral_idx, _ in enumerate(column_laterals):
                column_activation_at_idx = column.activation[column.idx_layer_mapping[lateral_idx]]
                activation[column.get_column_id()][lateral_idx] = self.dropout(column_laterals[lateral_idx](column_activation_at_idx))

        # 4. add all the activations at the same level
        num_levels = target_column.num_levels
        final_activations = []
        for level in range(1, num_levels):
            activations_at_level = [activation[column_id][level - 1] for column_id in column_ids]
            res = torch.zeros_like(activations_at_level[0])
            for activation_at_level in activations_at_level:
                res = torch.add(res, activation_at_level)
            final_activations.append(res)
        return final_activations"""
    """
    def forward(self, prev_columns, target_column):
        column_ids = [col.get_column_id() for col in prev_columns]
        activation = dict()
        # 3. store the activations of previous columns
        for column in prev_columns:
            column_laterals = self.lateral_connections[column.get_column_id()]
            activation[column.get_column_id()] = dict()
            for lateral_idx in column_laterals:
                column_activation_at_idx = column.activation[column.idx_layer_mapping[lateral_idx]]
                activation[column.get_column_id()][lateral_idx] = column_laterals[lateral_idx](column_activation_at_idx)

        # 4. add all the activations at the same level
        num_levels = target_column.num_levels
        final_activations = []
        for level in range(1, num_levels):
            activations_at_level = [activation[column_id][level - 1] for column_id in column_ids]
            res = torch.zeros_like(activations_at_level[0])
            for activation_at_level in activations_at_level:
                res = torch.add(res, activation_at_level)
            final_activations.append(res)
        return final_activations"""
