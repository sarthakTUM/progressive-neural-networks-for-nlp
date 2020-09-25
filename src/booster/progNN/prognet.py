import torch.nn as nn
import torch


class ProgressiveNet(nn.Module):
    def __init__(self,
                 prev_columns,
                 target_column,
                 adapter,
                 linear_adapter=True,
                 freeze_prev=True):
        super(ProgressiveNet, self).__init__()
        self.prev_columns = nn.ModuleList(prev_columns)
        self.target_column = target_column
        self.adapter = adapter
        self.linear_adapter = linear_adapter
        if freeze_prev:
            for col in self.prev_columns:
                col.freeze()

    def forward(self, train_X, train_Y, mode='train'):
        batch_ID = train_X['ID']
        activations = []
        for column_idx, column in enumerate(self.prev_columns):
            col_act = column.activate(train_X_ID=batch_ID, mode=mode)
            # activations.append(self.adapter(column.idx_layer_mapping, column.activation, column_idx))
            activations.append(self.adapter(col_act, column_idx))
        # activations = self.adapter(self.prev_columns, self.target_column)
        num_levels = self.target_column.num_levels
        final_activations = []
        for level in range(1, num_levels):
            activations_at_level = [a[level-1] for a in activations]
            if self.linear_adapter:
                res = torch.zeros_like(activations_at_level[0])
                for activation_at_level in activations_at_level:
                    res = torch.add(res, activation_at_level)
                final_activations.append(res)
            else:
                # activation_at_level_scaled = [self.adapter.scale(activation, col_idx, level) for col_idx, activation in enumerate(activations_at_level)]
                activations_at_level_cat = torch.cat(activations_at_level, dim=-1)
                final_activations.append(self.adapter.non_linear_forward(activations_at_level_cat, level))
        loss = self.target_column(train_X,
                                  train_Y,
                                  final_activations)

        del activations, final_activations, activations_at_level
        return loss

    def predict(self, train_X, mode='train'):
        batch_ID = train_X['ID']
        activations = []
        for column_idx, column in enumerate(self.prev_columns):
            col_act = column.activate(train_X_ID=batch_ID, mode=mode)
            # activations.append(self.adapter(column.idx_layer_mapping, column.activation, column_idx))
            activations.append(self.adapter(col_act, column_idx))
        # activations = self.adapter(self.prev_columns, self.target_column)
        num_levels = self.target_column.num_levels
        final_activations = []
        for level in range(1, num_levels):
            activations_at_level = [a[level-1] for a in activations]
            res = torch.zeros_like(activations_at_level[0])
            for activation_at_level in activations_at_level:
                res = torch.add(res, activation_at_level)
            final_activations.append(res)
        preds = self.target_column.predict(train_X,
                                           final_activations)
        return preds
    """def predict(self, train_X, mode='train'):
        batch_ID = train_X['ID']
        for column in self.prev_columns:
            column.activate(train_X_ID=batch_ID, mode=mode)
        activations = self.adapter(self.prev_columns, self.target_column)
        preds = self.target_column.predict(train_X,
                                           activations)
        return preds"""
