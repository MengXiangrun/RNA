import pandas as pd
import torch
import numpy as np
import random
import os


class RNADataset():
    def __init__(self, save_name='1', batch_size=1):
        self.save_name = save_name
        self.batch_size = batch_size

        self.train_rna_list = None
        self.valid_rna_list = None
        self.test_rna_list = None

        self.train_max_num_token = None
        self.valid_max_num_token = None
        self.test_max_num_token = None

        self.max_num_token = None

        # token = base or cls/eos/pad
        self.token_type = ['A', 'U', 'G', 'C', '-', 'X', '<cls>', '<eos>', '<pad>']
        self.token2index = {token: index for index, token in enumerate(self.token_type)}

        is_saved = self.load()

        if is_saved is False:
            train_rna_list, train_max_num_token = self.read(sequences_path='train_sequences.csv',
                                                            labels_path='train_labels.csv')
            valid_rna_list, valid_max_num_token = self.read(sequences_path='validation_sequences.csv',
                                                            labels_path='validation_labels.csv')
            test_rna_list, test_max_num_token = self.read(sequences_path='test_sequences.csv',
                                                          labels_path=None)

            self.train_max_num_token = train_max_num_token
            self.valid_max_num_token = valid_max_num_token
            self.test_max_num_token = test_max_num_token

            self.max_num_token = max(train_max_num_token, valid_max_num_token, test_max_num_token)

            if self.batch_size == 1:
                train_rna_list = self.preprocessing(rna_list=train_rna_list, is_test=False)
                valid_rna_list = self.preprocessing(rna_list=valid_rna_list, is_test=False)
                test_rna_list = self.preprocessing(rna_list=test_rna_list, is_test=True)

            if self.batch_size > 1:
                # 2 = one <'cls'> for head of sequence + one <'eos'> for end of sequence
                self.max_num_token = self.max_num_token + 2
                train_rna_list = self.preprocessing(rna_list=train_rna_list, is_test=False)
                valid_rna_list = self.preprocessing(rna_list=valid_rna_list, is_test=False)
                test_rna_list = self.preprocessing(rna_list=test_rna_list, is_test=True)

            self.train_rna_list = train_rna_list
            self.valid_rna_list = valid_rna_list
            self.test_rna_list = test_rna_list

            self.save()

        # a, b, c = self.dataloader(is_train=True)

        print()

    def read(self, sequences_path=None, labels_path=None):
        max_num_token = 0
        rna_list = []

        if sequences_path is not None and labels_path is None:
            sequences = pd.read_csv(filepath_or_buffer=sequences_path, na_filter=False)
            for row_index, row in sequences.iterrows():
                sequence = row['sequence']
                sequence = list(sequence)

                rna = {}
                rna['sequence'] = sequence
                print(rna)

                num_token = len(sequence)
                if num_token >= max_num_token:
                    max_num_token = num_token

                rna_list.append(rna)

        if labels_path is not None:
            labels = pd.read_csv(filepath_or_buffer=labels_path, na_filter=False)

            labels['ID'] = labels['ID'].astype(str)
            labels['resname'] = labels['resname'].astype(str)

            labels['resid'] = labels['resid'].astype(int)

            labels['x_1'] = labels['x_1'].replace('', np.nan)
            labels['x_1'] = labels['x_1'].fillna(np.inf)
            labels['x_1'] = labels['x_1'].astype(float)

            labels['y_1'] = labels['y_1'].replace('', np.nan)
            labels['y_1'] = labels['y_1'].fillna(np.inf)
            labels['y_1'] = labels['y_1'].astype(float)

            labels['z_1'] = labels['z_1'].replace('', np.nan)
            labels['z_1'] = labels['z_1'].fillna(np.inf)
            labels['z_1'] = labels['z_1'].astype(float)

            start_index_list = labels[labels['resid'] == 1].index.tolist()
            start_index_dict = {rna_index: row_index for rna_index, row_index in enumerate(start_index_list)}

            for rna_index, row_index in start_index_dict.items():
                start_index = row_index
                if rna_index == list(start_index_dict.keys())[-1]:
                    rna = labels.iloc[start_index:]
                else:
                    end_index = start_index_dict[rna_index + 1]
                    rna = labels.iloc[start_index:end_index]
                print(rna)

                sequence = rna['resname'].values.tolist()
                coordinate = rna[['x_1', 'y_1', 'z_1']].values.tolist()

                rna = {}
                rna['sequence'] = sequence
                rna['coordinate'] = coordinate
                print(rna)

                num_token = len(sequence)
                if num_token >= max_num_token:
                    max_num_token = num_token

                rna_list.append(rna)

                # reverse
                sequence.reverse()
                coordinate.reverse()

                rna = {}
                rna['sequence'] = sequence
                rna['coordinate'] = coordinate

                rna_list.append(rna)

        return rna_list, max_num_token

    def preprocessing(self, rna_list, is_test):
        rna_data_list = []
        if self.batch_size == 1:
            if is_test is False:
                for rna in rna_list:
                    sequence = rna['sequence']  # example: ['A', 'G', 'G', 'C', 'A', 'A', 'A', 'G', 'C', 'C', 'A']
                    coordinate = rna['coordinate']

                    pad_mask = [1.0] * len(sequence)

                    sequence = list(map(lambda x: self.token2index[x], sequence))

                    sequence = torch.tensor(sequence, dtype=torch.int64)
                    pad_mask = torch.tensor(pad_mask, dtype=torch.float32)
                    coordinate = torch.tensor(coordinate, dtype=torch.float32)

                    rna_data = (sequence, pad_mask, coordinate)
                    rna_data_list.append(rna_data)

            if is_test is True:
                for rna in rna_list:
                    sequence = rna['sequence']  # example: ['A', 'G', 'G', 'C', 'A', 'A', 'A', 'G', 'C', 'C', 'A']
                    pad_mask = [1.0] * len(sequence)
                    sequence = list(map(lambda x: self.token2index[x], sequence))

                    sequence = torch.tensor(sequence, dtype=torch.int64)
                    pad_mask = torch.tensor(pad_mask, dtype=torch.float32)
                    coordinate = torch.zeros((len(sequence), 3))
                    coordinate = coordinate.to(torch.float32)

                    rna_data = (sequence, pad_mask, coordinate)
                    rna_data_list.append(rna_data)

        if self.batch_size > 1:
            if is_test is False:
                for rna in rna_list:
                    sequence = rna['sequence']  # example: ['A', 'G', 'G', 'C', 'A', 'A', 'A', 'G', 'C', 'C', 'A']
                    coordinate = rna['coordinate']

                    sequence = ['<cls>'] + sequence + ['<eos>']
                    pad_mask = [1.0] * len(sequence)
                    coordinate = [[np.inf, np.inf, np.inf]] + coordinate + [[np.inf, np.inf, np.inf]]

                    while 1:
                        assert len(sequence) == len(coordinate)
                        assert len(sequence) == len(pad_mask)
                        assert len(pad_mask) == len(coordinate)
                        if len(sequence) == self.max_num_token: break
                        sequence.append('<pad>')
                        pad_mask.append(0.0)
                        coordinate.append([np.inf, np.inf, np.inf])

                    sequence = list(map(lambda x: self.token2index[x], sequence))

                    sequence = torch.tensor(sequence, dtype=torch.int64)
                    pad_mask = torch.tensor(pad_mask, dtype=torch.float32)
                    coordinate = torch.tensor(coordinate, dtype=torch.float32)

                    rna_data = (sequence, pad_mask, coordinate)
                    rna_data_list.append(rna_data)

            if is_test is True:
                for rna in rna_list:
                    sequence = rna['sequence']  # example: ['A', 'G', 'G', 'C', 'A', 'A', 'A', 'G', 'C', 'C', 'A']

                    sequence = ['<cls>'] + sequence + ['<eos>']
                    pad_mask = [1.0] * len(sequence)

                    while 1:
                        assert len(sequence) == len(pad_mask)
                        if len(sequence) == self.max_num_token: break
                        sequence.append('<pad>')
                        pad_mask.append(0.0)

                    sequence = list(map(lambda x: self.token2index[x], sequence))

                    sequence = torch.tensor(sequence, dtype=torch.int64)
                    pad_mask = torch.tensor(pad_mask, dtype=torch.float32)
                    coordinate = torch.zeros((len(sequence), 3))
                    coordinate = coordinate.to(torch.float32)

                    rna_data = (sequence, pad_mask, coordinate)
                    rna_data_list.append(rna_data)

        return rna_data_list

    def dataloader(self, is_train=False, is_valid=False, is_test=False, is_shuffle=True):
        batch_sequence_list = []
        batch_pad_mask_list = []
        batch_coordinate_list = []

        if is_train:  data_list = self.train_rna_list.copy()
        if is_valid:  data_list = self.valid_rna_list.copy()
        if is_test:  data_list = self.test_rna_list.copy()

        batch_size = self.batch_size

        if is_shuffle: random.shuffle(data_list)

        for index in range(0, len(data_list), batch_size):
            batch_data = data_list[index: index + batch_size]
            batch_sequence = [data[0] for data in batch_data]
            batch_pad_mask = [data[1] for data in batch_data]
            batch_coordinate = [data[2] for data in batch_data]

            batch_sequence = torch.stack(batch_sequence, dim=0)
            batch_pad_mask = torch.stack(batch_pad_mask, dim=0)
            batch_coordinate = torch.stack(batch_coordinate, dim=0)

            batch_sequence_list.append(batch_sequence)
            batch_pad_mask_list.append(batch_pad_mask)
            batch_coordinate_list.append(batch_coordinate)

        return batch_sequence_list, batch_pad_mask_list, batch_coordinate_list

    def save(self):
        save = {}
        variable_dict = vars(self)
        for variable_name, variable_value in variable_dict.items():
            if isinstance(variable_value, (int, float, str, bool, dict, list, tuple)):
                save[variable_name] = variable_value
        torch.save(save, f'{self.save_name}.pth')

    def load(self):
        if os.path.exists(f'{self.save_name}.pth'):
            save = torch.load(f'{self.save_name}.pth')
            for variable_name, variable_value in save.items():
                setattr(self, variable_name, variable_value)
            return True
        else:
            return False


RNA = RNADataset(save_name='1', batch_size=2)
print(RNA)
