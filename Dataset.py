import pandas as pd
import torch
import numpy as np


class RNADataset():
    def __init__(self, batch_size=1):
        self.batch_size = batch_size

        train_rna_list, train_max_num_token = self.read(sequences_path='train_sequences.csv',
                                                        labels_path='train_labels.csv')
        valid_rna_list, valid_max_num_token = self.read(sequences_path='validation_sequences.csv',
                                                        labels_path='validation_labels.csv')
        test_rna_list, test_max_num_token = self.read(sequences_path='test_sequences.csv',
                                                      labels_path=None)

        # token = base or cls/eos/pad
        self.token_type = ['A', 'U', 'G', 'C', '-', 'X', '<cls>', '<eos>', '<pad>']

        self.token2index = {}
        for index, token in enumerate(self.token_type):
            self.token2index[token] = index
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
        if self.batch_size == 1 and is_test is False:
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

        if self.batch_size > 1 and is_test is False:
            rna_data_list = []
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
                    coordinate.append([np.inf, np.inf, np.inf])
                    pad_mask.append(0.0)

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

        return rna_data_list


RNA = RNADataset()
