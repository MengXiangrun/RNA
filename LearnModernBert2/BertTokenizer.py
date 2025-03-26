import torch


class SimpleTokenizer:
    def __init__(self, max_len=128):
        self.vocab = {'[PAD]': 0, '[CLS]': 1, '[SEP]': 2, '[MASK]': 3}
        self.word_to_id = {'[PAD]': 0, '[CLS]': 1, '[SEP]': 2, '[MASK]': 3}
        self.id_to_word = {0: '[PAD]', 1: '[CLS]', 2: '[SEP]', 3: '[MASK]'}
        self.next_id = 4
        self.max_len = max_len

    def tokenize(self, sentence):
        max_len = self.max_len
        token_list = sentence.lower().split()
        token_list = ['[CLS]'] + token_list + ['[SEP]']
        if len(token_list) < max_len:
            padding = ['[PAD]'] * (max_len - len(token_list))
            token_list.extend(padding)  # 在这一步添加pad
        elif len(token_list) >= max_len:
            token_list = token_list[:max_len - 1]
            token_list = token_list + ['[SEP]']

        assert len(token_list) == max_len
        return token_list

    def convert_tokens_to_ids(self, token_list):
        token_ids = []
        for token in token_list:
            if token not in self.word_to_id:
                self.word_to_id[token] = self.next_id
                self.id_to_word[self.next_id] = token
                self.vocab[token] = 1
                self.next_id += 1
            token_ids.append(self.word_to_id[token])
        return token_ids

    def encode(self, sentence_list):

        token_index = []
        attention_mask = []
        for sentence in sentence_list:
            token_list = self.tokenize(sentence)
            token_ids = self.convert_tokens_to_ids(token_list)
            token_index.append(token_ids)
            attention_mask_list = []
            for token in token_ids:
                if token == self.word_to_id['[PAD]']:
                    attention_mask_list.append(0)
                else:
                    attention_mask_list.append(1)
            attention_mask.append(attention_mask_list)

        return torch.tensor(token_index), torch.tensor(attention_mask)

    def decode(self, token_index):
        sentence_list = []
        if torch.is_tensor(token_index):
            token_index = token_index.detach().cpu().numpy().tolist()
        for ids in token_index:
            sentence = []
            for id in ids:
                if id in self.id_to_word.keys():
                    token = self.id_to_word[id]
                    sentence.append(token)
                else:
                    token = '[UNK]'
                    sentence.append(token)
            sentence = ' '.join(sentence)
            sentence_list.append(sentence)
        return sentence_list
