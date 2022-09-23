import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

from config import CFG

class Dataset(Dataset):
    def __init__(self, df, tokenizer, is_label=True) -> None:
        super().__init__()
        self.df = df
        self.tokenizer = tokenizer
        self.texts = df['text'].values
        self.is_label = is_label

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        text = self.texts[index]
        encoding = self.tokenizer(text, max_length=CFG.max_length,
                                padding='max_length', add_special_tokens=True,
                                truncation=True)
        encoding['input_ids'] = torch.tensor(encoding['input_ids']).flatten()
        encoding['attention_mask'] = torch.tensor(encoding['attention_mask']).flatten()
        if self.is_label:
            label = self.df['target'].values[index]
            label = np.array(label)
            return {'input_ids': encoding['input_ids'],
                    'attention_mask': encoding['attention_mask'],
                    'label': torch.tensor(label)}

        return {'input_ids': encoding['input_ids'],
                'attention_mask': encoding['attention_mask']}

def create_dataloader(df, tokenizer, batch_size=CFG.batch_size, is_train=True, is_label=True):
    dataset = Dataset(df, tokenizer, is_label)
    return DataLoader(dataset, batch_size, shuffle=True if is_train else False)

# if __name__ == '__main__':
#     train_df = pd.read_csv('./data/train.csv')
#     tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
#     data = create_dataloader(train_df, tokenizer)
#     print(data)