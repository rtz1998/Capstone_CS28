import torch
import pandas as pd
import os
import numpy as np
from transformers import BertTokenizer
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader, TensorDataset

class Dataloader(object):
    def __init__(self, text_path='./all_train.tsv', image_path=''):
        self.text_path = text_path
        self.image_path = image_path

    def read_text_image(self):
        text = pd.read_csv(self.text_path, delimiter='\t').head(10)
        col = ['clean_title','id','image_url','2_way_label']
        text = text[col]
        text = text.dropna(axis=0)
        txt = list(text['clean_title'])
        id = np.array(text['id'])
        images = list()
        for i in range(len(id)):
            image_path = os.path.join(self.images_text, str(id[i]) + '.jpeg')
            image = Image.open(image_path)
            image = image.resize((224, 224))
            images.append(image)
        images = np.array(images)
        label = np.array(text['2_way_label'])
        return txt, images, label

    def txt2tensor(self, text, max_len=512):
        input_ids = list()
        token_type_ids = list()
        attention_mask = list()
        for i in text:
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            tokens = tokenizer.encode_plus(i, max_length=max_len, padding='max_length', truncation=True)
            input_ids.append(tokens['input_ids'])
            token_type_ids.append([tokens['token_type_ids']])
            attention_mask.append(tokens['attention_mask'])
        input_ids = torch.from_numpy(np.array(input_ids))
        token_type_ids = torch.from_numpy(np.array(token_type_ids))
        attention_mask = torch.from_numpy(np.array(attention_mask))
        return input_ids, token_type_ids, attention_mask

    def img2tensor(self, images):
        transform = transforms.Compose([transforms.RandomCrop(32,padding = 4),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.RandomRotation((-45, 45)),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.229, 0.224, 0.225))
                                        ])
        images = transform(images)
        return images

    def label2tensor(self, label):
        return torch.from_numpy(np.array(label))

    def loader(self):
        txt, images, labels = self.read_text_image()
        input_ids, token_type_ids, attention_mask = self.txt2tensor(txt)
        images = self.img2tensor(images)
        labels = self.label2tensor()
        dataset = TensorDataset(input_ids, token_type_ids, attention_mask, images, labels)
        loader = DataLoader(dataset=dataset, batch_size=64, shuffle=True, num_workers=0, drop_last=True)

        return loader



