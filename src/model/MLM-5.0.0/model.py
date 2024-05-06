import os
import torch
from tqdm import tqdm

from transformers import MBartForConditionalGeneration, AdamW
from tokenization_mlm import MLMTokenizer

path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


Config = {"batch_size": 32,
          "epoch_number": 10,
          "cuda_index": 0,
          "lr": 1e-4
          }


class Dataset(torch.utils.data.Dataset):
    def __init__(self, input_file_path):
        # combine input: original text with masked sbn
        print("Reading lines...")
        with open(input_file_path, encoding="utf-8") as f:
            self.text = f.readlines()

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        text = self.text[idx].split("\t")[0]
        sbn = self.text[idx].split("\t")[1].replace("\n", "")
        return text, sbn


def get_dataloader(input_file_path, batch_size=16):
    dataset = Dataset(input_file_path)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, pin_memory=True)
    return dataloader


class Generator:

    def __init__(self, lang="en", load_path=""):
        """
        :param train: train or test
        """
        self.epoch_number = Config["epoch_number"]
        self.device = torch.device(f"cuda:{Config['cuda_index']}" if torch.cuda.is_available() else "cpu")

        src_lang = {"en": "en_XX", "de": "de_DE", "it": "it_IT", "nl": "nl_XX"}
        self.tokenizer = MLMTokenizer.from_pretrained('XiaoZhang98/DRS-LMM-5.0.0', src_lang=src_lang.get(lang))
        self.target_tokenizer = MLMTokenizer.from_pretrained('XiaoZhang98/DRS-LMM-5.0.0', src_lang="<drs>")

        if len(load_path) == 0:
            self.model = MBartForConditionalGeneration.from_pretrained('XiaoZhang98/DRS-LMM-5.0.0')
        else:
            self.model = MBartForConditionalGeneration.from_pretrained(load_path)
        self.model.to(self.device)
        self.f1_list = []

    def evaluate(self, val_loader, save_path):
        with open(save_path, 'w+', encoding="utf-8") as f:
            self.model.eval()
            with torch.no_grad():
                for i, (text, target) in enumerate(tqdm(val_loader)):
                    x = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=256)['input_ids'].to(
                        self.device)
                    out_put = self.model.generate(x)
                    for j in range(len(out_put)):
                        o = out_put[j]
                        pred_text = self.target_tokenizer.decode(o, skip_special_tokens=True,
                                                                 clean_up_tokenization_spaces=False)
                        f.write(pred_text)
                        f.write('\n')

    def train(self, train_loader, val_loader, lr, epoch_number, accumulation_steps=4):
        optimizer = AdamW(self.model.parameters(), lr)
        for epoch in range(epoch_number):
            self.model.train()
            pbar = tqdm(train_loader)
            optimizer.zero_grad()  # Reset gradients tensors for each epoch
            for batch, (text, target) in enumerate(pbar):
                x = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=256)[
                    'input_ids'].to(
                    self.device)
                y = self.target_tokenizer(target, return_tensors='pt', padding=True, truncation=True, max_length=256)[
                    'input_ids'].to(
                    self.device)

                output = self.model(x, labels=y)
                loss = output.loss / accumulation_steps  # Normalize our loss (if averaged)
                loss.backward()

                if (batch + 1) % accumulation_steps == 0:  # Wait for several backward steps
                    optimizer.step()  # Now we can do an optimizer step
                    optimizer.zero_grad()  # Reset gradients tensors

                pbar.set_description(f"Loss: {format(loss.item(), '.3f')}")
