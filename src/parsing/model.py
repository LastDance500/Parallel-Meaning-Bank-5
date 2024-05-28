import os
import torch
from tqdm import tqdm
from transformers import (ByT5Tokenizer, T5ForConditionalGeneration,
                          MBartTokenizer, MBartForConditionalGeneration,
                          MT5Tokenizer, MT5ForConditionalGeneration,
                          AdamW)
from torch.optim.lr_scheduler import StepLR

path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

Config = {"batch_size": 6,
          "cuda_index": 0,
          "max_length": 512
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


def get_dataloader(input_file_path, batch_size=Config["batch_size"]):
    dataset = Dataset(input_file_path)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    return dataloader


class Generator:

    def __init__(self, lang, model, load_path=""):
        """
        Initialize the model and tokenizer.

        :param lang: The language code
        :param model: The model type (byt5, mt5, mbart)
        :param load_path: The path to load a model from (optional)
        """
        self.device = torch.device(f"cuda:{Config['cuda_index']}" if torch.cuda.is_available() else "cpu")
        self.tokenizer, self.model = self.initialize_model_and_tokenizer(model, load_path)
        self.model.to(self.device)

    def initialize_model_and_tokenizer(self, model, load_path):
        """
        Initialize the tokenizer and model based on model type.

        :param model: The model type (byt5, mt5, mbart)
        :param load_path: The path to load a model from (optional)
        :return: The tokenizer and model
        """
        if "byt5" in model:
            tokenizer = ByT5Tokenizer.from_pretrained(model)
            model_instance = T5ForConditionalGeneration.from_pretrained(
                load_path if load_path else model, max_length=Config["max_length"])
        elif "mt5" in model:
            tokenizer = MT5Tokenizer.from_pretrained(model)
            model_instance = MT5ForConditionalGeneration.from_pretrained(
                load_path if load_path else model, max_length=Config["max_length"])
        elif "mbart" in model:
            tokenizer = MBartTokenizer.from_pretrained(model)
            model_instance = MBartForConditionalGeneration.from_pretrained(
                load_path if load_path else model, max_length=Config["max_length"])
        else:
            raise ValueError("Model type not supported")

        return tokenizer, model_instance

    def evaluate(self, val_loader, save_path):
        with open(save_path, 'w', encoding="utf-8") as f:
            self.model.eval()
            with torch.no_grad():
                for i, (text, target) in enumerate(tqdm(val_loader)):
                    x = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True,
                                       max_length=Config["max_length"])['input_ids'].to(
                        self.device)
                    out_put = self.model.generate(x)
                    for j in range(len(out_put)):
                        o = out_put[j]
                        pred_text = self.tokenizer.decode(o, skip_special_tokens=True,
                                                          clean_up_tokenization_spaces=False)
                        f.write(pred_text)
                        f.write('\n')

    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch, (text, target) in enumerate(val_loader):
                x = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True,
                                   max_length=Config["max_length"])['input_ids'].to(self.device)
                y = self.tokenizer(target, return_tensors='pt', padding=True, truncation=True,
                                   max_length=Config["max_length"])['input_ids'].to(self.device)

                output = self.model(x, labels=y)
                total_loss += output.loss.item()

        average_loss = total_loss / len(val_loader)
        return average_loss

    def train(self, train_loader, val_loader, lr, epoch_number, patience=5, step_size=5, gamma=0.2, save_path="",
              min_epoch=10, min_delta=0.001):
        optimizer = AdamW(self.model.parameters(), lr)
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
        best_val_loss = float('inf')
        epochs_no_improve = 0

        for epoch in range(epoch_number):
            self.model.train()
            pbar = tqdm(train_loader)
            for batch, (text, target) in enumerate(pbar):
                x = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True,
                                   max_length=Config["max_length"])['input_ids'].to(self.device)
                y = self.tokenizer(target, return_tensors='pt', padding=True, truncation=True,
                                   max_length=Config["max_length"])['input_ids'].to(self.device)

                optimizer.zero_grad()
                output = self.model(x, labels=y)
                loss = output.loss
                loss.backward()
                optimizer.step()
                pbar.set_description(f"Loss: {loss.item():.3f}")

            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch + 1}, Current Learning Rate: {current_lr}")

            # Validation phase
            val_loss = self.validate(val_loader)
            print(f"val loss: {val_loss}")

            # Check if validation loss improved significantly
            loss_improvement = best_val_loss - val_loss
            if loss_improvement > min_delta:
                best_val_loss = val_loss
                epochs_no_improve = 0
                if len(save_path) != 0:
                    self.model.save_pretrained(save_path)
            else:
                epochs_no_improve += 1

            # Adjust learning rate
            scheduler.step()

            # Early stopping check
            if epochs_no_improve == patience and epoch > min_epoch:
                print("Early stopping triggered")
                break
