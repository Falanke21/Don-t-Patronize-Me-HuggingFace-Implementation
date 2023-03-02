# Make sure you have 
# !pip install transformers
# !pip install datasets
# !pip install evaluate

import evaluate
import pandas as pd
import numpy as np
import torch
from datasets import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_scheduler
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm


# Helper function, Do necessary actions to process the dataset
def prepare_dataset(tokenize_function, dataset):
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    prepared_dataset = tokenized_dataset.remove_columns(["text"])
    prepared_dataset = prepared_dataset.rename_column("label", "labels")
    prepared_dataset.set_format("torch")
    return prepared_dataset

# From pandas dataframe to pytorch dataloader
def get_dataloaders(args: dict, train_data: pd.DataFrame, val_data: pd.DataFrame):
    # Hugging face dataset object
    train_dataset = Dataset.from_pandas(train_data)
    val_dataset = Dataset.from_pandas(val_data)

    # Remove the __index_level_0__ column if it exists in datasets
    if "__index_level_0__" in train_dataset.column_names:
        train_dataset = train_dataset.remove_columns(["__index_level_0__"])
    if "__index_level_0__" in val_dataset.column_names:
        val_dataset = val_dataset.remove_columns(["__index_level_0__"])

    # Create the pre-trained model
    tokenizer = AutoTokenizer.from_pretrained(args["PRETRAINED_MODEL_NAME"])

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)
    
    train_prepared_dataset = prepare_dataset(tokenize_function, train_dataset)
    train_dataloader = DataLoader(train_prepared_dataset, shuffle=True, batch_size=args["BATCH_SIZE"])

    val_prepared_dataset = prepare_dataset(tokenize_function, val_dataset)
    val_dataloader = DataLoader(val_prepared_dataset, batch_size=args["BATCH_SIZE"])
    return train_dataloader, val_dataloader


def train_model(args: dict, device, train_dataloader: DataLoader, model_name = "model"):
    model = AutoModelForSequenceClassification.from_pretrained(args["PRETRAINED_MODEL_NAME"], num_labels=2)
    optimizer = AdamW(model.parameters(), lr=args["LEARNING_RATE"])

    # LR scheduler
    if args["USE_LR_SCHEDULER"]:
        num_training_steps = args["NUM_EPOCHS"] * len(train_dataloader)
        lr_scheduler = get_scheduler(
            name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
        )

    model.to(device)
    torch.cuda.empty_cache()

    model.train()
    for epoch in range(args["NUM_EPOCHS"]):
        print("Epoch: {}".format(epoch))
        pbar = tqdm(train_dataloader)
        for batch in pbar:
            # batch is a dictionary of keys: labels, input_ids,token_type_ids, attention_mask
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()

            optimizer.step()
            if args["USE_LR_SCHEDULER"]:
                lr_scheduler.step()
            optimizer.zero_grad()
            pbar.set_description(f"train loss: {loss.item()}")

            # Save the model at the end of each epoch
            model.save_pretrained("model-epoch-{}".format(epoch))

    # Save the model
    model.save_pretrained(model_name)
    return model


def calculate_f1(model_name: str, device, val_dataloader: DataLoader) -> float:
    # Load the model
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    model.to(device)

    # Use the validation dataset to evaluate the model
    model.eval()

    # We are interested in the F1 score
    metric = evaluate.combine(["accuracy", "f1", "precision", "recall"])
    for batch in val_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        # print(predictions)
        # print(batch["labels"])
        metric.add_batch(predictions=predictions, references=batch["labels"])

    result_dict = metric.compute()
    return result_dict["f1"]
