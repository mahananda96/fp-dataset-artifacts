from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
import torch
import numpy as np

# Load the dataset and tokenizer
dataset = load_dataset('snli')
tokenizer = AutoTokenizer.from_pretrained("google/electra-small-discriminator")

def tokenize_and_encode(example):
    return tokenizer(example['premise'], example['hypothesis'], truncation=True, padding="max_length", max_length=128)

# Tokenize and preprocess data
encoded_dataset = dataset.map(tokenize_and_encode, batched=True)

# Artifact model training
artifact_model = AutoModelForSequenceClassification.from_pretrained("google/electra-small-discriminator", num_labels=3)
artifact_training_args = TrainingArguments(
    output_dir="./artifact_model",
    evaluation_strategy="steps",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=32,
    num_train_epochs=3,
    weight_decay=0.01,
)

# Train artifact model
artifact_trainer = Trainer(
    model=artifact_model,
    args=artifact_training_args,
    train_dataset=encoded_dataset['train'],
    eval_dataset=encoded_dataset['validation'],
    tokenizer=tokenizer
)
artifact_trainer.train()


# Generate artifact model predictions
artifact_trainer.model.eval()
def get_artifact_logits(batch):
    inputs = {k: torch.tensor(v).to(artifact_trainer.model.device) for k, v in batch.items() if k in tokenizer.model_input_names}
    with torch.no_grad():
        logits = artifact_trainer.model(**inputs).logits
    return logits.cpu().numpy()

# Add artifact model logits as features to the dataset
def add_artifact_logits(batch):
    logits = get_artifact_logits(batch)
    batch['artifact_logits'] = logits.tolist()
    return batch

encoded_dataset_with_artifacts = encoded_dataset.map(add_artifact_logits, batched=True)

from transformers import Trainer


class ResidualTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        artifact_logits = inputs.pop("artifact_logits")
        outputs = model(**inputs)
        logits = outputs.logits

        # Compute residual logits
        residual_logits = logits - torch.tensor(artifact_logits).to(logits.device)
        loss = torch.nn.functional.cross_entropy(residual_logits, labels)

        return (loss, outputs) if return_outputs else loss

main_model = AutoModelForSequenceClassification.from_pretrained("google/electra-small-discriminator", num_labels=3)
main_training_args = TrainingArguments(
    output_dir="./main_model",
    evaluation_strategy="steps",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=32,
    num_train_epochs=3,
    weight_decay=0.01,
)

main_trainer = ResidualTrainer(
    model=main_model,
    args=main_training_args,
    train_dataset=encoded_dataset_with_artifacts['train'],
    eval_dataset=encoded_dataset_with_artifacts['validation'],
    tokenizer=tokenizer
)
main_trainer.train()

results = main_trainer.evaluate()
print("Main model evaluation results:", results)
