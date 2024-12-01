import datasets
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, HfArgumentParser
import evaluate
from helpers import prepare_dataset_hypothesis_only, compute_accuracy
import os
import json
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


NUM_PREPROCESSING_WORKERS = 2


def main():
    argp = HfArgumentParser(TrainingArguments)
    argp.add_argument('--model', type=str, default='google/electra-small-discriminator', help="The base model to fine-tune.")
    argp.add_argument('--task', type=str, choices=['nli', 'qa'], required=True, help="Task to train/evaluate on.")
    argp.add_argument('--dataset', type=str, default=None, help="Overrides the default dataset.")
    argp.add_argument('--max_length', type=int, default=128, help="Max sequence length.")
    argp.add_argument('--max_train_samples', type=int, default=None, help='Limit the number of examples to train on.')
    argp.add_argument('--max_eval_samples', type=int, default=None, help='Limit the number of examples to evaluate on.')

    training_args, args = argp.parse_args_into_dataclasses()
    dataset_id = tuple(args.dataset.split(':')) if args.dataset is not None else \
            default_datasets[args.task]

    # Dataset selection
    dataset = datasets.load_dataset('snli')  # Directly load SNLI dataset
    
    # Initialize model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=3)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)

    # Use hypothesis-only dataset preparation
    prepare_train_dataset = prepare_eval_dataset = \
        lambda exs: prepare_dataset_hypothesis_only(exs, tokenizer, args.max_length)

    # Preprocess dataset
    print("Preprocessing data... (this takes a little bit, should only happen once per dataset)")
    if dataset_id == ('snli',):
        # remove SNLI examples with no label
        dataset = dataset.filter(lambda ex: ex['label'] != -1)
    train_dataset = dataset['train']
    eval_dataset = dataset['validation']
    
    train_dataset_featurized = train_dataset.map(
        prepare_train_dataset,
        batched=True,
        num_proc=NUM_PREPROCESSING_WORKERS,
        remove_columns=train_dataset.column_names
    )
    
    eval_dataset_featurized = eval_dataset.map(
        prepare_eval_dataset,
        batched=True,
        num_proc=NUM_PREPROCESSING_WORKERS,
        remove_columns=eval_dataset.column_names
    )

    compute_metrics = compute_accuracy
    # This function wraps the compute_metrics function, storing the model's predictions
    # so that they can be dumped along with the computed metrics
    eval_predictions = None
    def compute_metrics_and_store_predictions(eval_preds):
        nonlocal eval_predictions
        eval_predictions = eval_preds
        return compute_metrics(eval_preds)

    # Check for the latest checkpoint in the output directory
    checkpoint_dir = os.path.join(training_args.output_dir, 'checkpoint-12876')  # default checkpoint folder

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset_featurized,
        eval_dataset=eval_dataset_featurized,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics_and_store_predictions,
    )


    # Train and/or evaluate
    if training_args.do_train:
        # Check if there's a checkpoint to resume from
        if os.path.exists(checkpoint_dir):
            print(f"Resuming training from checkpoint: {checkpoint_dir}")
            trainer.train(resume_from_checkpoint=checkpoint_dir)
        else:
            print("No checkpoint found, starting training from scratch.")
            trainer.train()

        # Save the model after training
        trainer.save_model()

    if training_args.do_eval:
        results = trainer.evaluate()

        # To add custom metrics, replace "compute_metrics" function.
        print('Evaluation results:')
        print(results)

        os.makedirs(training_args.output_dir, exist_ok=True)

        with open(os.path.join(training_args.output_dir, 'eval_metrics.json'), encoding='utf-8', mode='w') as f:
            json.dump(results, f)
        
        with open(os.path.join(training_args.output_dir, 'eval_predictions.jsonl'), encoding='utf-8', mode='w') as f:
            for i, example in enumerate(eval_dataset):
                example_with_prediction = dict(example)
                example_with_prediction['predicted_scores'] = eval_predictions.predictions[i].tolist()
                example_with_prediction['predicted_label'] = int(eval_predictions.predictions[i].argmax())
                f.write(json.dumps(example_with_prediction))
                f.write('\n')
        
        
        # Generate true and predicted labels
        true_labels = [example['label'] for example in eval_dataset]
        predicted_labels = [int(pred.argmax()) for pred in eval_predictions.predictions]

        # Save confusion matrix and error rates
        cm = confusion_matrix(true_labels, predicted_labels, labels=[0, 1, 2])
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", ax=ax)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix for SNLI Evaluation')
        plt.show()

        # Save confusion matrix as an image
        cm_path = os.path.join(training_args.output_dir, 'confusion_matrix.png')
        fig.savefig(cm_path)
        print(f"Confusion matrix saved at {cm_path}")

        # Calculate per-label error rates
        total_per_label = cm.sum(axis=1)  # Total true occurrences for each label
        errors_per_label = total_per_label - np.diag(cm)  # Errors per label (off-diagonal sum)
        error_rates_per_label = errors_per_label / total_per_label

        # Display error rates
        for i, label in enumerate(['Entailment', 'Neutral', 'Contradiction']):
            print(f"Error rate for '{label}': {error_rates_per_label[i]:.2%}")

        # Save error rates to a file
        error_rates_path = os.path.join(training_args.output_dir, 'error_rates.json')
        with open(error_rates_path, 'w') as f:
            json.dump(
                {label: f"{error_rates_per_label[i]:.2%}" for i, label in enumerate(['Entailment', 'Neutral', 'Contradiction'])},
                f,
                indent=4
            )
        print(f"Error rates saved at {error_rates_path}")

if __name__ == "__main__":
    main()
