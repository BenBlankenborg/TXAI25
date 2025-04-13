import pandas as pd
import numpy as np
import csv
import torch
import accelerate
import optuna
from transformers import DistilBertForSequenceClassification, AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from datasets import Dataset
import matplotlib.pyplot as plt

DISTILBERT_DROPOUT = 0.1
DISTILBERT_ATT_DROPOUT = 0.1
device = "cuda"


def model_init():
    class CustomDistilBertForSequenceClassification(DistilBertForSequenceClassification):
        def __init__(self, config):
            super().__init__(config)

            # Add an additional dropout layer before classification
            self.custom_dropout = torch.nn.Dropout(DISTILBERT_DROPOUT)

        def forward(self, input_ids=None, attention_mask=None, labels=None):
            outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
            hidden_state = outputs[0]  # Extract hidden states
            pooled_output = hidden_state[:, 0]  # CLS token representation

            pooled_output = self.pre_classifier(pooled_output)  # Pre-classifier layer
            pooled_output = torch.relu(pooled_output)  # Activation to avoid negative values
            pooled_output = self.custom_dropout(pooled_output)  # Dropout before classification

            logits = self.classifier(pooled_output)  # Pass through classifier

            loss = None
            if labels is not None:
                loss_fct = torch.nn.MSELoss()  # Regression loss
                loss = loss_fct(logits.squeeze(), labels.squeeze())

            return {'loss': loss, 'logits': logits} if loss is not None else {'logits': logits}

    model = CustomDistilBertForSequenceClassification.from_pretrained("dbmdz/distilbert-base-turkish-cased", num_labels=1)
    model.to(device)
    
    return model

tokenizer = AutoTokenizer.from_pretrained("dbmdz/distilbert-base-turkish-cased")

# Read data
df = pd.read_csv('turkish_movie_sentiment_dataset_processed.csv', on_bad_lines='warn')
x = df['comment'].tolist()
y_ = df['point']
y = [float(score) for score in y_]

# FOR TESTING CODE ONLY
# x = x[0:10]
# y = y[0:10]

# Prepare data
data = {'review': x, 'labels': y}
dataset = Dataset.from_dict(data)

def tokenize_function(examples):
    return tokenizer(examples['review'], truncation=True, padding='max_length', max_length=512)

tokenized_dataset = dataset.map(tokenize_function, batched=True)
tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

# Train/Validation/Test split 80/20/10_samples
train_test = tokenized_dataset.train_test_split(test_size=10)
train_validation = train_test['train'].train_test_split(test_size=0.2)
x_train = train_validation['train']
x_val = train_validation['test']
x_test = train_test['test']

for data in x_test:
    print(data['labels'].item())

# Found using Optuna
nr_train_epochs = 2 #was 2
lr = 3.5e-05
wd = 0.02
batch_size = 16

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=nr_train_epochs,
    per_device_train_batch_size=batch_size,
    learning_rate=lr,
    weight_decay=wd,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=1,
    load_best_model_at_end=True,
    logging_dir="./logs",
)

def compute_mse(pred):
    predictions = pred.predictions.squeeze(-1)
    labels = pred.label_ids
    mse = ((predictions - labels) ** 2).mean()
    return {"mse": mse}

trainer = Trainer(
    model_init=model_init,
    args=training_args,
    train_dataset=x_train,
    eval_dataset=x_val,
    processing_class=tokenizer,
    compute_metrics=compute_mse
)

trainer.train() 

def enable_mc_dropout(model):
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.train()
            
def mc_dropout_eval(test_set):
    trained_model = trainer.model
    trained_model.eval()
    enable_mc_dropout(trained_model)
    amount_of_mc_dropout_forward_passes = 100
    total_nll = 0
    nlls = []
    total_dists = []
    labels = np.array([data_point.pop('labels', None) for data_point in test_set])
    for d, data_point in enumerate(test_set):
        distribution = []
        label = data_point.pop('labels', None)
        data_point = {key: value.unsqueeze(0).to(device) for key, value in data_point.items()}
        for i in range(amount_of_mc_dropout_forward_passes):
            with torch.no_grad():
                output = trained_model(**data_point)
            distribution.append(output['logits'].item())
            print(f"\rProgress: {i+1}/{amount_of_mc_dropout_forward_passes}", end='', flush=True)
        mean = np.mean(distribution)
        std = np.std(distribution)
        std = max(std, 1e-6)
        nll = 0.5 * np.log(2 * np.pi * std**2) + ((label - mean)**2) / (2 * std**2)
        total_nll += nll
        total_dists.append(distribution)
        
        nlls.append(nll)
        print(f"Data point {d+1}: NLL = {nll:.4f}")
        plot_and_save_distributions(distribution, label, d, amount_of_mc_dropout_forward_passes)

    avg_nll = total_nll / len(test_set)
    print(f"\nAverage NLL over test set: {avg_nll:.4f}")
    calibration_plot(total_dists, labels)

def plot_and_save_distributions(data, label, num, amount):
    plt.figure(figsize=(6, 4))
    plt.hist(data, bins=20, density=True, alpha=0.6, color='b', edgecolor='black')
    plt.title(f"Distribution {num+1}, Label = {label}")
    plt.xlabel("Value")
    plt.ylabel("Density")
    mean = np.mean(data)
    std = np.std(data)
    plt.subplots_adjust(bottom=0.25)
    txt=f"Over {amount} runs, the mean of this distibution is {mean:.2f} and the standard deviation is {std:.2f}."
    plt.figtext(0.5, 0.03, txt, wrap=True, horizontalalignment='center', fontsize=12)

    file_name = f"UQ_Distribution_ExtraDropout_{num+1}.png"
    plt.savefig(file_name)
    plt.close()
    
def calibration_plot(predictions_list, ground_truths, num_bins=10):
    """
    Creates a calibration plot for Monte Carlo Dropout uncertainty.

    Args:
        predictions_list (list of np.ndarray): List of arrays, where each array is MC Dropout predictions for a single input.
        ground_truths (np.ndarray): Array of true labels.
        num_bins (int): Number of bins for uncertainty grouping.
    """
    uncertainties = []
    errors = []

    for preds, gt in zip(predictions_list, ground_truths):
        mu = np.mean(preds)  # Mean prediction
        sigma = np.std(preds, ddof=1)  # Standard deviation (uncertainty)
        uncertainties.append(sigma)
        errors.append(abs(mu - gt))  # Absolute error

    # Bin predictions based on uncertainty
    bin_edges = np.linspace(min(uncertainties), max(uncertainties), num_bins + 1)
    binned_uncertainties = []
    binned_errors = []

    for i in range(num_bins):
        mask = (uncertainties >= bin_edges[i]) & (uncertainties < bin_edges[i + 1])
        if np.any(mask):
            binned_uncertainties.append(np.mean(np.array(uncertainties)[mask]))
            binned_errors.append(np.mean(np.array(errors)[mask]))

    # Plot calibration curve
    plt.figure(figsize=(6, 6))
    plt.plot(binned_uncertainties, binned_errors, 'o-', label="Model Calibration")
    plt.plot([0, max(binned_uncertainties)], [0, max(binned_uncertainties)], 'k--', label="Ideal Calibration")

    plt.xlabel("Predicted Standard Deviation (Uncertainty)")
    plt.ylabel("Actual Absolute Error")
    plt.title("Calibration Plot")
    plt.legend()
    plt.grid(True)
    file_name = f"CalibrationPlotMCD1.png"
    plt.savefig(file_name)
    plt.close()

mc_dropout_eval(x_test)



