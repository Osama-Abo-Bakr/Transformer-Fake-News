# Transformer Fake News Detection

This repository implements a **fake news detection** model using **DistilBERT** from Hugging Face's Transformers library. The model is fine-tuned on a dataset to classify news articles as either real or fake.

## Repository Link
[Transformer-Fake-News](https://github.com/Osama-Abo-Bakr/Transformer-Fake-News)

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Dataset](#dataset)
- [Data Preprocessing](#data-preprocessing)
- [Model Architecture](#model-architecture)
- [Training Process](#training-process)
- [Evaluation](#evaluation)
- [Results](#results)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Overview
This project aims to classify news articles as either **fake** or **real** using a Transformer-based model, specifically DistilBERT. The workflow includes data preprocessing, tokenization, model training, evaluation, and saving the model for future use.

## Installation
Install the required dependencies:

```bash
!pip install transformers torch datasets
```

## Dataset
The model is trained on the **WELFake Dataset**, which contains labeled news articles.

### Dataset Preparation
- The dataset is loaded from a CSV file.
- Missing values in the `title` and `text` columns are filled with empty strings.
- The `text` column is created by concatenating the `title` and `text` columns.
- All text data is converted to lowercase.

Example preprocessing steps:
```python
# Filling missing values
data['title'] = data['title'].fillna('')
data['text'] = data['text'].fillna('')

# Combining title and text
data['text'] = data['title'] + ' ' + data['text']

# Dropping unnecessary columns
data = data.drop(columns=['Unnamed: 0', 'title'], axis=1)

# Converting text to lowercase
data['text'] = data['text'].apply(lambda x: x.lower())
```

### Dataset Splitting
The dataset is split into training and testing sets:
```python
from datasets import Dataset

dataset = Dataset.from_pandas(data)
dataset = dataset.train_test_split(0.2)
```

## Model Architecture
This project uses the **DistilBERT** model for binary sequence classification:
- **Tokenizer**: `DistilBertTokenizer`
- **Model**: `DistilBertForSequenceClassification`

### Tokenization
The dataset is tokenized with padding and truncation:
```python
def tokenize_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True)

# Apply tokenization
tokenized_datasets = dataset.map(tokenize_function, batched=True)
```

## Training Process
The model is fine-tuned using the following training arguments:
- **Number of Epochs**: 2
- **Batch Size (Train)**: 8
- **Batch Size (Eval)**: 16
- **Learning Rate Warmup Steps**: 500
- **Weight Decay**: 0.01

```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=2,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=1000
)
```

The `Trainer` class is used for training and evaluation:
```python
from transformers import Trainer

def compute_metrics(pred):
    prediction, label = pred
    prediction = prediction.argmax(axis=1)
    return {'accuracy': (prediction == label).mean()}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['test'],
    compute_metrics=compute_metrics,
)
trainer.train()
```

## Evaluation
The model is evaluated on the test dataset to measure performance:
```python
# Evaluate the model
trainer.evaluate(tokenized_datasets['test'])

# Generate predictions
prediction = trainer.predict(tokenized_datasets['test'])[1]
```

## Results
- **Test Accuracy**: `<calculated_accuracy>%`

## Model Saving
The trained model is saved for future use:
```python
trainer.save_model('spam-ham_model')
```

## Usage
To use this project in your applications, follow these steps:
1. Clone the repository and install dependencies.
2. Load and preprocess your data.
3. Fine-tune the model using the training script.
4. Evaluate the model and use it for predictions.

Example prediction pipeline:
```python
text = "Breaking news: stock market crashes!"
inputs = tokenizer(text, return_tensors="pt")
outputs = model(**inputs)
predicted_class = outputs.logits.argmax(axis=1).item()
print("Predicted Class:", "Fake" if predicted_class == 1 else "Real")
```

## Contributing
Contributions to this project are welcome. Please fork the repository and submit a pull request with your changes.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

