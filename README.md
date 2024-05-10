# LoRA Fine-Tuning on SAMSum Dataset

This project demonstrates the fine-tuning of a pre-trained Transformer model(T5) using LoRA (Low-Rank Adaptation) on the SAMSum dataset. 
LoRA allows for parameter-efficient training by adapting only a small subset of the model's weights.

## The requirements file should include:

peft==0.2.0\
transformers==4.27.2 \
datasets==2.9.0 \
accelerate==0.17.1 \
evaluate==0.4.0 \
loralib \
rouge-score \
tensorboard \
py7zr 

## Dataset
The SAMSum dataset, which contains conversational summaries, is used for training and evaluation. It can be loaded directly from Hugging Face's datasets library:

python

from datasets import load_dataset
dataset = load_dataset("samsum")
## Preprocessing
Tokenization and preparation of the data are crucial steps before training:

```python

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")

def preprocess_function(examples):
    inputs = ["summarize: " + dialog for dialog in examples["dialogue"]]
    model_inputs = tokenizer(inputs, max_length=256, truncation=True, padding="max_length")

    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["summary"], max_length=50, truncation=True, padding="max_length")

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_datasets = dataset.map(preprocess_function, batched=True)
## Model Training
Fine-tuning is done using the LoRA technique, which is integrated with the Hugging Face transformers library:

python
Copy code
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqTrainingArguments
from peft import LoraConfig, get_peft_model

# Load the pre-trained model
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
lora_config = LoraConfig(r=4, lora_alpha=16)
model = get_peft_model(model, lora_config)

training_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=3,
    predict_with_generate=True
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer
)

trainer.train()
Evaluation
After training, evaluate the model's performance using the ROUGE metric:

python
Copy code
import evaluate

rouge = evaluate.load("rouge")
results = rouge.compute(predictions=trainer.predict(tokenized_datasets["test"]).predictions, references=tokenized_datasets["test"]["summary"])
print(results)
Saving and Loading the Model
You can save the model to Hugging Face's Model Hub for easy access and sharing:

python
Copy code
model.push_to_hub("YourModelName")
To load the model:

python
Copy code
from transformers import AutoModelForSeq2SeqLM

model = AutoModelForSeq2SeqLM.from_pretrained("YourModelName")
Conclusion
This project highlights how LoRA can be effectively utilized for fine-tuning language models on summary tasks, achieving substantial performance gains with minimal computational resources.
