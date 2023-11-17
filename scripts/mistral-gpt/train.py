from datasets import load_dataset
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training
from peft import LoraConfig, get_peft_model
import transformers
from datetime import datetime

MAX_LENGTH = 2048

def tokenize(prompt):
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length",
    )
    result["labels"] = result["input_ids"].copy()
    return result


def generate_and_tokenize_prompt(data_point):
    messages = [
        {"role": "user", "content": f"Tóm tắt đoạn văn ngắn gọn:\n{data_point['Content']}"},
        {"role": "assistant", "content": data_point["Summary"]}
        ]
    message_prompt = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=False)
    
    return tokenize(message_prompt)

def init_wandb(wandb_project: None):
    import wandb
    import os
    wandb.login()
    if wandb_project:
        os.environ["WANDB_PROJECT"] = wandb_project



if __name__ == "__main__":

    TRANSFORMERS_CACHE="/datadrive04/temp_huggingface/models"
    PROJECT_ID = "summary-finetune"
    BASE_MODEL_NAME = "mistral"
    MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.1"
    DATASET_PATH = 'datasets/cleaned-summary-vietnamese_17_11_2023.csv'
    use_wandb = True
    
    if use_wandb:
        init_wandb(f"{PROJECT_ID}-{BASE_MODEL_NAME}")
    # CONFIG
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            "lm_head",
        ],
        bias="none",
        lora_dropout=0.05,  # Conventional
        task_type="CAUSAL_LM",
    )


    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, 
        trust_remote_code=True,
        cache_dir = TRANSFORMERS_CACHE)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        model_max_length=512,
        padding_side="left",
        trust_remote_code=True,
        add_eos_token=True,
        cache_dir =TRANSFORMERS_CACHE)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token = tokenizer.eos_token


    # DATASET                               
    dataset  = load_dataset('csv', data_files=[DATASET_PATH], split="train")
    splited = dataset.train_test_split(test_size=0.1)
    train_dataset, test_dataset = splited['train'], splited['test']

    tokenized_train_dataset = train_dataset.map(generate_and_tokenize_prompt)
    tokenized_test_dataset = test_dataset.map(generate_and_tokenize_prompt)

    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, config)

    if torch.cuda.device_count() > 1: # If more than 1 GPU
        model.is_parallelizable = True
        model.model_parallel = True


    trainer = transformers.Trainer(
        model=model,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_test_dataset,
        args=transformers.TrainingArguments(
            output_dir=f"./{BASE_MODEL_NAME}/{PROJECT_ID}" ,
            warmup_steps=50,
            per_device_train_batch_size=2,
            gradient_checkpointing=True,
            gradient_accumulation_steps=4,
            max_steps=1000,
            learning_rate=2.5e-5, # Want about 10x smaller than the Mistral learning rate
            logging_steps=50,
            bf16=False,
            optim="paged_adamw_8bit",
            logging_dir="./logs",        # Directory for storing logs
            save_strategy="steps",       # Save the model checkpoint every logging step
            save_steps=50,                # Save checkpoints every 50 steps
            evaluation_strategy="steps", # Evaluate the model every logging step
            eval_steps=5,               # Evaluate and save checkpoints every 50 steps
            do_eval=True,                # Perform evaluation at the end of training
            report_to="wandb",           # Comment this out if you don't want to use weights & baises
            run_name=f"{BASE_MODEL_NAME}-{PROJECT_ID}-{datetime.now().strftime('%Y-%m-%d-%H-%M')}"          # Name of the W&B run (optional)
        ),
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
    trainer.train()