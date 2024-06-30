import os

from transformers import TrainingArguments
from trl import SFTTrainer

def setup_training_arguments(output_dir):
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=10,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=1,
        optim="paged_adamw_32bit",
        save_steps=25,
        logging_steps=25,
        learning_rate=2e-3,
        weight_decay=0.001,
        fp16=True,
        bf16=False,
        max_grad_norm=0.3,
        max_steps=-1,
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="constant",
        report_to=["tensorboard"],
        logging_dir=os.path.join(output_dir, 'logs')
    )

def setup_trainer(model, training_data, tokenizer, train_params, rank):
    from peft import LoraConfig, get_peft_model

    peft_parameters = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=rank,
        bias="none",
        task_type="CAUSAL_LM"
    )

    lora_model = get_peft_model(model, peft_parameters)

    return SFTTrainer(
        model=lora_model,
        train_dataset=training_data,
        peft_config=peft_parameters,
        dataset_text_field="text",
        tokenizer=tokenizer,
        args=train_params
    )
