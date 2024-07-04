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
        save_steps=10000,
        logging_steps=10000,
        learning_rate=2e-3,
        weight_decay=0.001,
        fp16=True,
        bf16=False,
        max_grad_norm=0.3,
        max_steps=-1,
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="constant",
    )

def setup_trainer(lora_model, training_data, tokenizer, train_params, peft_parameters):

    return SFTTrainer(
        model=lora_model,
        train_dataset=training_data,
        peft_config=peft_parameters,
        dataset_text_field="text",
        tokenizer=tokenizer,
        args=train_params
    )


def get_lora_target_modules(specific_layers, base_model):
    attention_layers = []
    for name, module in base_model.named_modules():
        if 'attention' in name.lower():
            attention_layers.append((name, module))
    return attention_layers
    return specific_layers if specific_layers else None
