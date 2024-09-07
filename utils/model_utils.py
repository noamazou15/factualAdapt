import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

def setup_tokenizer(model_id, cache_directory):
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, cache_dir=cache_directory, use_auth_token='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')
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer

def load_base_model(model_id, cache_directory):
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=False
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quant_config,
        cache_dir=cache_directory,
        use_auth_token='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'
    )
    model.config.use_cache = True
    model.config.pretraining_tp = 1
    return model


def load_adapter(base_model, adapter_path):
    
    model = PeftModel.from_pretrained(base_model, adapter_path)
    
    return model
