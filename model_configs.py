from dataclasses import dataclass
from typing import Dict, Any, Optional

@dataclass
class ModelConfig:
    """Configuración para cada modelo"""
    name: str
    model_id: str
    model_type: str  # "seq2seq", "causal", "nllb", "m2m"
    tokenizer_class: str
    model_class: str
    batch_size: int
    gradient_accumulation: int
    learning_rate: float
    max_length: int
    use_lora: bool = False
    lora_config: Optional[Dict[str, Any]] = None
    special_tokens: Optional[Dict[str, str]] = None
    prefix_template: Optional[str] = None
    device_map: str = "auto"
    torch_dtype: str = "float16"

#El primer modelo es el que se usa en caso de -s
MODEL_CONFIGS = {        
    "m2m-100": ModelConfig(
        name="M2M-100",
        model_id="facebook/m2m100_418M",
        model_type="m2m",
        tokenizer_class="M2M100Tokenizer",
        model_class="M2M100ForConditionalGeneration",
        batch_size=16,
        gradient_accumulation=2,
        learning_rate=5e-5,
        max_length=256,
        special_tokens={"src_lang": "ca", "tgt_lang": "zh"},
        torch_dtype="float32",
        device_map="cuda"
    ),
    
    "llama-3.2": ModelConfig(
        name="Llama-3.2-3B",
        model_id="meta-llama/Llama-3.2-3B",
        model_type="causal",
        tokenizer_class="AutoTokenizer",
        model_class="AutoModelForCausalLM",
        batch_size=2,  # Reducido para RTX 4060
        gradient_accumulation=16,
        learning_rate=2e-4,
        max_length=512,
        use_lora=True,
        lora_config={
            "r": 16,
            "lora_alpha": 32,
            "target_modules": ["q_proj", "v_proj"],
            "lora_dropout": 0.1,
            "bias": "none"
        },
        torch_dtype="float32",
        device_map="cuda"

    ),
        "mt5-large": ModelConfig(
        name="mT5-Large",
        model_id="google/mt5-large",
        model_type="seq2seq",
        tokenizer_class="T5Tokenizer",
        model_class="AutoModelForSeq2SeqLM",
        batch_size=4,  # Reducido para RTX 4060
        gradient_accumulation=8,
        learning_rate=3e-5,
        max_length=256,
        prefix_template="translate Catalan to Chinese: {text}",
        torch_dtype="float32",  # Cambiado de float16 a float32 para evitar problemas
        device_map="cuda"  # Cambiado de "auto" a "cuda"
    ),
    
    "mt5-xl": ModelConfig(
        name="mT5-XL",
        model_id="google/mt5-xl",
        model_type="seq2seq",
        tokenizer_class="T5Tokenizer",
        model_class="AutoModelForSeq2SeqLM",
        batch_size=2,  # Muy reducido para RTX 4060
        gradient_accumulation=16,
        learning_rate=2e-5,
        max_length=128,
        prefix_template="translate Catalan to Chinese: {text}",
        torch_dtype="float32",
        device_map="cuda"
    ),
    
    "nllb-200": ModelConfig(
        name="NLLB-200",
        model_id="facebook/nllb-200-distilled-600M",
        model_type="nllb",
        tokenizer_class="AutoTokenizer",
        model_class="AutoModelForSeq2SeqLM",
        batch_size=16,
        gradient_accumulation=2,
        learning_rate=5e-5,
        max_length=256,
        special_tokens={"src_lang": "cat_Latn", "tgt_lang": "zho_Hans"},
        torch_dtype="float32",
        device_map="cuda"
    ),

        "madlad-400": ModelConfig(
        name="MADLAD-400",
        model_id="google/madlad400-3b-mt",
        model_type="seq2seq",
        tokenizer_class="T5Tokenizer",
        model_class="T5ForConditionalGeneration",
        batch_size=4,
        gradient_accumulation=8,
        learning_rate=3e-5,
        max_length=256,
        prefix_template="<2zh> {text}",
        torch_dtype="float32",
        device_map="cuda"
    ),
}