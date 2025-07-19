from torch.utils.data import Dataset
import torch

class UniversalTranslationDataset(Dataset):
    """Dataset universal para todos los modelos de traducción"""
    
    def __init__(self, dataframe, tokenizer, config, source_col="catalan", target_col="chino"):
        self.df = dataframe
        self.tokenizer = tokenizer
        self.config = config
        self.source_col = source_col
        self.target_col = target_col
        self.max_length = config.max_length
        
    def __len__(self):
        return len(self.df)
    
    def prepare_source_text(self, text):
        """Prepara el texto fuente según el modelo"""
        if self.config.prefix_template:
            return self.config.prefix_template.format(text=text)
        return text
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        source_text = str(row[self.source_col]).strip()
        target_text = str(row[self.target_col]).strip()
        
        # Preparar texto según modelo
        if self.config.model_type == "causal":
            # Formato instrucción para modelos causales como Llama
            instruction = f"""<s>[INST] Translate the following Catalan text to Chinese:
{source_text}
[/INST]
{target_text}</s>"""
            
            encoding = self.tokenizer(
                instruction,
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt"
            )
            
            return {
                "input_ids": encoding["input_ids"].squeeze(),
                "attention_mask": encoding["attention_mask"].squeeze(),
                "labels": encoding["input_ids"].squeeze()
            }
        
        else:
            # Modelos seq2seq
            source_text = self.prepare_source_text(source_text)
            
            # Nueva API sin as_target_tokenizer
            model_inputs = self.tokenizer(
                source_text,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            
            # Tokenizar target
            labels = self.tokenizer(
                text_target=target_text,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )["input_ids"]
            
            labels = labels.squeeze()
            labels[labels == self.tokenizer.pad_token_id] = -100
            
            return {
                "input_ids": model_inputs["input_ids"].squeeze(),
                "attention_mask": model_inputs["attention_mask"].squeeze(),
                "labels": labels
            }