import torch
from transformers import (
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    TrainerCallback
)
from torch.utils.data import Dataset
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import json
from datetime import datetime
import os
import gc

class TranslationDataset(Dataset):
    """Dataset genérico para traducción"""
    def __init__(self, dataframe, tokenizer, source_lang, target_lang, max_length=256, prefix=""):
        self.df = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.prefix = prefix
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Añadir prefijo si existe
        source_text = f"{self.prefix}{str(row[self.source_lang]).strip()}" if self.prefix else str(row[self.source_lang]).strip()
        target_text = str(row[self.target_lang]).strip()
        
        # Tokenizar
        source = self.tokenizer(
            source_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Para modelos que requieren as_target_tokenizer
        if hasattr(self.tokenizer, 'as_target_tokenizer'):
            with self.tokenizer.as_target_tokenizer():
                target = self.tokenizer(
                    target_text,
                    max_length=self.max_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt"
                )
        else:
            target = self.tokenizer(
                target_text,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
        
        labels = target["input_ids"].squeeze()
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            "input_ids": source["input_ids"].squeeze(),
            "attention_mask": source["attention_mask"].squeeze(),
            "labels": labels
        }

class BaseTrainer:
    """Clase base para entrenamiento de modelos de traducción"""
    
    def __init__(self, model_name, model_type="seq2seq"):
        self.model_name = model_name
        self.model_type = model_type
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def load_and_prepare_data(self, csv_path, tokenizer, source_col="catalan", target_col="chino", 
                              sample_size=None, prefix="", test_size=0.1):
        """Carga y prepara los datos"""
        print(f"\n📂 Cargando datos desde: {csv_path}")
        
        # Cargar CSV
        try:
            df = pd.read_csv(csv_path, encoding='utf-8')
        except:
            df = pd.read_csv(csv_path, encoding='utf-8-sig')
        
        inicial = len(df)
        
        # Limpieza
        df = df.dropna(subset=[source_col, target_col])
        df[source_col] = df[source_col].astype(str)
        df[target_col] = df[target_col].astype(str)
        df = df[df[source_col].str.strip() != '']
        df = df[df[target_col].str.strip() != '']
        
        # Filtrar longitud
        df = df[df[source_col].str.len() < 1000]
        df = df[df[target_col].str.len() < 1000]
        
        df = df.drop_duplicates(subset=[source_col])
        print(f"✅ Datos limpios: {len(df)}/{inicial} ({len(df)/inicial*100:.1f}%)")
        
        if sample_size and sample_size < len(df):
            df = df.sample(n=sample_size, random_state=42)
            print(f"📊 Usando muestra de {sample_size} ejemplos")
        
        # División
        train_df, temp_df = train_test_split(df, test_size=test_size, random_state=42)
        val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
        
        print(f"📈 División: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
        
        # Crear datasets
        train_dataset = TranslationDataset(train_df, tokenizer, source_col, target_col, prefix=prefix)
        val_dataset = TranslationDataset(val_df, tokenizer, source_col, target_col, prefix=prefix)
        test_dataset = TranslationDataset(test_df, tokenizer, source_col, target_col, prefix=prefix)
        
        return train_dataset, val_dataset, test_dataset
    
    def calculate_metrics(self, eval_preds, tokenizer):
        """Calcula métricas de evaluación"""
        predictions, labels = eval_preds
        
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        exact_match = sum(p.strip() == l.strip() for p, l in zip(decoded_preds, decoded_labels)) / len(decoded_preds)
        
        return {"exact_match": exact_match * 100}
    
    def get_training_args(self, output_dir, batch_size=4, gradient_accumulation=8, 
                         learning_rate=3e-5, num_epochs=3, warmup_ratio=0.1,
                         save_steps=500, eval_steps=500, max_length=256, **kwargs):
        """Configura argumentos de entrenamiento"""
        
        # Detectar soporte BF16
        bf16_supported = torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False
        
        args = Seq2SeqTrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=True,
            
            # Entrenamiento
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation,
            
            # Optimización
            bf16=bf16_supported,
            fp16=not bf16_supported and torch.cuda.is_available(),
            gradient_checkpointing=True,
            optim="adamw_torch",
            
            # Learning rate
            learning_rate=learning_rate,
            warmup_ratio=warmup_ratio,
            lr_scheduler_type="cosine",
            
            # Evaluación y guardado
            eval_strategy="steps",
            eval_steps=eval_steps,
            save_strategy="steps",
            save_steps=save_steps,
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            
            # Logging
            logging_dir=f"{output_dir}/logs",
            logging_steps=50,
            disable_tqdm=False,
            
            # Otros
            dataloader_num_workers=4,
            remove_unused_columns=False,
            
            # Generación
            predict_with_generate=True,
            generation_max_length=max_length,
            
            # Semilla
            seed=42,
        )
        
        # Actualizar con kwargs adicionales
        for key, value in kwargs.items():
            setattr(args, key, value)
            
        return args
    
    def cleanup_memory(self):
        """Limpia memoria GPU"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()