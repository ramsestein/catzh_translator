import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    AutoTokenizer,
    T5Tokenizer,
    T5ForConditionalGeneration,
    M2M100Tokenizer,
    M2M100ForConditionalGeneration,
    Seq2SeqTrainer,
    Trainer,
    Seq2SeqTrainingArguments,
    TrainingArguments,
    DataCollatorForSeq2Seq,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import gc
import time
import json
from transformers import TrainerCallback
from datetime import datetime
from model_configs import MODEL_CONFIGS, ModelConfig
from dataset_utils import UniversalTranslationDataset
from checkpoint_manager import CheckpointManager, AutoSaveCallback

class UniversalTrainer:
    """Entrenador universal para todos los modelos"""
    
    def __init__(self, checkpoint_base_dir="./training_outputs"):
        self.checkpoint_manager = CheckpointManager(checkpoint_base_dir)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def load_tokenizer(self, config: ModelConfig):
        """Carga el tokenizer según la configuración"""
        print(f"📥 Cargando tokenizer {config.tokenizer_class}...")
        
        tokenizer_map = {
            "AutoTokenizer": AutoTokenizer,
            "T5Tokenizer": T5Tokenizer,
            "M2M100Tokenizer": M2M100Tokenizer,
        }
        
        tokenizer_class = tokenizer_map.get(config.tokenizer_class, AutoTokenizer)
        
        if config.tokenizer_class == "T5Tokenizer":
            tokenizer = tokenizer_class.from_pretrained(config.model_id, use_fast=False, legacy=True)
        else:
            tokenizer = tokenizer_class.from_pretrained(config.model_id)
        
        # Configurar tokens especiales
        if config.special_tokens:
            for key, value in config.special_tokens.items():
                setattr(tokenizer, key, value)
        
        # Para modelos causales
        if config.model_type == "causal" and not tokenizer.pad_token:
            tokenizer.pad_token = tokenizer.eos_token
            
        return tokenizer
    
    def load_model(self, config: ModelConfig, checkpoint_path=None):
        """Carga el modelo según la configuración"""
        print(f"📥 Cargando modelo {config.name}...")
        
        model_map = {
            "AutoModelForSeq2SeqLM": AutoModelForSeq2SeqLM,
            "AutoModelForCausalLM": AutoModelForCausalLM,
            "T5ForConditionalGeneration": T5ForConditionalGeneration,
            "M2M100ForConditionalGeneration": M2M100ForConditionalGeneration,
        }
        
        model_class = model_map[config.model_class]
        
        # Configurar dtype
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32
        }
        torch_dtype = dtype_map.get(config.torch_dtype, torch.float16)
        
        # Cargar modelo
        model_path = checkpoint_path if checkpoint_path else config.model_id
        
        if config.device_map == "auto":
            model = model_class.from_pretrained(
                model_path,
                torch_dtype=torch_dtype,
                device_map="auto"
            )
        else:
            model = model_class.from_pretrained(
                model_path,
                torch_dtype=torch_dtype
            ).to(self.device)
        
        # Aplicar LoRA si está configurado
        if config.use_lora and not checkpoint_path:
            print("🔧 Aplicando LoRA...")
            task_type = TaskType.CAUSAL_LM if config.model_type == "causal" else TaskType.SEQ_2_SEQ_LM
            
            lora_config = LoraConfig(
                task_type=task_type,
                **config.lora_config
            )
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()
        
        return model
    
    def prepare_data(self, csv_path, tokenizer, config: ModelConfig):
        """Prepara los datasets"""
        print(f"📂 Cargando datos desde: {csv_path}")
        
        # Cargar CSV
        try:
            df = pd.read_csv(csv_path, encoding='utf-8')
        except:
            df = pd.read_csv(csv_path, encoding='utf-8-sig')
        
        # Limpieza básica
        inicial = len(df)
        df = df.dropna(subset=['catalan', 'chino'])
        df['catalan'] = df['catalan'].astype(str)
        df['chino'] = df['chino'].astype(str)
        df = df[df['catalan'].str.strip() != '']
        df = df[df['chino'].str.strip() != '']
        df = df[df['catalan'].str.len() < 1000]
        df = df[df['chino'].str.len() < 1000]
        df = df.drop_duplicates(subset=['catalan'])
        
        print(f"✅ Datos limpios: {len(df)}/{inicial} ({len(df)/inicial*100:.1f}%)")
        
        # División
        train_df, temp_df = train_test_split(df, test_size=0.1, random_state=42)
        val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
        
        print(f"📈 División: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
        
        # Crear datasets
        train_dataset = UniversalTranslationDataset(train_df, tokenizer, config)
        val_dataset = UniversalTranslationDataset(val_df, tokenizer, config)
        test_dataset = UniversalTranslationDataset(test_df, tokenizer, config)
        
        return train_dataset, val_dataset, test_dataset
    
    def get_training_args(self, output_dir, config: ModelConfig, steps_per_epoch):
        """Obtiene argumentos de entrenamiento según el modelo"""
        
        # Para RTX 4060 con 8GB, es mejor no usar mixed precision
        common_args = {
            "output_dir": output_dir,
            "overwrite_output_dir": True,
            "num_train_epochs": 3,
            "per_device_train_batch_size": config.batch_size,
            "per_device_eval_batch_size": config.batch_size,
            "gradient_accumulation_steps": config.gradient_accumulation,
            "learning_rate": config.learning_rate,
            "warmup_ratio": 0.1,
            "logging_steps": 50,
            "logging_first_step": True,  # Añadir esto
            "eval_strategy": "steps",
            "eval_steps": 500,
            "save_strategy": "steps",
            "save_steps": 500,
            "save_total_limit": 3,
            "load_best_model_at_end": True,
            "metric_for_best_model": "eval_loss",
            "bf16": False,
            "fp16": False,
            "gradient_checkpointing": True,
            "gradient_checkpointing_kwargs": {"use_reentrant": False},  # Más eficiente
            "dataloader_num_workers": 0,  # Cambiar a 0 para Windows/WSL
            "dataloader_pin_memory": False,  # Desactivar para acelerar inicio
            "remove_unused_columns": False,
            "push_to_hub": False,
            "report_to": [],
            "logging_dir": f"{output_dir}/logs",
            "ddp_find_unused_parameters": False,
            "optim": "adamw_torch",
            "adam_beta1": 0.9,
            "adam_beta2": 0.999,
            "adam_epsilon": 1e-8,
        }
        
        if config.model_type in ["seq2seq", "nllb", "m2m"]:
            return Seq2SeqTrainingArguments(
                **common_args,
                predict_with_generate=True,
                generation_max_length=config.max_length,
            )
        else:
            return TrainingArguments(**common_args)
    
    def train_model_on_dataset(self, model_key: str, dataset_path: str):
        """Entrena un modelo específico con un dataset"""
        
        config = MODEL_CONFIGS[model_key]
        dataset_name = os.path.basename(dataset_path).replace('.csv', '')
        model_dir = self.checkpoint_manager.get_model_dir(config.model_id, dataset_name)
        
        print(f"\n{'='*70}")
        print(f"🚀 ENTRENANDO: {config.name}")
        print(f"📊 Dataset: {dataset_name}")
        print(f"📁 Directorio: {model_dir}")
        print(f"{'='*70}\n")
        
        try:
            # Buscar checkpoint
            checkpoint = self.checkpoint_manager.find_latest_checkpoint(model_dir)
            
            # Cargar tokenizer y modelo
            tokenizer = self.load_tokenizer(config)
            model = self.load_model(config, checkpoint)
            
            # Preparar datos
            train_dataset, val_dataset, test_dataset = self.prepare_data(
                dataset_path, tokenizer, config
            )
            
            # Configurar entrenamiento
            steps_per_epoch = len(train_dataset) // (config.batch_size * config.gradient_accumulation)
            training_args = self.get_training_args(model_dir, config, steps_per_epoch)
            
            # Data collator
            print("🔧 Preparando data collator...")
            if config.model_type == "causal":
                data_collator = DataCollatorForLanguageModeling(
                    tokenizer=tokenizer,
                    mlm=False,
                    pad_to_multiple_of=8
                )
            else:
                data_collator = DataCollatorForSeq2Seq(
                    tokenizer=tokenizer,
                    model=model,
                    padding=True,
                    max_length=config.max_length
                )
            
            # Crear trainer con más logging
            print("🔧 Creando trainer...")
            trainer_class = Trainer if config.model_type == "causal" else Seq2SeqTrainer
            
            trainer = trainer_class(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                processing_class=tokenizer,
                data_collator=data_collator,
                compute_metrics=lambda p: self.calculate_metrics(p, tokenizer) if config.model_type != "causal" else None,
                callbacks=[AutoSaveCallback(self.checkpoint_manager, model_dir)]
            )
            
            print("✅ Trainer creado")
            
            # Entrenar con logging detallado
            print("\n🏃 Iniciando entrenamiento...")
            print("⏳ Preparando modelo y optimizador (puede tardar 1-2 minutos)...")
            
            if checkpoint:
                print(f"📍 Resumiendo desde: {checkpoint}")
                print("⏳ Cargando estado del optimizador...")
                
                # Añadir callback para mostrar progreso
                start_time = time.time()
                trainer.train(resume_from_checkpoint=checkpoint)
                
            else:
                print("🆕 Entrenando desde cero")
                print("⏳ Inicializando optimizador y preparando primera época...")
                
                # Temporalmente reducir logging para la inicialización
                original_logging = training_args.logging_steps
                training_args.logging_steps = 1  # Log desde el primer step
                
                import time
                start_time = time.time()
                
                # Callback personalizado para el primer step
                from transformers import TrainerCallback
                
                class FirstStepCallback(TrainerCallback):
                    def on_step_begin(self, args, state, control, **kwargs):
                        if state.global_step == 0:
                            elapsed = time.time() - start_time
                            print(f"\n✅ Primera iteración iniciada después de {elapsed:.1f} segundos")
                            print("🚀 El entrenamiento está en marcha!\n")
                        return control
                
                trainer.add_callback(FirstStepCallback())
                trainer.train()
                
                # Restaurar logging original
                training_args.logging_steps = original_logging
            
            # Guardar modelo final
            print("\n💾 Guardando modelo final...")
            self.checkpoint_manager.save_final_model(trainer, tokenizer, model_dir)
            
            # Evaluar si no es causal
            if config.model_type != "causal":
                print("\n📊 Evaluando en test set...")
                test_results = trainer.predict(test_dataset)
                print(f"Test metrics: {test_results.metrics}")
                
                # Guardar métricas
                with open(f"{model_dir}/test_metrics.json", 'w') as f:
                    json.dump(test_results.metrics, f, indent=2)
            
            print(f"\n✅ {config.name} con {dataset_name} completado!")
            
        except Exception as e:
            print(f"\n❌ Error entrenando {config.name} con {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
            
        finally:
            # Limpiar memoria
            if 'model' in locals():
                del model
            if 'trainer' in locals():
                del trainer
            self.cleanup_memory()
            
    def calculate_metrics(self, eval_preds, tokenizer):
        """Calcula métricas para modelos seq2seq"""
        predictions, labels = eval_preds
        
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        labels = torch.where(torch.tensor(labels) != -100, torch.tensor(labels), tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        exact_match = sum(p.strip() == l.strip() for p, l in zip(decoded_preds, decoded_labels)) / len(decoded_preds)
        
        return {"exact_match": exact_match * 100}
    
    def cleanup_memory(self):
        """Limpia memoria GPU"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    def train_all_combinations(self, models=None, datasets=None):
        """Entrena todas las combinaciones de modelos y datasets"""
        
        if models is None:
            models = list(MODEL_CONFIGS.keys())
        if datasets is None:
            datasets = [
                "./datasets/dataset_limpio.csv",
                "./datasets/dataset_optimizado.csv"
            ]
        
        total_combinations = len(models) * len(datasets)
        current = 0
        
        print(f"\n{'='*70}")
        print(f"🚀 INICIANDO ENTRENAMIENTO COMPLETO")
        print(f"📊 {len(models)} modelos × {len(datasets)} datasets = {total_combinations} entrenamientos")
        print(f"{'='*70}\n")
        
        # Crear resumen
        summary = {
            "start_time": datetime.now().isoformat(),
            "models": models,
            "datasets": datasets,
            "results": {}
        }
        
        for model_key in models:
            summary["results"][model_key] = {}
            
            for dataset_path in datasets:
                current += 1
                dataset_name = os.path.basename(dataset_path).replace('.csv', '')
                
                print(f"\n{'='*70}")
                print(f"📊 PROGRESO: {current}/{total_combinations}")
                print(f"{'='*70}")
                
                start_time = datetime.now()
                
                try:
                    self.train_model_on_dataset(model_key, dataset_path)
                    status = "success"
                    error = None
                except Exception as e:
                    status = "failed"
                    error = str(e)
                
                end_time = datetime.now()
                duration = (end_time - start_time).total_seconds()
                
                summary["results"][model_key][dataset_name] = {
                    "status": status,
                    "duration_seconds": duration,
                    "error": error,
                    "timestamp": end_time.isoformat()
                }
                
                # Guardar resumen actualizado
                with open("./training_summary.json", 'w') as f:
                    json.dump(summary, f, indent=2)
                
                print(f"\n⏱️ Tiempo: {duration/3600:.2f} horas")
                
                # Pausa entre entrenamientos
                print("\n⏸️ Pausa de 30 segundos...")
                import time
                time.sleep(30)
        
        summary["end_time"] = datetime.now().isoformat()
        
        # Guardar resumen final
        with open("./training_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n{'='*70}")
        print(f"✅ ENTRENAMIENTO COMPLETO FINALIZADO")
        print(f"📊 Resumen guardado en: ./training_summary.json")
        print(f"{'='*70}\n")