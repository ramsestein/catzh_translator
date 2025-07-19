import os
import glob
import json
import shutil
from datetime import datetime
from transformers import TrainerCallback

class CheckpointManager:
    """Gestor de checkpoints para entrenamiento"""
    
    def __init__(self, base_dir="./checkpoints"):
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)
    
    def get_model_dir(self, model_name, dataset_name):
        """Obtiene directorio para modelo y dataset específicos"""
        safe_model_name = model_name.replace("/", "_")
        safe_dataset_name = os.path.splitext(dataset_name)[0]
        return os.path.join(self.base_dir, f"{safe_model_name}_{safe_dataset_name}")
    
    def find_latest_checkpoint(self, model_dir):
        """Busca el checkpoint más reciente"""
        checkpoints = glob.glob(f"{model_dir}/checkpoint-*")
        if not checkpoints:
            return None
        
        checkpoints.sort(key=lambda x: int(x.split("-")[-1]))
        return checkpoints[-1]
    
    def save_training_info(self, model_dir, info):
        """Guarda información del entrenamiento"""
        info['timestamp'] = datetime.now().isoformat()
        info_path = os.path.join(model_dir, "training_info.json")
        
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=2)
    
    def save_final_model(self, trainer, tokenizer, model_dir):
        """Guarda el modelo final"""
        final_dir = os.path.join(model_dir, "modelo_final")
        trainer.save_model(final_dir)
        tokenizer.save_pretrained(final_dir)
        print(f"✅ Modelo guardado en: {final_dir}")

class AutoSaveCallback(TrainerCallback):
    """Callback para auto-guardado"""
    
    def __init__(self, checkpoint_manager, model_dir):
        self.checkpoint_manager = checkpoint_manager
        self.model_dir = model_dir
        
    def on_save(self, args, state, control, **kwargs):
        """Ejecuta acciones adicionales al guardar"""
        info = {
            'current_step': state.global_step,
            'current_epoch': state.epoch,
            'best_metric': state.best_metric,
            'best_model_checkpoint': state.best_model_checkpoint
        }
        self.checkpoint_manager.save_training_info(self.model_dir, info)