#!/usr/bin/env python3
"""
Script principal para entrenar todos los modelos con todos los datasets
"""

import argparse
import sys
import os
from universal_trainer import UniversalTrainer
from model_configs import MODEL_CONFIGS

def verify_environment():
    """Verifica que el entorno esté configurado correctamente"""
    print("🔍 Verificando entorno...")
    
    # Verificar GPU
    import torch
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"✅ GPU detectada: {gpu_name} ({gpu_memory:.1f}GB)")
    else:
        print("⚠️ No se detectó GPU. El entrenamiento será muy lento.")
        response = input("¿Continuar de todos modos? (s/n): ")
        if response.lower() != 's':
            sys.exit(1)
    
    # Verificar datasets
    datasets = [
        "./datasets/dataset_limpio.csv",
        "./datasets/dataset_optimizado.csv"]
    
    missing = []
    for dataset in datasets:
        if os.path.exists(dataset):
            print(f"✅ {dataset} encontrado")
        else:
            print(f"❌ {dataset} NO encontrado")
            missing.append(dataset)
    
    if missing:
        print("\n❌ Faltan datasets. Por favor, asegúrate de tener todos los archivos.")
        sys.exit(1)
    
    print("\n✅ Entorno verificado correctamente\n")

def main():
    parser = argparse.ArgumentParser(description="Entrenar modelos de traducción CA-ZH")
    
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(MODEL_CONFIGS.keys()),
        default=None,
        help="Modelos específicos a entrenar (por defecto: todos)"
    )
    
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Rutas a datasets específicos (por defecto: los 3 predefinidos)"
    )
    
    parser.add_argument(
        "--single",
        action="store_true",
        help="Entrenar solo una combinación (primer modelo, primer dataset)"
    )
    
    parser.add_argument(
        "--skip-verification",
        action="store_true",
        help="Saltar verificación del entorno"
    )
    
    args = parser.parse_args()
    
    # Verificar entorno
    if not args.skip_verification:
        verify_environment()
    
    # Crear entrenador
    trainer = UniversalTrainer()
    
    # Definir modelos y datasets
    models = args.models
    datasets = args.datasets
    
    if args.single:
        # Modo single: solo primera combinación
        models = models[:1] if models else list(MODEL_CONFIGS.keys())[:1]
        datasets = datasets[:1] if datasets else ["./datasets/dataset_optimizado.csv"]
    
    # Entrenar
    try:
        trainer.train_all_combinations(models=models, datasets=datasets)
    except KeyboardInterrupt:
        print("\n\n⚠️ Entrenamiento interrumpido por usuario")
        print("Los checkpoints se han guardado y puedes continuar más tarde")
    except Exception as e:
        print(f"\n❌ Error fatal: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()