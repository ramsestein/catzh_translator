#!/usr/bin/env python3
"""
Script para pre-descargar todos los modelos antes del entrenamiento
"""

from download_utils import pre_download_all_models
from model_configs import MODEL_CONFIGS
import argparse

def main():
    parser = argparse.ArgumentParser(description="Pre-descargar modelos")
    parser.add_argument("--models", nargs="+", help="Modelos específicos a descargar")
    args = parser.parse_args()
    
    if args.models:
        print(f"📥 Descargando modelos específicos: {args.models}")
        for model_key in args.models:
            if model_key in MODEL_CONFIGS:
                config = MODEL_CONFIGS[model_key]
                from download_utils import download_with_progress
                download_with_progress(config.model_id)
            else:
                print(f"❌ Modelo no reconocido: {model_key}")
    else:
        pre_download_all_models()

if __name__ == "__main__":
    main()