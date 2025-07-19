#!/bin/bash

# Script mejorado para entrenar modelos

echo "================================================"
echo "🚀 SISTEMA DE ENTRENAMIENTO UNIVERSAL"
echo "📅 Fecha: $(date)"
echo "================================================"

# Función para mostrar uso
show_usage() {
    echo "Uso: $0 [opciones]"
    echo ""
    echo "Opciones:"
    echo "  -a, --all              Entrenar todos los modelos con todos los datasets (defecto)"
    echo "  -m, --models          Modelos específicos (ej: -m mt5-large nllb-200)"
    echo "  -d, --datasets        Datasets específicos (ej: -d dataset_limpio.csv)"
    echo "  -s, --single          Entrenar solo una combinación (prueba rápida)"
    echo "  -h, --help            Mostrar esta ayuda"
    echo ""
    echo "Modelos disponibles:"
    echo "  mt5-large, mt5-xl, nllb-200, madlad-400, m2m-100, llama-3.2"
    echo ""
    echo "Ejemplos:"
    echo "  $0                    # Entrenar todo"
    echo "  $0 -s                 # Prueba rápida con 1 modelo"
    echo "  $0 -m mt5-large       # Solo mT5-Large con todos los datasets"
    echo "  $0 -m mt5-large nllb-200 -d dataset_limpio.csv  # Combinación específica"
}

# Parsear argumentos
MODELS=""
DATASETS=""
SINGLE_MODE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -a|--all)
            # Modo por defecto
            shift
            ;;
        -m|--models)
            shift
            while [[ $# -gt 0 ]] && [[ ! "$1" =~ ^- ]]; do
                MODELS="$MODELS $1"
                shift
            done
            ;;
        -d|--datasets)
            shift
            while [[ $# -gt 0 ]] && [[ ! "$1" =~ ^- ]]; do
                DATASETS="$DATASETS $1"
                shift
            done
            ;;
        -s|--single)
            SINGLE_MODE="--single"
            shift
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        *)
            echo "Opción desconocida: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Construir comando
CMD="python train_all.py"

if [ -n "$SINGLE_MODE" ]; then
    CMD="$CMD --single"
fi

if [ -n "$MODELS" ]; then
    CMD="$CMD --models $MODELS"
fi

if [ -n "$DATASETS" ]; then
    CMD="$CMD --datasets $DATASETS"
fi

# Crear directorios
mkdir -p training_outputs
mkdir -p logs

# Mostrar configuración
echo ""
echo "📋 Configuración:"
echo "  Comando: $CMD"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'No detectada')"
echo ""

# Confirmar
read -p "¿Iniciar entrenamiento? (s/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Ss]$ ]]; then
    echo "Cancelado."
    exit 1
fi

# Ejecutar con logging
LOG_FILE="logs/training_$(date +%Y%m%d_%H%M%S).log"
echo ""
echo "📄 Log guardado en: $LOG_FILE"
echo "💡 Puedes monitorear con: tail -f $LOG_FILE"
echo ""
echo "🏃 Iniciando entrenamiento..."
echo ""

# Ejecutar
$CMD 2>&1 | tee "$LOG_FILE"

# Mostrar resumen final
echo ""
echo "================================================"
echo "✅ PROCESO COMPLETADO"
echo "📊 Resumen: ./training_summary.json"
echo "📁 Modelos: ./training_outputs/"
echo "📄 Log: $LOG_FILE"
echo "⏰ Finalizado: $(date)"
echo "================================================"