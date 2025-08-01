# Sistema Universal de Entrenamiento para Traducción Catalán-Chino

## 📋 Descripción

Sistema modular y escalable para entrenar múltiples modelos de traducción automática entre catalán y chino. Diseñado para ejecutarse en estaciones de trabajo con GPUs potentes (A600, A100, etc.) y gestionar automáticamente el entrenamiento de múltiples combinaciones modelo-dataset, con recuperación automática y monitorización avanzada.

### 🌟 Características Principales

- **🤖 6 Modelos Preconfigurados:** mT5 (Large/XL), NLLB-200, MADLAD-400, M2M-100, Llama 3.2
- **📊 Gestión Multi-Dataset:** Entrenamiento automatizado con múltiples datasets
- **💾 Recuperación Automática:** Continúa desde checkpoints tras interrupciones
- **🔧 Configuración Centralizada:** Fácil añadir nuevos modelos en `model_configs.py`
- **📈 Monitoreo en Tiempo Real:** Logs detallados y métricas, resumen automático de experimentos
- **🚀 Optimizado para GPUs Modernas:** Soporte para BF16, TF32, gradient checkpointing, LoRA

### ⚠️ Aviso Importante

**La primera vez que se ejecuta cada modelo, el sistema descargará los pesos (400MB a 3.7GB). Esta descarga puede tardar minutos y sólo ocurre una vez por modelo (caché local).**

---

## 🛠️ Requisitos

### Hardware

- **GPU:** NVIDIA A600/A100 (recomendado) o RTX 4060+ (mínimo)
- **VRAM:**
  - Mínimo: 8GB (modelos pequeños: M2M-100, NLLB-200)
  - Recomendado: 24GB+
  - Óptimo: 48GB+ (A600/A100)
- **RAM:** 32GB+
- **Almacenamiento:** 500GB+ SSD
- **Internet:** Necesario para descargar modelos la primera vez

### Software

- Python 3.8+
- CUDA 11.8+
- Ubuntu 20.04+ o WSL2

---

## 📦 Instalación

1. **Clonar el repositorio:**

   ```bash
   git clone https://github.com/ramsestein/catalan-chinese-translator.git
   cd catalan-chinese-translator
   ```

2. **Crear entorno virtual:**

   ```bash
   python -m venv venv
   source venv/bin/activate  # En Windows: venv\Scripts\activate
   ```

3. **Instalar dependencias:**

   ```bash
   pip install -r requeriments.txt
   ```

4. **Verificar instalación:**

   ```bash
   python -c "import torch; print(f'PyTorch: {torch.__version__}')"
   python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
   python -c "import torch; print(f'CUDA disponible: {torch.cuda.is_available()}')"
   ```

---

## 📁 Estructura del Proyecto

```
catalan-chinese-translator/
├── train_all.py              # Script principal de entrenamiento
├── universal_trainer.py      # Lógica principal del entrenamiento universal
├── model_configs.py          # Configuración de modelos (añade aquí nuevos modelos)
├── dataset_utils.py          # Utilidades y loader de datasets
├── trainer_base.py           # Base para entrenadores y métricas
├── checkpoint_manager.py     # Gestión de checkpoints y recuperación
├── run_training.sh           # Script bash para lanzar experimentos fácilmente
├── requeriments.txt          # Dependencias del proyecto
├── README.md                 # Este archivo
│
├── datasets/                 # Datasets de entrenamiento (en repositorio como .zip)
│   ├── dataset_optimizado.csv    # 2.5M pares de frases, curado
│   └── dataset_limpio.csv        # 5M pares de frases, filtrado
│
├── training_outputs/         # Salida de modelos entrenados (se crea automáticamente)
│   └── {modelo}_{dataset}/
│       ├── checkpoint-*/
│       ├── modelo_final/
│       └── logs/
│
└── logs/                     # Logs detallados de entrenamiento
```

---

## 📊 Datasets

Se encuentran en la carpeta datasets en formato .zip, deben descomprimirse para su uso

### Datasets Disponibles

| Dataset                 | Tamaño | Descripción                    | Calidad |
| ----------------------- | ------ | ------------------------------ | ------- |
| dataset\_optimizado.csv | 2.5M   | Dataset curado de alta calidad | ⭐⭐⭐⭐⭐   |
| dataset\_limpio.csv     | 5M     | Dataset limpio y filtrado      | ⭐⭐⭐⭐    |

#### Formato de los Datasets

CSV con columnas `catalan` y `chino`:

```csv
catalan,chino
"Bon dia, com estàs?","早上好，你好吗？"
"M'agrada molt aprendre idiomes","我很喜欢学习语言"
"Catalunya és un país meravellós","加泰罗尼亚是一个美妙的国家"
```

#### Preparar los Datasets

```bash
# Crear directorio datasets si no existe
mkdir -p datasets

# Mover tus archivos CSV al directorio datasets
mv dataset_optimizado.csv datasets/
mv dataset_limpio.csv datasets/
```

---

## ⚙️ Configuración

### Modelos Disponibles (editable en `model_configs.py`)

| Modelo     | ID         | Parámetros | VRAM Min | Batch Size | Descarga (aprox.) |
| ---------- | ---------- | ---------- | -------- | ---------- | ----------------- |
| M2M-100    | m2m-100    | 418M       | 8GB      | 16         | 5 min             |
| NLLB-200   | nllb-200   | 600M       | 8GB      | 16         | 7 min             |
| mT5-Large  | mt5-large  | 1.2B       | 16GB     | 4-8        | 10 min            |
| MADLAD-400 | madlad-400 | 3B         | 24GB     | 4-8        | 20 min            |
| mT5-XL     | mt5-xl     | 3.7B       | 40GB     | 2-4        | 25 min            |
| Llama 3.2  | llama-3.2  | 3B         | 24GB     | 2-4        | 20 min            |

**Nota:** Los tiempos de descarga dependen de la conexión y sólo ocurren la primera vez.

#### Agregar un Nuevo Modelo

Edita `model_configs.py` siguiendo la plantilla existente:

```python
MODEL_CONFIGS = {
    "nuevo-modelo": ModelConfig(
        name="Mi Nuevo Modelo",
        model_id="organizacion/modelo-id",
        model_type="seq2seq",  # o "causal"
        tokenizer_class="AutoTokenizer",
        model_class="AutoModelForSeq2SeqLM",
        batch_size=8,
        gradient_accumulation=4,
        learning_rate=3e-5,
        max_length=256,
        torch_dtype="float32",
        device_map="cuda"
    ),
    # ...otros modelos
}
```

---

## 🚀 Uso

### Usar el script Bash (`run_training.sh`)

**Opciones principales:**

- `-a`, `--all`: Entrenar todos los modelos con todos los datasets (por defecto)
- `-m`, `--models`: Entrenar modelos específicos (`-m mt5-large nllb-200`)
- `-d`, `--datasets`: Usar datasets específicos (`-d dataset_limpio.csv`)
- `-s`, `--single`: Prueba rápida (primer modelo + primer dataset)
- `-h`, `--help`: Ayuda

#### Ejemplos de uso

```bash
# Prueba rápida con M2M-100 (recomendado para empezar)
./run_training.sh --single

# Entrenar solo M2M-100
./run_training.sh --models m2m-100

# Entrenar varios modelos
./run_training.sh -m mt5-large nllb-200 m2m-100

# Entrenar todos los modelos con dataset optimizado
./run_training.sh --datasets datasets/dataset_optimizado.csv

# Entrenar todos los modelos y todos los datasets
./run_training.sh
```

### Uso directo en Python

```bash
# Modo single (prueba rápida)
python train_all.py --single

# Modelos específicos
python train_all.py --models m2m-100 nllb-200

# Saltar verificación de entorno
python train_all.py --skip-verification
```

### Ejecución en Background

```bash
# Con nohup
nohup ./run_training.sh -m mt5-large > training.log 2>&1 &

# Con screen
screen -S training
./run_training.sh --models m2m-100 nllb-200
# Ctrl+A, D para detach

# Con tmux
tmux new -s training
./run_training.sh
# Ctrl+B, D para detach
```

---

## ⏱️ Qué Esperar Durante el Entrenamiento

**Primera ejecución (con descarga):**

```
🚀 ENTRENANDO: M2M-100
📊 Dataset: dataset_optimizado
📁 Directorio: ./training_outputs/facebook_m2m100_418M_dataset_optimizado

📥 Cargando tokenizer M2M100Tokenizer...
📥 Cargando modelo M2M-100...
⚠️  Primera vez: Descargando ~418MB - puede tardar varios minutos...
...
✅ Modelo cargado
📂 Cargando datos desde: ./datasets/dataset_optimizado.csv
✅ Datos limpios: 2449584/2466428 (99.3%)
📈 División: Train=2204625, Val=122479, Test=122480
...
🏃 Iniciando entrenamiento...
⏳ Preparando modelo y optimizador (puede tardar 1-2 minutos)...
🆕 Entrenando desde cero
```

**Ejecuciones posteriores** serán mucho más rápidas (modelo ya descargado y compilado).

---

## ⏱️ Tiempos Estimados de Entrenamiento

En NVIDIA A600 (48GB VRAM):

| Dataset             | Modelo     | 1 Época | 3 Épocas | Total Estimado |
| ------------------- | ---------- | ------- | -------- | -------------- |
| dataset\_optimizado | M2M-100    | 2-3 h   | 6-9 h    | 7-10 h         |
|                     | NLLB-200   | 3-4 h   | 9-12 h   | 10-14 h        |
|                     | mT5-Large  | 5-7 h   | 15-21 h  | 18-24 h        |
|                     | MADLAD-400 | 8-10 h  | 24-30 h  | 26-32 h        |
|                     | mT5-XL     | 12-15 h | 36-45 h  | 40-48 h        |
|                     | Llama 3.2  | 10-12 h | 30-36 h  | 32-38 h        |
| dataset\_limpio     | M2M-100    | 4-6 h   | 12-18 h  | 14-20 h        |
| ...                 | ...        | ...     | ...      | ...            |

**Total estimado (entrenar todo): \~18-23 días**

---

## 📊 Monitoreo

- **Último log:**
  ```bash
  tail -f logs/training_*.log
  ```
- **Ver métricas:**
  ```bash
  grep "loss" logs/training_*.log | tail -20
  ```
- **Uso de GPU:**
  ```bash
  watch -n 1 nvidia-smi
  nvidia-smi dmon -s pucvmet
  ```
- **Ver checkpoints guardados:**
  ```bash
  find training_outputs -name "checkpoint-*" -type d
  cat training_outputs/*/training_info.json
  ```

---

## 🔧 Solución de Problemas

- **Entrenamiento lento al inicio:**
  - Normal: primera vez descarga pesos, compila modelo y optimizador.
- **Error: CUDA out of memory:**
  - Reduce batch\_size en `model_configs.py`.
  - Aumenta gradient\_accumulation.
  - Reduce max\_length.
  - Cierra otras aplicaciones que usen GPU.
- **Warnings tokenizer:**
  - Warnings sobre as\_target\_tokenizer y max\_length son normales.
- **Entrenamiento interrumpido:**
  - El sistema recupera automáticamente desde el último checkpoint.

---

## 📈 Resultados Esperados

Durante el entrenamiento se monitoriza `loss` y otras métricas. El loss debe bajar de \~5-6 a \~1-2.

Estructura típica de salida:

```
training_outputs/
├── facebook_m2m100_418M_dataset_optimizado/
│   ├── checkpoint-500/
│   ├── checkpoint-1000/
│   ├── modelo_final/
│   │   ├── config.json
│   │   ├── pytorch_model.bin
│   │   ├── tokenizer_config.json
│   │   └── sentencepiece.bpe.model
│   ├── training_info.json
│   └── test_metrics.json
└── ...
```

---

## 📋 Archivo requeriments.txt

```
torch>=2.0.0
transformers>=4.35.0
accelerate>=0.24.0
sentencepiece>=0.1.99
protobuf>=3.20.0
pandas>=2.0.0
scikit-learn>=1.3.0
numpy>=1.24.0
peft>=0.5.0
tensorboard>=2.13.0
bitsandbytes>=0.41.0
datasets>=2.14.0
```

---

## 🤝 Contribuciones

- **Agregar un nuevo modelo:**
  - Añadir configuración en `model_configs.py`
  - Probar primero con `--single`
  - Verificar uso de memoria antes de escalar
  - Crear PR incluyendo resultados y logs
- **Reportar problemas:**
  - Incluye logs completos, especificaciones del sistema y el comando exacto usado.

---

## 📄 Licencia

Este proyecto está bajo licencia MIT. Ver archivo LICENSE para más detalles.

---
