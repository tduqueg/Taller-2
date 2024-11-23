# Fraud Detection with PyTorch

Este proyecto implementa modelos de clasificación con PyTorch para detectar fraudes en transacciones de tarjetas de crédito. Se incluyen implementaciones de regresión lineal, regresión logística (binaria y multiclase) y análisis discriminante lineal (LDA). El código está diseñado para ejecutarse en dispositivos con GPU NVIDIA, utilizando PyTorch para aprovechar la aceleración por hardware.

## Características principales

- **Modelos implementados:**
  - Regresión lineal para clasificación.
  - Regresión logística binaria y multiclase.
  - Análisis discriminante lineal (LDA).
- **Entrenamiento optimizado para GPU NVIDIA.**
- Métricas de evaluación detalladas, como AUC-ROC y PR-AUC.
- Uso de `tqdm` para el seguimiento del progreso durante el entrenamiento.

## Requisitos previos

1. **Hardware:**
   - GPU NVIDIA compatible con CUDA.
2. **Software:**
   - Python 3.8 o superior.
   - **PyTorch** instalado con soporte para CUDA.
   - Dependencias listadas en `requirements.txt`:
     - `pandas`
     - `numpy`
     - `torch`
     - `scikit-learn`
     - `tqdm`

## Instalación

1. Clonar el repositorio:
   ```bash
   https://github.com/tduqueg/Taller-2.git
   cd Taller-2
   ```
2. Crear y activar un entorno virtual (opcional):
   ```bash
    python -m venv venv
    source venv/bin/activate  # En Windows: venv\Scripts\activate
   ```
3. Instalar las dependencias:
   ```bash
   pip install -r requirements.txt
   ```
4. Asegurarse de tener PyTorch instalado con soporte para CUDA:
   Instalar según las instrucciones oficiales: https://pytorch.org/get-started/locally/.

## Uso

1. Colocar el archivo de datos `card_transdata.csv` en el directorio principal del proyecto. Este archivo debe contener una columna `fraud` como variable objetivo y el resto como características.

2. Ejecutar el script principal:

```bash
python main.py
```

3. Durante la ejecución, el código:

- Entrena cada modelo con los datos proporcionados.
- Muestra el progreso del entrenamiento.
- Calcula métricas de evaluación, como AUC-ROC y PR-AUC.
- Imprime un reporte de clasificación detallado para cada modelo.

## Resultados esperados

Al final de la ejecución, se mostrará un resumen de métricas como:

- AUC-ROC (Área bajo la curva ROC).
- PR-AUC (Área bajo la curva de precisión-recall).
- Reporte de clasificación con precisión, recall y F1-score.

## Notas importantes

- El script detectará automáticamente si hay una GPU disponible y la usará. Si no hay GPU, se usará la CPU, aunque el entrenamiento será más lento.
- Ajusta el hiperparámetro `batch_size` en `main()` para optimizar el uso de memoria según tu dispositivo.
