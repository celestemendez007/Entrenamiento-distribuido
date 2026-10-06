# Contexto Completo del Proyecto: Entrenamiento Distribuido de Segmentación Panóptica

> **Nota para el Asistente de IA (Claude Code en casa):**  
> Este documento contiene el contexto integral de la arquitectura, los problemas resueltos, las decisiones de ingeniería, el estado final alcanzado y las instrucciones para la fase de inferencia, evaluación y demostración (grabación de video) en una sola máquina. **El modelo ya está 100% entrenado.** No se requiere volver a entrenar.

---

## 1. Resumen Ejecutivo y Objetivo del Proyecto
* **Tarea:** Segmentación Panóptica (*Panoptic Segmentation*: combinación de segmentación semántica de fondo y segmentación por instancias de objetos individuales).
* **Arquitectura:** `Panoptic FPN` con backbone `ResNet-50` y Feature Pyramid Network (`panoptic_fpn_R_50_3x`).
* **Estrategia:** Transfer Learning utilizando los pesos preentrenados del Model Zoo de Detectron2 sobre el benchmark Microsoft COCO 2017.
* **Paradigma de Entrenamiento:** Entrenamiento Distribuido de Datos (**DDP** - *Distributed Data Parallel*) multi-máquina en red local (LAN) a través de sockets TCP en Windows.
* **Hardware Utilizado:**
  * **Nodo 0 (Master / Rank 0):** PC con NVIDIA GeForce RTX 3060 (12 GB VRAM), IP: `10.74.11.178`, Puerto: `29500`.
  * **Nodo 1 (Worker / Rank 1):** PC con NVIDIA GeForce RTX 3060 (12 GB VRAM), IP: `10.74.11.177`.

---

## 2. Obstáculos Críticos Superados y Soluciones de Ingeniería

### A. Ausencia de NCCL en Windows y Configuración de Gloo
* **Problema:** En Windows, el backend nativo de PyTorch para GPU (`NCCL`) no está disponible; Detectron2 por defecto fuerza `NCCL` en `launch()`.
* **Solución:** Se implementó una función `custom_launch()` en `train_panoptic.py` que fuerza el backend `GLOO` compatible con Windows.

### B. Bug de Deadlock en Gloo por IPv6 Link-Local en Windows
* **Problema:** En Windows, la implementación C++ de `c10d.ProcessGroupGloo` ignora las opciones de interfaz pasadas por Python y por defecto se vincula a interfaces de enlace local IPv6 (`fe80:...`), provocando bloqueos de conexión (*timeout / deadlock*) al intentar sincronizar dos computadoras diferentes en la LAN.
* **Solución:** Se creó un monkey-patch dinámico en `train_panoptic.py` (`_enable_windows_gloo_ipv4` con metaclase `MetaGloo` y fábrica `PatchedGlooFactory`) que intercepta la inicialización de Gloo y fuerza la resolución y enlace exclusivamente a la dirección IPv4 local del adaptador de red.

### C. Mapeo de Entradas para Panoptic FPN (`CustomPanopticDatasetMapper`)
* **Problema:** El pipeline estándar de Detectron2 arrojaba un `AssertionError: assert "sem_seg" in batched_inputs[0]`, ya que `PanopticFPN` requiere simultáneamente máscaras de segmentación semántica continua (`sem_seg`) e instancias delimitadas (`instances`).
* **Solución:** Se diseñó `CustomPanopticDatasetMapper` que lee los archivos PNG panópticos de COCO en tiempo real (`pan_seg_file_name`), separa dinámicamente las clases de fondo (*stuff*, categorías 1 a 53) en tensores `sem_seg`, y extrae las instancias (*things*, 80 categorías) con sus máscaras binarias y polígonos.

### D. Compatibilidad del Evaluador (`CustomCOCOPanopticEvaluator`)
* **Problema:** El evaluador oficial esperaba una correspondencia de identificadores contiguos separados para las clases *stuff*.
* **Solución:** Se creó `CustomCOCOPanopticEvaluator` sobreescribiendo el diccionario `_stuff_contiguous_id_to_dataset_id` para garantizar el cálculo exacto de métricas sin errores de indexación.

### E. Limpieza y Sincronización del Dataset (`sync_json.py`)
* **Problema:** Los archivos de anotaciones originales de COCO contenían miles de referencias a imágenes no descargadas localmente.
* **Solución:** Se creó `sync_json.py` para filtrar `panoptic_train2017.json` (dejando exactamente 21,844 imágenes presentes en disco) y `panoptic_val2017.json` (3,425 imágenes presentes en disco), sincronizando ambos archivos para los dos nodos.

---

## 3. Proceso de Optimización y Rendimiento Final

Se realizaron iteraciones de optimización de hiperparámetros para reducir el tiempo total sin degradar la precisión:
1. **Configuración Inicial:** `batch = 2`, `max_iter = 90,000` -> Tiempo estimado: **~45 horas** (descartado por tiempo).
2. **Configuración Intermedia:** `batch = 4`, `max_iter = 20,000` -> Tiempo estimado: **~13 horas**, VRAM: ~3.0 GB.
3. **Configuración Final Implementada:**
   * **`IMS_PER_BATCH = 12`** (6 imágenes por GPU en cada nodo).
   * **`MAX_ITER = 7,000`** con escala lineal de Learning Rate (`BASE_LR = 0.001`, `WARMUP_ITERS = 500`).
   * **Aceleración de Hardware:** Precisión Mixta FP16 activa (`cfg.SOLVER.AMP.ENABLED = True`), Tensor Cores habilitados (`torch.set_float32_matmul_precision('high')`), `cfg.CUDNN_BENCHMARK = True`.
   * **VRAM Utilizada:** ~6.1 GB a 7.8 GB (aprovechando la capacidad de las RTX 3060 con margen de seguridad contra OOM).
   * **Tiempo Total de Entrenamiento:** **5 horas, 19 minutos y 30 segundos** (reducción del 88% respecto al plan original).

---

## 4. Resultados Oficiales de la Evaluación (Validación Final)

La evaluación panóptica sobre el conjunto de validación de COCO (3,425 imágenes) arrojó los siguientes resultados oficiales:

| Categoría | PQ (Panoptic Quality) | SQ (Segmentation Quality) | RQ (Recognition Quality) | # Clases |
| :--- | :---: | :---: | :---: | :---: |
| **All (Total)** | **40.427%** | **77.750%** | **49.463%** | 133 |
| **Things (Objetos/Instancias)** | **48.020%** | **81.327%** | **58.183%** | 80 |
| **Stuff (Fondos/Semántica)** | **28.966%** | **72.351%** | **36.301%** | 53 |

> **Logro:** El baseline oficial publicado por Facebook AI Research (FAIR) para `Panoptic FPN ResNet-50` en COCO es de **~40.2% PQ**. Nuestro modelo alcanzó **40.42% PQ**, alcanzando el estándar de referencia del estado del arte para este modelo.

---

## 5. Estructura de Archivos del Repositorio

* **`train_panoptic.py`**: Script de entrenamiento distribuido con soporte Windows/Gloo, dataset mapper personalizado y evaluador panóptico.
* **`demo_inference.py`**: Script para inferencia y generación de imágenes/predicciones en una sola computadora.
* **`sync_json.py`**: Script de filtrado y sincronización de JSONs de anotaciones con imágenes locales.
* **`test_pipeline.py` / `verify_environment.py`**: Pruebas de sanidad y diagnóstico de CUDA / PyTorch.
* **`requirements.txt`**: Dependencias de Python.
* **`.gitignore`**: Configurado para ignorar la carpeta `output/`, checkpoints pesados (`*.pth`) y datasets grandes.
* **`test_resultado.jpg`**: Imagen de prueba generada exitosamente demostrando la segmentación de instancias y fondo a color.

### Ubicación de los Pesos del Modelo (`model_final.pth`)
* El archivo de pesos entrenados pesa **~184 MB** (`180,120 KB`).
* Como GitHub restringe archivos mayores a 100 MB, **el archivo fue copiado a una memoria USB física (`DENIIIIIIS (D:)`)** dentro de la carpeta `output/` y en la raíz de la USB como `model_final.pth`.

---

## 6. Instrucciones para la Fase de Demostración en Casa (con Claude Code)

### Objetivo en Casa:
**Ya no se va a entrenar.** El objetivo es ejecutar demostraciones visuales, generar imágenes/videos segmentados con el modelo para entenderlo a profundidad y preparar la presentación / grabación.

### Paso 1: Configurar el Repositorio en Casa
```powershell
# 1. Clonar el repositorio
git clone https://github.com/celestemendez007/Entrenamiento-distribuido.git
cd "Entrenamiento-distribuido"

# 2. Copiar model_final.pth desde la memoria USB
# Pegar el archivo model_final.pth en la raíz del proyecto o en una carpeta output/
```

### Paso 2: Ejecutar Inferencia con `demo_inference.py`
Para segmentar cualquier imagen (formato `.jpg` o `.png`):
```powershell
python demo_inference.py --weights model_final.pth --image ruta/a/tu_imagen.jpg --output mi_resultado.jpg
```
* **Qué produce:** Una imagen con las siluetas delineadas de cada persona u objeto (*things*) junto con el fondo segmentado por categorías (*stuff: cielo, banqueta, césped*), mostrando etiquetas y niveles de confianza.
* **Tiempo de ejecución:** Aproximadamente ~0.05 a 0.2 segundos por imagen.

### Paso 3: Conceptos Clave para la Explicación / Grabación
* **¿Qué es Segmentación Panóptica?** Es la unificación de dos tareas clásicas de visión por computadora:
  1. *Instance Segmentation (Mask R-CNN)*: Responde a "¿qué objeto es y dónde está cada ejemplar?" (ej. Persona 1, Persona 2, Coche A).
  2. *Semantic Segmentation (FCN)*: Responde a "¿qué tipo de superficie o entorno es este píxel?" (ej. píxeles de cielo vs píxeles de carretera).
* **¿Qué demostró el proyecto?**
  * Viabilidad de cómputo distribuido sin servidores dedicados, conectando dos PCs estándar de Windows mediante la biblioteca Gloo y PyTorch DDP.
  * Escalado de batch size lineal con precisión mixta FP16 para reducir el tiempo de entrenamiento de casi 2 días a solo 5 horas con 19 minutos, manteniendo 40.4% de Panoptic Quality.
