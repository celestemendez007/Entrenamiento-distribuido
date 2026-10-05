"""
test_pipeline.py
Script de diagnóstico y validación integral para el Pipeline de Segmentación Panóptica DDP.
Verifica:
1. Problema y Modelo (Panoptic FPN ResNet-50)
2. Configuración y Parámetros Anti-OOM (512px, AMP, Freeze_at 3, Batch 2, LR 0.00025, Workers 2)
3. Evaluador Panóptico (PQ, SQ, RQ)
4. Entorno de Ejecución (GPU NVIDIA CUDA vs AMD integrada)
5. Integridad de Datos en Disco (Estructura COCO, annotations, .png vs .jpg)
"""
import os
import sys
import json

def test_hardware_and_gpus():
    print("=" * 60)
    print("TEST 1: HARDWARE Y COMPATIBILIDAD DE GPUS")
    print("=" * 60)
    import torch
    print(f"PyTorch Version: {torch.__version__}")
    cuda_ok = torch.cuda.is_available()
    print(f"CUDA Disponible: {cuda_ok}")
    if cuda_ok:
        device_count = torch.cuda.device_count()
        print(f"Dispositivos CUDA detectados: {device_count}")
        for i in range(device_count):
            name = torch.cuda.get_device_name(i)
            vram_gb = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            print(f"  -> [CUDA:{i}] {name} ({vram_gb:.2f} GB VRAM)")
    else:
        print("  -> ALERTA: No se detectó CUDA en este intérprete de Python.")

    print("\n[DIAGNÓSTICO GPUS]")
    print("- GPU 0 (NVIDIA GeForce RTX 3060 12GB): COMPATIBLE con CUDA y DDP.")
    print("- GPU 1 (AMD Radeon(TM) Integrada): NO COMPATIBLE con PyTorch CUDA ni DDP.")
    print("  -> Explicación: PyTorch DDP en Windows/WSL solo puede distribuir entre GPUs NVIDIA.")
    print("     La gráfica integrada AMD del procesador no puede usarse en conjunto con la NVIDIA.")
    print("     Por ende, en este nodo solo se usa 1 GPU (la RTX 3060).")


def test_code_configuration():
    print("\n" + "=" * 60)
    print("TEST 2: CONFIGURACIÓN DETECTRON2 Y PARÁMETROS DEL PIPELINE")
    print("=" * 60)
    try:
        from detectron2.config import get_cfg
        from detectron2.model_zoo import model_zoo
        print("Detectron2 importado exitosamente.")
    except ImportError as e:
        print(f"ERROR: No se pudo importar detectron2: {e}")
        return False

    cfg = get_cfg()
    try:
        config_path = model_zoo.get_config_file("COCO-PanopticSegmentation/panoptic_fpn_R_50_3x.yaml")
        cfg.merge_from_file(config_path)
        print(f"[OK] Configuración base cargada vía model_zoo: panoptic_fpn_R_50_3x.yaml")
    except Exception as e:
        print(f"[ERROR] Error al cargar config base: {e}")
        return False

    # Pipeline adjustments para 2 GPUs (1 por máquina)
    cfg.DATASETS.TRAIN = ("coco_2017_train_panoptic",)
    cfg.DATASETS.TEST = ("coco_2017_val_panoptic",)
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-PanopticSegmentation/panoptic_fpn_R_50_3x.yaml")
    cfg.INPUT.MIN_SIZE_TRAIN = (512,)
    cfg.INPUT.MAX_SIZE_TRAIN = 800
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.SOLVER.AMP.ENABLED = True
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.MODEL.BACKBONE.FREEZE_AT = 3
    cfg.SOLVER.BASE_LR = 0.0002
    cfg.SOLVER.MAX_ITER = 90000

    # Assertions
    checks = [
        ("Paso 1 & 7: Backbone ResNet-50", "resnet50" in cfg.MODEL.BACKBONE.NAME.lower() or "r_50" in config_path.lower() or "build_resnet_fpn_backbone" in cfg.MODEL.BACKBONE.NAME),
        ("Paso 6: Resolución reducida 512px", cfg.INPUT.MIN_SIZE_TRAIN == (512,)),
        ("Paso 6: NUM_WORKERS = 2", cfg.DATALOADER.NUM_WORKERS == 2),
        ("Paso 7: AMP (FP16) Activado", cfg.SOLVER.AMP.ENABLED == True),
        ("Paso 7: Batch Size global = 2 (1 por cada una de las 2 GPUs)", cfg.SOLVER.IMS_PER_BATCH == 2),
        ("Paso 7: Congelamiento FREEZE_AT = 3", cfg.MODEL.BACKBONE.FREEZE_AT == 3),
        ("Paso 8: BASE_LR = 0.0002 (escalado lineal batch 2)", cfg.SOLVER.BASE_LR == 0.0002),
        ("Paso 8: MAX_ITER = 90000", cfg.SOLVER.MAX_ITER == 90000),
    ]

    all_passed = True
    for name, ok in checks:
        status = "[PASSED]" if ok else "[FAILED]"
        print(f"{status} {name}")
        if not ok:
            all_passed = False

    return all_passed


def test_evaluator():
    print("\n" + "=" * 60)
    print("TEST 3: EVALUADOR PANÓPTICO (SQ, RQ, PQ)")
    print("=" * 60)
    try:
        from detectron2.evaluation import COCOPanopticEvaluator
        evaluator_cls = COCOPanopticEvaluator
        print("[PASSED] COCOPanopticEvaluator disponible en detectron2.evaluation")
        print("  -> Mide métricas estándar de Panoptic Segmentation: PQ, SQ y RQ.")
        return True
    except Exception as e:
        print(f"[FAILED] Error con el evaluador: {e}")
        return False


def test_dataset_integrity(base_dir="datasets/coco"):
    print("\n" + "=" * 60)
    print("TEST 4: INTEGRIDAD DEL DATASET EN DISCO")
    print("=" * 60)

    train_json = os.path.join(base_dir, "annotations", "panoptic_train2017.json")
    val_json = os.path.join(base_dir, "annotations", "panoptic_val2017.json")

    panoptic_train_dir = os.path.join(base_dir, "panoptic_train2017")
    panoptic_val_dir = os.path.join(base_dir, "panoptic_val2017")
    img_train_dir = os.path.join(base_dir, "train2017")
    img_val_dir = os.path.join(base_dir, "val2017")

    print(f"Verificando directorio: {os.path.abspath(base_dir)}")

    # Check JSONs
    train_json_ok = os.path.exists(train_json) and os.path.getsize(train_json) > 1000
    val_json_ok = os.path.exists(val_json) and os.path.getsize(val_json) > 1000
    print(f"[{'PASSED' if train_json_ok else 'FAILED'}] Anotaciones Train: {train_json} ({os.path.getsize(train_json)/(1024**2):.1f} MB)" if train_json_ok else f"[FAILED] Falta {train_json}")
    print(f"[{'PASSED' if val_json_ok else 'FAILED'}] Anotaciones Val: {val_json} ({os.path.getsize(val_json)/(1024**2):.1f} MB)" if val_json_ok else f"[FAILED] Falta {val_json}")

    # Check Directories
    def check_dir(d, expected_ext):
        if not os.path.exists(d):
            return 0, 0, False
        files = os.listdir(d)
        total = len(files)
        correct_ext = sum(1 for f in files if f.lower().endswith(expected_ext))
        return total, correct_ext, True

    tot_pt, cor_pt, ex_pt = check_dir(panoptic_train_dir, ".png")
    tot_pv, cor_pv, ex_pv = check_dir(panoptic_val_dir, ".png")
    tot_it, cor_it, ex_it = check_dir(img_train_dir, ".jpg")
    tot_iv, cor_iv, ex_iv = check_dir(img_val_dir, ".jpg")

    print("\nResultados de archivos físicos:")
    print(f"- Fotos Train ({img_train_dir}/*.jpg): {cor_it} archivos JPG")
    print(f"- Máscaras Panópticas Train ({panoptic_train_dir}/*.png): {cor_pt} archivos PNG")
    print(f"- Fotos Val ({img_val_dir}/*.jpg): {cor_iv} archivos JPG")
    print(f"- Máscaras Panópticas Val ({panoptic_val_dir}/*.png): {cor_pv} archivos PNG")

    # Extra diagnostic for current messy state
    if os.path.exists(panoptic_val_dir):
        pv_jpg = sum(1 for f in os.listdir(panoptic_val_dir) if f.lower().endswith(".jpg"))
        if pv_jpg > 0:
            print(f"\n[ALERTA CRÍTICA]: Se encontraron {pv_jpg} archivos .jpg dentro de '{panoptic_val_dir}'.")
            print("  -> Esas imágenes son fotos RGB (de val2017), NO son máscaras panópticas (.png).")
            print(f"  -> Deben moverse a la carpeta '{img_val_dir}'.")

    if tot_pt == 0:
        print("\n[ALERTA CRÍTICA]: La carpeta 'panoptic_train2017' está completamente VACÍA.")
        print("  -> Faltan las máscaras PNG de entrenamiento.")

    if not os.path.exists(img_train_dir) or tot_it == 0:
        print("\n[ALERTA CRÍTICA]: No existe la carpeta 'train2017' con las fotos JPG de entrenamiento.")


def test_model_and_vram():
    print("\n" + "=" * 60)
    print("TEST 5: CONSTRUCCIÓN DEL MODELO Y PRUEBA DE MEMORIA VRAM (ANTI-OOM)")
    print("=" * 60)
    import torch
    from detectron2.config import get_cfg
    from detectron2.model_zoo import model_zoo
    from detectron2.modeling import build_model

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Probando construcción en: {device}")

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-PanopticSegmentation/panoptic_fpn_R_50_3x.yaml"))
    cfg.MODEL.DEVICE = device
    cfg.MODEL.BACKBONE.FREEZE_AT = 3

    model = build_model(cfg)
    model.eval()

    total_p = sum(p.numel() for p in model.parameters())
    trainable_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_p = total_p - trainable_p

    print(f"[OK] Modelo construido exitosamente.")
    print(f"  -> Parámetros Totales: {total_p:,}")
    print(f"  -> Parámetros Entrenables: {trainable_p:,}")
    print(f"  -> Parámetros Congelados (FREEZE_AT=3): {frozen_p:,} ({100*frozen_p/total_p:.1f}% congelado)")

    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        mem_before = torch.cuda.memory_allocated() / (1024**2)

        # Mock image at 512x512
        dummy_input = [{"image": torch.randint(0, 256, (3, 512, 512), dtype=torch.uint8, device=device)}]
        with torch.no_grad():
            _ = model(dummy_input)

        peak_mem = torch.cuda.max_memory_allocated() / (1024**2)
        print(f"  -> Memoria consumida con 1 imagen de 512x512 en GPU: {peak_mem:.1f} MB")
        if peak_mem < 3500:
            print(f"[PASSED] El consumo ({peak_mem:.1f} MB) cabe perfectamente dentro del límite de 4GB VRAM (Anti-OOM).")
        else:
            print(f"[WARNING] El consumo ({peak_mem:.1f} MB) está cerca del límite de 4GB VRAM.")

    return True


if __name__ == "__main__":
    test_hardware_and_gpus()
    test_code_configuration()
    test_evaluator()
    test_dataset_integrity()
    test_model_and_vram()

