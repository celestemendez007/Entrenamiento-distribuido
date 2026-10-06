import os
import cv2
import argparse
import torch
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog
from detectron2.model_zoo import model_zoo

def run_demo(image_path, weights_path, output_path="resultado_panoptic.jpg"):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-PanopticSegmentation/panoptic_fpn_R_50_3x.yaml"))
    
    # Cargar pesos entrenados
    cfg.MODEL.WEIGHTS = weights_path
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Umbral de confianza
    
    # Usar GPU si está disponible, sino CPU
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Ejecutando inferencia en: {cfg.MODEL.DEVICE.upper()}")
    
    # Crear predictor
    predictor = DefaultPredictor(cfg)
    
    # Cargar imagen
    if not os.path.exists(image_path):
        print(f"[ERROR] No se encontró la imagen: {image_path}")
        return
        
    im = cv2.imread(image_path)
    print(f"[INFO] Procesando imagen: {image_path} ({im.shape[1]}x{im.shape[0]})...")
    
    # Inferencia
    panoptic_seg, segments_info = predictor(im)["panoptic_seg"]
    
    # Visualización
    coco_metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0] if len(cfg.DATASETS.TRAIN) else "coco_2017_val_panoptic")
    v = Visualizer(im[:, :, ::-1], coco_metadata, scale=1.2, instance_mode=ColorMode.IMAGE_BW)
    out = v.draw_panoptic_seg_predictions(panoptic_seg.to("cpu"), segments_info)
    
    # Guardar resultado
    cv2.imwrite(output_path, out.get_image()[:, :, ::-1])
    print(f"[ÉXITO] Imagen resultante guardada en: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Demo de Inferencia Panoptic Segmentation")
    parser.add_argument("--image", type=str, default="datasets/coco/val2017/000000000139.jpg", help="Ruta de la imagen a probar")
    parser.add_argument("--weights", type=str, default="output/model_final.pth", help="Ruta de los pesos (.pth)")
    parser.add_argument("--output", type=str, default="resultado_panoptic.jpg", help="Ruta donde guardar la imagen resultante")
    args = parser.parse_args()
    
    run_demo(args.image, args.weights, args.output)
