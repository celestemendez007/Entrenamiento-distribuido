import os
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.evaluation import COCOPanopticEvaluator

class PanopticTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        9. Evaluation: Medición utilizando las métricas panópticas (PQ, SQ y RQ).
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOPanopticEvaluator(dataset_name, output_folder)

def setup(args):
    """
    Configura el pipeline con los requerimientos específicos para GPUs de 4GB VRAM.
    """
    cfg = get_cfg()
    
    # Asegurar que Detectron2 encuentre la carpeta datasets local si no está en variables de entorno
    if "DETECTRON2_DATASETS" not in os.environ:
        os.environ["DETECTRON2_DATASETS"] = os.path.abspath("datasets")

    # Cargar config base de ResNet-50 mediante model_zoo (compatible con pip y git)
    from detectron2.model_zoo import model_zoo
    cfg.merge_from_file(model_zoo.get_config_file("COCO-PanopticSegmentation/panoptic_fpn_R_50_3x.yaml"))
    
    cfg.DATASETS.TRAIN = ("coco_2017_train_panoptic",)
    cfg.DATASETS.TEST = ("coco_2017_val_panoptic",)
    
    # 7. Transfer Learning: Pesos iniciales del Model Zoo
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-PanopticSegmentation/panoptic_fpn_R_50_3x.yaml")
    
    # 6. Reducción extrema de resolución para 4GB VRAM
    cfg.INPUT.MIN_SIZE_TRAIN = (512,)
    cfg.INPUT.MAX_SIZE_TRAIN = 800 # Limitar también el lado máximo
    
    # 6. Ajuste conservador de NUM_WORKERS
    cfg.DATALOADER.NUM_WORKERS = 2 
    
    # 7. Precisión Mixta (AMP - FP16)
    cfg.SOLVER.AMP.ENABLED = True
    
    # 7. Tamaño de lote: 3 total (1 imagen por cada una de las 3 GPUs/PCs)
    # Detectron2 exige que IMS_PER_BATCH sea divisible por el total de GPUs (3 / 3 = 1 por GPU)
    cfg.SOLVER.IMS_PER_BATCH = 3 
    
    # 7. Backbone ligero y congelado (Congelar los primeros 3 bloques ahorra VRAM)
    cfg.MODEL.BACKBONE.FREEZE_AT = 3
    
    # 8. Hyperparameter Tuning: Tasa de aprendizaje ajustada para batch size 3
    cfg.SOLVER.BASE_LR = 0.0003 
    cfg.SOLVER.MAX_ITER = 90000 
    
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg

def main(args):
    cfg = setup(args)

    if args.eval_only:
        model = PanopticTrainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = PanopticTrainer.test(cfg, model)
        return res

    trainer = PanopticTrainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()

if __name__ == "__main__":
    parser = default_argument_parser()
    args = parser.parse_args()
    print("Argumentos de linea de comandos:", args)
    
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
