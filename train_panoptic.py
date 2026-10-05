import os
import sys
import logging
from datetime import timedelta
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

# Windows PyTorch support: Desactivar libuv para TCPStore (evita RuntimeError por falta de libuv en PyTorch para Windows)
os.environ["USE_LIBUV"] = "0"

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup
from detectron2.evaluation import COCOPanopticEvaluator
from detectron2.utils import comm

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
    
    # 7. Tamaño de lote: 2 total (1 imagen por cada una de las 2 GPUs/PCs)
    # Detectron2 exige que IMS_PER_BATCH sea divisible por el total de GPUs (2 / 2 = 1 por GPU)
    cfg.SOLVER.IMS_PER_BATCH = 2 
    
    # 7. Backbone ligero y congelado (Congelar los primeros 3 bloques ahorra VRAM)
    cfg.MODEL.BACKBONE.FREEZE_AT = 3
    
    # 8. Hyperparameter Tuning: Tasa de aprendizaje ajustada para batch size 2
    cfg.SOLVER.BASE_LR = 0.0002 
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

def _distributed_worker(
    local_rank,
    main_func,
    world_size,
    num_gpus_per_machine,
    machine_rank,
    dist_url,
    args,
    timeout=timedelta(minutes=30),
):
    assert torch.cuda.is_available(), "CUDA no está disponible en este sistema."
    global_rank = machine_rank * num_gpus_per_machine + local_rank
    
    # En Windows, PyTorch oficial solo soporta GLOO (NCCL es exclusivo de Linux)
    backend = "nccl" if dist.is_nccl_available() else "gloo"
    os.environ["USE_LIBUV"] = "0"
    
    try:
        dist.init_process_group(
            backend=backend,
            init_method=dist_url,
            world_size=world_size,
            rank=global_rank,
            timeout=timeout,
        )
    except Exception as e:
        logger = logging.getLogger("detectron2")
        logger.error(f"Error inicializando grupo de procesos con URL: {dist_url}")
        raise e

    comm.synchronize()

    assert num_gpus_per_machine <= torch.cuda.device_count()
    torch.cuda.set_device(local_rank)

    assert comm._LOCAL_PROCESS_GROUP is None
    num_machines = world_size // num_gpus_per_machine
    for i in range(num_machines):
        ranks_on_i = list(range(i * num_gpus_per_machine, (i + 1) * num_gpus_per_machine))
        pg = dist.new_group(ranks_on_i)
        if i == machine_rank:
            comm._LOCAL_PROCESS_GROUP = pg

    main_func(*args)

def custom_launch(
    main_func,
    num_gpus_per_machine,
    num_machines=1,
    machine_rank=0,
    dist_url=None,
    args=(),
    timeout=timedelta(minutes=30),
):
    os.environ["USE_LIBUV"] = "0"
    world_size = num_machines * num_gpus_per_machine
    if world_size > 1:
        if dist_url and dist_url.startswith("tcp://") and "use_libuv" not in dist_url:
            separator = "&" if "?" in dist_url else "?"
            dist_url = f"{dist_url}{separator}use_libuv=0"

        mp.spawn(
            _distributed_worker,
            nprocs=num_gpus_per_machine,
            args=(
                main_func,
                world_size,
                num_gpus_per_machine,
                machine_rank,
                dist_url,
                args,
                timeout,
            ),
            daemon=False,
        )
    else:
        main_func(*args)

if __name__ == "__main__":
    parser = default_argument_parser()
    args = parser.parse_args()
    print("Argumentos de linea de comandos:", args)
    
    custom_launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )

