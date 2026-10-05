import os
import sys
import logging
from datetime import timedelta
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

# Windows PyTorch support: Desactivar libuv para TCPStore (evita RuntimeError por falta de libuv en PyTorch para Windows)
os.environ["USE_LIBUV"] = "0"

import copy
import numpy as np
from PIL import Image

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup
from detectron2.evaluation import COCOPanopticEvaluator
from detectron2.utils import comm
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.data import build_detection_train_loader
from detectron2.structures import Instances, Boxes, BitMasks

class CustomPanopticDatasetMapper:
    """
    Extrae simultáneamente 'instances' (foreground objects) y 'sem_seg' (background stuff)
    a partir de las máscaras panópticas COCO PNG sin requerir archivos adicionales de sem_seg.
    """
    def __init__(self, cfg, is_train=True):
        self.is_train = is_train
        if is_train:
            self.augmentations = [
                T.ResizeShortestEdge(
                    cfg.INPUT.MIN_SIZE_TRAIN,
                    cfg.INPUT.MAX_SIZE_TRAIN,
                    cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING,
                ),
                T.RandomFlip(),
            ]
        else:
            self.augmentations = [
                T.ResizeShortestEdge(
                    cfg.INPUT.MIN_SIZE_TEST,
                    cfg.INPUT.MAX_SIZE_TEST,
                    "choice",
                )
            ]
        self.img_format = cfg.INPUT.FORMAT

    def __call__(self, dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        utils.check_image_size(dataset_dict, image)

        # 1. Decodificar la máscara panóptica PNG (R + G*256 + B*256^2)
        pan_img = np.array(Image.open(dataset_dict["pan_seg_file_name"]), dtype=np.uint32)
        pan_id = pan_img[:, :, 0] + (pan_img[:, :, 1] << 8) + (pan_img[:, :, 2] << 16)

        H, W = pan_id.shape
        sem_seg = np.zeros((H, W), dtype=np.uint8)

        boxes = []
        classes = []
        masks = []

        for seg in dataset_dict.get("segments_info", []):
            seg_id = seg["id"]
            cat_id = seg["category_id"]
            is_thing = seg.get("isthing", False)
            mask = (pan_id == seg_id)
            if not mask.any():
                continue
            if is_thing:
                y_idx, x_idx = np.where(mask)
                box = [float(x_idx.min()), float(y_idx.min()), float(x_idx.max() + 1), float(y_idx.max() + 1)]
                boxes.append(box)
                classes.append(cat_id)
                masks.append(mask.astype(np.uint8))
            else:
                # Mapear categorías de stuff al rango [1, 53] de SemSegFPNHead
                stuff_id = cat_id - 80 + 1
                sem_seg[mask] = stuff_id

        # 2. Aplicar aumentos de imagen y máscara semántica
        aug_input = T.AugInput(image, sem_seg=sem_seg)
        transforms = T.AugmentationList(self.augmentations)(aug_input)
        image = aug_input.image
        sem_seg = aug_input.sem_seg

        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1).astype("float32")))
        dataset_dict["sem_seg"] = torch.as_tensor(sem_seg.astype("int64"), dtype=torch.int64)

        # 3. Aplicar transformaciones a las máscaras y cajas de instancias
        inst = Instances(image.shape[:2])
        if len(masks) > 0:
            transformed_masks = []
            transformed_boxes = []
            transformed_classes = []
            for m, b, c in zip(masks, boxes, classes):
                m_tfm = transforms.apply_segmentation(m) > 0
                if m_tfm.any():
                    transformed_masks.append(m_tfm)
                    transformed_classes.append(c)
                    y_i, x_i = np.where(m_tfm)
                    transformed_boxes.append([float(x_i.min()), float(y_i.min()), float(x_i.max() + 1), float(y_i.max() + 1)])

            if len(transformed_masks) > 0:
                inst.gt_boxes = Boxes(torch.as_tensor(transformed_boxes, dtype=torch.float32))
                inst.gt_classes = torch.as_tensor(transformed_classes, dtype=torch.int64)
                inst.gt_masks = BitMasks(torch.as_tensor(np.stack(transformed_masks), dtype=torch.bool))
            else:
                inst.gt_boxes = Boxes(torch.zeros((0, 4), dtype=torch.float32))
                inst.gt_classes = torch.zeros((0,), dtype=torch.int64)
                inst.gt_masks = BitMasks(torch.zeros((0, image.shape[0], image.shape[1]), dtype=torch.bool))
        else:
            inst.gt_boxes = Boxes(torch.zeros((0, 4), dtype=torch.float32))
            inst.gt_classes = torch.zeros((0,), dtype=torch.int64)
            inst.gt_masks = BitMasks(torch.zeros((0, image.shape[0], image.shape[1]), dtype=torch.bool))

        dataset_dict["instances"] = inst
        return dataset_dict

class CustomCOCOPanopticEvaluator(COCOPanopticEvaluator):
    """
    Evaluador panóptico compatible con PanopticFPN (mapea las 53 clases de stuff al rango COCO estándar).
    """
    def __init__(self, dataset_name, output_dir=None):
        super().__init__(dataset_name, output_dir)
        from detectron2.data.datasets.builtin_meta import _get_builtin_metadata
        m_sep = _get_builtin_metadata("coco_panoptic_separated")
        self._stuff_contiguous_id_to_dataset_id = {
            v: k for k, v in m_sep["stuff_dataset_id_to_contiguous_id"].items()
        }

class PanopticTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        9. Evaluation: Medición utilizando las métricas panópticas (PQ, SQ y RQ).
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return CustomCOCOPanopticEvaluator(dataset_name, output_folder)

    @classmethod
    def build_train_loader(cls, cfg):
        mapper = CustomPanopticDatasetMapper(cfg, is_train=True)
        return build_detection_train_loader(cfg, mapper=mapper)

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
    
    # 6. Reducción de resolución y optimizaciones de hardware
    cfg.INPUT.MIN_SIZE_TRAIN = (512,)
    cfg.INPUT.MAX_SIZE_TRAIN = 800
    cfg.DATALOADER.NUM_WORKERS = 2 

    # Activar optimizaciones Ampere (Tensor Cores y cuDNN Benchmark)
    torch.set_float32_matmul_precision("high")
    cfg.CUDNN_BENCHMARK = True

    # 7. Precisión Mixta (AMP - FP16)
    cfg.SOLVER.AMP.ENABLED = True
    
    # 7. Tamaño de lote: 4 total (2 imágenes por cada una de las 2 GPUs/PCs)
    # Detectron2 exige que IMS_PER_BATCH sea divisible por el total de GPUs (4 / 2 = 2 por GPU)
    cfg.SOLVER.IMS_PER_BATCH = 4 
    
    # 7. Backbone ligero y congelado (Congelar los primeros 3 bloques ahorra VRAM)
    cfg.MODEL.BACKBONE.FREEZE_AT = 3
    
    # 8. Hyperparameter Tuning optimizado: 20,000 iteraciones y LR escalado
    cfg.SOLVER.BASE_LR = 0.0004 
    cfg.SOLVER.MAX_ITER = 20000 
    cfg.SOLVER.STEPS = (14000, 18000)
    cfg.SOLVER.CHECKPOINT_PERIOD = 2500
    
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

def _enable_windows_gloo_ipv4():
    """
    Corrige el bug de PyTorch Gloo en Windows donde dist.init_process_group ignora pg_options
    y por defecto intenta enlazar por IPv6 link-local (fe80:...), bloqueando la comunicación LAN.
    """
    import socket
    import torch.distributed.distributed_c10d as c10d

    _orig_PGGloo = c10d.ProcessGroupGloo

    class MetaGloo(type):
        def __instancecheck__(cls, instance):
            return isinstance(instance, _orig_PGGloo)

    class PatchedGlooFactory(metaclass=MetaGloo):
        def __new__(cls, store, rank, size, options=None, timeout=None):
            if options is None:
                options = _orig_PGGloo._Options()
                if timeout is not None:
                    options._timeout = timeout
                try:
                    local_ip = socket.gethostbyname(socket.gethostname())
                    options._devices = [_orig_PGGloo.create_device(hostname=local_ip)]
                except Exception:
                    pass
            return _orig_PGGloo(store, rank, size, options)

    c10d.ProcessGroupGloo = PatchedGlooFactory

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
    
    if backend == "gloo":
        _enable_windows_gloo_ipv4()

    try:
        print(f"[Rank {global_rank}] Conectando al grupo distribuido ({backend.upper()})...")
        dist.init_process_group(
            backend=backend,
            init_method=dist_url,
            world_size=world_size,
            rank=global_rank,
            timeout=timeout,
        )
        print(f"[Rank {global_rank}] Conectado exitosamente. Sincronizando nodos...")
    except Exception as e:
        logger = logging.getLogger("detectron2")
        logger.error(f"Error inicializando grupo de procesos con URL: {dist_url}")
        raise e

    comm.synchronize()
    print(f"[Rank {global_rank}] Sincronización completada. Iniciando modelo...")

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

