import json
import os
import shutil

splits = ["train", "val"]

for split in splits:
    print("=" * 60)
    print(f"PROCESANDO SPLIT: {split.upper()}2017")
    print("=" * 60)
    
    img_dir = os.path.join("datasets", "coco", f"{split}2017")
    mask_dir = os.path.join("datasets", "coco", f"panoptic_{split}2017")
    json_path = os.path.join("datasets", "coco", "annotations", f"panoptic_{split}2017.json")
    
    if not os.path.exists(json_path):
        print(f"AVISO: No se encontró el JSON: {json_path}")
        continue

    if not os.path.exists(img_dir) or len(os.listdir(img_dir)) == 0:
        print(f"ERROR: La carpeta de imágenes '{img_dir}' está vacía o no existe.")
        continue

    if not os.path.exists(mask_dir) or len(os.listdir(mask_dir)) == 0:
        print(f"ERROR: La carpeta de máscaras '{mask_dir}' está vacía o no existe.")
        continue

    jpg_files = set(os.listdir(img_dir))
    png_bases = set(os.path.splitext(f)[0] for f in os.listdir(mask_dir) if f.endswith(".png"))
    
    print(f"Imágenes JPG en '{img_dir}': {len(jpg_files)}")
    print(f"Máscaras PNG en '{mask_dir}': {len(png_bases)}")

    print("Cargando JSON (esto puede tomar unos segundos)...")
    with open(json_path, "r") as f:
        data = json.load(f)

    original_images_count = len(data.get("images", []))
    print(f"Imágenes originales en JSON: {original_images_count}")

    # Filtrar imágenes que tienen TANTO el .jpg como la máscara .png
    imagenes_filtradas = []
    ids_validos = set()

    for img in data.get("images", []):
        fname = img["file_name"]
        base_name = os.path.splitext(fname)[0]
        if fname in jpg_files and base_name in png_bases:
            imagenes_filtradas.append(img)
            ids_validos.add(img["id"])

    print(f"Imágenes completas (JPG + PNG) coincidentes: {len(imagenes_filtradas)}")

    if len(imagenes_filtradas) == 0:
        print("ERROR CRÍTICO: 0 imágenes coincidieron. Se aborta para no corromper el JSON.")
        continue

    data["images"] = imagenes_filtradas

    # Filtrar anotaciones correspondientes
    original_ann_count = len(data.get("annotations", []))
    anotaciones_filtradas = [ann for ann in data.get("annotations", []) if ann["image_id"] in ids_validos]
    print(f"Anotaciones originales: {original_ann_count} -> Filtradas: {len(anotaciones_filtradas)}")
    data["annotations"] = anotaciones_filtradas

    # Backup de seguridad
    backup_path = json_path + ".bak"
    if not os.path.exists(backup_path):
        shutil.copyfile(json_path, backup_path)
        print(f"Backup creado en: {backup_path}")

    # Guardar JSON actualizado
    print(f"Guardando nuevo JSON en: {json_path} ...")
    with open(json_path, "w") as f:
        json.dump(data, f)

    print(f"¡{split.upper()} sincronizado exitosamente!\n")

print("=" * 60)
print("¡TODOS LOS SPLITS SINCRONIZADOS Y LISTOS PARA ENTRENAMIENTO!")
print("=" * 60)
