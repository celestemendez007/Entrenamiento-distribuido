import json
import os

# Directorios de referencia (ajustar si se filtra por fotos o por máscaras)
img_dir = os.path.join("datasets", "coco", "train2017")
panoptic_dir = os.path.join("datasets", "coco", "panoptic_train2017")
json_path = os.path.join("datasets", "coco", "annotations", "panoptic_train2017.json")

if not os.path.exists(json_path):
    print(f"ERROR: No se encontró el archivo JSON en: {json_path}")
    exit(1)

# Determinar cuál carpeta existe y tiene archivos físicos para sincronizar
target_dir = None
is_panoptic_mask = False

if os.path.exists(img_dir) and len(os.listdir(img_dir)) > 0:
    target_dir = img_dir
    is_panoptic_mask = False
elif os.path.exists(panoptic_dir) and len(os.listdir(panoptic_dir)) > 0:
    target_dir = panoptic_dir
    is_panoptic_mask = True
else:
    print("ALERTA DE SEGURIDAD: Ni 'train2017' ni 'panoptic_train2017' contienen archivos.")
    print("La sincronización se CANCELA para evitar vaciar y corromper el archivo JSON.")
    exit(0)

print(f"Leyendo archivos físicos desde: {target_dir}")
archivos_fisicos = set(os.listdir(target_dir))
print(f"Archivos encontrados en la carpeta: {len(archivos_fisicos)}")

print("Cargando JSON original (esto puede tomar unos segundos)...")
with open(json_path, 'r') as f:
    data = json.load(f)

print(f"Imágenes originales en JSON: {len(data.get('images', []))}")

# Filtrar solo las imágenes que aún existen en la carpeta
imagenes_filtradas = []
ids_validos = set()

for img in data.get('images', []):
    fname = img['file_name']
    # Si estamos comparando contra máscaras panópticas, reemplazar extensión a .png
    if is_panoptic_mask:
        fname_compare = os.path.splitext(fname)[0] + ".png"
    else:
        fname_compare = fname

    if fname_compare in archivos_fisicos:
        imagenes_filtradas.append(img)
        ids_validos.add(img['id'])

if len(imagenes_filtradas) == 0:
    print("ERROR CRÍTICO: 0 imágenes coincidieron. Se ABORTA el guardado para proteger el archivo JSON.")
    exit(1)

print(f"Imágenes válidas que se guardarán en el nuevo JSON: {len(imagenes_filtradas)}")
data['images'] = imagenes_filtradas

# Filtrar las anotaciones que corresponden a esas imágenes válidas
print(f"Anotaciones originales en JSON: {len(data.get('annotations', []))}")
anotaciones_filtradas = [ann for ann in data.get('annotations', []) if ann['image_id'] in ids_validos]
print(f"Anotaciones válidas que se guardarán: {len(anotaciones_filtradas)}")
data['annotations'] = anotaciones_filtradas

# Crear backup de seguridad antes de sobreescribir
backup_path = json_path + ".bak"
if not os.path.exists(backup_path):
    import shutil
    shutil.copyfile(json_path, backup_path)
    print(f"Copia de seguridad guardada en: {backup_path}")

print("Guardando JSON actualizado...")
with open(json_path, 'w') as f:
    json.dump(data, f)

print("¡Sincronización completada exitosamente! Tu dataset está listo.")
