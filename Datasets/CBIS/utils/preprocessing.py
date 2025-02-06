import os
import cv2
import numpy as np
import pandas as pd

# Diretórios
base_dir = "/home/eflammere/BreastCancerQuanvolution/Datasets/CBIS/"
csv_dir = os.path.join(base_dir, "csv")
image_dir = os.path.join(base_dir, "original_png")
mask_dir = os.path.join(base_dir, "masks")
output_dir = os.path.join(base_dir, "png")

# Criar diretórios de saída
for split in ["train", "test"]:
    for label in ["0", "1"]:  # 0 = Benign, 1 = Malign
        os.makedirs(os.path.join(output_dir, split, label), exist_ok=True)

# Função para somar múltiplas máscaras
def sum_masks(mask_paths):
    masks = []
    for mask_path in mask_paths:
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is not None:
            masks.append(mask)
    
    if not masks:
        return None  # Nenhuma máscara válida encontrada
    
    summed_mask = np.sum(masks, axis=0, dtype=np.uint8)
    _, binary_mask = cv2.threshold(summed_mask, 1, 255, cv2.THRESH_BINARY)
    return binary_mask

# Função para processar imagens
def process_images(csv_path, split):
    df = pd.read_csv(csv_path)
    grouped = df.groupby("full_path")  # Agrupar múltiplas máscaras da mesma imagem
    
    for image_name, group in grouped:
        if pd.isna(image_name):
            print("Nome da imagem ausente, pulando...")
            continue
        
        image_path = os.path.join(image_dir, str(image_name))
        mask_paths = [os.path.join(mask_dir, str(mask)) for mask in group["mask_path"].dropna()]
        
        if not os.path.exists(image_path) or not all(os.path.exists(m) for m in mask_paths):
            print(f"Arquivo ausente: {image_path} ou uma das máscaras")
            continue
        
        # Carregar imagem e somar máscaras
        image = cv2.imread(image_path)
        mask = sum_masks(mask_paths)
        
        if image is None or mask is None:
            print(f"Erro ao carregar: {image_path} ou máscara ausente")
            continue
        
        # Encontrar bounding box da máscara
        x, y, w, h = cv2.boundingRect(mask)
        
        # Garantir mínimo de 64x64
        min_size = 64
        if w < min_size:
            x = max(0, x - (min_size - w) // 2)
            w = min(image.shape[1] - x, min_size)
        if h < min_size:
            y = max(0, y - (min_size - h) // 2)
            h = min(image.shape[0] - y, min_size)
        
        # Recortar e redimensionar a imagem
        cropped_image = image[y:y+h, x:x+w]
        resized_image = cv2.resize(cropped_image, (64, 64), interpolation=cv2.INTER_AREA)
        
        # Definir rótulo (0 = Benign, 1 = Malign)
        label = "0" if group.iloc[0]["pathology"].strip().lower() == "benign" else "1"
        output_path = os.path.join(output_dir, split, label, os.path.basename(image_path))
        
        # Salvar imagem processada
        cv2.imwrite(output_path, resized_image)
        print(f"Imagem salva: {output_path}")

# Processar treino e teste
process_images(os.path.join(csv_dir, "combined_case_description_train.csv"), "train")
process_images(os.path.join(csv_dir, "combined_case_description_test.csv"), "test")
