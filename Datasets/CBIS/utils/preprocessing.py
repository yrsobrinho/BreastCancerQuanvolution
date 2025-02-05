import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

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

# Função para processar imagens
def process_images(csv_path, split):
    df = pd.read_csv(csv_path)
    
    for _, row in df.iterrows():
        image_path = os.path.join(image_dir, row["full_path"])
        mask_path = os.path.join(mask_dir, row["mask_path"])
        
        if not os.path.exists(image_path) or not os.path.exists(mask_path):
            print(f"Arquivo ausente: {image_path} ou {mask_path}")
            continue
        
        # Carregar imagem e máscara
        image = cv2.imread(image_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        if image is None or mask is None:
            print(f"Erro ao carregar: {image_path} ou {mask_path}")
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
        
        # Recortar a imagem original
        cropped_image = image[y:y+h, x:x+w]
        
        # Redimensionar para 64x64
        resized_image = cv2.resize(cropped_image, (64, 64), interpolation=cv2.INTER_AREA)
        
        # Definir rótulo (0 = Benign, 1 = Malign)
        label = "0" if row["pathology"].strip().lower() == "benign" else "1"
        output_path = os.path.join(output_dir, split, label, os.path.basename(image_path))
        
        # Salvar imagem processada
        cv2.imwrite(output_path, resized_image)
        print(f"Imagem salva: {output_path}")

# Processar treino e teste
process_images(os.path.join(csv_dir, "combined_case_description_train.csv"), "train")
process_images(os.path.join(csv_dir, "combined_case_description_test.csv"), "test")