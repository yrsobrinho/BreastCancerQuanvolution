import os
import pandas as pd
from PIL import Image
import cv2
import numpy as np

# Caminhos das pastas e do arquivo CSV
base_images_dir = '/home/eflammere/QuantumIC/Datasets/VinDr-Mammo/png_from_dicom'
output_base_dir = '/home/eflammere/QuantumIC/Datasets/VinDr-Mammo/png'
annotations_path = '/home/eflammere/QuantumIC/Datasets/VinDr-Mammo/csv/finding_annotations.csv'

# Carrega o CSV com as coordenadas de corte
annotations = pd.read_csv(annotations_path)

# Função para mapear 'split' do CSV para a estrutura das pastas
def get_split_folder(split):
    return 'train' if split.strip().lower() == 'training' else 'test'

# Função para aplicar CLAHE
def apply_clahe(image):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(image.astype(np.uint8))

# Processa as imagens, realiza o corte, aplica CLAHE e resize, mantendo o crop exato
for _, row in annotations.iterrows():
    image_name = row['image_id']
    split = get_split_folder(row['split'])

    # Caminhos para as pastas de entrada (0 e 1)
    for label in ['0', '1']:
        input_path = os.path.join(base_images_dir, split, label, f"{image_name}.png")

        # Caminho de saída
        output_dir = os.path.join(output_base_dir, split, label)
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{image_name}.png")

        # Verifica se a imagem existe
        if os.path.exists(input_path):
            try:
                # Carrega a imagem usando Pillow e converte para NumPy
                with Image.open(input_path) as img:
                    img_np = np.array(img)

                    # Realiza o corte **exato** usando coordenadas originais
                    xmin, ymin, xmax, ymax = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
                    cropped_img = img_np[ymin:ymax, xmin:xmax]

                    # Redimensiona o resultado do crop para 64x64
                    resized_img = cv2.resize(cropped_img, (64, 64), interpolation=cv2.INTER_LINEAR)

                    # Aplica CLAHE na imagem redimensionada
                    clahe_img = apply_clahe(resized_img)

                    # Salva a imagem processada
                    Image.fromarray(clahe_img).save(output_path)
                    print(f"Imagem {image_name}.png salva em {output_path}")
            except Exception as e:
                print(f"Erro ao processar {image_name}.png: {e}")
            break  # Sai do loop após encontrar a imagem
    else:
        print(f"Imagem não encontrada em nenhuma pasta: {image_name}.png")
