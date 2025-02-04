import os
import cv2
import pydicom
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
from skimage import exposure, filters
from sklearn.model_selection import train_test_split

def rename_dicom_files(dicom_dir):
    dicom_files = [f for f in os.listdir(dicom_dir) if f.endswith('.dcm')]

    for dicom_file in dicom_files:
        old_path = os.path.join(dicom_dir, dicom_file)

        new_file_name = dicom_file.split('_')[0] + '.dcm'
        new_path = os.path.join(dicom_dir, new_file_name)

        if old_path != new_path:
            os.rename(old_path, new_path)
            print(f"Renamed: {dicom_file} -> {new_file_name}")

def preprocess_and_save_images(csv_file, root_dir, xml_dir, output_dir, test_size=0.2):
    data = pd.read_csv(csv_file, sep=';')
    data = data[data['Bi-Rads'].notnull()]
    data['Label'] = data['Bi-Rads'].apply(lambda x: 0 if x in ['1', '2'] else 1)
    data['File Name'] = data['File Name'].astype(str)

    train_data, test_data = train_test_split(data, test_size=test_size, stratify=data['Label'], random_state=42)
    train_labels = dict(zip(train_data['File Name'], train_data['Label']))
    test_labels = dict(zip(test_data['File Name'], test_data['Label']))

    process_and_save(train_data, 'train', train_labels, root_dir, xml_dir, output_dir)
    process_and_save(test_data, 'test', test_labels, root_dir, xml_dir, output_dir)

def process_and_save(dataset, split, labels_dict, root_dir, xml_dir, output_dir):
    split_output_dir = os.path.join(output_dir, split)
    os.makedirs(split_output_dir, exist_ok=True)

    for label in [0, 1]:
        os.makedirs(os.path.join(split_output_dir, str(label)), exist_ok=True)

    for file_name in dataset['File Name']:
        dicom_file = file_name + '.dcm'
        dicom_path = os.path.join(root_dir, dicom_file)
        xml_file = os.path.join(xml_dir, file_name + '.xml')

        if not os.path.isfile(dicom_path) or not os.path.isfile(xml_file):
            continue

        img, _ = read_dicom_img(dicom_path)
        rois = extract_regions_from_xml(xml_file, img)

        if not rois:
            continue

        label = labels_dict[file_name]
        label_dir = os.path.join(split_output_dir, str(label))
        for i, roi in enumerate(rois):
            output_path = os.path.join(label_dir, f"{file_name}_roi{i}.png")
            cv2.imwrite(output_path, roi * 255)

def read_dicom_img(path):
    dcm = pydicom.dcmread(path)
    img = dcm.pixel_array
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img.astype(np.uint8))

    image = cv2.convertScaleAbs(img - np.min(img), alpha=(255.0 / min(np.max(img) - np.min(img), 10000)))

    if dcm.PhotometricInterpretation == "MONOCHROME1":
        image = np.invert(image)

    return image / 255.0, image.shape

def extract_regions_from_xml(xml_file, img):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    rois = []

    for region in root.findall(".//array"):
        points = region.findall("string")
        coords = [tuple(map(float, point.text.strip("()").split(", "))) for point in points]

        if len(coords) < 2:
            continue

        try:
            x_coords, y_coords = zip(*coords)
        except ValueError:
            continue

        x_min, x_max = int(min(x_coords)), int(max(x_coords))
        y_min, y_max = int(min(y_coords)), int(max(y_coords))

        cropped = img[y_min:y_max, x_min:x_max]
        if cropped.size == 0 or cropped.shape[0] < 64 or cropped.shape[1] < 64:
            continue

        resized = cv2.resize(cropped, (64, 64), interpolation=cv2.INTER_AREA)
        rois.append(resized)

    return rois

def main():

    dicom_dir = '/home/eflammere/BreastCancerQuanvolution/Datasets/INbreast/dicom'
    csv_file = '/home/eflammere/BreastCancerQuanvolution/Datasets/INbreast/csv/INbreast.csv'
    root_dir = '/home/eflammere/BreastCancerQuanvolution/Datasets/INbreast/dicom'
    xml_dir = '/home/eflammere/BreastCancerQuanvolution/Datasets/INbreast/masks'
    output_dir = '/home/eflammere/BreastCancerQuanvolution/Datasets/INbreast/png'
    

    # rename_dicom_files(dicom_dir)
    preprocess_and_save_images(csv_file, root_dir, xml_dir, output_dir)

if __name__ == "__main__":
    main()
