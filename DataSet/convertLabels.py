import os
import xml.etree.ElementTree as ET

label2idx = dict({'face_mask': 1, 'face': 0})
idx2label = dict({'0': 'face', '1': 'face_mask'})

# Convert labels to COCO standard

def convertLabels(data_path):
    target_dir = data_path + '/' + "labels/"
    for rel_path in os.listdir(data_path + '/' + "Annotations"):
        if not rel_path.endswith(".xml"): continue
        annotation_path = data_path + '/' + "Annotations" + '/' + rel_path
        annotation = ET.parse(annotation_path).getroot()
        if not annotation.find('object'):
            img_path = data_path + '/' + "images" + '/' + rel_path.replace(".xml", ".jpg")
            if os.path.exists(img_path):
                os.remove(img_path)
            continue
        text_label = annotation.find('object').find('name').text
        label = label2idx[text_label]
        bndbox = annotation.find('object').find('bndbox')
        width = float(annotation.find('size').find('width').text)
        height = float(annotation.find('size').find('height').text)
        xmin = float(bndbox.find('xmin').text) / width
        ymin = float(bndbox.find('ymin').text) / height
        xmax = float(bndbox.find('xmax').text) / width
        ymax = float(bndbox.find('ymax').text) / height
        target_file_name = target_dir + rel_path
        f = open(target_file_name.replace(".xml", ".txt"), "w")
        list_to_write = [label, xmin, ymin, xmax, ymax]
        for field in list_to_write:
            f.write(str(field) + ' ')
        f.close()


if __name__ == '__main__':
    cwd = os.getcwd()
    data_dir = os.path.join(os.path.join(cwd, os.pardir), os.pardir)
    my_data_path = os.path.join(data_dir, "data/face_mask/test")
    convertLabels(my_data_path)
