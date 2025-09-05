import os
import json
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from ultralytics import YOLO
from custom_layers.MinPooling import MinPooling2D




class IDCardProcessor:
    def __init__(self ,
                yolo_model_path="../runs/detect/train10/weights/best.pt",
                gender_model_path="../models/cnn_model_gender_2.keras",
                religion_model_path="../models/cnn_model_religion_2.keras",
                marital_model_path="../models/cnn_model_marital_status_2.keras",
                field_map_path="../models/class_en_to_ar.json",
                class_names_gender_path="../models/class_names_gender.json",
                class_names_religion_path="../models/class_names_religion.json",
                class_names_marital_path="../models/class_names_marital_status.json",
                output_path="../outputs"):
        
        self.yolo_model = YOLO(yolo_model_path)
        self.cnn_model_gender = load_model(gender_model_path)
        self.cnn_model_religion=load_model(religion_model_path)
        self.cnn_model_marital_status=load_model(marital_model_path)


        with open(field_map_path, "r", encoding="utf-8") as f:
            self.field_map = json.load(f)
        with open(class_names_gender_path, "r", encoding="utf-8") as f:
            self.class_names_gender = json.load(f)
        with open(class_names_religion_path, "r", encoding="utf-8") as f:
            self.class_names_religion = json.load(f)
        with open(class_names_marital_path, "r", encoding="utf-8") as f:
            self.class_names_marital_status = json.load(f)

        self.output_folder = output_path
        os.makedirs(self.output_folder , exist_ok=True)


    def predict_field(self , img_path, field_name):
        img = image.load_img(img_path, target_size=(64, 64))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        if field_name == "gender":
            preds = self.cnn_model_gender.predict(img_array, verbose=0)
            label = self.class_names_gender[np.argmax(preds, axis=1)[0]]
        elif field_name == "religion":
            preds = self.cnn_model_religion.predict(img_array, verbose=0)
            label = self.class_names_religion[np.argmax(preds, axis=1)[0]]
        elif field_name == "marital status":
            preds = self.cnn_model_marital_status.predict(img_array, verbose=0)
            label = self.class_names_marital_status[np.argmax(preds, axis=1)[0]]
        else:
            return None
        
        label_ar = self.field_map[label]
        return label_ar
    

    def process_id_card(self , img_path, visualize=False):
        results = self.yolo_model(img_path , verbose=False)
        predictions = {}

        img = results[0].plot()
        basename = os.path.splitext(os.path.basename(img_path))[0]

        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                field_name = self.yolo_model.names[cls_id]  

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                crop = img[y1:y2, x1:x2]

                crop_path = f"{self.output_folder}/{basename}_{field_name}.jpg"
                cv2.imwrite(crop_path, crop)


                label = self.predict_field(crop_path, field_name)
                if label:
                    predictions[field_name] = label


        json_path = os.path.join(self.output_folder, f"{basename}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(predictions, f, ensure_ascii=False, indent=4)
        if visualize:
            return img , predictions
        else:
            return predictions


    def process_folder(self, folder_path , visualize =False):
        results_all = {}
        for filename in os.listdir(folder_path):
            img_path = os.path.join(folder_path, filename)
            if not os.path.isfile(img_path):
                continue
            try:
                if visualize:
                    img, predictions = self.process_id_card(img_path, visualize=True)
                    results_all[filename] = {"result" : predictions, "image": img}
                else:
                    predictions = self.process_id_card(img_path)
                    results_all[filename] = predictions
            except Exception as e:
                print(f"Error processing {filename}: {e}")
        return results_all
