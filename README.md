# EgyptianID-OCR  

A deep learning project for detecting and recognizing key fields in Egyptian National ID cards.  
The system uses **YOLO** for object detection and custom **CNN models** for classification.  

---

## Project Pipeline  

1. **Input Image**  
2. Pass through **YOLO model** to detect ID fields  
3. Crop detected fields and send each to its dedicated **CNN model**  
   - Gender Model  
   - Religion Model  
   - Marital Status Model  
4. CNN models output predictions in English  
5. A JSON file maps predictions to Arabic labels  
6. Final results are saved in the `output/` folder as JSON  

---

## Steps  

### 1. Data Collection & Augmentation  
- Collected real Egyptian ID cards  
- Edited synthetic cards with the same font to expand dataset  
- Applied augmentation to increase dataset size  

### 2. Annotation  
- Annotated selected samples for three fields: gender, religion, marital status  
- Used annotated + unannotated data to train YOLO  

### 3. YOLO Training  
- Trained multiple YOLO models  
- Best weights stored in: `train10/best.pt`  

### 4. CNN Models  
- Cropped YOLO fields used to train CNNs built from scratch  
- Designed **two architectures per field** (Gender, Religion, Marital Status)  

### 5. Final Integration  
- Implemented a class in `notebooks/final/`  
- Key functions:  
  - `process_card(image)` → process a single ID  
  - `process_folder(folder)` → process multiple IDs  

---

## Tools and Frameworks  
- Python  
- OpenCV  
- TensorFlow / Keras  
- YOLO  
- CNN  

---

## Usage Example  

```python
from final.model import IDCardProcessor  

# Initialize processor  
processor = IDCardProcessor(  
    yolo_model_path="../runs/detect/train10/weights/best.pt",  
    gender_model_path="../models/cnn_model_gender_2.keras",  
    religion_model_path="../models/cnn_model_religion_2.keras",  
    marital_model_path="../models/cnn_model_marital_status_2.keras",  
)  

# Process single ID card  
result = processor.process_id_card("sample_id.jpg")  
print(result)  

# Process folder of ID cards  
results = processor.process_folder("ids_folder/")  
print(results)  


## Project Structure

```
EgyptianID-OCR/
│── notebooks/
│ └── final/ <sub><i>Final pipeline class</i></sub>
│ └── runs/detect/train10/ <sub><i>YOLO best model</i></sub>
│── output/ <sub><i>JSON results</i></sub>
│── data/ <sub><i>Raw and augmented dataset</i></sub>
│── models/ <sub><i>CNN architectures</i></sub>
│── ocr.yml <sub><i>Conda environment file</i></sub>
│── README.md

```

## Future Improvements  

- Add support for additional fields  
- Improve CNN architectures with transfer learning  
