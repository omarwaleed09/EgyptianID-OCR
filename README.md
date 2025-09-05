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

## Example Usage  

```python
from final.pipeline import IDProcessor

processor = IDProcessor()

# Single image
processor.process_card("sample_id.jpg")

# Folder of images
processor.process_folder("ids_folder/")

Results will be saved in the output/ folder as JSON.

Repository Structure
```
EgyptianID-OCR/
│── notebooks/
│   └── final/                        # Final pipeline class
│   └── runs/detect/train10/          # YOLO best model
│── output/                           # JSON results
│── data/                             # Raw and augmented dataset
│── models/                           # CNN architectures
│── ocr.yml                           # Conda environment file
│── README.md
```

## Future Improvements  

- Add support for additional fields  
- Improve CNN architectures with transfer learning  
