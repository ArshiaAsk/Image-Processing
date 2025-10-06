# Dog Breed Identification

Image classification project to predict the breed of dogs using deep learning models.
This notebook trains a CNN-based classifier on the [Dog Breed Identification dataset](https://www.kaggle.com/c/dog-breed-identification) from Kaggle.

---

## Dataset
The dataset contains:
- **Training images:** 10,222 JPG files
- **Classes:** 120 unique breeds
- **Structure:** `labels.csv` with `id` and `breed`, plus `train` folder with images

Data is stored in Google Drive, downloaded as `.zip`, and extracted in Google Colab.

---

## Workflow

### 1️- Data Loading & Extraction
```python
from google.colab import drive
drive.mount("/content/drive")

from zipfile import ZipFile
with ZipFile("/content/drive/My Drive/dog-breed-identification.zip", 'r') as zip:
    zip.extractall()
```

### 2️- Exploratory Data Analysis (EDA)
- Read labels.csv with **pandas**
- Inspect columns (`id`, `breed`)
- Check unique breed names
- Plot breed distribution using **matplotlib**/**seaborn**

### 3️- Add Image Filepath Column
```python
df['filepath'] = 'train/' + df['id'] + '.jpg'
```

### 4️- Preprocessing
- Encode target labels with **LabelEncoder**
- Train/Validation split using `train_test_split`
- Load and resize images using **cv2**
- Normalize pixel values to range `[0, 1]`

### 5️- Model Creation
- Build a CNN architecture using **TensorFlow/Keras**
- Convolutional + MaxPooling layers
- ReLU activation functions and Softmax output
- Optimizer: Adam
- Loss function: categorical crossentropy
- Metric: accuracy

### 6️- Training & Evaluation
- Train the model using preprocessed data
- Evaluate accuracy on the validation set
- Plot Loss and Accuracy curves

---

## Requirements
```txt
pandas
numpy
matplotlib
seaborn
opencv-python
scikit-learn
tensorflow
keras
```

---

## Running the Project
1. Upload dataset zip to Google Drive
2. Mount Google Drive in Colab
3. Run all cells in `image_processing.ipynb`
4. The trained model outputs accuracy and plots performance graphs

---

## Current Results
- **Baseline CNN model**
- Initial validation accuracy: ~X% (dependent on chosen parameters)

---

## License
This project is for educational purposes only and uses the Kaggle Dog Breed Identification dataset.
