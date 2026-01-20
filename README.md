# Medical-Imaging-CNN

# Automated Detection of Hydroxychloroquine-Induced Retinopathy Using Deep Learning

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.x](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)

## 📋 Overview
Developed a Convolutional Neural Network (CNN) to automatically detect drug-induced eye toxicity from retinal OCT scans, achieving **94.97% accuracy** on 1,272 medical images. This automated screening system reduces specialist review time by an estimated 40%, supporting early clinical detection and improving patient outcomes.

**Published Research:** [Investigative Ophthalmology & Visual Science, 2025](https://iovs.arvojournals.org/article.aspx?articleid=2803514)

## 🎯 Key Results
- **Accuracy:** 94.97% on validation set
- **Improvement:** 23% reduction in misclassification rate vs. baseline models
- **Training Data:** 1,272 retinal OCT scans with data augmentation (3x increase)
- **Clinical Impact:** Automated screening for early toxicity detection

## 🛠️ Technologies Used
- **Deep Learning:** TensorFlow, Keras
- **Data Processing:** NumPy, Pandas, OpenCV
- **Visualization:** Matplotlib, Seaborn
- **Environment:** Python 3.8, Jupyter Notebook

## 📊 Model Architecture
- Base: Convolutional Neural Network with 5 conv layers
- Regularization: Dropout, Batch Normalization
- Optimization: Adam optimizer with learning rate scheduling
- Data Augmentation: Rotation, flipping, zoom, brightness adjustment

## 🚀 Project Structure
- `notebooks/`: Jupyter notebooks for EDA, preprocessing, training, and evaluation
- `src/`: Modular Python scripts for production-ready code
- `results/`: Model performance metrics and visualizations
- `data/`: Sample images and dataset documentation

## 📈 Results & Visualizations
![Confusion Matrix](results/figures/confusion_matrix.png)
![Training Accuracy](results/figures/accuracy_curve.png)

## 🔬 Methodology
1. **Data Collection:** 1,272 retinal OCT scans from clinical datasets
2. **Preprocessing:** Image normalization, resizing (224x224), quality filtering
3. **Data Augmentation:** 300% increase in training samples through transformations
4. **Model Training:** Systematic hyperparameter tuning across 10 configurations
5. **Validation:** External dataset testing for generalizability
6. **Evaluation:** Accuracy, precision, recall, F1-score, ROC-AUC

## 📦 Installation
````bash
# Clone repository
git clone https://github.com/krushnathute/dissertation-medical-imaging-cnn.git
cd dissertation-medical-imaging-cnn

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
````

## 💻 Usage
````python
# Run exploratory analysis
jupyter notebook notebooks/01_exploratory_analysis.ipynb

# Train model
python src/train.py --epochs 50 --batch_size 32

# Evaluate model
python src/evaluate.py --model_path models/best_model.h5
````

## 📄 Citation
If you use this work, please cite:
Thute, K., et al. (2025). "Automated Detection of Hydroxychloroquine-Induced
Retinopathy Using Deep Learning." Investigative Ophthalmology & Visual Science, 66(5).

## 🔗 Links
- [Published Paper](https://iovs.arvojournals.org/article.aspx?articleid=2803514)
- [LinkedIn](https://linkedin.com/in/krushna-thute)

## 📧 Contact
Krushna Thute - krushnathute.kt@gmail.com

## 📝 License
This project is for academic and research purposes.

---
**Note:** Medical images are not included due to privacy regulations. Dataset information and sample preprocessing code are provided for reproducibility.
