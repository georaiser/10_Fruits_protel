# 🍎 Fruits ProTEL Classification with EfficientNetV2 🍊

## 🌟 Project Overview
This project implements a deep learning image classification model to identify fresh fruits using the ProTEL dataset. It leverages the EfficientNetV2 architecture, employing transfer learning and fine-tuning techniques to achieve optimal performance in fruit classification.

## 🛠️ Technologies Used

- PyTorch
- torchvision
- scikit-learn
- matplotlib
- NumPy

## 📋 Requirements
- Python 3.8+
- PyTorch 1.9+
- torchvision 0.10+
- scikit-learn 0.24+
- matplotlib 3.4+
- NumPy 1.20+

## 📁 Project Structure

- Data preparation and splitting
- Model setup and configuration
- Custom dataset and dataloader creation
- Model training with early stopping
- Performance visualization
- Inference on test images

## 🔑 Key Features

- 📸 Image classification of fresh fruits
- Transfer learning with EfficientNetV2 pre-trained weights
- 🔧 Custom dataset handling and data splitting
- 🎨 Data augmentation and transformations
- Model fine-tuning and training
- Employs learning rate scheduling and early stopping
- 📊 Provides visualization of training/validation loss and accuracy
- Allows inference on new images with probability scores

## 🛠️ Technical Implementation

### 📊 Data Preparation and Handling

- Custom Dataset Class: CustomDataset_classification for efficient data loading and preprocessing.
- Data Splitting: Utilizes create_data_splits function for train/val/test separation.
- Class Selection: Filters for "Fresh" fruit categories only.

### 🧠 Model Architecture

- Base Model: EfficientNetV2-S with pre-trained weights.
- Fine-tuning: Base layers frozen to preserve learned features.
- Custom classifier head is added to match the number of fruit classes.

**Classifier Structure:**
```
Copynn.Sequential(
  nn.Dropout(p=0.2, inplace=True),
  nn.Linear(in_features=1280, out_features=len(classes))
)
```

### 🏋️ Training Process

- Loss Function: CrossEntropyLoss
- Optimizer: Adam
- Learning Rate Scheduling: ReduceLROnPlateau for adaptive learning rate
- Early Stopping: Custom implementation to prevent overfitting
- Training Loop: Encapsulated in run_model function from engine_classification module

### 📈 Visualization and Evaluation

The project implements:

- Loss and accuracy curve plotting
- Sample prediction visualization on test images

## 🔬 Key Features and Innovations

- **🎚️ Adaptive Learning Rate**: Uses ReduceLROnPlateau scheduler to adjust learning rate based on validation loss.
- **🛑 Early Stopping**: Custom early stopping mechanism to prevent overfitting.
- **🧠 Transfer Learning**: Leverages pre-trained EfficientNetV2 weights.
- **📁 Custom Data Handling**: Bespoke data splitting and loading pipeline.
- **🔮 Inference Capabilities**: Function for making predictions on new images.

## 🏗 Code Structure

- custom_dataset.py: Contains CustomDataset_classification class
- create_data_splits.py: Implements data splitting functionality
- engine_classification.py: Houses the run_model training function
- Main script: Orchestrates the entire training and evaluation process

## 📊 Performance Analysis
The project includes functionality to analyze model performance:

- 📉 Training and validation loss curves
- 📉 Training and validation accuracy curves
- 📈 Accuracy progression over epochs
- 👁️ Visual inspection of predictions on test images
- 👁️ Sample predictions

## 🚀 Running the Project

- 📦 Ensure all dependencies are installed.
- 🗂️ Prepare your dataset in the specified directory structure.
- 🏃‍ Run the main script to train the model and visualize results.
- 🔍 Use the inference function to make predictions on new images.

## 🔮 Future Enhancements

- 📦 Experiment with other state-of-the-art architectures
- 🔄 Implement data augmentation techniques.
- 📦 Explore techniques for handling class imbalance
- 🤝 Explore ensemble methods with multiple architectures.
- 🌐 Develop a web interface for real-time classification.
- 📦 Develop a user-friendly interface for real-time predictions
- 🔍 Investigate model interpretability techniques (e.g., Grad-CAM).
- 🌐 Optimize for mobile deployment
- 📦 Implement cross-validation for more robust evaluation

## 🤝 Contributing
Contributions to improve the model's performance, add new features, or enhance documentation are welcome. Please feel free to submit pull requests or open issues for discussion.
## 📜 License
[license here]

This project demonstrates the application of modern deep learning techniques to the task of fruit classification, showcasing the power of transfer learning and the effectiveness of the EfficientNetV2 architecture in handling image data. 🍎🍊🍌🥝
