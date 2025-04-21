import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from models.classifier import WasteClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import pandas as pd

def train_waste_classifier(data_dir, model_save_path, epochs=50, batch_size=16):
    """
    Train a waste classification model using transfer learning
    
    Parameters:
        data_dir: Directory containing training data in subdirectories
        model_save_path: Path to save the trained model
        epochs: Number of training epochs
        batch_size: Batch size for training
        
    Returns:
        Trained model and training history
    """
    # Get the number of classes from the directory structure
    class_names = [d for d in os.listdir(data_dir) 
                  if os.path.isdir(os.path.join(data_dir, d))]
    num_classes = len(class_names)
    print(f"Detected {num_classes} classes: {', '.join(class_names)}")
    
    # Create and configure the classifier with the correct number of classes
    classifier = WasteClassifier(num_classes=num_classes)
    
    # Get training and validation data generators
    train_generator, validation_generator = classifier.get_training_data_generator(
        data_dir,
        batch_size=batch_size
    )
    
    # Add callbacks for early stopping and model checkpointing
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=model_save_path,
            monitor='val_accuracy',
            save_best_only=True,
            save_weights_only=False
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir='./logs',
            histogram_freq=1
        ),
        # Add LearningRateScheduler to track learning rate
        tf.keras.callbacks.LambdaCallback(
            on_epoch_end=lambda epoch, logs: logs.update({'lr': tf.keras.backend.get_value(classifier.model.optimizer.lr)})
        )
    ]
    
    # Train the model
    history = classifier.train(
        train_generator,
        validation_generator,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks
    )
    
    # Save the model
    classifier.save_model(model_save_path)
    
    # Plot training history
    plot_training_history(history)
    
    # Create confusion matrix
    plot_confusion_matrix(classifier, validation_generator)
    
    # Generate classification report
    generate_classification_report(classifier, validation_generator)
    
    return classifier, history

def plot_training_history(history):
    """
    Plot training and validation accuracy and loss with enhanced visualization
    
    Parameters:
        history: History object returned by model.fit()
    """
    # Create a figure with two subplots
    plt.figure(figsize=(15, 6))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], 'o-', label='Training Accuracy', linewidth=2)
    plt.plot(history.history['val_accuracy'], 'o-', label='Validation Accuracy', linewidth=2)
    plt.title('Model Accuracy Over Epochs', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.ylim([0, 1.0])
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Add epoch markers
    epochs = range(1, len(history.history['accuracy']) + 1)
    plt.xticks(range(0, len(epochs), max(1, len(epochs) // 10)))
    
    # Highlight best validation accuracy
    best_val_acc_epoch = np.argmax(history.history['val_accuracy'])
    best_val_acc = history.history['val_accuracy'][best_val_acc_epoch]
    plt.plot(best_val_acc_epoch, best_val_acc, 'r*', markersize=15, 
             label=f'Best Val Acc: {best_val_acc:.4f}')
    
    plt.legend(loc='lower right', fontsize=10)
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], 'o-', label='Training Loss', linewidth=2)
    plt.plot(history.history['val_loss'], 'o-', label='Validation Loss', linewidth=2)
    plt.title('Model Loss Over Epochs', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Add epoch markers
    plt.xticks(range(0, len(epochs), max(1, len(epochs) // 10)))
    
    # Highlight best validation loss
    best_val_loss_epoch = np.argmin(history.history['val_loss'])
    best_val_loss = history.history['val_loss'][best_val_loss_epoch]
    plt.plot(best_val_loss_epoch, best_val_loss, 'r*', markersize=15, 
             label=f'Best Val Loss: {best_val_loss:.4f}')
    
    plt.legend(loc='upper right', fontsize=10)
    
    plt.tight_layout()
    
    # Save high-resolution figure
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    
    # Create a separate figure for learning rate if it exists in history
    if 'lr' in history.history:
        plt.figure(figsize=(10, 4))
        plt.plot(history.history['lr'], 'o-', linewidth=2)
        plt.title('Learning Rate Over Epochs', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Learning Rate', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.xticks(range(0, len(epochs), max(1, len(epochs) // 10)))
        plt.yscale('log')
        plt.tight_layout()
        plt.savefig('learning_rate_history.png', dpi=300, bbox_inches='tight')
        print(f"Learning rate history plot saved to learning_rate_history.png")
    
    # Create a combined metrics plot
    plt.figure(figsize=(12, 8))
    
    # Training metrics
    plt.subplot(2, 1, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
    plt.plot(history.history['loss'], label='Training Loss', linewidth=2)
    plt.title('Training Metrics Over Epochs', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Value', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='center right', fontsize=10)
    
    # Validation metrics
    plt.subplot(2, 1, 2)
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
    plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    plt.title('Validation Metrics Over Epochs', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Value', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='center right', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('combined_metrics.png', dpi=300, bbox_inches='tight')
    print(f"Combined metrics plot saved to combined_metrics.png")
    
    plt.close('all')
    print(f"Training history plots saved to training_history.png")


def plot_confusion_matrix(classifier, validation_generator):
    """Plot confusion matrix to visualize model performance on each class"""
    # Reset the generator
    validation_generator.reset()
    
    # Get predictions for the validation set
    y_true = validation_generator.classes
    steps = validation_generator.samples // validation_generator.batch_size + 1
    y_pred = classifier.model.predict(validation_generator, steps=steps)
    y_pred = np.argmax(y_pred, axis=1)[:validation_generator.samples]
    
    # Create confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=classifier.class_labels,
               yticklabels=classifier.class_labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    plt.close()
    
    print(f"Confusion matrix saved to confusion_matrix.png")

def generate_classification_report(classifier, validation_generator):
    """Generate a detailed classification report"""
    # Reset the generator
    validation_generator.reset()
    
    # Get predictions for the validation set
    y_true = validation_generator.classes
    steps = validation_generator.samples // validation_generator.batch_size + 1
    y_pred = classifier.model.predict(validation_generator, steps=steps)
    y_pred = np.argmax(y_pred, axis=1)[:validation_generator.samples]
    
    # Generate report
    report = classification_report(y_true, y_pred, 
                                  target_names=classifier.class_labels,
                                  output_dict=True)
    
    # Convert to dataframe for better visualization
    df_report = pd.DataFrame(report).transpose()
    
    # Save report to CSV
    df_report.to_csv('classification_report.csv')
    
    print(f"Classification report saved to classification_report.csv")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=classifier.class_labels))

def test_on_image(classifier, image_path):
    """Test the trained model on a single image"""
    class_label, class_index, confidence = classifier.predict(image_path)
    
    # Load and display the image
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    plt.title(f'Prediction: {class_label} ({confidence:.2%})')
    plt.axis('off')
    plt.savefig('prediction_example.png')
    plt.close()
    
    print(f"Predicted class: {class_label}")
    print(f"Confidence: {confidence:.2%}")
    print(f"Prediction visualization saved to prediction_example.png")

def prepare_sample_dataset():
    """
    Prepare a sample dataset structure for waste classification
    """
    # Create directory structure for the dataset
    dataset_dir = 'sample_dataset'
    categories = ['recyclable', 'compostable', 'general_waste']
    
    os.makedirs(dataset_dir, exist_ok=True)
    for category in categories:
        os.makedirs(os.path.join(dataset_dir, category), exist_ok=True)
    
    print(f"Created sample dataset structure at {dataset_dir}")
    print("In a real application, you would need to populate this with actual images")
    print("For each category (recyclable, compostable, general_waste)")
    
    # Guidance on dataset collection
    print("\nRecommendations for dataset collection:")
    print("1. Aim for at least 1000 images per category for good performance")
    print("2. Include various lighting conditions and backgrounds")
    print("3. Capture objects from different angles")
    print("4. Include different types of items within each category")
    print("5. Consider using data augmentation to expand your dataset")
    
    return dataset_dir

if __name__ == "__main__":
    # Prepare dataset structure
    data_dir = prepare_sample_dataset()
    
    # Set the path to save the trained model
    model_save_path = 'models/saved_model/waste_classifier.h5'
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    
    # Train the model
    classifier, history = train_waste_classifier(data_dir, model_save_path)
    
    print("\nModel training complete.")
    print(f"Model saved to {model_save_path}")
    print("You can now use this model for waste classification.")
    
    # If you have a test image, uncomment the following line to test it
    # test_on_image(classifier, 'path/to/test_image.jpg')
