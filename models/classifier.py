import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
import gc

class WasteClassifier:
    """Waste classification model using MobileNetV2 with transfer learning"""
    
    def __init__(self, model_path=None, num_classes=3):
        """
        Initialize the model, loading a pre-trained model if available
        
        Parameters:
            model_path: Path to a saved model file
            num_classes: Number of waste categories to classify
        """
        self.img_size = (224, 224)  # MobileNetV2 recommended input size
        self.model = None
        self.num_classes = num_classes
        self.class_labels = ['compostable', 'general_waste', 'recyclable']
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            # If no model is available, create a new one
            self._create_model()
    
    def _create_model(self):
        """Create and compile the model architecture"""
        # Base MobileNetV2 model (pre-trained on ImageNet)
        base_model = MobileNetV2(
            input_shape=(224, 224, 3),
            include_top=False,
            weights='imagenet'
        )
        
        # Freeze the base model layers
        for layer in base_model.layers:
            layer.trainable = False
        
        # Add custom classification layers
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.3)(x)
        predictions = Dense(self.num_classes, activation='softmax')(x)
        
        # Final model
        self.model = Model(inputs=base_model.input, outputs=predictions)
        
        # Compile model
        self.model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print(f"Created new model with {self.num_classes} classes. This model needs to be trained before use.")
    
    def load_model(self, model_path):
        """Load a saved model from disk"""
        try:
            # Clear any existing model
            if self.model is not None:
                del self.model
                gc.collect()
                tf.keras.backend.clear_session()
            
            # Load new model
            self.model = load_model(model_path)
            print(f"Model loaded successfully from {model_path}")
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            self._create_model()
    
    def save_model(self, model_path):
        """Save the model to disk"""
        if self.model:
            # Ensure directory exists
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            self.model.save(model_path)
            print(f"Model saved to {model_path}")
        else:
            print("No model to save. Create or train a model first.")
    
    def train(self, train_data, validation_data, epochs=20, batch_size=32, callbacks=None):
        """
        Train the model on waste image data
        
        Parameters:
            train_data: Training data generator or tuple (x_train, y_train)
            validation_data: Validation data generator or tuple (x_val, y_val)
            epochs: Number of training epochs
            batch_size: Batch size for training
            callbacks: List of tf.keras.callbacks to use during training
            
        Returns:
            Training history object
        """
        if not self.model:
            self._create_model()
        
        # First train with frozen base model
        history1 = self.model.fit(
            train_data,
            validation_data=validation_data,
            epochs=int(epochs/2),
            batch_size=batch_size,
            callbacks=callbacks
        )
        
        # Unfreeze some layers for fine-tuning
        base_model = self.model.layers[0]
        # Unfreeze the top 30% of the base model
        for layer in base_model.layers[-int(len(base_model.layers) * 0.3):]:
            layer.trainable = True
        
        # Recompile with a lower learning rate for fine-tuning
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(1e-5),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Continue training with unfrozen layers
        history2 = self.model.fit(
            train_data,
            validation_data=validation_data,
            epochs=int(epochs/2),
            batch_size=batch_size,
            callbacks=callbacks
        )
        
        # Combine histories
        combined_history = {}
        for k in history1.history.keys():
            combined_history[k] = history1.history[k] + history2.history[k]
        
        return type('obj', (object,), {'history': combined_history})
    
    def preprocess_image(self, image_path):
        """
        Preprocess an image for model prediction
        
        Parameters:
            image_path: Path to the image file
            
        Returns:
            Preprocessed image array ready for model input
        """
        img = load_img(image_path, target_size=self.img_size)
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        return preprocess_input(img_array)
    
    def predict(self, image):
        """
        Classify an image
        
        Parameters:
            image: Preprocessed image array or path to image file
            
        Returns:
            class_label: String label (recyclable, compostable, general_waste)
            predicted_class: Class index (0=recyclable, 1=compostable, 2=general_waste)
            confidence: Confidence score for the prediction
        """
        if not self.model:
            raise ValueError("Model not loaded. Please load or train a model first.")
        
        try:
            # Process image if a file path is provided
            if isinstance(image, str):
                image = self.preprocess_image(image)
            
            # Get predictions
            predictions = self.model.predict(image)
            
            # Get class with highest confidence
            predicted_class = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_class])
            class_label = self.class_labels[predicted_class]
            
            # Clean up
            del predictions
            gc.collect()
            tf.keras.backend.clear_session()
            
            return class_label, predicted_class, confidence
        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            # Clean up
            gc.collect()
            tf.keras.backend.clear_session()
            raise e
    
    def evaluate(self, test_data):
        """
        Evaluate the model on test data
        
        Parameters:
            test_data: Test data generator or tuple (x_test, y_test)
            
        Returns:
            Dictionary of evaluation metrics
        """
        if not self.model:
            raise ValueError("Model not loaded. Please load or train a model first.")
        
        results = self.model.evaluate(test_data, verbose=1)
        metrics = dict(zip(self.model.metrics_names, results))
        print(f"Test accuracy: {metrics['accuracy']:.4f}")
        return metrics
    
    def get_training_data_generator(self, data_dir, batch_size=32):
        """
        Create a data generator for training from a directory structure
        
        Expects a structure like:
            data_dir/
                recyclable/
                    image1.jpg
                    image2.jpg
                    ...
                compostable/
                    image1.jpg
                    ...
                general_waste/
                    image1.jpg
                    ...
                    
        Returns:
            train_generator, validation_generator
        """
        from tensorflow.keras.preprocessing.image import ImageDataGenerator
        
        # Data augmentation for training
        train_datagen = ImageDataGenerator(
            preprocessing_function=preprocess_input,
            rotation_range=40,
            width_shift_range=0.3,
            height_shift_range=0.3,
            shear_range=0.3,
            zoom_range=0.3,
            horizontal_flip=True,
            vertical_flip=True,
            brightness_range=[0.7, 1.3],
            fill_mode='nearest',
            validation_split=0.2  # Use 20% for validation
        )
        
        # Training generator
        train_generator = train_datagen.flow_from_directory(
            data_dir,
            target_size=self.img_size,
            batch_size=batch_size,
            class_mode='categorical',
            subset='training'
        )
        
        # Validation generator
        validation_generator = train_datagen.flow_from_directory(
            data_dir,
            target_size=self.img_size,
            batch_size=batch_size,
            class_mode='categorical',
            subset='validation'
        )
        
        # Update class labels from the generator
        self.class_labels = list(train_generator.class_indices.keys())
        print(f"Class labels detected: {self.class_labels}")
        
        return train_generator, validation_generator
