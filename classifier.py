# classifier.py 

import tensorflow as tf 
from tensorflow import keras 
import numpy as np 
from PIL import Image 
import cv2 
import matplotlib.pyplot as plt 

class ImageClassifier: 
    """ 
    Image classification using computer vision 
    Demonstrates concepts from Chapter 24 
    """ 
     
    def __init__(self, model_path='model/keras_model.h5', labels_path='model/labels.txt'): 
        """Load the trained model and labels""" 
        print("Loading model...") 
        self.model = keras.models.load_model(model_path, compile=False) 
         
        # Load labels 
        with open(labels_path, 'r') as f: 
            self.labels = [line.strip() for line in f.readlines()] 
         
        print(f"Model loaded with {len(self.labels)} classes: {self.labels}") 

        # Model expects 224x224 images 
        self.image_size = (224, 224) 
     
    def preprocess_image(self, image_path): 
        """ 
        Prepare image for classification 
        This is like the preprocessing we did for text! 
        """ 
        # Open and resize image 
        img = Image.open(image_path).convert('RGB') 
        img = img.resize(self.image_size) 
         
        # Convert to array and normalize 
        img_array = np.array(img) 
        img_array = img_array / 255.0  # Normalize pixel values 
         
        # Add batch dimension 
        img_array = np.expand_dims(img_array, axis=0) 
         
        return img_array, img 
     
    def classify_image(self, image_path): 
        """Classify a single image""" 
        # Preprocess 
        processed_image, original_image = self.preprocess_image(image_path) 
         
        # Predict 
        predictions = self.model.predict(processed_image, verbose=0) 
         
        # Get results 
        results = [] 
        for i, label in enumerate(self.labels): 
            confidence = float(predictions[0][i]) * 100 
            results.append({ 
                'class': label, 
                'confidence': confidence 
            }) 
         
        # Sort by confidence 
        results.sort(key=lambda x: x['confidence'], reverse=True) 
         
        return results, original_image 
     
    def visualize_prediction(self, image_path): 
        """Show image with prediction results""" 
        results, img = self.classify_image(image_path) 
         
        # Create visualization 
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5)) 
         
        # Show image 
        ax1.imshow(img) 
        ax1.axis('off') 
        ax1.set_title('Input Image') 
         
        # Show predictions 
        classes = [r['class'] for r in results[:3]] 
        confidences = [r['confidence'] for r in results[:3]] 
         
        ax2.barh(classes, confidences) 
        ax2.set_xlabel('Confidence (%)') 
        ax2.set_title('Top 3 Predictions') 
        ax2.set_xlim(0, 100) 
         
        # Add percentage labels 
        for i, (c, conf) in enumerate(zip(classes, confidences)): 
            ax2.text(conf + 1, i, f'{conf:.1f}%') 
         
        plt.tight_layout() 
        plt.savefig('prediction_result.png') 
        plt.show() 
         
        print("\n🎯 Classification Results:") 
        print("-" * 40) 
        for r in results: 
            print(f"{r['class']:15} {r['confidence']:6.2f}%") 
         
        return results[0]  # Return top prediction 
     
    def classify_from_webcam(self): 
        """Real-time classification from webcam""" 
        print("\n📸 Starting webcam classifier...") 
        print("Press 'q' to quit, 'c' to capture and classify") 
         
        cap = cv2.VideoCapture(0) 
         
        while True: 
            ret, frame = cap.read() 
            if not ret: 
                break 
             
            # Show feed 
            cv2.imshow('Webcam - Press C to Classify, Q to Quit', frame) 
             
            key = cv2.waitKey(1) & 0xFF 
             
            if key == ord('q'): 
                break 
            elif key == ord('c'): 
                # Save frame 
                cv2.imwrite('webcam_capture.jpg', frame) 
                 
                # Classify 
                results, _ = self.classify_image('webcam_capture.jpg') 
                 
                print("\n📸 Webcam Classification:") 
                for r in results[:3]: 
                    print(f"{r['class']:15} {r['confidence']:6.2f}%") 
         
        cap.release() 
        cv2.destroyAllWindows() 
     
    def explain_process(self): 
        """Explain how computer vision works""" 
        print("\n" + "="*60) 
        print("🤖 HOW COMPUTER VISION WORKS") 
        print("="*60) 
         
        explanation = """ 
        1. IMAGE CAPTURE 
           - Image is captured as a grid of pixels 
           - Each pixel has RGB (Red, Green, Blue) values 
            
        2. PREPROCESSING 
           - Resize to standard size (224x224 for our model) 
           - Normalize pixel values (0-255 → 0-1) 
           - Prepare shape for neural network 
            
        3. FEATURE EXTRACTION 
           - Neural network finds patterns: 
             * Edges and corners (early layers) 
             * Shapes and textures (middle layers) 
             * Objects and concepts (deep layers) 
            
        4. CLASSIFICATION 
           - Final layer outputs probability for each class 
           - Highest probability is the prediction 
            
        5. CHALLENGES 
           - Lighting changes appearance 
           - Different angles look different 
           - Partial occlusion (things blocking view) 
           - Similar looking objects 
        """ 
         
        print(explanation)