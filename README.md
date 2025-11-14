## Computer Visison Image Classifier

## Project Overview
This project implements image classification using computer vision concepts from Chapter
24. The AI can recognize objects in images using a neural network trained with Teachable
Machine.


<p align="center">
  <img width="500" height="auto" alt="Screenshot 2025-11-13 171053" src="https://github.com/user-attachments/assets/3e94af74-85d4-4714-b3a4-9aaf7c07cfc0" />
</p>


## My Classification Task

**Classes:** Phone, Plug, Sticky Notes

**Training Images:** ~130 images per class

**Accuracy:** 60% ~ 99.8% during testing

<p align="center"><img width="500" height="auto" alt="Screenshot 2025-11-13 171457" src="https://github.com/user-attachments/assets/24e72af8-a005-47fe-9115-2d98d9fb0fb9" /></p>
<p align="center"><img width="500" height="auto" alt="Screenshot 2025-11-13 180356" src="https://github.com/user-attachments/assets/1acec9d2-7028-44ed-a950-c2e27d2d9f5e" /></p>
<p align="center"><img width="500" height="auto" alt="Screenshot 2025-11-13 180633" src="https://github.com/user-attachments/assets/a03ccfd9-832c-4644-8c57-72f7e2f11e61" /></p>
## How to Run

1. **Install Requirements**
```bash
pip install -r requirements.txt
```
2. **Test the Classifier**
```bash
python test_classifier.py
```
3. **Web Interface**
```bash
python web_interface.py
```
Visit http://localhost:5000

## How Computer Vision Works

### Image as Numbers
Every image is a grid of pixels. Each pixel has RGB values (0-255 for Red, Green, Blue). The neural network uses tehse numbers to detect patterns.

### Feature Detection
The neural network learns to detect:

- **Edges**: Where colors change sharply
- **Shapes**: Combinations of edges
- **Patterns**: Repeated features
- **Objects**: Complex combinations like the differences between a phone, plug and sticky notes.

### My Model's Process
1. Resize image to 224x224 pixels
2. Normalize pixel values
3. Pass through neural network
4. Get probability for each class
5. Return highest probability as prediction

## Challenges I Encountered
When entering training data, I found that sixty images was not enough data to train the model. After adding more data, the model was able to more easily detect the objects with better accuracy. I struggled to get the model to distinguish between the plug and the sticky notes, but by giving more variety in angles, the model showed noticeable improvement. This demonstrates that the more data a model receives, the better it can learn.

Additionally, I encountered a compatibility issue when trying to run the provided .h5 model. The model was created using older versions of TensorFlow and Keras, which caused errors in my current environment. I resolved this by downgrading to Python 3.10.11 and installing TensorFlow/Keras 2.12 to match the model’s requirements.

## Real-World Applications

1. **Medical**: Image classifiers can be used medically to review different types of moles and detect abnormal shapes, colors and sizes to determine if it needs a closer look or biopsy. Used in conjunction with other tools, it could also help to determine a better diagnosis or the right path of treatment.
2. **Security**: Facial recognition could provide better security and monitoring through high traffic areas like malls, airports and busy streets to identify wanted individuals, criminals or on-going crime.
3. **Retail**: Retail stores can use image classifying to improve checkout systems and help move customers through their lines quicker while simultaneously updating their inventories.

## Ethical Considerations
When computers train on specific data they may not get all the information they need and bias can occur. For example, if only one angle is viewed at a time it may not detect the same object when angled in the opposite direction or if it is flipped or if the object it is training on can have various shapes, sizes or a different color. The more information the image classifer has, the better data and the decrease in bias. There are significant privacy concerns if webcams are used to collect personal information or any varying amount of data. Users should be aware of when it is being used and how it is being used. Other privacy issues include the image classifier being used to track anyone without their knowledge.

## What I Learned
I learned that computer vision models will break down images into pixels to detect patterns thorugh neural networks. I used Google's Teachable Machine to build my own dataset and export a trained model which trained on my own data. Through the Teachable Machine I was able to visualize how the model trained and was able to predict different objects (phone, plug and sticky notes).
