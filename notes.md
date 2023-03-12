##  Here's a detailed explanation of the code in Emotion_recognization.py:

1. The first few lines import the required Python modules, including pandas (for data analysis), numpy (for numerical operations), and Keras (for building and training neural networks).
2. Next, the script loads the FER2013 dataset from a CSV file using pandas' read_csv() method. This dataset contains 48x48 grayscale images of faces and their corresponding emotional labels (anger, disgust, fear, happiness, sadness, surprise, and neutral). The dataset is split into three sets: training, validation, and testing.
3. The script then loops through each row in the dataset and splits the pixel values (stored as a string) into an array. It then appends the image array and its corresponding label to separate lists (X_train, train_y, X_test, test_y), depending on whether the image is part of the training or testing set.
4. Next, the script converts the label data to categorical format using Keras' to_categorical() function. This is required because the neural network needs the labels to be in categorical format for training and prediction.
5. The script then normalizes the image data by subtracting the mean and dividing by the standard deviation. This is a standard preprocessing step for image data.
6. The image data is then reshaped into a 4D tensor with shape (num_samples, width, height, num_channels). In this case, the num_channels is 1 because the images are grayscale.
7. The neural network architecture is then defined using Keras' Sequential model. This is a simple stack of layers where each layer has exactly one input tensor and one output tensor.
8. The model consists of several convolutional layers, followed by max pooling layers, and then fully connected layers. Each convolutional layer applies a set of learnable filters to the input image and produces an output feature map. The max pooling layer downsamples the output feature map by taking the maximum value within a small region. The fully connected layers at the end of the network use the features extracted by the convolutional layers to predict the class labels.
9. The compile() method is then called to configure the model for training. It takes three arguments: the loss function, the optimizer, and the evaluation metric. In this case, the categorical crossentropy loss function is used, along with the Adam optimizer and accuracy metric.
10. Finally, the model is trained on the training data using the fit() method. This method takes several arguments, including the input data (X_train), the target labels (train_y), the batch size, the number of epochs, and the validation data (X_test and test_y). During training, the model updates its weights to minimize the loss function, and the accuracy is computed and displayed after each epoch. Once training is complete, the model is saved to disk in JSON and HDF5 format using the to_json() and save_weights() methods, respectively.


## videoTester.py program

This is a Python program that performs facial emotion analysis in real-time using a pre-trained deep learning model. The program uses the OpenCV library to capture video from the webcam and detect faces in each frame using a pre-trained Haar cascade classifier. Once a face is detected, the program crops the image to the area containing the face, resizes it to 48x48 pixels, and passes it through a deep learning model to predict the facial expression.

The deep learning model used is loaded from two files, 'fer.json' and 'fer.h5', which contain the architecture and weights of the model, respectively. The predicted emotion is then displayed on the screen next to a rectangle drawn around the detected face.

The program continuously captures frames from the webcam until the user presses the 'q' key, at which point the program exits and the webcam is released.
