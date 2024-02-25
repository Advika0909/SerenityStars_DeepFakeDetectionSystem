# SerenityStars_DeepFakeDetectionSystem
Training of model for deepfake image detection

Activation Functions:
ReLU (Rectified Linear Activation):

Purpose: Used in the convolutional layers for introducing non-linearity.
Description: ReLU sets all negative values to zero and leaves positive values unchanged.
Used in: Convolutional layers (Conv2D).
Sigmoid:

Purpose: Used in the last Dense layer for binary classification.
Description: Sigmoid squashes the output between 0 and 1, suitable for binary classification where the output represents the probability of the positive class.
Used in: Last Dense layer.
Input Function:
Conv2D Layers:

Conv2D layers with ReLU activation are used for feature extraction from the input images.
Each Conv2D layer applies a specified number of filters to the input image.
MaxPooling2D Layers:

MaxPooling2D layers follow Conv2D layers and reduce the spatial dimensions of the input.
They help in reducing the computational complexity and control overfitting.
Flatten Layer:

Flatten layer is used to flatten the 2D feature maps into a 1D vector.
Necessary before passing the data to the Dense layers.
Output Function:
Sigmoid Activation:
The output layer uses a single neuron with a sigmoid activation function.
For binary classification tasks like this one, sigmoid is commonly used.
Sigmoid outputs values between 0 and 1, indicating the probability of the image belonging to the positive class (e.g., deepfake).
![image](https://github.com/Advika0909/SerenityStars_DeepFakeDetectionSystem/assets/141475413/b2d6696b-7ae8-4b4a-b02c-209b8b45409a)

**PLATFORM INTERFACE**
1.Our primary frontend strategy involves seamlessly integrating the ML model into the web application through TensorFlow.js. The proposed interface is illustrated in a preliminary screenshot, showcasing the HTML layout. This approach aims to provide a user-friendly and interactive experience, leveraging the capabilities of TensorFlow.js for efficient machine learning model deployment on the web.
2.Gradio is being considered as our secondary option for creating interfaces to test our models. While it offers simplicity and ease of use, we are evaluating it alongside another option to determine the most suitable interface for our needs. Factors such as ease of implementation, flexibility, community support, compatibility with our existing tech stack, scalability, and security are being taken into account during this evaluation process. By exploring both options, we aim to make an informed decision that aligns with our specific requirements and goals for testing and showcasing our machine learning models.
![ss_gradio](https://github.com/Advika0909/SerenityStars_DeepFakeDetectionSystem/assets/139324446/a44a4d7d-4f3f-4abc-9160-72c0e2c68e56)

**VIDEO OF IMPLEMENTATION**
https://drive.google.com/file/d/1_mkaLr1XcusAAwj0lg8nMgC2F6VrVw_n/view?usp=drivesdk

PPT
https://docs.google.com/presentation/d/1FFyDn4zPQpWBaAI0TII-jWgIlFwcsRXu/edit?usp=sharing&ouid=100837408989732526935&rtpof=true&sd=true
