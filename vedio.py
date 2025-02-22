import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import os
from models import *  # Assuming you have a custom model file
import transforms as transforms
from PIL import Image
from skimage import io
from skimage.transform import resize
from datetime import datetime
# Initialize OpenCV for webcam
cap = cv2.VideoCapture(0)



# Load cascade classifier for face detection
face_cascade = cv2.CascadeClassifier('frontalface/haarcascade_frontalface_default.xml')

# Define the image transformations
cut_size = 44
transform_test = transforms.Compose([
    transforms.TenCrop(cut_size),
    transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
])

pret=0
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

before_class=''

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        break

    # Convert to grayscale for face detection
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray_frame)

    # Process each face found
    for x, y, w, h in faces:
        # Extract face region
        face = frame[y:y+h, x:x+w]

        # Convert the face region to grayscale and resize it
        gray = rgb2gray(face)
        gray_face = resize(gray, (48, 48), mode='symmetric').astype(np.uint8)
        img = gray_face[:, :, np.newaxis]
        img = np.concatenate((img, img, img), axis=2)
        img = Image.fromarray(img)

        class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        emo_weight = [-1, -1.5, -2.5, 3, -2, 1, 2]
        # Load pre-trained model
        net = VGG('VGG19')
        checkpoint = torch.load(os.path.join('FER2013_VGG19', 'PrivateTest_model.t7'))
        net.load_state_dict(checkpoint['net'])
        net.cuda()
        net.eval()
        # Apply transformations
        inputs = transform_test(img)
        ncrops, c1, h1, w1 = np.shape(inputs)
        inputs = inputs.view(-1, c1, h1, w1)
        inputs = inputs.cuda()
        with torch.no_grad():
            inputs = Variable(inputs)

        # Get predictions from the network
        outputs = net(inputs)
        outputs_avg = outputs.view(ncrops, -1).mean(0)  # avg over crops
        score = F.softmax(outputs_avg,dim=0)
        _, predicted = torch.max(outputs_avg.data, 0)

        # Get the predicted emotion class
        emos=score.data.cpu().numpy()
        predicted_class = class_names[int(predicted.cpu().numpy())]
        t = score.data.cpu().numpy() * emo_weight
        if predicted_class!=before_class:

            print(str(datetime.now())+"当前表情为"+predicted_class)

            # for i in range(len(class_names)):
            #     print(class_names[i]+' gets '+str(emos[i]))
            before_class=predicted_class
        if abs(pret-sum(t))>0.1:
            pret=sum(t)
            print(sum(t))
        # Draw rectangle around face and display the predicted emotion
        # cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        # cv2.putText(frame, f"Expression: {predicted_class}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Real-time Face Expression Recognition', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
