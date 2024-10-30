#!/usr/bin/env python3

import os
import rospy
from duckietown.dtros import DTROS, NodeType
from sensor_msgs.msg import CompressedImage
import urllib.request
import json
import cv2
from cv_bridge import CvBridge

from PIL import Image
import torch
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights

# Load the ResNet18 model with the most up-to-date ImageNet weights


class CameraReaderNode(DTROS):

    def __init__(self, node_name):
        # initialize the DTROS parent class
        super(CameraReaderNode, self).__init__(node_name=node_name, node_type=NodeType.VISUALIZATION)
        # load pretrained ResNet
        # self.model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', weights=ResNet18_Weights.IMAGENET1K_V1)
        self.model = resnet18(weights=ResNet18_Weights.DEFAULT)
        # static parameters
        print('1')
        url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
        with urllib.request.urlopen(url) as f:
            self.labels = json.load(f)
        self._vehicle_name = os.environ['VEHICLE_NAME']
        # print('2')
        self._camera_topic = f"/{self._vehicle_name}/camera_node/image/compressed"
        print('3')
        # bridge between OpenCV and ROS
        self._bridge = CvBridge()
        print('4')
        # create window
        self._window = "camera-reader"
        print('5')
        cv2.namedWindow(self._window, cv2.WINDOW_AUTOSIZE)
        # construct subscriber
        self.sub = rospy.Subscriber(self._camera_topic, CompressedImage, self.callback)

	
    def callback(self, msg):
        # convert JPEG bytes to CV image
        image = self._bridge.compressed_imgmsg_to_cv2(msg)

        input_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        input_tensor = preprocess(input_image)
        input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model


        # move the input and model to GPU for speed if available
        if torch.cuda.is_available():
            input_batch = input_batch.to('cuda')
            self.model.to('cuda')

        with torch.no_grad():
            self.model.eval()
            output = self.model(input_batch)
        # Tensor of shape 1000, with confidence scores over ImageNet's 1000 classes
        #print(output[0])
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        #print(probabilities)
        predicted_class = probabilities.argmax().item()
        class_name = self.labels[predicted_class]
        # display frame
        cv2.putText(image, class_name, (10,10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, 255)
        cv2.imshow(self._window, image)
        cv2.waitKey(1)

if __name__ == '__main__':
    # create the node
    node = CameraReaderNode(node_name='camera_reader_node')
    # keep spinning
    rospy.spin()
