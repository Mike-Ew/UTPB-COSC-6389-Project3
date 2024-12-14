# UTPB-COSC-6389-Project3
This repo contains the assignment and provided code base for Project 3 of the graduate Computational Biomimicry class.

The goals of this project are:
1) Understand how convolutional neural networks are constructed and used, and the particulars of their implementation.

Description:
Using the code from your Project 2, create an extension which implements convolutions in your networks.  This time, your goal is to create an image classifier network, which accepts single rectangular images as input and outputs which of the object classes the network believes the image depicts.  You are allowed to select "toy" problems for this, such as the famous handwritten numerical digit dataset (example available here: https://www.kaggle.com/datasets/jcprogjava/handwritten-digits-dataset-not-in-mnist).

This site (https://www.kaggle.com/datasets) seems to be a database of datasets that you will likely find useful for both projects 2 and 3.

As with Projects 1 and 2, your application must generate the neural networks, display them on a canvas, and update them in real time as the weight values change.

You are not allowed to make use of any libraries related to neural networks in the code for this project.  The implementation of the network construction, operation, forward and backward propagation, training, and testing must all be your own.

Grading criteria:
1) If the code submitted via your pull request does not compile, the grade is zero.
2) If the code crashes due to forseeable unhandled exceptions, the grade is zero.
3) For full points, the code must correctly perform the relevant algorithms and display the network in real time, via the UI.

Deliverables:
A Python application which provides a correct implementation of a neural network generation and training system and is capable of training an image classifier which has good accuracy for the problem set selected.
