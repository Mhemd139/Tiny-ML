# Deep learning using Tensorflow Lite on Raspberry Pi

![alt text](https://github.com/Zaheer505/Deep-learning-using-Tensorflow-Lite-on-Raspberry-Pi/blob/main/images/thumbnail.png)


## About this Repository
This course is focused on Embedded Deep learning in Python . Raspberry PI 4 is utilized as a main hardware and we will be building practical projects with custom data .

- We will start with trigonometric functions approximation . In which we will generate random data and produce a model for Sin function approximation

- Next is a calculator that takes images as input and builds up an equation and produces a result .This Computer vision based project is going to be using convolution network architecture for Categorical classification

- Another amazing project is focused on convolution network but the data is custom voice recordings . We will involve a little bit of electronics to show the output by controlling our multiple LEDs using own voice .

- Unique learning point in this course is Post Quantization applied on Tensor flow models trained on Google Colab . Reducing size of models to 3 times and increasing inferencing speed up to 0.03 sec per input .

Note: This repo contains step by step approach to teach different things to students of our course. You may find some raw data / codes which are meant for learning purposes of students.


## Installations
- Laptop/PC Installations
    - Rpi-Imager for installing RPI OS on SD CARD
        ```
        sudo apt install rpi-imager
        ```
    - Tensorflow
        ```
        pip install tensorflow
        ```

- Raspberry PI 4 installations
    - Tensorflow Lite Interpreter
        ```
        python3 -m pip install tflite-runtime
        ```
    - Install tightvnc server
        ```
        sudo apt-get install tightvncserver
        ```
- Common Installations
    - OPENCV
        ```
        pip3 install opencv-python
        sudo apt-get install libcblas-dev
        sudo apt-get install libhdf5-dev
        sudo apt-get install libhdf5-serial-dev
        sudo apt-get install libatlas-base-dev
        sudo apt-get install libjasper-dev
        sudo apt-get install libqtgui4
        sudo apt-get install libqt4-test
        sudo apt-get install libatlas-base-dev
        ```
    - Upgrade Numpy
        ```
        pip install -U numpy
        ```
    - Audio processing Dependencies
        ```
        pip install sounddevice
        sudo apt-get install libportaudio2
        pip install scipy
        ```
----
## Using Repository
- SSH into your RPI
    ```
    ssh pi@<IP_of_RPI>
    ```
- Access RPI through TeamViewer on PC
---

![Equation GIF Demo](https://github.com/Mhemd139/Tiny-ML/blob/main/Number.gif)


![Equation GIF Demo](https://github.com/Mhemd139/Tiny-ML/raw/main/Equation.gif)

## Pre-Course Requirments
- PC   : Ubuntu 22.04
- RPI4 : RPI Full OS
    - SD-CARD 16GB
    - RPI Camera V2
    - Power Bank with Type C cable
    - 3D printed Parts for Camera Holding
    - Fan on RPI for better thermals


## Instructors
Muhammad Luqman - [Profile Link](https://www.linkedin.com/in/muhammad-luqman-9b227a11b/)
Zaheer Ahmed - [Profile Link](https://www.linkedin.com/in/zaheer-ahmed505/)

----
