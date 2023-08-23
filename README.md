# <img src="logo-usu.png" alt="Logo" width="25px"/> Math Symbol Detection Web Apps

This application enables the detection of math symbols within mathematical expressions. It utilizes object detection techniques to identify and locate math symbols present in the expressions.
<br>
app demo : https://drive.google.com/file/d/1d-wkoeIVz26g9aiM9Yq2YH8JGK2MMEsD/preview 
<br>
This web-based application utilizes the Faster R-CNN ResNet50 model trained with TensorFlow Object Detection API to perform object detection. It operates by performing inference from a Saved Model and can be accessed directly through a web browser using Flask as the server. The application identifies objects in images and provides predictions of symbols along with their corresponding bounding boxes. 
<br>
The application is developed as the final thesis project for the Information Technology program at the Universitas Sumatera Utara by Meina Lisa 191402032.


## Table of Contents
- [Math Symbol Detection Web Apps](#math-symbol-detection-web-apps)
  - [Table of Contents](#table-of-contents)
  - [How to Install and Use](#how-to-install-and-use)
    - [Install](#install)
    - [Use](#use)
    - [Custom Detection](#custom-detection)
  - [Contributions](#contributions)


## How to Install and Use

### Install
open command prompt, then run `pip install -r requirement.txt`

### Use
1. Run `python app.py` or `flask run`
2. Open http://127.0.0.1:5000/ in browser

### Custom Detection
1. Train your model using TensorFlow Object Detection API https://github.com/tensorflow/models
2. Export the trained model as a Saved Model and save the corresponding label_map.pbtxt file
3. Replace the existing Saved Model folder with the newly exported one


## Contributions
If you want to contribute, please email meinalisa02@gmail.com.
