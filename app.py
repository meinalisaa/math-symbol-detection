import os, sys, io, csv, base64, ast, urllib.parse, json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for
from object_detection.utils import label_map_util, visualization_utils as viz_utils
from PIL import Image
from io import BytesIO
from markupsafe import Markup
from flask_cors import CORS

# Flask app setup
app = Flask(__name__)
CORS(app)

# Define Paths
current_dir = os.getcwd()
sys.path.append(current_dir)
PATH_TO_SAVED_MODEL = os.path.join(current_dir, "saved_model")
PATH_TO_LABELS = os.path.join(PATH_TO_SAVED_MODEL, "saved_model.pbtxt")
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)
detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)
print('Loading model...', end='')


@app.route('/')
def index():
    return render_template('beranda.html')

@app.route('/tutorial', methods=['GET'])
def tutorial():
    return render_template('tutorial.html')

@app.route('/deteksi', methods=['GET'])
def deteksi():
    return render_template('deteksi.html')

@app.route('/inference', methods=['POST'])
def inference():
    print("Inference triggered")
    image_data = request.form['image_data']
    image_data = image_data.split(',')[1] 

    image = Image.open(BytesIO(base64.b64decode(image_data)))

    image_np = np.array(image)
    input_tensor = tf.convert_to_tensor(image_np)[tf.newaxis, ...]

    # Perform object detection
    detections = detect_fn(input_tensor)
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy() for key, value in detections.items()}
    detections['num_detections'] = num_detections
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    # Define the class IDs to exclude
    exclude_ids = [1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 24, 26, 27, 29, 31, 33, 36, 37, 38, 39, 40, 41, 42, 46, 54, 64, 65, 66, 67, 68, 69, 74, 78, 83, 86, 87, 90, 91, 93, 95, 96, 98, 99, 101, 102, 108, 113, 114, 116, 118, 126, 130, 134]

    # Filter out the excluded class IDs from the detection results
    indices_to_keep = [i for i, cls_id in enumerate(detections['detection_classes']) if cls_id not in exclude_ids]
    detections['detection_boxes'] = detections['detection_boxes'][indices_to_keep]
    detections['detection_classes'] = detections['detection_classes'][indices_to_keep]
    detections['detection_scores'] = detections['detection_scores'][indices_to_keep]

    # Prepare data for rendering in HTML template
    detection_results = []
    for detection_class, detection_score, detection_box in zip(detections['detection_classes'], detections['detection_scores'], detections['detection_boxes']):
        if detection_score >= 0.50:
            class_name = category_index[detection_class]['name']
            ymin, xmin, ymax, xmax = detection_box
            result = {
                'class': detection_class,
                'class_name': class_name,
                'score': f'{detection_score:.4f}',
                'ymin': f'{ymin:.4f}',
                'xmin': f'{xmin:.4f}',
                'ymax': f'{ymax:.4f}',
                'xmax': f'{xmax:.4f}'
            }
            detection_results.append(result)

    # Visualize detections
    image_np_with_detections = image_np.copy()
    viz_utils.visualize_boxes_and_labels_on_image_array(
        image_np_with_detections,
        detections['detection_boxes'],
        detections['detection_classes'],
        detections['detection_scores'],
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=200,
        min_score_thresh=.50,
        agnostic_mode=False
    )

    # Normalize and save the image with detections
    image_np_with_detections_normalized = colors.Normalize()(image_np_with_detections)
    image_path = os.path.join(current_dir, 'static', 'detection_result.jpg')
    plt.imsave(image_path, image_np_with_detections_normalized)

    # Serialize the detection_results as a string
    detection_results_str = urllib.parse.quote(str(detection_results))

    print(detection_results_str)
    print("Done inference, go to the result page")
    return redirect(url_for('result', detection_results=detection_results_str))


@app.route('/result')
def result():
    print("Calling /result")
    detection_results_str = request.args.get('detection_results')

    # Deserialize the detection_results string back into a list
    detection_results = ast.literal_eval(urllib.parse.unquote(detection_results_str))

    # Add cache-control header to prevent caching
    headers = {
        'Cache-Control': 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0',
        'Pragma': 'no-cache',
        'Expires': '0'
    }

    data_path = os.path.join(current_dir, "static/symbols-data.txt")
    data_row = []
    with open(data_path, 'r') as txt_file:
        reader = csv.reader(txt_file, delimiter='\t')
        for row in reader:
            processed_row = []
            for cell in row:
                if '&' in cell:
                    cell = Markup(cell)
                else:
                    cell = Markup.escape(cell)
                processed_row.append(cell)
            data_row.append(processed_row)

    print("Done /result")
    # print("Detection result = ", detection_results)
    # print("Data row = ", data_row)
    return render_template('result.html', detection_results=detection_results, data_row=data_row), 200, headers

@app.route('/informasi', methods=['GET'])
def informasi():
    data_path = os.path.join(current_dir, "static/symbols-data.txt")
    data_row = []
    with open(data_path, 'r') as txt_file:
        reader = csv.reader(txt_file, delimiter='\t')
        for row in reader:
            processed_row = []
            for cell in row:
                if '&' in cell:
                    cell = Markup(cell)
                else:
                    cell = Markup.escape(cell)
                processed_row.append(cell)
            data_row.append(processed_row)

    return render_template('informasi.html', data_row=data_row)


@app.route('/image')
def serve_image():
    image_path = os.path.join(current_dir, 'static', 'detection_result.jpg')
    return send_file(image_path, mimetype='image/jpeg')


if __name__ == '__main__':
    app.run(debug=True, port=5000, host='0.0.0.0')
    # app.run(debug=True)
    # app.run(debug=True, host= '192.168.43.64')
    # app.run(debug=True, port=5000, host='116.206.31.79')
    # Navigate to http://[your-ipv4-address]:5000 from another machine in the same network, and you should be able to access your app.

