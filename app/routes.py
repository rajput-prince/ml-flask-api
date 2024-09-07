import os
from flask import Blueprint, request, render_template
from PIL import Image
import torchvision.transforms as transforms
from ultralytics import YOLO

bp = Blueprint('main', __name__)

# Load the Ultralytics YOLO model
model = YOLO('model/injury_prediction.pt')  # Use your path to the YOLO model

# Define image transformation
transform = transforms.Compose([
    transforms.Resize((640, 640)),  # YOLO models expect larger input images
    transforms.ToTensor(),
])

@bp.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file part in the request'
        
        file = request.files['file']
        if file.filename == '':
            return 'No selected file'
        
        if file:
            filepath = os.path.join('uploads', file.filename)
            file.save(filepath)

            # Perform the prediction using the YOLO model
            results = model.predict(filepath)

            # Check if any objects were detected
            if len(results[0].boxes) > 0:
                # Extract the class and confidence from the results
                predicted_class = results[0].names[int(results[0].boxes.cls[0])]
                confidence = results[0].boxes.conf[0]
                return render_template('index.html', prediction=f'{predicted_class} ({confidence:.2f})')
            else:
                # Handle case where no objects were detected
                return render_template('index.html', prediction='No injury detected in the image.')

    return render_template('index.html', prediction=None)
