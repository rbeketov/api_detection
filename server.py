from flask import Flask, request
from PIL import Image
import io

app = Flask(__name__)

@app.route('/upload', methods=['POST'])
def upload():
    try:
        image_data = request.data
        image = Image.open(io.BytesIO(image_data))
        image.show()
        return "Image received and displayed!"
    except Exception as e:
        return str(e), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
