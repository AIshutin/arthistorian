from flask import jsonify
from PIL import Image
import os
from model import predict_date
os.system('mkdir tmp')

def resize_img(img, size=224):
    width, height = img.size
    scale_f = max(width, height) / size
    width_new = int(width / scale_f)
    height_new = int(height / scale_f)
    img = img.resize((width_new, height_new))
    return img

def do_something(img):
    return {'result': predict_date(img)}

def main(request):
    if len(request.files) == 0:
        return jsonify({
            'error': 'no "image" in files'
        })

    for key in request.files:
        file = request.files[key]
        key = os.path.basename(key)
        filepath = f'tmp/{key}'
        file.save(filepath)
        image = Image.open(filepath)
        image = resize_img(image)
        os.system(f'rm {filepath}')
        break

    result = do_something(image)
    return jsonify(result)

if __name__ == "__main__":
    import flask
    from flask import Flask, request
    from flask_cors import CORS
    app = Flask(__name__)
    CORS(app)


    @app.route('/', methods=["POST"])
    def process():
        return main(request)

    app.run(host="0.0.0.0", port=8080, debug=True)
