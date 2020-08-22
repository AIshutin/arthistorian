from flask import jsonify
from PIL import Image
import os
os.system('mkdir tmp')

def resize_img(img, size=224):
    width, height = img.size
    scale_f = max(width, height) / size
    width_new = int(width / scale_f)
    height_new = int(height / scale_f)
    img = img.resize((width_new, height_new))
    return img

def do_something(img):
    return {
        'geo': 'France',
        'date': 'XVI - XVII',
        'important': [
            {'bbox': [0, 0, 24, 24], 'reason': 'obvious'},
            {'bbox': [50, 50, 74, 74], 'reason': 'obvious'}
        ],
        'is_like': [
            'Rafael',
            'Michelangelo',
            'Russo'
        ]
    }

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
    app = Flask(__name__)

    @app.route('/', methods=["POST"])
    def process():
        return main(request)

    app.run(port=8080, debug=True)
