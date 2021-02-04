import object_detection_api
import os
from PIL import Image
from flask import Flask, request, Response
from string import Template
app = Flask(__name__)
STATIC_MAP_TEMPLATE = Template("""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>person activity tracker</title>
    <script src="https://webrtc.github.io/adapter/adapter-latest.js"></script>
    <style>
        video {
            position: absolute;
            top: 0;
            left: 0;
            z-index: -1;
        }
        canvas{
            position: absolute;
            top: 0;
            left: 0;
            z-index:1
        }
    </style>
</head>
<body>
<video id="myVideo" crossOrigin="anonymous" src="${place_name}" muted controls></video>
<script id="objDetect" src="/static/objDetectonMotion.js" data-source="myVideo"  ></script>
</body>
</html>
""")

# for CORS
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,POST') # Put any other methods you need here
    return response


@app.route('/')
def index():
    return Response(open('./static/home.html').read(), mimetype="text/html")


@app.route('/local')
def local():
    return Response(open('./static/local.html').read(), mimetype="text/html")



@app.route('/video',methods=['POST','GET'])
def remote():
    videourl = request.form['videourl']
    return(STATIC_MAP_TEMPLATE.substitute(place_name=videourl))


@app.route('/image', methods=['POST'])
def image():
    try:
        image_file = request.files['image']  # get the image
        image_object = Image.open(image_file)
        objects = object_detection_api.get_objects(image_object)
        return objects

    except Exception as e:
        print('POST /image error: %e' % e)
        return e


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0')

    
