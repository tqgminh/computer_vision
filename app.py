from flask import Flask, render_template, request, redirect, send_from_directory, url_for
import os
import cv2
from PIL import Image
import PIL
import pathlib
from clustering_based.algorithms import kmeans, mean_shift
from region_based.algorithms import auto_seeded_region_growing
from mask_rcnn.model import mask_rcnn_segmentation
from unet.model import unet_segmentation

app = Flask(__name__)

app.config["UPLOAD_FOLDER"] = "static/upload"
app.config["ANNOTATION_FOLDER"] = "static/annotation"


@app.route("/")
def main():
    return render_template("index.html")


def create_new_folder(local_dir):
    newpath = local_dir
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    return newpath


@app.route("/", methods=["POST"])
def uploader():
    try:
        if request.method == "POST":

            file = request.files["file"]
            if file.filename == "":
                return redirect(request.url)

            file_name = file.filename
            img_path = app.config['UPLOAD_FOLDER'] + "/" + file_name
            file.save(img_path)

            res_img = mask_rcnn_segmentation(img_path)
            res_path = app.config['ANNOTATION_FOLDER'] + "/" + file_name
            cv2.imwrite(res_path, res_img)

            return render_template("index.html", res_path=res_path, img_path=img_path)

    except:
        return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
