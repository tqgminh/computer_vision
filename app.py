from flask import Flask, render_template, request, redirect
import os
import cv2
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

            method = request.form['method']
            res_img = 0
            note = ""
            if method == 'kmeans':
                numofclus = request.form['nums']
                numofclus = int(numofclus)
                res_img = kmeans(img_path, numofclus)
                note = "K-mean clustering with " + str(numofclus) + "clusters"
            elif method == 'meanshift':
                res_img = mean_shift(img_path)
                note = "Mean shift clustering"
            elif method == 'regiongrowing':
                res_img = auto_seeded_region_growing(img_path)
                note = "Region growing"
            elif method == 'unet':
                res_img = unet_segmentation(img_path)
                note = "Unet segmentation, green: dog, red: cat"
            elif method == 'maskrcnn':
                res_img = mask_rcnn_segmentation(img_path)
                note = "Mask-RCNN segmentation"
            else:
                return render_template("index.html")

            res_path = app.config['ANNOTATION_FOLDER'] + "/" + file_name
            cv2.imwrite(res_path, res_img)

            return render_template("result.html", res_path=res_path, img_path=img_path, note=note)

    except:
        return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
