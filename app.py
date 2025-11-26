# app.py
import os
import json
import numpy as np
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

# TensorFlow lazy loading - only loaded when needed
try:
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing import image
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("Warning: TensorFlow not available. Model prediction will be disabled.")

# --------------------------------------------------------------------
# FLASK SETUP
# --------------------------------------------------------------------
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = "static/uploads"
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
app.secret_key = "replace_with_a_random_secret"

# --------------------------------------------------------------------
# DATABASE SETUP
# --------------------------------------------------------------------
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///agroviz.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

class Post(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user = db.Column(db.String(100))
    title = db.Column(db.String(200))
    body = db.Column(db.Text)
    image = db.Column(db.String(200), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class Comment(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    post_id = db.Column(db.Integer, db.ForeignKey('post.id'))
    user = db.Column(db.String(100))
    text = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class Order(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    product_id = db.Column(db.Integer)
    qty = db.Column(db.Integer)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

with app.app_context():
    db.create_all()

# --------------------------------------------------------------------
# LOAD MODEL + SUPPORT FILES
# --------------------------------------------------------------------
if TENSORFLOW_AVAILABLE:
    MODEL = load_model("model/plant_disease_model.h5")
else:
    MODEL = None
    print("Model loading skipped - TensorFlow not available")

CLASS_INDICES = json.load(open("ml/class_indices.json"))
IDX_TO_CLASS = {v: k for k, v in CLASS_INDICES.items()}

DISEASE_INFO = json.load(open("data/disease_info.json"))
PRODUCTS = json.load(open("data/products.json"))

# --------------------------------------------------------------------
# ROUTES
# --------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if not TENSORFLOW_AVAILABLE:
        return "Model prediction is not available. Please install TensorFlow.", 503
    
    if request.method == "POST":
        if "image" not in request.files:
            return "No image uploaded", 400

        f = request.files["image"]
        if f.filename == "":
            return "No selected file", 400

        # Save uploaded image
        filename = secure_filename(f.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        f.save(filepath)

        # --------------------------------------------------------------
        # IMAGE PREPROCESSING — MUST MATCH MODEL INPUT (128x128)
        # --------------------------------------------------------------
        img = image.load_img(filepath, target_size=(128, 128))
        x = image.img_to_array(img) / 255.0
        x = np.expand_dims(x, axis=0)

        preds = MODEL.predict(x)[0]
        idx = int(np.argmax(preds))
        class_name = IDX_TO_CLASS[idx]
        confidence = float(preds[idx]) * 100

        # --------------------------------------------------------------
        # DISEASE INFO + RECOMMENDED PRODUCTS
        # --------------------------------------------------------------
        info = DISEASE_INFO.get(class_name, {
            "display_name": class_name,
            "description": "No information available.",
            "prevention": "No details.",
            "treatment": "No details.",
            "recommended_products": []
        })

        recommended = []
        for pname in info.get("recommended_products", []):
            for p in PRODUCTS:
                if p["name"] == pname:
                    recommended.append(p)

        return render_template(
            "result.html",
            img=filepath,
            info=info,
            products=recommended,
            conf=round(confidence, 2),
            class_name=class_name
        )

    return render_template("predict.html")

# --------------------------------------------------------------------
# SHOP
# --------------------------------------------------------------------
@app.route("/shop")
def shop():
    return render_template("shop.html", products=PRODUCTS)

@app.route("/buy/<int:product_id>")
def buy(product_id):
    order = Order(product_id=product_id, qty=1)
    db.session.add(order)
    db.session.commit()
    return redirect(url_for("shop"))

# --------------------------------------------------------------------
# COMMUNITY
# --------------------------------------------------------------------
@app.route("/community", methods=["GET", "POST"])
def community():
    if request.method == "POST":
        user = request.form.get("user", "Anonymous")
        title = request.form.get("title")
        body = request.form.get("body")

        img_file = request.files.get("image")
        imgpath = None
        if img_file and img_file.filename:
            fn = secure_filename(img_file.filename)
            imgpath = os.path.join(app.config["UPLOAD_FOLDER"], fn)
            img_file.save(imgpath)

        post = Post(user=user, title=title, body=body, image=imgpath)
        db.session.add(post)
        db.session.commit()
        return redirect(url_for("community"))

    posts = Post.query.order_by(Post.created_at.desc()).all()
    return render_template("community.html", posts=posts)

@app.route("/post/<int:post_id>", methods=["GET", "POST"])
def post_detail(post_id):
    post = Post.query.get_or_404(post_id)

    if request.method == "POST":
        user = request.form.get("user", "Anonymous")
        text = request.form.get("text")
        comment = Comment(post_id=post.id, user=user, text=text)
        db.session.add(comment)
        db.session.commit()
        return redirect(url_for("post_detail", post_id=post.id))

    comments = Comment.query.filter_by(post_id=post.id).order_by(Comment.created_at.asc()).all()
    return render_template("post_detail.html", post=post, comments=comments)

# --------------------------------------------------------------------
# RUN SERVER
# --------------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)
