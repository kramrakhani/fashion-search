from flask import Flask, render_template, request
from fashion_clip.fashion_clip import FashionCLIP
import numpy as np
import pandas as pd
from PIL import Image
import os
from flask import jsonify  # Add this import at the top of the file

app = Flask(__name__)

# Load the FashionCLIP model
fclip = FashionCLIP('fashion-clip')

# Load the dataset
subset = pd.read_csv("subset_data.csv")

# Load the precomputed embeddings
image_embeddings = np.load("image_embeddings.npy")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/search", methods=["POST"])
def search():
    search_query = request.form.get("search_query")
    text_embedding = fclip.encode_text([search_query], 32)[0]

    similarities = text_embedding.dot(image_embeddings.T)
    top_indices = np.argsort(similarities)[::-1][:5]

    results = []
    for i in top_indices:
        article_id = int(subset.iloc[i]["article_id"])  # cast to int
        image_path = os.path.join("static", "images", f"{article_id}.jpg")
        results.append({"id": article_id, "image": image_path})

    return jsonify(results)


if __name__ == "__main__":
    app.run(debug=True)
