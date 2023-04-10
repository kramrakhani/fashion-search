from flask import Flask, render_template, request
from fashion_clip.fashion_clip import FashionCLIP
import numpy as np
import pandas as pd
from PIL import Image
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

    id_of_matched_object = np.argmax(text_embedding.dot(image_embeddings.T))
    found_object = subset["article_id"].iloc[id_of_matched_object]

    fixed_height = 224
    image = Image.open(f"data_for_fashion_clip/{found_object}.jpg")
    height_percent = (fixed_height / float(image.size[1]))
    width_size = int((float(image.size[0]) * float(height_percent)))
    image = image.resize((width_size, fixed_height), Image.NEAREST)
    image.save(f"static/images/{found_object}.jpg")

    image_url = f"static/images/{found_object}.jpg"
    return jsonify(image_url=image_url)


if __name__ == "__main__":
    app.run(debug=True)


# from flask import Flask, render_template, request
# from fashion_clip.fashion_clip import FashionCLIP
# import os
# import pandas as pd
# import numpy as np
# from PIL import Image

# # Initialize Flask app
# app = Flask(__name__)

# # Load the FashionCLIP model
# fclip = FashionCLIP('fashion-clip')

# articles = pd.read_csv("data_for_fashion_clip/articles.csv")

# # Load image embeddings
# image_embeddings = np.load("image_embeddings.npy")
# subset = pd.read_csv('subset_data.csv')


# @app.route("/", methods=["GET", "POST"])
# def index():
#     if request.method == "POST":
#         # Get the user input (text or image)
#         user_input = request.form["user_input"]

#         # Encode the user input
#         text_embedding = fclip.encode_text([user_input], 32)[0]

#         # Calculate similarities
#         similarities = text_embedding.dot(image_embeddings.T)
#         top_matches = np.argsort(similarities)[-5:][::-1]

#         # Get the matching items from the dataset
#         results = subset.iloc[top_matches]
#         print(results)

#         return render_template("results.html", results=results)

#     return render_template("index.html")


# if __name__ == "__main__":
#     app.run(debug=True)
