from flask import Flask, request, make_response, render_template, jsonify
import os
import io
import subprocess

IMAGE_FOLDER = os.path.join('static', 'image')

# Initialize Flask app
app = Flask(__name__)
app.config['IMAGE_FOLDER'] = IMAGE_FOLDER

# Define a route for your web page
@app.route("/")
def index():
    full_image_name = os.path.join(app.config['IMAGE_FOLDER'], 'out.jpg')
    return render_template("upload.html", current_image = full_image_name)  # Render HTML template

@app.route("/chatgpt", methods=["GET"])
def chatgpt():
    return render_template("chat_gpt.html") 

@app.route("/upload", methods=["POST"])
def upload():
    if 'image' not in request.files:
        return "No image selected!"

    # Get the uploaded image file
    image_file = request.files['image']

    # Check if a valid image file
    if image_file.filename.lower().endswith(('.png', '.jpg', '.gif', '.jpeg')):

        import numpy as np
        import pandas as pd
        from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
        from keras.utils import to_categorical
        from sklearn.model_selection import train_test_split
        import matplotlib.pyplot as plt
        import random
        import os

        from keras.models import Sequential
        from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, BatchNormalization

        model = Sequential()

        model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(128, (3, 3), activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        model.add(Dense(2, activation='softmax'))  # 2 because we have cat and dog classes

        model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

        model.load_weights("model.h5")


        # Save the image to a designated folder (implement security checks)
        image_data = io.BytesIO(image_file.read())
        from PIL import Image  # Install Pillow library: pip install Pillow
        image = Image.open(image_data)
        flipped_image = image.transpose(Image.FLIP_LEFT_RIGHT)  # Flip horizontally

        test_image = image.resize((128, 128))

        test_image = test_image.convert('RGB')

        test_image.save('./static/image/output.jpg')

        test_filenames = os.listdir("./static/image")
        test_df = pd.DataFrame({
            'filename': test_filenames
        })
        nb_samples = test_df.shape[0]

        test_gen = ImageDataGenerator(rescale=1. / 255)
        test_generator = test_gen.flow_from_dataframe(
            test_df,
            "./static/image/",
            x_col='filename',
            y_col=None,
            class_mode=None,
            target_size=(128, 128),
            batch_size=15,
            shuffle=False
        )

        predict = model.predict(test_generator, steps=int(np.ceil(nb_samples / 15)))
        test_df['category'] = np.argmax(predict, axis=-1)

        ###############
        submission_df = test_df.copy()
        submission_df['id'] = submission_df['filename'].str.split('.').str[0]
        submission_df['label'] = submission_df['category']
        submission_df.drop(['filename', 'category', 'id'], axis=1, inplace=True)

        result_value = submission_df.values[0, 0]

        if result_value == 1:
            result = "Dog"
        elif result_value == 0:
            result = "Cat"
        else:
            result = "Invalid value"

        full_image_name = os.path.join(app.config['IMAGE_FOLDER'], 'output.jpg')
        return render_template("upload.html", current_image = full_image_name, classification = result)
    else:
        return "Invalid image format!"

@app.route("/app", methods=["GET"])
def myApp():
    message = request.args.get("message")
    return render_template("chat_gpt.html", msg=message)

@app.route("/process", methods=["GET"])
def proc():
    try:
        process = subprocess.run(['pip', 'install', 'subprocess'], capture_output=True, text=True)
        output = process.stdout
    except subprocess.CalledProcessError as e:
        output = f"Error"
    return render_template("chat_gpt.html", msg=output)
                           

# Define a route for processing user input (replace with your PyTorch logic)
@app.route("/predict", methods=["GET"])
def predict():
    # Get user input from the form (replace with your form field names)
    #data = request.form.get("data")

    # Preprocess data (convert to tensor, etc.)
    # ... your PyTorch processing here ...

    # Generate prediction using PyTorch model
    prediction = "test" # Your PyTorch model prediction logic

    return jsonify({'user_id': 'bseo', 'prediction': {
      'gender' : 'male', 'age': 30  
    }})

if __name__ == "__main__":
    app.run(debug=True)  # Run the Flask app in debug mode
