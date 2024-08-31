# # # from flask import Flask, request, jsonify
# # # from flask import send_file
# # # from werkzeug.utils import secure_filename
# # # import numpy as np
# # # import pickle
# # import tensorflow as tf
# # # from tensorflow.keras.preprocessing import image as ku
# # # import matplotlib.pyplot as plt
# # # import io

# # # app = Flask(__name__)

# # # # Load your model
# # # with open('knn.pkl', 'rb') as model_file:
# # #     model = pickle.load(model_file)

# # # # Define your labels
# # # labels = ['Overcrowded', 'coach_cleaniness', 'toilet_cleaniness', 'window_damaged']

# # # def preprocess_image(img_path):
# # #     img = ku.load_img(img_path, target_size=(300, 300))
# # #     img = ku.img_to_array(img, dtype=np.uint8)
# # #     img = np.array(img) / 255.0
# # #     return img
# # # @app.route('/')
# # # def HomePage():
# # #     return "<p> hello world</p>"




# # # @app.route('/predict', methods=['POST'])
# # # def predict():
# # #     if 'file' not in request.files:
# # #         return jsonify({'error': 'No file provided'}), 400

# # #     file = request.files['file']
# # #     if file.filename == '':
# # #         return jsonify({'error': 'No selected file'}), 400

# # #     filename = secure_filename(file.filename)
# # #     img_path = f'temp_{filename}'
# # #     file.save(img_path)

# # #     # Preprocess the image
# # #     img = preprocess_image(img_path)

# # #     # Make prediction
# # #     prediction = model.predict(img[np.newaxis, ...])
# # #     predicted_class = labels[np.argmax(prediction[0], axis=-1)]
    
# # #     probability = np.max(prediction[0], axis=-1)
# # #     return prediction

# # #     # Create an image with title
# # #     plt.axis('off')
# # #     plt.imshow(img.squeeze())
# # #     plt.title(f"Classified: {predicted_class}")
# # #     plt.savefig('result.png')

# # #     # Send the result image back to the client
# # #     return send_file('output.png', mimetype='image/png')

# # # if __name__ == '__main__':
# # #     app.run(debug=True)
# # from flask import Flask, request, jsonify
# # from flask_restful import Api, Resource
# # from PIL import Image
# # import numpy as np
# # import pickle
# # import tf.keras.utils as ku
# # import numpy as np

# # app = Flask(__name__)
# # api = Api(app)

# # # Load the model
# # with open('knn.pkl', 'rb') as f:
# #     model = pickle.load(f)

# # class Predict(Resource):
# #     def post(self):
# #         if 'file' not in request.files:
# #             return jsonify({'error': 'No file part'})

# #         file = request.files['file']

# #         if file.filename == '':
# #             return jsonify({'error': 'No selected file'})

# #         try:
# #             # Load and preprocess the image
# #             # image = Image.open(file)
# #             # image = image.convert('RGB')
# #             # image = image.resize((300, 300))  # Resize if needed
# #             # image_array = np.array(image)
# #             # image_array = image_array / 255.0  # Normalize if needed
# #             # image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
# #             # if image_array.shape != (1, 224, 224, 3):
# #             #     return jsonify({'error': f'Input shape mismatch: {image_array.shape}'})
# #             # Make prediction
            
# #             img = ku.load_img(file, target_size = (300,300))
# #             img = ku.img_to_array(img, dtype=np.uint8)
# #             img = np.array(img)/255.0
# #             prediction = model.predict(img[np.newaxis, ...])
# #             # label = np.argmax(prediction)  # Get the index of the highest score
            
            
# #             # Map the label to the class names
# #             # class_names = ['Overcrowded', 'coach_cleaniness', 'toilet_cleaniness', 'window_damaged']
# #             # result = class_names[label]

# #             # return jsonify({'prediction': result})
# #             return np.max(prediction[0], axis=-1)

# #         except Exception as e:
# #             return jsonify({'error': str(e)})

# # api.add_resource(Predict, '/predict')

# # if __name__ == '__main__':
# #     app.run(debug=True)



# # ashhutosh code

# # from flask import Flask, request, jsonify
# # from flask_restful import Api, Resource
# # from PIL import Image
# # import numpy as np
# # import pickle

# # app = Flask(__name__)
# # api = Api(app)

# # # Load the model
# # with open('knn.pkl', 'rb') as f:
# #     model = pickle.load(f)

# # # Define your labels
# # labels = ['Overcrowded', 'coach_cleaniness', 'toilet_cleaniness', 'window_damaged']

# # class Predict(Resource):
# #     def post(self):
# #         if 'file' not in request.files:
# #             return jsonify({'error': 'No file part'}), 400

# #         file = request.files['file']

# #         if file.filename == '':
# #             return jsonify({'error': 'No selected file'}), 400

# #         try:
# #             # Directly open the FileStorage object with PIL
# #             image = Image.open(file)
# #             image = image.convert('RGB')  # Convert image to RGB
# #             image = image.resize((300, 300))  # Resize to match model input size
# #             image_array = np.array(image) / 255.0  # Normalize pixel values
# #             image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

# #             # Make prediction
# #             prediction = model.predict(image_array)
# #             predicted_class_index = np.argmax(prediction[0], axis=-1)
# #             predicted_class = labels[predicted_class_index]
# #             probability = np.max(prediction[0], axis=-1)

# #             # Return prediction result
# #             return jsonify({
# #                 'prediction': predicted_class,
# #                 'probability': float(probability)
# #             })

# #         except Exception as e:
# #             return jsonify({'error': str(e)}), 500

# # api.add_resource(Predict, '/predict')

# # if __name__ == '__main__':
# #     app.run(debug=True)






# #Edited code




# from flask import Flask, request, jsonify
# from flask_restful import Api, Resource
# from PIL import Image
# import numpy as np
# import pickle

# app = Flask(__name__)
# api = Api(app)

# # Load the model from .pkl file
# with open('knn.pkl', 'rb') as f:
#     model = pickle.load(f)

# # Define your labels directly
# labels = {
#     0: 'Overcrowded',
#     1: 'coach_cleaniness',
#     2: 'toilet_cleaniness',
#     3: 'window_damaged'
# }

# class Predict(Resource):
#     def post(self):
#         if 'file' not in request.files:
#             return jsonify({'error': 'No file part'}), 400

#         file = request.files['file']

#         if file.filename == '':
#             return jsonify({'error': 'No selected file'}), 400

#         try:
#             # Open the image file
#             image = Image.open(file)
#             image = image.convert('RGB')
#             image = image.resize((300, 300))
#             image_array = np.array(image) / 255.0
#             image_array = np.expand_dims(image_array, axis=0)

#             # Make prediction
#             prediction = model.predict(image_array)
#             predicted_class_index = np.argmax(prediction[0], axis=-1)
#             predicted_class = labels[predicted_class_index]
#             probability = np.max(prediction[0], axis=-1)

#             return jsonify({
#                 'prediction': predicted_class,
#                 'probability': float(probability)
#             })

#         except Exception as e:
#             return jsonify({'error': str(e)}), 500

# api.add_resource(Predict, '/predict')

# if __name__ == '__main__':
#     app.run(debug=True, port=5050)

from flask import Flask, request, jsonify
from flask_restful import Api, Resource
from PIL import Image
import numpy as np
import pickle

app = Flask(__name__)
api = Api(app)

# Load the model from .pkl file
with open('knn.pkl', 'rb') as f:
    model = pickle.load(f)

# Define your labels dictionary directly
labels = {
    0: 'Overcrowded',
    1: 'coach_cleaniness',
    2: 'toilet_cleaniness',
    3: 'window_damaged'
}

class Predict(Resource):
    def post(self):
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400

        file = request.files['file']

        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        try:
            # Open the image file
            image = Image.open(file)
            image = image.convert('RGB')
            image = image.resize((300, 300))
            image_array = np.array(image) / 255.0
            image_array = np.expand_dims(image_array, axis=0)

            # Make prediction
            prediction = model.predict(image_array)
            predicted_class_index = np.argmax(prediction[0], axis=-1)
            predicted_class = labels[predicted_class_index]
            probability = np.max(prediction[0], axis=-1)

            return jsonify({
                'prediction': predicted_class,
                'probability': float(probability)
            })

        except Exception as e:
            return jsonify({'error': str(e)}), 500

api.add_resource(Predict, '/predict')

if __name__ == '__main__':
    app.run(debug=True, port=5001)