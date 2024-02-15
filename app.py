from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
from transformers import pipeline
import pytesseract
import datetime

app = Flask(__name__)
CORS(app)

qa_pipeline = pipeline('question-answering', model='bert-large-uncased-whole-word-masking-finetuned-squad', tokenizer='bert-large-uncased-whole-word-masking-finetuned-squad')

@app.route('/now', methods=['GET'])
def get_current_time():
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return jsonify({"current_time": current_time})

@app.route('/process-image', methods=['POST'])
def process_image():
    if 'image' not in request.files:
            return jsonify({"error": "Missing image file"}), 400

    image_file = request.files['image']
    image = Image.open(image_file.stream)
    extracted_text = pytesseract.image_to_string(image)

    question_key_map = {
           'What is the State Name?': 'state',
           'What is the Full Name?': 'name',
           'What is the Address?': 'address',
           'What is the Height?': 'height',
           'What is the Gender?': 'gender',
           'What is the Date of Birth?': 'date_of_birth',
           'What is the Issued Date?': 'issued_date',
           'What is the Expiry Date?': 'expiry_date',
       }
    answers = {}
    for question, key in question_key_map.items():
        answer = qa_pipeline({'question': question, 'context': extracted_text})
        answers[key] = answer['answer']

    return jsonify(answers)


if __name__ == '__main__':
    app.run(debug=True)
