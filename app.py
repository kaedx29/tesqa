from flask import Flask, render_template, request, jsonify
from qa_train import *

app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/qa', methods=['POST'])
def question_answering():
    try:
        data = request.json

        if 'context' not in data or 'question' not in data:
            raise ValueError("Request tidak lengkap. Pastikan menyertakan 'konteks' dan 'pertanyaan'.")
        
        context = data['context']
        question = data['question']

        answer = predict_answer(context, question)

        print(answer)
        return jsonify({
            "success": True,
            "message": "Pertanyaan berhasil dijawab.",
            "answer": answer
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "message": str(e),
            "answer": None
        })

if __name__ == '__main__':
    app.run(debug=True)