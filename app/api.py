from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict_endpoint():
    data = request.get_json()
    text = data.get('text')

    return jsonify({
        'received_text': text,
        'message': 'Text extracted successfully'
    }), 200


if __name__ == '__main__':
    app.run(debug=True, port=5000)
