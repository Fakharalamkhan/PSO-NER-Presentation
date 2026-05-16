from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import nlp_engine

app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/health')
def health():
    return jsonify({"status": "ok", "model": "en_core_web_sm"})

@app.route('/analyse', methods=['POST'])
def analyse():
    data = request.get_json()
    if not data or 'sentence' not in data or 'task' not in data:
        return jsonify({"error": "Invalid request, missing sentence or task"}), 400
    
    sentence = data['sentence']
    task = data['task']
    
    try:
        if task == 'pos':
            result = nlp_engine.analyse_pos(sentence)
        elif task == 'ner':
            result = nlp_engine.analyse_ner(sentence)
        else:
            return jsonify({"error": "Unknown task. Must be 'pos' or 'ner'"}), 400
            
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(port=5000, debug=True)
