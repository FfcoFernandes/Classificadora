from flask import Flask, request, jsonify, Response, send_file
from flask_cors import CORS
import os
from model import SimpleCNN as CNN
from io import BytesIO

print("Carregando modelo de I.A...")
model = CNN()

app = Flask(__name__)
CORS(app)
    
@app.route('/classificar', methods=['POST'])
def classificar():
    if 'file' not in request.files:
        return "Nenhuma imagem encontrada", 400
    
    file = request.files['file']

    try:
        imagem = BytesIO(file.read())
        classificacao, pctg = model.classificar_imagem(imagem)

        return jsonify({"classificacao" : classificacao, "pctg" : pctg}), 200
    except Exception as e:
        return f'Erro ao classificar a imagem: {str(e)}', 500

if __name__ == '__main__':
    app.run(debug=True)