from flask import Flask, request, render_template, send_file, Response
from werkzeug.utils import secure_filename
import io
from ultralytics import YOLO
import numpy as np
from PIL import Image
import cv2
import os
from datetime import datetime
import logging
from pymongo import MongoClient
from bson.json_util import dumps
from flask_cors import CORS
from bson import ObjectId

app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = 'uploads/'

# Configuração de logs 
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MONGO_URI = 'mongodb+srv://userDB:diva739GJçq@formulario.oe2v2.mongodb.net/?retryWrites=true&w=majority&appName=formulario'



try:
    client = MongoClient(MONGO_URI)
    client.admin.command('ping')
    logger.info("Conexão estabelecida com o MongoDB Atlas ")
except Exception as e:
    logger.error('Erro ao conectar ao MongoDB Atlas'.format(e))

db = client ['formulario']
collection = db['dados_inspeccao']




# Certifique-se de que a pasta de uploads existe
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])


class Detection:
    def __init__(self):
        # Baixe os pesos do YOLO e altere o caminho conforme necessário
        self.model = YOLO(r"yolov11_custom.pt")

    def predict(self, img, classes=[], conf=0.5):
        if classes:
            results = self.model.predict(img, classes=classes, conf=conf)
        else:
            results = self.model.predict(img, conf=conf)
        return results

    
    def predict_and_detect(self, img, classes=[], conf=0.5, rectangle_thickness=2, text_thickness=1):
        results = self.predict(img, classes, conf=conf)
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                class_name = result.names[class_id]

                # Define a cor do quadrado com base no status do objeto
                if class_name == "Intacto":
                    box_color = (0, 255, 0)  # Verde para objetos intactos
                else:
                    box_color = (255, 0, 0)  # Vermelho para outros objetos

                # Desenha o quadrado ao redor do objeto
                cv2.rectangle(img, (int(box.xyxy[0][0]), int(box.xyxy[0][1])),
                            (int(box.xyxy[0][2]), int(box.xyxy[0][3])), box_color, rectangle_thickness)

                # Adiciona o nome da classe acima do quadrado
                cv2.putText(img, f"{class_name}",
                            (int(box.xyxy[0][0]), int(box.xyxy[0][1]) - 10),
                            cv2.FONT_HERSHEY_PLAIN, 1, box_color, text_thickness)
        return img, results   
    
    
    
    
    
    
    
    
    
    
    
    def detect_from_image(self, image):
        result_img, results = self.predict_and_detect(image, classes=[], conf=0.5)
        return result_img, results

    def detect_from_video_frame(self, frame):
        result_img, _ = self.predict_and_detect(frame, classes=[], conf=0.5)
        return result_img


detection = Detection()


@app.route('/')
def index():
    return render_template('teste.html')


@app.route('/object-detection/', methods=['POST'])
def apply_detection():
    print("Requisição recebida em /object-detection")  # Log para depuração
    if 'image' not in request.files:
        return 'No file part', 400

    file = request.files['image']
    
    if file.filename == '':
        return 'No selected file', 400

    if file:
        try:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Processar a imagem
            img = Image.open(file_path).convert("RGB")
            img = np.array(img)
            img = cv2.resize(img, (512, 512))
            img, results = detection.detect_from_image(img)
            
            # Contar objetos intactos e danificados
            intact_count = 0
            damaged_count = 0
            for result in results:
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    class_name = result.names[class_id]
                    confidence = float(box.conf[0])
                    if class_name == "Intacto":
                        intact_count += 1
                    elif class_name == "Danificado":
                        damaged_count += 1

                    # Salvar cada objeto detectado no MongoDB
                    detection_data = {
                        "filename": filename,
                        "detected_object": class_name,
                        "confidence": confidence,
                        "timestamp": datetime.now(),
                        "status": "intacto" if class_name == "Intacto" else "defeito"
                    }
                    collection.insert_one(detection_data)

            # Converter a imagem processada para PNG
            output = Image.fromarray(img)
            buf = io.BytesIO()
            output.save(buf, format="PNG")
            buf.seek(0)

            # Remover arquivo original
            os.remove(file_path)

            # Retornar a imagem processada
            return send_file(buf, mimetype='image/png')

        except Exception as e:
            print(f"Erro ao processar a imagem: {str(e)}")
            if 'file_path' in locals() and os.path.exists(file_path):
                os.remove(file_path)
            return f"Erro interno: {str(e)}", 500
    
@app.route('/video')
def index_video():
    return render_template('video.html')


def gen_frames():
    cap = cv2.VideoCapture(0)  # Use 0 para a câmera padrão
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Redimensiona o frame para 512x512
        frame = cv2.resize(frame, (512, 512))

        # Aplica a detecção de objetos no frame
        frame = detection.detect_from_video_frame(frame)

        # Converte o frame para JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue

        # Converte o buffer para bytes
        frame_bytes = buffer.tobytes()

        # Retorna o frame no formato de streaming
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    # Libera a câmera quando o loop termina
    cap.release()


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/api-info', methods=['GET'])
def get_inspection_data():
    try:
        # Consultar os dados no MongoDB
        inspections = list(collection.find({}))  # Remove {'_id': 0}
        logger.info(f"Dados encontrados: {len(inspections)} registros")

        # Formatar os dados para JSON
        for inspection in inspections:
            inspection['_id'] = str(inspection['_id'])  # Converte ObjectId para string
            inspection['timestamp'] = inspection['timestamp'].isoformat()  # Converte datetime para string ISO

        return dumps(inspections), 200  # Retorna os dados em formato JSON
    
    except Exception as e:
        logger.error(f"Erro ao buscar dados: {str(e)}")
        return f"Erro ao buscar dados: {str(e)}", 500




if __name__ == '__main__':
    app.run(host="0.0.0.0", port=17000, debug=True)