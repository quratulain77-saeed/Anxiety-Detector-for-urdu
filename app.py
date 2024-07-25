import os
import io
import json
import torch
import base64
import requests
from utils import preprocess
from database import db, Chat
from datetime import datetime
import torch.nn.functional as F
from transformers import pipeline
import soundfile as sf
from flask import (
    Flask, 
    request, 
    jsonify,
    render_template, 
)
from transformers import (
    VitsModel,
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

messages = []

system_message = "آپ ایک اے آئی انزائٹی تھراپسٹ ہیں جس کا نام اوللاما ہے۔ آپ کا کام ان لوگوں کو تسلی دینا ہے جو انزائٹی کا شکار ہیں۔ آپ کا مقصد انہیں پرسکون کرنا اور بات چیت کے ذریعے مدد فراہم کرنا ہے۔ اگر وہ مشورہ طلب کریں، تو انہیں بہترین مشورہ دیں۔ آپ صرف اردو میں بات کریں گے۔ ایک بھی لفظ یا حرف انگریزی میں نہ ہو۔ کسی بھی صورت میں انگریزی کا استعمال نہ کریں۔ انگریزی میں بات کرنے سے مریض کو شدید تکلیف ہوگی، اور آپ یہاں مدد کرنے کے لیے ہیں نہ کہ تکلیف دینے کے لیے۔ یہ انتہائی ضروری ہے کہ آپ ہر صورت میں صرف اردو میں جواب دیں۔ صارف اردو میں لکھے گا، اور آپ اردو میں ہی جواب دیں گے۔ یہ نہ بتائیں کہ آپ ایک اے آئی ماڈل ہیں۔ آپ کو ایک انسان کی طرح بات کرنی ہے۔"

BASE_DIR = 'audio_files/'
min_val = 0
max_val = 100
TTS_SAMPLE_RATE = 18_000

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

stt_model_name = 'openai/whisper-small'
anxiety_model_name = 'model/xlm-roberta-urdu'
tts_model_name = 'facebook/mms-tts-urd-script_arabic'

pipe = pipeline("automatic-speech-recognition", model=stt_model_name, device=device)
tokenizer = AutoTokenizer.from_pretrained(anxiety_model_name)
tts_tokenizer = AutoTokenizer.from_pretrained(tts_model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    anxiety_model_name,
    num_labels=2,
    ignore_mismatched_sizes=True,
).to(device)
tts_model = VitsModel.from_pretrained(tts_model_name).to(device)

def create_app():
    app = Flask(__name__)
    app.config.from_object('config.Config')
    db.init_app(app)

    os.makedirs(os.path.dirname(app.config['SQLALCHEMY_DATABASE_URI'].replace('sqlite:///', '')), exist_ok=True)

    with app.app_context():
        db.create_all()

    OLLAMA_API_URL = "http://127.0.0.1:11434/api/chat"

    @app.route('/')
    def home():
        return render_template('chat.html')

    @app.route('/save_audio', methods=["POST"])
    def save_audio():
        if 'audio_data' in request.files:
            file = request.files['audio_data']
            filename = datetime.now().strftime("%m-%d-%Y-%H-%M-%S") + '.wav'

            os.makedirs(BASE_DIR, exist_ok=True)

            file.save(os.path.join(BASE_DIR, filename))
            transcription = transcribe(os.path.join(BASE_DIR, filename))['text']

            return jsonify({"status": True, "transcription": transcription})
        return jsonify({"status": False}), 400

    def transcribe(filename):
        transcription = pipe(filename, generate_kwargs={'language': 'urdu'})
        return transcription

    def calculate_angle(value):
        value = max(min_val, min(max_val, value))
        angle = (value - min_val) / (max_val - min_val) * 180
        return angle

    def get_anxiety_level():
        texts = []
        for i in messages[-10:]:
            if i['role'] == 'user':
                text = i['content']
                clean_text = preprocess(text)
                texts.append(clean_text)

        if not texts:
            return calculate_angle(0)

        input_ids = tokenizer(texts, max_length=512, padding=True, truncation=True, return_tensors="pt").to(device)
        logits = model(**input_ids).logits
        out = F.softmax(logits, dim=1)[:, 1] * 100
        value = out.mean().item()

        return calculate_angle(value)

    @app.route('/speak', methods=["POST"])
    @torch.no_grad()
    def speak():
        msg = request.json.get('message')
        inputs = tts_tokenizer(msg, return_tensors="pt").to(device)
        output = tts_model(**inputs).waveform

        waveform_np = output.squeeze().cpu().numpy()
        audio_bytes = io.BytesIO()
        sf.write(audio_bytes, waveform_np, samplerate=TTS_SAMPLE_RATE, format='WAV')
        audio_bytes.seek(0)
        
        audio_base64 = base64.b64encode(audio_bytes.read()).decode('utf-8')

        return jsonify({"status": True, "audio": audio_base64})

    @app.route('/chat', methods=['POST'])
    def chat():
        user_message = request.json.get('message')
        if not user_message:
            return jsonify({'error': 'No message provided'}), 400

        payload = {
            "model": "llama3",
            "messages": messages
        }

        if not messages:
            payload["messages"].append({
                "role": "system",
                "content": system_message
            })

        payload['messages'].append({
            "role": "user",
            "content": user_message
        })

        messages.append({
            "role": "user",
            "content": user_message
        })

        response = requests.post(OLLAMA_API_URL, json=payload)

        if response.status_code != 200:
            return jsonify({'error': 'Error communicating with the AI model'}), 500

        response_content_decoded = response.content.decode('utf-8')

        chunks = response_content_decoded.strip().split('\n')
        full_text = ""

        for chunk in chunks:
            data = json.loads(chunk)
            content = data.get('message', {}).get('content', '')
            if content.endswith(('.', '!', '?', ',')):
                full_text += content + " "
            else:
                full_text += content

        bot_message = full_text.strip()
        
        messages.append({
            "role": "assistant",
            "content": bot_message
        })

        chat = Chat(user_message=user_message, bot_response=bot_message)
        db.session.add(chat)
        db.session.commit()

        return jsonify({
            'message': bot_message,
            'angle': get_anxiety_level()
        })

    return app

if __name__ == '__main__':
    app = create_app()
    app.run(debug=True)
