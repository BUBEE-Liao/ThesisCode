from model import *
import torch
from safetensors import safe_open
from transformers import AutoFeatureExtractor, Wav2Vec2BertModel
from transformers import MarianMTModel, MarianTokenizer

device = torch.device('cuda')
# 定義 preprocess 要用的模組
model_name = "Helsinki-NLP/opus-mt-zh-en"
tokenizer = MarianTokenizer.from_pretrained(model_name)
processor = AutoFeatureExtractor.from_pretrained("facebook/w2v-bert-2.0")
# 模型宣告，初始化
model = SpeechToTextTranslationModel()
# 載入訓練好的模型參數
tensors = {}
ckpt_path = "/home/bubee/Thesis_code/s2t/ctc_share_freeze_ckpt/checkpoint-200000/model.safetensors" # need to fill in your own ckpt_path
with safe_open(ckpt_path, framework="pt", device=0) as f:
    for k in f.keys():
        tensors[k] = f.get_tensor(k)
model.load_state_dict(tensors)
# 將模型轉到 GPU 上，否則會在 CPU 上跑，很慢
model = model.to(device)
model.eval()

# the audio you want to translate
audio_path = '/home/bubee/data/nas05/dataset/AISHELL-3/test/wav/SSB0693/SSB06930003.wav'

# load wav
audio, sample_rate = librosa.load(audio_path, sr=16000)
# resample wav to sample rate 16k
audio = librosa.resample(y=audio, orig_sr=sample_rate, target_sr=16000)
# preprocess
inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
# get preprocess result
inputs = inputs['input_features'].to(device)

# model prediction
speech_encoder_output, _, _= model.forward_speech(inputs)
res, ids = model.inference(encoder_hidden_state=speech_encoder_output, tokenizer=tokenizer, device=device)
print(res[0])