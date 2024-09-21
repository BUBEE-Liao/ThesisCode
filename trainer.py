import torch
import numpy as np
from model import SpeechToTextTranslationModel
from transformers import AutoFeatureExtractor, MarianTokenizer
from transformers import AutoProcessor, Wav2Vec2Model
from transformers import Trainer, TrainingArguments
import librosa
import torch.nn as nn
from datasets import load_metric
import random
metric = load_metric("sacrebleu")
import os

# load 前處理模組，tokenizer 處理文字端， processor 處理語音端
tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-zh-en")
processor = AutoFeatureExtractor.from_pretrained("facebook/w2v-bert-2.0")

# 類似 dataloader，將傳進來的 data list 做前處理, 在 training 的時候餵給模型做 forward
# data list 是在下面的 line 85 處理的
def data_collator(features: list):
    path = [f[0] for f in features] # audio path
    src_txt = [f[1] for f in features] # zh
    tgt_txt = [f[2] for f in features] # en
    ### audio 前處理
    # load wav
    audio = [load_wav(audio_path) for audio_path in path]
    # preprocess
    audio_feature = processor(audio, sampling_rate=16000, return_tensors="pt", return_attention_mask=True, padding='longest')
    # get preprocess result
    audio_feature_padded = audio_feature['input_features']
    # get audio attention mask
    audio_feature_mask = audio_feature['attention_mask']
    # src text
    ### src text(中文) 前處理
    src_txt_feature = tokenizer(src_txt, add_special_tokens=False, return_tensors="pt", padding='longest')
    # get src text preprocess result
    src_txt_padded = src_txt_feature['input_ids']
    # get src text attention mask
    src_txt_mask = src_txt_feature['attention_mask']
    ### target text(英文) 前處理
    decoder_input_ids, decoder_input_mask, label = generate_decoder_input_and_label(tgt_txt)
    # huggingface trainer 規定回傳的結果要是dictionary, 而且參數的數量跟順序要跟 model forward 傳入的參數一模一樣
    return {'text_input_ids':src_txt_padded, 'text_attention_mask':src_txt_mask, 'speech_input':audio_feature_padded, \
            'speech_attention_mask':audio_feature_mask, 'decoder_input_ids':decoder_input_ids, 'decoder_attention_mask':decoder_input_mask, \
            'label':label}

def load_wav(audio_path):
    # 利用 library 'librosa' 來載入 audio
    sample_rate = 16000
    audio, sr = librosa.load(audio_path, sr=sample_rate)
    return audio
    
def generate_decoder_input_and_label(tgt_txt):
    # 產生 label(要加上EOS), padding decoder input(要加上BOS) 以及 attention mask
    # 我這邊寫法比較複雜，可以用自己的寫法，不用照我的
    batch_size = len(tgt_txt)
    bos_token = tokenizer(tokenizer.pad_token, add_special_tokens=False, return_tensors="pt").input_ids
    eos_token = tokenizer(tokenizer.eos_token, add_special_tokens=False, return_tensors="pt").input_ids

    tgt_ids = [tokenizer(txt, add_special_tokens=False, return_tensors="pt").input_ids for txt in tgt_txt]
    label = [torch.cat((txt_ids, eos_token), -1).squeeze() for txt_ids in tgt_ids]
    decoder_input = [torch.cat((bos_token, txt_ids), -1).squeeze() for txt_ids in tgt_ids]

    max_decoder_input_ids_len = max([len(x) for x in decoder_input])
    max_label_len = max([len(x) for x in label])

    decoder_input_ids_padded = torch.IntTensor(batch_size, max_decoder_input_ids_len)
    label_padded = torch.IntTensor(batch_size, max_label_len)
    decoder_input_mask = torch.IntTensor(batch_size, max_decoder_input_ids_len)

    decoder_input_ids_padded.fill_(65000)
    label_padded.fill_(65000)
    decoder_input_mask.zero_()
    for i in range(batch_size):
        decoder_input_ids = decoder_input[i]
        decoder_input_ids_padded[i, :decoder_input_ids.size(0)] = decoder_input_ids

        decoder_input_mask[i, : decoder_input_ids.size(0)] = torch.ones(decoder_input_ids.size(0))
        
        label_ids = label[i]
        label_padded[i, :label_ids.size(0)] = label_ids
    return decoder_input_ids_padded, decoder_input_mask, label_padded

# load datalist
train_data = []
eval_data = []
with open('train.txt', 'r')as f:
    for line in f.readlines():
        if(line!=''):
            data = line.split('|') # path, zh, en
            train_data.append(
            [data[0],
             data[1],
             data[2].rstrip('\n')
            ]
        )
        
with open('eval.txt', 'r')as f:
    for line in f.readlines():
        if(line!=''):
            data = line.split('|') # path, zh, en
            eval_data.append(
            [data[0],
             data[1],
             data[2].rstrip('\n')
            ]
        )
# 這個 function 是 huggingface trainer 為了預防 GPU memory leak 而加上的
def preprocess_logits_for_metrics(logits, labels):
    """
    Original Trainer may have a memory leak. 
    This is a workaround to avoid storing too many tensors that are not needed.
    """
    return logits

# train 到一半的時候需要觀看訓練結果( evaluation ), 傳入的參數來自下面 line 194
def compute_metrics(eval_pred):
    # prediction : 模型輸出
    prediction, label = eval_pred
    prediction = torch.from_numpy(prediction)
    label = torch.from_numpy(label)

    # 計算 NLLloss，看訓練結果如何
    logSoftmax = nn.LogSoftmax(dim=-1)
    nllloss = nn.NLLLoss(reduction='mean', ignore_index=65000)
    speech_logits_logSoftmax = logSoftmax(prediction)
    speech_logits_logSoftmax = speech_logits_logSoftmax.permute(0, 2, 1)
    label = label.type(torch.LongTensor)
    # label 原本 padding 的地方是填入 65001，但是在傳遞的過程中會變成 -100，這裡把 -100 的地方轉回 65001
    label[label==-100] = 65000
    # 計算 loss
    loss_speech = nllloss(speech_logits_logSoftmax, label)

    # 將模型欲結果轉成人看得懂的字，看翻譯得怎麼樣
    batch_size = 8
    predictions_txt = tokenizer.batch_decode(torch.argmax(prediction, dim=-1),skip_special_tokens=True)
    labels_txt = tokenizer.batch_decode(label,skip_special_tokens=True)
    idx = random.randint(0,batch_size-1)
    print('prediction:',predictions_txt[idx])
    print('label:',labels_txt[idx])
    return {
        'loss_speech ': loss_speech.item(),
    }

# 如果要用 huggingface trainer 傳入自定義的模型，要繼承 class Trainer 並複寫 compute_loss( model forward ), prediction_step ( evaluation )
class MyTrainer(Trainer):

  def compute_loss(self, model, inputs, return_outputs=False): ## define forward batch data
      # line 20 回傳的結果會變成 parameter 'inputs', 這裡依照 dictionary 的 key 取出相應的值
      label = inputs.get("label")
      text_logits, speech_logits, speech_encoder_logit, speech_encoder_length = model(inputs.get('text_input_ids'), inputs.get('text_attention_mask'), inputs.get('speech_input'), inputs.get('speech_attention_mask'), inputs.get('decoder_input_ids'), inputs.get('decoder_attention_mask'))
      
      ### 計算 loss 相應的宣告
      # for NLLloss
      logSoftmax = nn.LogSoftmax(dim=-1)
      nllloss = nn.NLLLoss(reduction='mean', ignore_index=65000)
      # for KL divergence loss
      softmax = nn.Softmax(dim=-1)
      kl = torch.nn.KLDivLoss(reduction='mean')
      # for CTC loss
      ctc = torch.nn.CTCLoss(reduction='mean', zero_infinity=False)
      ### ctc loss ###
      # bzs, seq, dim -> seq, bzs, dim
      input = speech_encoder_logit.permute(1, 0, 2).log_softmax(2).detach().requires_grad_()
      target = inputs.get('text_input_ids').type(torch.LongTensor)
      input_lengths = speech_encoder_length.type(torch.LongTensor)
      target_lengths = torch.sum(inputs.get('text_attention_mask'), 1).type(torch.LongTensor)
      ############## 計算 CTC loss ##############
      ctc_loss = ctc(input, target, input_lengths, target_lengths)
      ##########################################
      
      ############## 計算 NLLloss ###############
      text_logits_logSoftmax = logSoftmax(text_logits) 
      speech_logits_logSoftmax = logSoftmax(speech_logits)
      
      text_logits_logSoftmax = text_logits_logSoftmax.permute(0, 2, 1)
      speech_logits_logSoftmax = speech_logits_logSoftmax.permute(0, 2, 1)
      label = label.type(torch.LongTensor).to(torch.device('cuda'))
      loss_text = nllloss(text_logits_logSoftmax, label)
      loss_speech = nllloss(speech_logits_logSoftmax, label)
      ##########################################

      ############## 計算 KL divergence loss ##############
      text_logits_softmax = softmax(text_logits)
      speech_logits_softmax = softmax(speech_logits)
      loss_kl = kl(speech_logits_softmax, text_logits_softmax)
      ####################################################

      total_loss = loss_text + loss_speech + loss_kl + 0.05*ctc_loss
      
      return (total_loss, {'outputs':speech_logits}) if return_outputs else total_loss

      # 做 evaluation, 結果會傳到 line 117
  def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys):
      label = inputs.get("label")
      text_logits, speech_logits, _, _ = model(inputs.get('text_input_ids'), inputs.get('text_attention_mask'), inputs.get('speech_input'), inputs.get('speech_attention_mask'), inputs.get('decoder_input_ids'), inputs.get('decoder_attention_mask')) 
      return (None, speech_logits.detach(), label)
      

### Define Hyperparameters ###
model = SpeechToTextTranslationModel()
# freeze text encoder
for param in model.text_encoder.parameters():
        param.requires_grad = False 
batch_size = 2
learning_rate = 1e-5
epochs = 10
result_dir = './ctc_share_freeze_ckpt'
##############################

### 傳入訓練參數
training_args = TrainingArguments(output_dir=result_dir,
                         do_train=True,
                         do_eval=True,
                         evaluation_strategy='steps',
                         eval_steps=200,
                         prediction_loss_only=False,
                         per_device_train_batch_size=batch_size,
                         per_device_eval_batch_size=batch_size,
                         gradient_accumulation_steps=4,
                         learning_rate=learning_rate,
                         weight_decay=1e-6,
                         adam_beta1=0.9,
                         adam_beta2=0.98,
                         num_train_epochs=epochs,
                         save_strategy="steps",
                         save_steps=20000,
                         save_total_limit=10,
                         fp16=True,
                         half_precision_backend ="auto",
                         dataloader_num_workers=8,
                         dataloader_drop_last=False,
                         logging_dir='./ctc_share_freeze_log',
                         logging_strategy='steps',
                         logging_steps=200,
                         max_grad_norm=5.0,
                         )

# 使用 huggingface trainer
trainer = MyTrainer(
        model=model, 
        args=training_args,  # training arguments, defined above
        data_collator=data_collator,
        train_dataset=train_data,
        eval_dataset=eval_data,
        compute_metrics = compute_metrics,
        preprocess_logits_for_metrics = preprocess_logits_for_metrics
    )
### 如果是第一次訓練，resume_from_checkpoint 設成 False (重頭訓練), 如果訓到一半被中斷，要恢復之前的訓練進度則設成 True
# trainer.train(resume_from_checkpoint=True)
trainer.train(resume_from_checkpoint=False)
# 存入訓練的模型參數
trainer.save_model(result_dir)