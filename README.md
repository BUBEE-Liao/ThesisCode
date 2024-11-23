# Leveraging Pre-trained Models and Various Types of Data to Improve Speech Translation

This project is associated with the paper:  
**"利用預訓練模型和多種類型的數據改進語音翻譯"**  
**"Leveraging Pre-trained Models and Various Types of Data to Improve Speech Translation"**

![image](https://github.com/BUBEE-Liao/ThesisCode/blob/main/Training.png)

Contains the following components:
- List of datasets
- Model architecture
- Training script (`trainer.py`)

## Datasets

- our dataset contains : AISHELL-3 , ner-trs, ner-trs-pro

## Environment Setup

To train the model, you need to have a proper Python environment. The recommended environment uses Python 3.9.

If you don’t have the required environment, follow these steps to create it:

1. Create a new conda environment:
    ```bash
    conda create --name MyEnv python=3.9
    ```
2. Activate the newly created environment:
    ```bash
    conda activate MyEnv
    ```
3. Install pytorch
   ```bash
   pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118
   ```
   
4. Install sentencepiece model
    ```bash
   pip install sentencepiece
   ```
   
5. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Training the Model

To start training the model, ensure that the conda environment is activated and follow these steps:


1. Activate the conda environment:
    ```bash
    conda activate MyEnv
    ```
    if you don't have the enviroment 'MyEnv', go check the previous step 'Environment Setup'
    
2. Start the training by specifying the GPU to be used (replace `x` with the desired GPU index):
    ```bash
    CUDA_VISIBLE_DEVICES=x python trainer.py
    ```
    which x means which GPU you want to use

### Modifying Training Arguments

All training arguments can be modified in the `trainer.py` file, including but not limited to:
- `batch_size`
- `stored_steps`
- `result_directory`
- `log_directory`

These parameters can be adjusted according to your experiment needs.
if you want to know each meaning of training args, refer to : https://huggingface.co/docs/transformers/main_classes/trainer

## Model

The model architecture contains two encoders and a text decoder
one is text encoder - transformer encoder layer x6
another is speech encoder - w2v-bert 2.0
text decoder - transformer decoder layer x6


## Inference

In 'inference.py', you need to specify the 1.ckpt_path 2.path of audio you want to translate
and then follow the command :
```bash
CUDA_VISIBLE_DEVICS=x python inference.py
```

## Evaluate with BLEU metric
evaluation dataset we use common-voice 17 zh-CN and zh-TW, has a total numbers of 500.
evaluation dataset is default set to 'BLEU_metric_latest.txt', each data point has two label for calculation BLEU.
you can run 'EvalSacreBLEU.ipynb' to run the code.
you can specify 1. model ckpt 2. evaluation dataset in 'EvalSacreBLEU.ipynb'.


------------
For any further details, refer to the paper or the included scripts.
