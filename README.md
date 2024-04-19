# Setup

### Environment setup
```
conda create -n text-distillation python=3.10
conda activate text-distillation
pip3 install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
pip3 install -r requirements.txt
```

### Download Needed Dataset
```
python3 wikitext.py download
python3 wikitext.py encode_data
```

# Training the Model
### Pre-trained Model
The pretraining stage.
```
python3 train.py
```
This will output a model into /trained_out.
Note: Change these three variables if you want to use gpu or speed up the training. Same with condense dataset
```
device = "cpu"  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = "bfloat16"  # float32|bfloat16|float16
compile = False  # use PyTorch 2.0 to compile the model to be faster
```

### Condense dataset
The condensation stage. Before you run this you have to have the pretrained model in /trained_out
```
python3 condense.py
```
This will output a model into /out. 

# Evaluation
### Finetuning
To evaluate the model using commonsenseqa, we have to finetune it first.
Remember to backup the ckpt.pt file in your out folder, if you still need it.
```
python3 commonsense_finetune.py
```


### Evaluate model performance
Run this to evaluate the model (that's currently in your out folder, called ckpt.pt) on commonsenseqa.
```
python3 commonsense.py
```
