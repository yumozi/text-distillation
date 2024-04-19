## Text Distillation

### Environment setup
```
conda create -n text-distillation python=3.10
conda activate text-distillation
pip3 install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
pip3 install -r requirement.txt
```

### Download Needed Dataset
```
python3 wikitext.py download
python3 wikitext.py encode_data
```

### Pre-trained Model
```
python3 train.py
```

### Condense dataset
Before run it before you have the pre-trained model in `/trained_out/`
```
python3 condense.py
```

### Evaluate model performance
```
python3 commonsense.py
```