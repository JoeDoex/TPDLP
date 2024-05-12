# TPDLP
An efficient forecasting model for long-term time series forecasting.

# Get Started

1.Install Python 3.6, PyTorch 1.9.0.  
2.Download data. You can obtain all the six benchmarks from [Google Drive](https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy?usp=sharing). All the datasets are well pre-processed and can be used easily.  
3.Train the model. You can reproduce the experiment results by:  

```python  
python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path ETTm2.csv \
  --model_id ETTm2_$seq_len'_'96 \
  --model $model_name \
  --data ETTm2 \
  --features M \
  --seq_len $seq_len \
  --pred_len 96 \
  --enc_in 7 \
  --des 'Exp' \
  --itr 1 --batch_size 32 --learning_rate 0.001 >logs/LongForecasting/$model_name'_'ETTm2_$seq_len'_'96.log
```

# Requirement
* numpy
* matplotlib
* pandas
* scikit-learn
* torch==1.9.0
