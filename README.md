# TPDLP
An efficient forecasting model for long-term time series forecasting.

# Get Started

1.Install Python 3.6, PyTorch 1.9.0.  
2.Download data. You can obtain all the six benchmarks from Google Drive. All the datasets are well pre-processed and can be used easily.  
3.Train the model. We provide the experiment scripts of all benchmarks under the folder ./scripts. You can reproduce the experiment results by:  
`bash ./scripts/ETT_script/ETTm1.sh
bash ./scripts/ECL_script/ECL.sh
bash ./scripts/Exchange_script/Exchange.sh
bash ./scripts/Traffic_script/Traffic.sh
bash ./scripts/Weather_script/Weather.sh
bash ./scripts/ILI_script/Illness.sh`

# Requirement
*numpy
*matplotlib
*pandas
*scikit-learn
*torch==1.9.0
