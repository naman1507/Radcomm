This folder consist of code and related files related to finetuning a LLM with 3GPP specification files. 
Here the LLM used is a Masked Language Model (MLM) "distilroberta-base" for finetuning. 
DataCollatorForLanguageModeling has been used to prepare datasets for mask fill task.

# Model Architecture: https://huggingface.co/distilroberta-base

No. of layers: 6
Hidden Layer Dimension: 768
Vocabulary Size : 50265
Model Size: 82.8M params

# Finetuning Key Points: 

The model has been finetuned on a fill masking task where 0.15% of words have been masked from a particular chunk/batch and needs to be predicted by the model.
Out of 6 layers, only last two layers were finetuned for 10 epochs.
The Perplexity before finetuning was around 7-8 which has been reduced to 3.79 after finetuning.
