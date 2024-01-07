a) This folder consist of code and related files related to finetuning a LLM with 3GPP specification files. 
b) Here the LLM used is a Masked Language Model (MLM) "distilroberta-base" for finetuning. 
c) DataCollatorForLanguageModeling has been used to prepare datasets for mask fill task.

# Model Architecture: https://huggingface.co/distilroberta-base

a) No. of layers: 6
b) Hidden Layer Dimension: 768
c) Vocabulary Size : 50265
d) Model Size: 82.8M params

# Finetuning Key Points: 

a) The model has been finetuned on a fill masking task where 0.15% of words have been masked from a particular chunk/batch and needs to be predicted by the model.
b) Out of 6 layers, only last two layers were finetuned for 10 epochs.
c) The Perplexity before finetuning was around 7-8 which has been reduced to 3.79 after finetuning.
