import torch
from transformers import AutoTokenizer, AutoModel, pipeline
import re
import numpy as np
import os
import requests
from tqdm.auto import tqdm
tokenizer = AutoTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
model = AutoModel.from_pretrained("Rostlab/prot_bert")
fe = pipeline('feature-extraction', model=model, tokenizer=tokenizer,device=0)
sequences_Example = "MTCTLVLLIASVLHFRMRGSCLLDIERFPVIPGTIYAGHIAYCAILYFLHDLEILPTRYRSRMGWLTTFLIELVLGVAFMEV"
embedding = fe(sequences_Example)
embedding = np.array(embedding)
print(embedding)