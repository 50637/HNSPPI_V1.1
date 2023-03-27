HNSPPI: A Hybrid Computational Model Combing Network and Sequence Information for Predicting Protein-Protein Interaction

HNSPPI is a novel computational model for PPI predictions which comprehensively characterizes the intrinsic relationship between two proteins by integrating protein sequence and PPI network connection properties.

HNSPPI Version: V1.1
Date: 2023-03-25
Platform: Tensorflow 2.7.0 and Python 3.8.0

[1] The folders in the HNSPPI package:
1) data: This folder contains seven benchmark datasets, including two PPI information files (positive and negative cases) and sequence information file. 
2) embeddings: This folder stores the embedding features generated in the training.
3) OpenNE: This folder contains the node2vec method source code.
4) SOTA: This folder contains 5 state-of-the-art algorithms.

[2] Scripts:
1) The source code files of HNSPPI model

main.py - This script is the main entry of the program.

parallel.py - This script is used to perform parallel experiments.

utils.py - This script is used to prepare data and features for the model.

evaluation.py – This script is used to evaluate the performance of HNSPPI.

2) The source code files for SOTA algorithms

DeepFE.py – This script is to evaluate the performance of DeepFE-PPI. The complete source code can be downloaded from https://github.com/xal2019/DeepFE-PPI.

DCONV.py- This script is to evaluate the performance of DCONV. The source code can be downloaded from https://gitlab.univnantes.fr/richoux-f/DeepPPI/tree/v1.tcbb.

DFC.py- This script is to evaluate the performance of DFC. The source code can be downloaded from https://gitlab.univnantes.fr/richoux-f/DeepPPI/tree/v1.tcbb.

DeepPur(AAC).py- This script is to evaluate the performance of DeepPur(AAC). The code is downloaded from https://github.com/kexinhuang12345/DeepPurpose.

DeepPur(CNN).py- This script is to evaluate the performance of DeepPur(CNN). The code is downloaded from https://github.com/kexinhuang12345/DeepPurpose.

[3] Datasets: 

(1) In this study, we provided seven testing datasets. The whole package of data can be downloaded from our official website: http://cdsic.njau.edu.cn/data/PPIDataBankV1.0.

(2) The data profiles for each dataset also can be downloaded from the folder ‘data’ in the Github. 

-------Sharing/access Information-------

S.cerevisiae:	PMID: 25657331

M.musculus:		PMID: 34536380

H.pylori :      PMID: 11196647

D.melanogaster:	PMID: 19171120

Fly:	        PMID: 34536380

Human:		    http://www.csbio.sjtu.edu.cn/bioinf/LR_PPI/Dara.htm

[4] Running:

--Running the HNSPPI model requires two edgelist files (one is for positive samples, and the other is for negative samples) and a csv file for protein sequences.

--Command line:

run main.py script with --input1 <positive edgelist> --input2 <negative edgelist> --output <output file> --species <species name> --seed <seed>
  
--Model output: will generate a file called results.csv
  
--For example: 
  
python main.py --input1 data/mouse/mouse_pos.edgelist --input2 data/mouse/mouse_neg.edgelist --output embeddings/mouse --species ‘mouse’ --seed 0
