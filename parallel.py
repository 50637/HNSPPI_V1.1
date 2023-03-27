import os
#Set the number of parallel
for i in range(100):
   '''
   parallel experiments.
   '''
   cmp="python main.py --input1 data/mouse/mouse_pos.edgelist --input2 data/mouse/mouse_neg.edgelist --output embeddings/mouse --species 'mouse' --seed "+str(i)
   os.system(cmp)










