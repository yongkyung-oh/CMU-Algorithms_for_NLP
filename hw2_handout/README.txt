naivebayes.py dev_text.txt dev_label.txt heldout_text.txt heldout_pred_nb.txt
neuralnet.py dev_text.txt dev_label.txt heldout_text.txt heldout_pred_nn.txt

requirments.txt - for naivebayes.py
requirments_nn.txt - for neuralnet.py

- Tested on ubuntu 16.04
- Used 'python' rather than 'python3'
- In the neuralnet.py, I used cuda:1 instead of cuda:0 due to hardware issue