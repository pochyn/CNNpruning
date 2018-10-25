# CNNpruning

Five layerd CNN trained on MNIST dataset.  
Layer sizes are 1000 1000 500 200 10  
  
  
  
function weight_variable() performs weighted pruning of weight variable pruning SPARSITY% of nodes.  
SPARSITY variable is global variable defined on the top of the file.  
function l2_weight_variable() performs unit pruning, and it prunes SPARSITY% of columns.  
  
output
weighted pruning:
![alt text](https://github.com/pochyn/CNNpruning/blob/master/weighted.png) 

l2 norm pruning:
![alt text](https://github.com/pochyn/CNNpruning/blob/master/l2.png) 

