# CNNpruning

Five layerd CNN trained on MNIST dataset.  
Layer sizes are 1000 1000 500 200 10  
  
  
  
function weight_variable() performs weighted pruning of weight variable pruning SPARSITY% of nodes.  
SPARSITY variable is global variable defined on the top of the file.  
function l2_weight_variable() performs unit pruning, and it prunes SPARSITY% of columns.  
  
PROBLEMS:
Neural network is trained in MNIST with 98% accuracy, but evaluation of images always around 10%  
which is procent of randomn gues <-- TODO fix this
