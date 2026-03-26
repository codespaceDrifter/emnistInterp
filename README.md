mechanistic interpretability of different models trained on EMNIST  

uses model, training, probe code from the svnapse repo  
uses visualization code from the intvrpviz repo  

models included:  
MLP  
CNN  
ViT  

we run hyperparameter search to find the smallest size model for each to interpret.  

then we use probes from the distill circuit thread and anthropic transformer circuit thread and maybe original things to interpret them  

then we visualize everything in intvrpviz  

so the weights visualization didn't work. I think the problem is first, superposition. we need probes. second, like visualizing later layers just ignores all nonlinearity. 
