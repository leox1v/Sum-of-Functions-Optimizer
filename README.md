Sum of Functions Optimizer (SFO) for pytorch
================================

This is an extension to use Sohl-Dickstein et al. Sum of Function optimization for pytorch. The original repository with an extensive documentation on how to use the code in a general setting can be found [here](https://github.com/Sohl-Dickstein/Sum-of-Functions-Optimizer).

## Use SFO as a pytorch Optimizer

A Simple example code on MNIST is given in **sfo_pytorch_demo.py**.

### Python package

To use TorchSFO, you should first import TorchSFO,  
`from sfo_pytorch import TorchSFO`  
then initialize it,    
`optimizer = TorchSFO(model.parameters(), (data, target), batch_size)`    
then call the optimizer with a closure of the form

`    def closure(x, y_):
        f = model.forward(x)
        loss = F.nll_loss(f, y_)
        loss.backward()
        return loss

    SFO_opt.step(closure)`.

The three required initialization parameters are:    
- *params* - Iterable of parameters to optimize.
- *dataset* - (tuple(Tensor, Tensor)) Tuple of Tensor of full dataset and Tensor of full labels.
- *batch_size* - Batch size to use.
