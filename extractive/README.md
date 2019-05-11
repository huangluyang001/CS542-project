This code is developed from  https://github.com/ChenRocks/fast_abs_rl/edit/master/README.md  
We implement self critical and NN-SE on their code base   

## Dependencies
- **Python 3** (tested on python 3.6)
- [PyTorch](https://github.com/pytorch/pytorch) 0.4.0
    - with GPU and CUDA enabled installation (though the code is runnable on CPU, it would be way too slow)
- [gensim](https://github.com/RaRe-Technologies/gensim)
- [cytoolz](https://github.com/pytoolz/cytoolz)
- [tensorboardX](https://github.com/lanpa/tensorboard-pytorch)
- [pyrouge](https://github.com/bheinzerling/pyrouge) (for evaluation)

You can use the python package manager of your choice (*pip/conda*) to install the dependencies.
The code is tested on the *Linux* operating system.
First set data repository
```
export DATA=[your data repository]
```
Then run
```
python train_word2vec.py --path=[path/to/word2vec]
```
train *extractor* using ML objectives
```
python train_extractor_ml.py --path=[path/to/extractor/model] --w2v=[path/to/word2vec/word2vec.128d.226k.bin] --combine (--model_type nnse if you want to train nnse model)
```
train rl
```
python train_full_rl.py --path=[path/to/save/model] --ext_dir=[path/to/extractor/model] --sc
```
You can also refer to ../results/pointnet(pointnet+rl nnse) to directly get our results or run
```
python eval_full_model.py --rouge --decode_dir ../results/pointnet
```
Give a path for rouge perl script before you run 
```
export ROUGE=[your rouge path]/ROUGE1.5.5
```
