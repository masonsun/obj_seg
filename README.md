# Object Segmentation using Deep Reinforcement Learning

## Downloading the dataset

...to be completed...

## Running standalone CRF-RNN

To independently run the CRF-RNN segmentation method, which is an implementation of the ICCV 2015 paper [Conditional Random Fields as Recurrent Neural Networks](http://www.robots.ox.ac.uk/~szheng/papers/CRFasRNN.pdf) in Tensorflow/Keras, run the commands as follows:

```
# activate Python 2.7.X virtualenv
source <root_of_repo>/bin/activate
pip install -r requirements.txt

# build CRF-RNN custom C++ code (outputs `high_dim_filter.so` if successful)
cd ./cpp
./compile.sh
cd ..
```

As per the original model, you can download the weights from [here](https://goo.gl/ciEYZi) and place it in the root of the repo.

```
# run a demo
python eval.py
```

If successful, a file with the segmentation named "labels.png" will be created.