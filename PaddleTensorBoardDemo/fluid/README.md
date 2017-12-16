## Use Tensorboard to visualize PaddlePaddle Fluid data

This is a prototype to demonstrate the feasibility to use Tensorboard to display PaddlePaddle's training data metrics and graph.


#### 1)  Startup development PaddlePaddle docker image in board/PaddleTensorBoardDemo/fluid directory.  Install dependencies inside docker image.

```
cd board/PaddleTensorBoardDemo
docker run -p 6006:6006 -it -v `pwd`:/paddle paddlepaddle/paddle:latest /bin/bash
apt install libatlas-base-dev
pip install tensorflow
``` 

#### 2) Run PaddlePaddle Fluid unit tests

```
cd /paddle
python ./fluid/test_fit_a_line.py
python ./fluid/test_recognize_digits_conv.py
```

#### 3) Launch TensorBoard after training is complete

```
tensorboard --logdir=/paddle/logs
```

#### 4) Launch browser and navigate to http://localhost:6006/
