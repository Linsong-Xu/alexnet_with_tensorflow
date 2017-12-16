# AlexNet with Tensorflow

## How to use:
* First you should download the file [bvlc_alexnet.npy](http://www.cs.toronto.edu/%7Eguerzhoy/tf_alexnet/) or [here](https://pan.baidu.com/s/1o8KbYEA)
* If you want to use my data, you can download it from [train.csv](https://pan.baidu.com/s/1b5riz0), this data contains 28709 images and labels:
  
  * images: contain 28709 human facial expression images
  
  * labels: '0' -> 'angry', '1' -> 'hate', '2' -> 'scared', '3' -> 'happy', '4' -> 'sad', '5' -> 'surprised', '6' -> 'unclear'
  
  * **For example:** this is a image with label '3'
  
  <div align=center>
  <img src='https://github.com/Linsong-Xu/alexnet_with_tensorflow/blob/master/zebra.jpeg'>
  </div>
  
  * I found this data from kaggle, but this competition may be temporary, because it is a machine learning course of NTU, maybe you can study machine learning or get some information from [here](http://speech.ee.ntu.edu.tw/~tlkagk/courses.html)
  
* You can run use_default_para.py, this code use the already trained parameters to predict the category of the image which you input

  * **For example:**
  if you input 'python use_default_para.py --path zebra.jpeg' in commind line, it will output 'zebra'
  
  <div align=center>
  <img src='https://github.com/Linsong-Xu/alexnet_with_tensorflow/blob/master/zebra.jpeg'>
  </div>
  
* If you want to use AlexNet with our own data, you should note that:

  * The input image size should be (227, 227)
  
  * You should train your parameter with your own train data, in the file finetune.py, you can edit the train_layers to decide which layer should be retrained, and other layer still use the original trained parameter in 'bvlc_alexnet.npy'
  
* Then you can run the code with your own data

  * First run change_img.py to change my data to the required type
  
  * Then run finetune.py to train the model, it will take a long time
  
  * Finally, run use_my_para.py, it use the trained model to predict the human facial expression(In my experiment: I just use 1000 images as train data, 200 images as valid data, and epoch is 10, I get the accuracy is about 50%, I don't know can I get a better result if I use the whole data to train...I hope so)
 
