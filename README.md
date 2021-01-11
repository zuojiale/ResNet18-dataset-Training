# ResNet18-dataset-Training
how to make the DATASET and TRAIN
1.CvSaveImg:
  this dir includes some code that using OPENCV to capture pictures and rename them into "label_0000XXX.jpeg" format.
2.MakeTrainDataTxt：
  this dirctionary includes the python code which can make the NECCESSARY .txt file before traning the ResNet18.
  Note:you should devide these pics saved into diffrent DIRS which's name is the pic's LABEL.
  
3.ResNet18
  the dir includes some RESNET18's python code:
  cnn.py: the Structure of ResNet18
  work.py:the Train code with PYTORCH
  
