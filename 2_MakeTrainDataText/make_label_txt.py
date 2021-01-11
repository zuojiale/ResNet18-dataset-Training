#!/usr/bin/python
# -*- coding:utf-8 -*-

import os


def generate(dir, label):
    files = os.listdir(dir)
    files.sort()
    print('*********')
    print('input:', dir)
    listText = open('/home/zjl/HRG_hand/1_OpencvTof/samples/opencv/Test_Pics/Train_list.txt', 'a')
    for file in files:
        fileType = os.path.split(file)
        if fileType[1] == '.txt':
            continue
        name = file + ' ' + str(int(label)) + '\n'
        listText.write(name)
    listText.close()
    print('down')



outer_path = '/home/zjl/HRG_hand/1_OpencvTof/samples/opencv/Test_Pics1/' # 这里是你的图片的目录

if __name__ == '__main__':
    i = 0
    folderlist = os.listdir(outer_path)  # 列举文件夹
    folderlist.sort()
    for folder in folderlist:
        generate(os.path.join(outer_path, folder), i)
        i += 1
