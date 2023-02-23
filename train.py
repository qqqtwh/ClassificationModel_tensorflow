import numpy as np
from tensorflow.keras.layers import *
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LearningRateScheduler
import argparse
from dataloader import get_dataloader
from models import *
import matplotlib.pyplot as plt

def opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_size',type=int,default=224)
    parser.add_argument('--epochs',type=int,default=100)
    parser.add_argument('--batch_size',type=int,default=32)

    return parser.parse_args()

def adjustLR(epoch):
    if epoch<=30:
        lr = 1e-4
    elif epoch>30 and epoch<=70:
        lr = 1e-5
    else:
        lr = 1e-6
    return lr

def train(model,args,train_data_loader,test_data_loader):

    model.compile(optimizer=Adam(1e-4),loss='categorical_crossentropy',metrics=['accuracy'])
    history = model.fit(
        x=train_data_loader,
        batch_size=args.batch_size,
        epochs=args.epochs,
        verbose=1,
        callbacks=[LearningRateScheduler(adjustLR)],
        validation_data=test_data_loader,
        shuffle=True,
    )

    plt.plot(np.arange(args.epochs),history.history['accuracy'],c='r',label='train_accuracy')
    plt.plot(np.arange(args.epochs),history.history['val_accuracy'],c='b',label='val_accuracy')
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.show()


if __name__ == '__main__':
    # 参数初始化
    args = opt()
    # 创建数据集
    train_data_loader,test_data_loader,num_classes = get_dataloader(args.img_size,args.batch_size)
    # 创建模型
    alexnet = AlexNet(args.img_size,num_classes)
    # 训练模型
    train(alexnet,args,train_data_loader,test_data_loader)
