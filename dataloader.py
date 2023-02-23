from tensorflow.keras.preprocessing.image import ImageDataGenerator

def get_dataloader(img_size = 224,batch_size = 32):
    train_data_transform = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rescale=1/255,
        zoom_range=0.1,
        horizontal_flip=True
    )

    test_data_transform = ImageDataGenerator(rescale=1/255)

    train_data_loader = train_data_transform.flow_from_directory(
        'datasets/flowers_data/train',
        target_size=(img_size,img_size),
        batch_size=batch_size,
        shuffle=True
    )
    test_data_loader = test_data_transform.flow_from_directory(
        'datasets/flowers_data/test',
        target_size=(img_size,img_size),
        batch_size=batch_size,
        shuffle=True
    )

    num_classes = train_data_loader.num_classes

    return train_data_loader,test_data_loader,num_classes
