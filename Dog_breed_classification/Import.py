from keras.models import *
from keras.layers import *
from keras.applications import *
from keras.preprocessing.image import *
import tensorflow as tf
import h5py

'''
with tf.device('/gpu:0'):
    a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
    b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
    c = tf.matmul(a, b)
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    print (sess.run(c))
'''
def write_gap(MODEL, image_size, lambda_func=None):
    width = image_size[0]
    height = image_size[1]
    input_tensor = (height, width, 3)
#        input_tensor = Input((height, width, 3))
#        print(input_tensor)
    x = input_tensor
    if lambda_func:
        x = Lambda(lambda_func)(x)
    print(x)
    base_model = MODEL(input_shape=x, weights="imagenet", include_top=False)
#        base_model = MODEL(input_tensor=x, weights='imagenet', include_top=False)
    model = Model(base_model.input, GlobalAveragePooling2D()(base_model.output))
    gen = ImageDataGenerator()
    train_generator = gen.flow_from_directory("train_class", image_size, shuffle=False,
                                              batch_size=1)
    test_generator = gen.flow_from_directory("test", image_size, shuffle=False,
                                             batch_size=1, class_mode=None)
    #train = model.predict_generator(train_generator, train_generator.samples)
    #test = model.predict_generator(test_generator, test_generator.samples)
    with h5py.File("gap_%s.h5" % MODEL.__name__) as h:
        h.create_dataset("train", data=train)
        h.create_dataset("test", data=test)
        h.create_dataset("label", data=train_generator.classes)

write_gap(ResNet50, (224, 224))
#write_gap(InceptionV3, (299, 299), inception_v3.preprocess_input)
#write_gap(Xception, (299, 299), xception.preprocess_input)
#write_gap(MobileNet, (224, 224))

