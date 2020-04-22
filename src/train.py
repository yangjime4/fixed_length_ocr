# coding=gbk
from PIL import Image
from keras.utils import np_utils
import random
from keras.models import *
from keras.layers import *
import keras

from src.model import build_model

CAPTCHA_ARRAY = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
# ѵ��������
TRAIN_SIZE = 2000
# ����������
TEST_SIZE = 1000
# ѵ��ÿ������
BATCH_SIZE = 10
# ѵ������
TRAIN_BATCHES = 200
IMAGE_HEIGHT = 30
IMAGE_WIDTH = 100
# CAPTCHA_PATH = 'D:\\captcha_imgs'
CAPTCHA_PATH = '../data/no_img'
CAPTCHA_LEN = 7
# ��֤�����������0123abc...etc��
CAPTCHA_TYPE_NUM = 10


def load_captcha(root, train_size, test_size):
    # ����Ԥ����
    dirs = os.listdir(root)
    dirs = dirs[:train_size + test_size]
    print('����ȡ����������')
    print(len(dirs))
    x = []
    y = []
    for path in dirs:

        array = trans_x(root + '\\' + path)
        if '_' in path:
            cap_str = path.split('_')[1].split('.')[0].replace('s', '5')
            if len(cap_str) == 7:
                try:
                    y_captcha = []
                    for captcha in cap_str:
                        captcha_key = CAPTCHA_ARRAY.index(captcha)
                        y_captcha.append(np_utils.to_categorical(captcha_key, num_classes=CAPTCHA_TYPE_NUM))
                    y_captcha = np.array(y_captcha)
                    x.append(array)
                    y.append(y_captcha)
                except Exception:
                    pass
    # תΪnp����
    x = np.array(x)
    y = np.array(y)
    # ����˳��
    index = [i for i in range(len(x))]
    random.shuffle(index)
    x = x[index]
    y = y[index]
    # ��ʾ��һ������֤�������Ч��
    # plt.imshow(Image.fromarray(x[0]))
    # plt.show()
    # print(y[0])

    # �ָ�ѵ���������Լ�
    x_train = x[:train_size]
    y_train = y[:train_size]
    y_train = [y_train[:, 0, :], y_train[:, 1, :], y_train[:, 2, :], y_train[:, 3, :], y_train[:, 4, :],
               y_train[:, 5, :], y_train[:, 6, :]]
    x_test = x[train_size:train_size + test_size + 1]
    y_test = y[train_size:train_size + test_size + 1]
    y_test = [y_test[:, 0, :], y_test[:, 1, :], y_test[:, 2, :], y_test[:, 3, :], y_test[:, 4, :], y_test[:, 5, :],
              y_test[:, 6, :]]
    if train_size == 0:
        return x_test, y_test
    else:
        return (x_train, y_train), (x_test, y_test)


# �������ݴ�������ͼƬ·�������ش���������
def trans_x(path):
    img = Image.open(path)
    img = img.resize((100, 30))
    # img.show()
    # ����ͼƬ,��������ֱ���г�1�����ر߿�
    # crop_width = IMAGE_WIDTH
    # crop_height = IMAGE_HEIGHT
    # if IMAGE_HEIGHT % 2 == 1:
    #     crop_height = IMAGE_HEIGHT - 1
    # if IMAGE_WIDTH % 2 == 1:
    #     crop_width = IMAGE_WIDTH - 1
    # region = (0, 0, crop_width, crop_height)
    # region = (4, 4, 90, 28)
    # img = img.crop(region)
    # img.show()
    # ��ʽת��
    img = img.convert('L')
    # img.show()
    array = np.array(img)
    array = np.array(array)
    array = array.reshape(IMAGE_HEIGHT, IMAGE_WIDTH, 1)
    # ��ɫת��Ϊ0,����׼��
    array = (255 - array) / 255
    # ��ɫת��Ϊ0,��ֵ80��ֵ��
    # array[array >= 100] = 0
    # array[array < 100] = 1
    return array


def train():
    (x_train, y_train), (x_test, y_test) = load_captcha(CAPTCHA_PATH, TRAIN_SIZE, TEST_SIZE)
    # print('������������������')
    # print(len(x_train))
    # print(len(y_train))
    # print(len(x_test))
    # print(len(y_test))

    model_cnn = build_model()

    model_cnn.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

    model_cnn.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=TRAIN_BATCHES, verbose=1,
                  validation_data=(x_test, y_test), callbacks=[keras.callbacks.TensorBoard(log_dir='../logs')])
    score = model_cnn.evaluate(x_test, y_test, verbose=0)
    print('Test accuracy:', score)
    # �ĸ�׼ȷ����˵õ�����׼ȷ��
    return model_cnn, score[5] * score[6] * score[7] * score[8]


if __name__ == '__main__':
    accuracy = 0
    # while accuracy < 0.99:
    (model, accuracy) = train()
    model.save(os.getcwd() + "captcha_model_" + str(accuracy) + ".h5")
