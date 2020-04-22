# coding=gbk
from PIL import Image
from keras.utils import np_utils
import random
from keras.models import *
from keras.layers import *
import keras

from src.model import build_model

CAPTCHA_ARRAY = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
# 训练样本数
TRAIN_SIZE = 2000
# 测试样本数
TEST_SIZE = 1000
# 训练每批次数
BATCH_SIZE = 10
# 训练次数
TRAIN_BATCHES = 200
IMAGE_HEIGHT = 30
IMAGE_WIDTH = 100
# CAPTCHA_PATH = 'D:\\captcha_imgs'
CAPTCHA_PATH = '../data/no_img'
CAPTCHA_LEN = 7
# 验证码种类个数（0123abc...etc）
CAPTCHA_TYPE_NUM = 10


def load_captcha(root, train_size, test_size):
    # 数据预处理
    dirs = os.listdir(root)
    dirs = dirs[:train_size + test_size]
    print('共读取数据条数：')
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
    # 转为np数组
    x = np.array(x)
    y = np.array(y)
    # 打乱顺序
    index = [i for i in range(len(x))]
    random.shuffle(index)
    x = x[index]
    y = y[index]
    # 显示第一个，验证数组打乱效果
    # plt.imshow(Image.fromarray(x[0]))
    # plt.show()
    # print(y[0])

    # 分割训练集、测试集
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


# 输入数据处理，输入图片路径，返回处理后的数组
def trans_x(path):
    img = Image.open(path)
    img = img.resize((100, 30))
    # img.show()
    # 裁切图片,单数长宽直接切除1个像素边框
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
    # 格式转换
    img = img.convert('L')
    # img.show()
    array = np.array(img)
    array = np.array(array)
    array = array.reshape(IMAGE_HEIGHT, IMAGE_WIDTH, 1)
    # 白色转化为0,并标准化
    array = (255 - array) / 255
    # 白色转化为0,阈值80二值化
    # array[array >= 100] = 0
    # array[array < 100] = 1
    return array


def train():
    (x_train, y_train), (x_test, y_test) = load_captcha(CAPTCHA_PATH, TRAIN_SIZE, TEST_SIZE)
    # print('各个集合数据条数：')
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
    # 四个准确率相乘得到最终准确率
    return model_cnn, score[5] * score[6] * score[7] * score[8]


if __name__ == '__main__':
    accuracy = 0
    # while accuracy < 0.99:
    (model, accuracy) = train()
    model.save(os.getcwd() + "captcha_model_" + str(accuracy) + ".h5")
