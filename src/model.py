from keras import Input, Model
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense

IMAGE_HEIGHT = 30
IMAGE_WIDTH = 100
# 验证码长度
CAPTCHA_LEN = 7
# 验证码种类个数（0123abc...etc）
CAPTCHA_TYPE_NUM = 10


def build_model():
    input_tensor = Input((IMAGE_HEIGHT, IMAGE_WIDTH, 1))
    x = input_tensor
    x = Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, 1))(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.2)(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Flatten()(x)
    x = Dense(81, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = [Dense(CAPTCHA_TYPE_NUM, activation='softmax', name='c%d' % (i + 1))(x) for i in range(CAPTCHA_LEN)]
    model_cnn = Model(inputs=input_tensor, outputs=x)
    return model_cnn
