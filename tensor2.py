
import tensorflow as tf

key = [170,180,175,160]
shoes = [260,270,265,255]

# y = ax + b
# (신발 = a * 키 + b )

zl = 170
tlsqkf = 260

# tlsqkf = zl * a + b

a = tf.Variable(0.1)
b = tf.Variable(0.2)

def 손실함수():
    예측값 =  zl * a + b
    return tf.square(260 - 예측값)

opt = tf.keras.optimizers.Adam(learning_rate=0.1)

for i in range(300):
    opt.minimize(손실함수, var_list=[a,b])
    print(a.numpy(),b.numpy())






