
import tensorflow as tf

#상수 만들기
ten = tf.constant([3.0,4,5])
ten2 = tf.constant([6,7,8], tf.float32)
ten3 =tf.constant([[1,2],
                   [3,4]])
ten4 = tf.zeros([2,2,3,2])
# print(ten.shape)
# print(ten2.shape)
# print(ten3.shape)
# print(ten4.shape)

# print(ten)
# print(ten2)

#변수 만들기
w = tf.Variable(1.0)
print(w.numpy())
w.assign(2.0)
print(w.numpy())