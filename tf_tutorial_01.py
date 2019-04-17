import tensorflow as tf

def var_demo():
	a = tf.Variable(tf.linspace(-5., 5., 10), dtype=tf.float32)
	b = tf.Variable(tf.random_normal([3, 3], mean=1.0, stddev=2.0, dtype=tf.float32), dtype=tf.float32)
	c = tf.Variable(b.initialized_value(), dtype=tf.float32)
	d = tf.Variable(tf.zeros([3, 3], dtype=tf.float32), dtype=tf.float32)
	e = tf.assign(a, tf.linspace(-1., 1., 10))
	f = tf.cast(e, dtype=tf.int32)
	c1 = tf.constant(1)
	c2 = tf.constant([2, 3])

	init = tf.global_variables_initializer();
	sess = tf.Session()
	sess.run(init)
	print(sess.run(a))
	print(sess.run(b))
	print(sess.run(c))
	print(sess.run(d))
	print(sess.run(e))
	print(sess.run(f))
	print(sess.run(c1))
	print(sess.run(c2))


def ops_demo():
	a = tf.constant([[1, 2, 3], [4, 5, 6]])
	b = tf.constant(4)
	c = tf.add(a, b)
	d = tf.multiply(a, b)
	e = tf.divide(a, b)
	f = tf.Variable(tf.random_normal([2, 3], 1.0, 3.0), dtype=tf.float32)
	g = tf.add(f, tf.cast(tf.divide(a, b), dtype=tf.float32))
	init = tf.global_variables_initializer();
	sess = tf.Session()
	sess.run(init)
	print(sess.run(c))
	print(sess.run(d))
	print(sess.run(e))
	print(sess.run(g))


def mat_ops_demo():
	m1 = tf.Variable(tf.random_normal([3, 3], 1.0, 3), dtype=tf.float32)
	m2 = tf.Variable(tf.random_normal([3, 3], 3.0, 1), dtype=tf.float32)
	m3 = tf.add(m1, m2)
	m4 = tf.subtract(m1, m2)
	m5 = tf.multiply(m1, m2)
	m6 = tf.divide(m1, m2)

	mm = tf.matmul(m1, m2)

	init = tf.global_variables_initializer();
	sess = tf.Session()
	sess.run(init)
	print(sess.run([m3, m4, m5, m6]))
	print(sess.run(mm))


def placeholder_demo():
	a = tf.placeholder(shape=[3, 3], dtype=tf.float32)
	b = tf.placeholder(shape=[3, 2], dtype=tf.float32)
	mm = tf.matmul(a, b)
	init = tf.global_variables_initializer();
	sess = tf.Session()
	sess.run(init)
	result = sess.run(mm, feed_dict={a: [[1,1,1],[2,2,2],[3,3,3]], b:[[4,4],[5,5],[6,6]]})
	print(result)


placeholder_demo()