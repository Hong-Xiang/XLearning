import tensorflow as tf
import numpy as np

def foo():
    return np.ones([1], dtype=np.float32)

def main():
    tf_f = tf.py_func(foo, [], [tf.float32], name='sampler', stateful=False)
    batch_size = 32
    queue = tf.RandomShuffleQueue(
        capacity=200, 
        min_after_dequeue=100, 
        dtypes=[tf.float32], 
        shapes=[[1]], 
        name='data_queue'
        )

    enqueue_op = queue.enqueue(tf_f)
    tf_data = queue.dequeue_many(batch_size)
    qr = tf.train.QueueRunner(queue, [enqueue_op] * 4)
    tf.add_to_collection(tf.GraphKeys.QUEUE_RUNNERS, qr)
    sv = tf.train.Supervisor(logdir="logdir")
    # config = tf.ConfigProto(device_count={'GPU': -1})
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    with sv.managed_session() as sess:
        sv.start_queue_runners(sess)
        coord = tf.train.Coordinator()
    #     enqueue_threads = qr.create_threads(sess, coord=coord, start=True)

        for step in range(10):
            if coord.should_stop():
                break
            result = sess.run(tf_data)
            print(step, result)
        coord.request_stop()
        coord.join(enqueue_threads)
        pass
    
if __name__ == "__main__":
    main()