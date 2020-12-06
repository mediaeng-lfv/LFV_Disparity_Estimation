import tensorflow as tf
import tensorflow.keras.backend as K
import sobel


def cos_similarity(a, b, axis=-1):
    norm_a = tf.nn.l2_normalize(a, axis=axis)
    norm_b = tf.nn.l2_normalize(b, axis=axis)
    cos = tf.reduce_sum(tf.multiply(norm_a,norm_b), axis=axis)
    return cos

def get_loss_function():
    get_gradient = sobel.Sobel().get_gradient
    def loss_function(disp, pred):
        # disp loss
        loss_disp = K.mean(K.abs(pred - disp))

        # grad loss
        pred_grad = get_gradient(pred)
        disp_grad = get_gradient(disp)
        pred_grad_dx = K.expand_dims(pred_grad[..., 0], axis=-1)
        pred_grad_dy = K.expand_dims(pred_grad[..., 1], axis=-1)
        disp_grad_dx = K.expand_dims(disp_grad[..., 0], axis=-1)
        disp_grad_dy = K.expand_dims(disp_grad[..., 1], axis=-1)
        loss_dx = K.mean(K.abs(pred_grad_dx - disp_grad_dx))
        loss_dy = K.mean(K.abs(pred_grad_dy - disp_grad_dy))

        # normal loss
        ones = K.ones_like(disp_grad_dx)
        pred_normal = K.concatenate([-pred_grad_dx, -pred_grad_dy, ones], axis=-1)
        disp_normal = K.concatenate([-disp_grad_dx, -disp_grad_dy, ones], axis=-1)
        loss_normal = K.mean(K.abs(1 - cos_similarity(pred_normal, disp_normal, axis=-1)))

        return loss_disp + (loss_dx + loss_dy) + loss_normal
    return loss_function