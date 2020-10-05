import math

# reimplement the code of dynet

def is_almost_equal(grad, computed_grad):
    f = abs(grad - computed_grad)
    m = max(abs(grad), abs(computed_grad))

    if f > 0.01 and m > 0.:
        f /= m

    if f > 0.01 or math.isnan(f):
        return False
    else:
        return True

def check(function, weights, true_grad, alpha = 1e-2):
    if weights.size() != true_grad.size():
        raise RuntimeError("Incompatible dimensions")

    size = weights.size()
    weights_view = weights.view(-1)
    true_grad_view = true_grad.view(-1)
    for i in range(weights_view.size()[0]):
        old = weights_view[i].item()

        weights_view[i] = old - alpha
        value_left = function(weights).item()

        weights_view[i] = old + alpha
        value_right = function(weights).item()

        weights_view[i] = old
        grad = (value_right - value_left) / (2. * alpha)

        if not is_almost_equal(grad, true_grad_view[i]):
            return False

    return True
