import numpy as np
import scipy.optimize


def minimize(fun, x0, *args, **kwargs):
    x0 = np.asarray(x0)
    x0_shape = x0.shape

    def fwrap(x):
        x = x.reshape(x0_shape)
        val = fun(x)
        val_shape = np.asarray(val).shape
        if val_shape != ():
            raise ValueError(
                "Objective function must return a true scalar. "
                f"Got array of shape {val_shape}."
            )
        return val

    out = scipy.optimize.minimize(fwrap, x0, *args, **kwargs)
    out.x = out.x.reshape(x0.shape)
    return out
