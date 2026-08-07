import functools
from scipy._lib._array_api import (
    xp_result_device, is_cupy, is_jax, scipy_namespace_for, SCIPY_ARRAY_API,
    xp_capabilities,
)

import numpy as np
from ._ndimage_api import *   # noqa: F403
from . import _ndimage_api
from . import _delegators
__all__ = _ndimage_api.__all__


MODULE_NAME = 'ndimage'


def _maybe_convert_arg(arg, xp, device=None):
    """Convert arrays/scalars hiding in the sequence `arg`."""
    if isinstance(arg, np.ndarray | np.generic):
        return xp.asarray(arg, device=device)
    elif isinstance(arg, list | tuple):
        return type(arg)(_maybe_convert_arg(x, xp, device) for x in arg)
    else:
        return arg


# Some cupyx.scipy.ndimage functions don't exist or are incompatible with
# their SciPy counterparts
CUPY_BLOCKLIST = [
    'distance_transform_bf',
    'distance_transform_cdt',
    'find_objects',
    'geometric_transform',
    'vectorized_filter',
]


# `jax.scipy.ndimage.map_coordinates` cannot be handed our arguments verbatim.
# It takes `order` as a *required positional* argument, implements only a
# subset of what SciPy offers, and -- less obviously -- uses two of SciPy's
# mode names for different semantics.
#
# Forwarding arguments unchanged therefore had two failure modes:
#   * `map_coordinates(x, coords)` raised `TypeError: missing 1 required
#     positional argument: 'order'` on JAX while working everywhere else;
#   * `mode='constant'` (the default) and `mode='wrap'` returned *silently
#     different* values from SciPy for any out-of-bounds coordinate, because
#     JAX's meanings for those two names are SciPy's `grid-constant` and
#     `grid-wrap`.
#
# So delegate only what JAX implements with SciPy's semantics, translating the
# two names that do correspond, and let everything else fall through to the
# NumPy implementation -- which is what the rest of `ndimage` already does on a
# non-CuPy backend.
#: SciPy mode -> JAX mode, for modes whose semantics agree at *any* order.
#: `constant` and `wrap` are deliberately absent: SciPy's versions treat the
#: half-sample beyond each edge differently from anything JAX offers.
_JAX_MODES = {
    "nearest": "nearest",
    "grid-constant": "constant",
}

#: Modes that agree only for linear interpolation.  At order 0 these three fold
#: the coordinate before rounding it, and break ties in the other direction
#: from SciPy -- measurable only at exactly half-integer coordinates, and wrong
#: silently when it happens.
_JAX_MODES_ORDER1_ONLY = {
    "mirror": "mirror",
    "reflect": "reflect",
    "grid-wrap": "wrap",
}


def _jax_map_coordinates_args(
    input, coordinates, output=None, order=3, mode='constant', cval=0.0,
    prefilter=True
):
    """Translate a `map_coordinates` call for JAX, or None if JAX cannot do it.

    Returns ``(args, kwargs)`` for `jax.scipy.ndimage.map_coordinates`, or None
    to fall back to the NumPy implementation.

    `prefilter` is deliberately ignored: it only has an effect for order > 1,
    and those orders are never delegated.
    """
    if output is not None or order not in (0, 1):
        return None
    jax_mode = _JAX_MODES.get(mode)
    if jax_mode is None and order == 1:
        jax_mode = _JAX_MODES_ORDER1_ONLY.get(mode)
    if jax_mode is None:
        return None
    return (input, coordinates, order), {"mode": jax_mode, "cval": cval}


def delegate_xp(delegator, module_name):
    def inner(func):
        @functools.wraps(func)
        def wrapper(*args, **kwds):
            xp = delegator(*args, **kwds)

            # try delegating to a cupyx/jax namesake
            if is_cupy(xp) and func.__name__ not in CUPY_BLOCKLIST:
                # https://github.com/cupy/cupy/issues/8336
                import importlib
                cupyx_module = importlib.import_module(f"cupyx.scipy.{module_name}")
                cupyx_func = getattr(cupyx_module, func.__name__)
                return cupyx_func(*args, **kwds)
            elif (
                is_jax(xp)
                and func.__name__ == "map_coordinates"
                and (jax_args := _jax_map_coordinates_args(*args, **kwds))
                is not None
            ):
                spx = scipy_namespace_for(xp)
                jax_module = getattr(spx, module_name)
                jax_func = getattr(jax_module, func.__name__)
                return jax_func(*jax_args[0], **jax_args[1])
            else:
                # the original function (does all np.asarray internally)
                # XXX: output arrays
                # The NumPy round-trip must return results on the device of
                # the input arrays, not on the backend's default device
                device = xp_result_device(*args, *kwds.values())
                result = func(*args, **kwds)

                if isinstance(result, np.ndarray | np.generic):
                    # XXX: np.int32->np.array_0D
                    return xp.asarray(result, device=device)
                elif isinstance(result, int):
                    return result
                elif isinstance(result, dict):
                    # value_indices:
                    # result is {np.int64(1): (array(0), array(1))} etc
                    return {
                        k.item(): tuple(xp.asarray(vv, device=device) for vv in v)
                        for k,v in result.items()
                    }
                elif result is None:
                    # inplace operations
                    return result
                else:
                    # lists/tuples
                    return _maybe_convert_arg(result, xp, device)
        return wrapper
    return inner

default_capabilities = xp_capabilities(
    cpu_only=True, exceptions=["cupy"], allow_dask_compute=True, jax_jit=False
)

capabilities_dict = {
    "geometric_transform": xp_capabilities(
        cpu_only=True, allow_dask_compute=True, jax_jit=False
    ),
    "find_objects": xp_capabilities(
        cpu_only=True, allow_dask_compute=True, jax_jit=False
    ),
    "distance_transform_bf": xp_capabilities(
        cpu_only=True, allow_dask_compute=True, jax_jit=False
    ),
    "distance_transform_cdt": xp_capabilities(
        cpu_only=True, allow_dask_compute=True, jax_jit=False
    ),
    "vectorized_filter": xp_capabilities(
        cpu_only=True, allow_dask_compute=True, jax_jit=False
    ),
    "generate_binary_structure": xp_capabilities(out_of_scope=True),
    "map_coordinates": xp_capabilities(
        cpu_only=True, exceptions=["cupy", "jax.numpy"],
        allow_dask_compute=True, jax_jit=True
    ),
    "labeled_comprehension": xp_capabilities(np_only=True),
}

# ### decorate ###
for func_name in _ndimage_api.__all__:
    bare_func = getattr(_ndimage_api, func_name)
    delegator = getattr(_delegators, func_name + "_signature")

    capabilities = capabilities_dict.get(func_name, default_capabilities)

    # pyrefly:ignore[not-callable]
    f = capabilities(
        delegate_xp(delegator, MODULE_NAME)(bare_func)
        if SCIPY_ARRAY_API else bare_func
    )
    # add the decorated function to the namespace, to be imported in __init__.py
    vars()[func_name] = f
