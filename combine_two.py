import os
os.environ['XLA_FLAGS'] = "--xla_dump_hlo_as_proto --xla_dump_hlo_as_text --xla_dump_hlo_as_html --xla_dump_to=custom_part_ln_fwd_dump"

from functools import partial

import numpy as np
import jax
from jax import random
from jax import core, vmap
import jax.numpy as jnp
from jax.sharding import Mesh
from jax.experimental.custom_partitioning import custom_partitioning
from jax.experimental.pjit import pjit
from jax.sharding import PartitionSpec, NamedSharding
P = PartitionSpec

from transformer_engine.jax.cpp_extensions import _layernorm_fwd_p, _layernorm_bwd_p
from collections.abc import Sequence

P_MESH_SIZE = 2
D_MESH_SIZE = 2
T_MESH_SIZE = 2


# ------------------------------------------------------------------------
# Make TE's layernorm supports multi-leading batch dims
from functools import reduce
import operator
from jax.interpreters import xla, mlir
from jax.interpreters.mlir import ir
from jax.abstract_arrays import ShapedArray

from jax import dtypes
import transformer_engine_jax
from transformer_engine.jax.cpp_extensions import jax_dtype_to_te_dtype, custom_caller
from transformer_engine.jax.cpp_extensions import CustomCallArgsWrapper

def ln_fwd_abstract(x, gamma, beta, **kwargs):
    x_dtype = dtypes.canonicalize_dtype(x.dtype)
    assert x_dtype in [jnp.float32, jnp.float16, jnp.bfloat16]

    mu_dtype = jnp.float32
    rsigma_dtype = jnp.float32

    assert gamma.size == beta.size
    hidden_size = gamma.size
    assert x.size % hidden_size == 0
    batch_shape = x.shape[:-1]

    return (
        ShapedArray(x.shape, x_dtype, named_shape=x.named_shape),    # output
        ShapedArray(batch_shape, mu_dtype, named_shape=x.named_shape),    # mu
        ShapedArray(batch_shape, rsigma_dtype, named_shape=x.named_shape),    # rsigma
    )

def ln_fwd_lowering(ctx, x, gamma, beta, *, zero_centered_gamma, epsilon):
    x_aval, gamma_aval, beta_aval = ctx.avals_in
    assert gamma_aval.dtype == beta_aval.dtype
    x_type = ir.RankedTensorType(x.type)
    x_shape = x_type.shape
    w_type = ir.RankedTensorType(gamma.type)
    w_shape = w_type.shape
    b_type = ir.RankedTensorType(beta.type)
    b_shape = b_type.shape
    assert w_type == b_type
    assert w_shape == b_shape

    output_type = w_type.element_type
    ir_mu_dtype = ir.F32Type.get()
    ir_rsigma_dtype = ir.F32Type.get()

    out_shape = x_shape
    hidden_size = reduce(operator.mul, w_shape)
    batch_shape = out_shape[:-1]
    batch_size = reduce(operator.mul, x_shape) // hidden_size

    out_types = [
        ir.RankedTensorType.get(out_shape, output_type),
        ir.RankedTensorType.get(batch_shape, ir_mu_dtype),
        ir.RankedTensorType.get(batch_shape, ir_rsigma_dtype),
    ]
    operands = [x, gamma, beta]
    operand_shapes = [x_shape, w_shape, b_shape]
    args = CustomCallArgsWrapper(out_types, operands, operand_shapes)

    opaque = transformer_engine_jax.pack_norm_descriptor(
        batch_size,
        hidden_size,
        jax_dtype_to_te_dtype(x_aval.dtype),
        jax_dtype_to_te_dtype(gamma_aval.dtype),
        zero_centered_gamma,
        epsilon,
    )

    out = custom_caller("te_layernorm_forward", args, opaque, False)
    return out

_layernorm_fwd_p.def_abstract_eval(ln_fwd_abstract)
mlir.register_lowering(_layernorm_fwd_p, ln_fwd_lowering, platform='cuda')

def te_layernorm_fwd(x: jnp.ndarray, gamma: jnp.ndarray, beta: jnp.ndarray,
                     zero_centered_gamma: bool, epsilon: float):
    return _layernorm_fwd_p.bind(x, gamma, beta,
                                 zero_centered_gamma=zero_centered_gamma,
                                 epsilon=epsilon)
# ------------------------------------------------------------------------


# 0: ln-dot-dot, 1: dot-dot-ln
TEST_CASE = int(os.environ.get("TEST_CASE", 0))
assert TEST_CASE in (0, 1)

from jax._src.interpreters import batching

def layernorm_fwd_batcher(
    batched_args: Sequence[jax.Array],
    batch_dims: Sequence[int | None],
    *,
    zero_centered_gamma: bool,
    epsilon: float,
) -> tuple[Sequence[jax.Array], Sequence[int | None]]:
  x, gamma, beta = batched_args
  x_bdim, gamma_bdim, beta_bdim = batch_dims

  # Our first goal is only to support PP on Paxml.
  assert x.shape[x_bdim] == P_MESH_SIZE
  assert gamma.shape[gamma_bdim] == P_MESH_SIZE
  assert beta.shape[beta_bdim] == P_MESH_SIZE

  out_bdims = x_bdim, gamma_bdim, beta_bdim
  return te_layernorm_fwd(x, gamma, beta, zero_centered_gamma=zero_centered_gamma, epsilon=epsilon), out_bdims
batching.primitive_batchers[_layernorm_fwd_p] = layernorm_fwd_batcher


from jax._src.interpreters import mlir
from jax.experimental.custom_partitioning import custom_partitioning

_layernorm_fwd_lower = custom_partitioning(te_layernorm_fwd,
                                           static_argnums=(3, 4))

def get_padded_spec(arg_info: jax.ShapeDtypeStruct) -> tuple:
  if arg_info.sharding is None:
    return (None,) * arg_info.ndim
  ndim, spec = arg_info.ndim, arg_info.sharding.spec
  assert len(spec) <= ndim
  return spec + (None,) * (ndim - len(spec))

def infer_sharding_from_operands(zero_centered_gamma, epsilon, mesh, arg_infos, result_infos):
  del epsilon, result_infos  # Unused.
  x_spec = get_padded_spec(arg_infos[0])
  out_sharding = NamedSharding(mesh, P(*x_spec[:-1]))
  return [out_sharding,] * 3

def partition(zero_centered_gamma, epsilon, mesh, arg_infos, result_infos):
  x_spec = NamedSharding(mesh, P(*get_padded_spec(arg_infos[0])))
  g_spec = NamedSharding(mesh, P(*get_padded_spec(arg_infos[1])))
  b_spec = NamedSharding(mesh, P(*get_padded_spec(arg_infos[2])))
  out_spec = NamedSharding(mesh, P(*get_padded_spec(arg_infos[0])[:-1]))
  arg_shardings = (x_spec, g_spec, b_spec)
  out_shardings = [out_spec,] * 3
  impl = partial(te_layernorm_fwd, zero_centered_gamma=zero_centered_gamma,
                 epsilon=epsilon)
  return mesh, impl, out_shardings, arg_shardings

_layernorm_fwd_lower.def_partition(
    infer_sharding_from_operands=infer_sharding_from_operands,
    partition=partition)

mlir.register_lowering(_layernorm_fwd_p,
                       mlir.lower_fun(_layernorm_fwd_lower, multiple_results=True),
                       platform='cuda')


# Tests
def _layernorm(x, gamma, beta, layernorm_type, zero_centered_gamma, epsilon):
  assert layernorm_type is None
  del layernorm_type
  z, *_ = te_layernorm_fwd(x, gamma, beta,
                           zero_centered_gamma=zero_centered_gamma, epsilon=epsilon)
  return z

def layernorm_ref(x, scale, bias, layernorm_type, zero_centered_gamma, epsilon):
    assert layernorm_type is None
    del layernorm_type

    x_ = jnp.asarray(x, jnp.float32)
    mean = jnp.mean(x_, axis=-1, keepdims=True)
    var = jnp.mean(jnp.square(x_ - mean), axis=-1, keepdims=True)
    normed_input = (x_ - mean) * jax.lax.rsqrt(var + epsilon)
    if zero_centered_gamma:
        return jnp.asarray(normed_input * (scale + 1) + bias).astype(x.dtype)
    return jnp.asarray(normed_input * scale + bias).astype(x.dtype)

def func(x, gamma, beta, y1, y2, ln_fn):
    if TEST_CASE == 0:
        x = ln_fn(x, gamma, beta, None, False, 1e-6)
        x = jnp.dot(x, y1)
        out = jnp.dot(x, y2)
        return jnp.mean(out)
    else:
        x = jnp.dot(x, y1)
        x = jnp.dot(x, y2)
        out = ln_fn(x, gamma, beta, None, False, 1e-6)
        return jnp.mean(out)

def vmap_f(x, gamma, beta, y1, y2):
    partial_ln_func = partial(func, ln_fn=_layernorm, gamma=gamma, beta=beta, y1=y1, y2=y2)
    return vmap(partial_ln_func, in_axes=(0,), out_axes=0)(x)

def vmap_f_ref(x, gamma, beta, y1, y2):
    partial_ln_func = partial(func, ln_fn=layernorm_ref, gamma=gamma, beta=beta, y1=y1, y2=y2)
    return vmap(partial_ln_func, in_axes=(0,), out_axes=0)(x)

def vmap_f(x, gamma, beta, y1, y2):
    partial_ln_func = partial(func, ln_fn=_layernorm)
    return vmap(partial_ln_func, in_axes=(0, 0, 0, 0, 0), out_axes=0)(x, gamma, beta, y1, y2)

def vmap_f_ref(x, gamma, beta, y1, y2):
    partial_ln_func = partial(func, ln_fn=layernorm_ref)
    return vmap(partial_ln_func, in_axes=(0, 0, 0, 0, 0), out_axes=0)(x, gamma, beta, y1, y2)


x_ = random.normal(random.PRNGKey(1124), (P_MESH_SIZE, 32, 128))
gamma_ = jnp.ones((P_MESH_SIZE, 128))
beta_ = jnp.ones((P_MESH_SIZE, 128))
y1_ = random.normal(random.PRNGKey(1126), (P_MESH_SIZE, 128, 128))
y2_ = random.normal(random.PRNGKey(1127), (P_MESH_SIZE, 128, 128))

devices = np.array(jax.local_devices())
devices = devices.reshape((P_MESH_SIZE, D_MESH_SIZE, T_MESH_SIZE))
with Mesh(devices, ('p', 'd', 't')) as mesh:
    x = jax.device_put(x_, NamedSharding(mesh, PartitionSpec('p', 'd', None)))
    gamma = jax.device_put(gamma_, NamedSharding(mesh, PartitionSpec('p', None)))
    beta = jax.device_put(beta_, NamedSharding(mesh, PartitionSpec('p', None)))
    y1 = jax.device_put(y1_, NamedSharding(mesh, PartitionSpec('p', None, 't')))
    y2 = jax.device_put(y2_, NamedSharding(mesh, PartitionSpec('p', 't', None)))

    pjitter_ref = pjit(vmap_f_ref,
                    in_shardings=[PartitionSpec('p', 'd', None), PartitionSpec('p', None), PartitionSpec('p', None),
                                    PartitionSpec('p', None, 't'), PartitionSpec('p', 't', None)],
                    out_shardings=PartitionSpec('p',)
                   )

    pjitter = pjit(vmap_f,
                   in_shardings=[PartitionSpec('p', 'd', None), PartitionSpec('p', None), PartitionSpec('p', None),
                                 PartitionSpec('p', None, 't'), PartitionSpec('p', 't', None)],
                   out_shardings=PartitionSpec('p',)
              )

    ref_outs = pjitter_ref(x_, gamma_, beta_, y1_, y2_)
    test_outs = pjitter(x_, gamma_, beta_, y1_, y2_)

print("TEST_CASE:", TEST_CASE)
print("Outputs match:", jnp.allclose(ref_outs, test_outs))
