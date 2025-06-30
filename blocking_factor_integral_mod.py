# -*- coding: utf-8 -*-
"""
Created on Thu May 15 00:16:07 2025

@author: User
"""

import tensorflow as tf
import logging
from relative_permeability import RelativePermeability

# ----------------------------------------
# 1) Pure-TF root-finders, no Python loops
# ----------------------------------------

@tf.function
def _solve_chandrupatla(cost, ref, max_iters=20, tol=1e-6, max_value=1.):
    """
    Find root of cost(Sg)=0 in [0, 0.78] for each entry of `ref` tensor.
    - cost: fn(Sg: Tensor) -> Tensor (same shape as ref)
    - ref:   a Tensor whose shape we imitate (e.g., the pressure grid cell)
    """
    zero = tf.constant(0)
    lo = tf.zeros_like(ref)
    hi = tf.ones_like(ref) * 0.78
    f_lo = cost(lo)
    f_hi = cost(hi)
    # Ensure bracket: if f_lo * f_hi > 0, nudge hi
    bad = tf.greater(f_lo * f_hi, 0.0)
    hi = tf.where(bad, lo + 1e-3, hi)
    f_hi = tf.where(bad, cost(hi), f_hi)

    def cond(lo, hi, f_lo, f_hi, it):
        return tf.logical_and(
            tf.reduce_any(hi - lo > tol),
            tf.less(it, max_iters)
        )

    def body(lo, hi, f_lo, f_hi, it):
        d = (f_hi - f_lo) / (hi - lo + 1e-12)
        guess = hi - f_hi / d
        f_guess = cost(guess)
        replace_lo = tf.less(f_lo * f_guess, 0.0)
        new_lo = tf.where(replace_lo, lo, guess)
        new_f_lo = tf.where(replace_lo, f_lo, f_guess)
        new_hi = tf.where(replace_lo, guess, hi)
        new_f_hi = tf.where(replace_lo, f_guess, f_hi)
        return new_lo, new_hi, new_f_lo, new_f_hi, it + 1

    lo, hi, _, _, _ = tf.while_loop(
        cond, body,
        [lo, hi, f_lo, f_hi, zero],
        shape_invariants=[
            ref.get_shape(),  # lo
            ref.get_shape(),  # hi
            ref.get_shape(),  # f_lo
            ref.get_shape(),  # f_hi
            zero.get_shape()  # it
        ]
    )
    return 0.5 * (lo + hi)


@tf.function
def _solve_newton(cost, ref, max_iters=20, max_value=1.):
    """
    Newton-Raphson on cost(Sg)=0, starting from 0.1, clamped to [0, 0.78].
    - cost: fn(Sg) -> Tensor
    - ref:   a Tensor whose shape we imitate
    """
    zero = tf.constant(0)
    Sg = tf.fill(tf.shape(ref), 0.1)

    def cond(it, Sg):
        return tf.less(it, max_iters)

    def body(it, Sg):
        with tf.GradientTape() as g:
            g.watch(Sg)
            f = cost(Sg)
        df = g.gradient(f, Sg)
        Sg_new = Sg - f / (df + 1e-12)
        Sg_new = tf.clip_by_value(Sg_new, 0.0, max_value)
        return it + 1, Sg_new

    _, Sg = tf.while_loop(
        cond, body,
        [zero, Sg],
        shape_invariants=[zero.get_shape(), ref.get_shape()]
    )
    return Sg


# ----------------------------------------------------
# 2) PVT property extraction based on fluid type
# ----------------------------------------------------

@tf.function
def extract_pvt_properties(pvt_tensor, fluid_type='DG', enable_logging=True):
    """
    Extract PVT properties based on fluid type.
    Args:
        pvt_tensor: Tensor of shape [2, n_properties, ...], where index 0 is values, 1 is derivatives.
        fluid_type: String, 'DG' (dry gas) or 'GC' (gas condensate).
        enable_logging: Bool, whether to log property shapes (default: True).
    Returns:
        Tuple of PVT properties (values and derivatives) as tensors.
    """
    # Log PVT tensor shape for debugging
    # tf.print("extract_pvt_properties: pvt_tensor shape =", tf.shape(pvt_tensor))

    # Get shape for zero tensors (same as pvt_tensor[0, 0, ...])
    zero_shape = tf.shape(pvt_tensor)[2:]  # Skip [2, n_properties]
    zero_tensor = tf.zeros(zero_shape, dtype=tf.float32)

    if fluid_type == 'DG':
        # 2 properties: invBg, invug
        invBg = pvt_tensor[0, 0]  # value of invBg
        invug = pvt_tensor[0, 1]  # value of invug    
        if enable_logging:
            logging.info(f"DG: invBg shape: {invBg.shape}, invug shape: {invug.shape}")
        return invBg, zero_tensor, invug, zero_tensor, zero_tensor, zero_tensor
    elif fluid_type == 'GC':
        # 6 properties: invBg, invBo, invug, invuo, Rs, Rv
        invBg = pvt_tensor[0, 0]  # value of invBg
        invBo = pvt_tensor[0, 1]  # value of invBo
        invug = pvt_tensor[0, 2]  # value of invug
        invuo = pvt_tensor[0, 3]  # value of invuo
        Rs = pvt_tensor[0, 4]     # value of Rs
        Rv = pvt_tensor[0, 5]     # value of Rv
        if enable_logging:
            logging.info(f"GC: invBg shape: {invBg.shape}, invBo shape: {invBo.shape}")
            logging.info(f"GC: invug shape: {invug.shape}, invuo shape: {invuo.shape}")
            logging.info(f"GC: Rs shape: {Rs.shape}, Rv shape: {Rv.shape}")
        return invBg, invBo, invug, invuo, Rs, Rv
    else:
        if enable_logging:
            logging.warning(f"Unknown fluid type: {fluid_type}. Defaulting to DG.")
        return extract_pvt_properties(pvt_tensor, fluid_type='DG', enable_logging=enable_logging)


# ----------------------------------------------------
# 3) Block integral functions for each fluid type
# ----------------------------------------------------

@tf.function
def compute_blocking_integral_dg(
    p0, p1, mg_prev, mo_prev, sum_g, sum_o, Sg_n1, mg_n1, mo_n1,
    invBg1, invug1, krog1, krgo1, well_id, compute_mo, eps
):
    """
    Compute blocking integral for dry gas (DG).
    """
    mgg1 = krgo1 * invBg1 * invug1
    mg1 = mgg1  # No mgo term for DG
    mo1 = tf.zeros_like(mg1) if not compute_mo else tf.zeros_like(mg1)  # No oil phase
    dp = p0 - p1
    sum_g_new = sum_g + 0.5 * (mg_prev + mg1) * dp
    sum_o_new = sum_o  # No oil contribution
    return sum_g_new, sum_o_new, mg1, mo1


@tf.function
def compute_blocking_integral_gc(
    p0, p1, mg_prev, mo_prev, sum_g, sum_o, Sg_n1, mg_n1, mo_n1,
    invBg1, invBo1, invug1, invuo1, Rs1, Rv1, krog1, krgo1, well_id, compute_mo, eps
):
    """
    Compute blocking integral for gas condensate (GC).
    """
    mgg1 = krgo1 * invBg1 * invug1
    mgo1 = krog1 * invBo1 * invuo1 * Rs1
    moo1 = krog1 * invBo1 * invuo1
    mog1 = krgo1 * invBg1 * invug1 * Rv1
    mg1 = mgg1 + mgo1
    mo1 = moo1 + mog1 if compute_mo else tf.zeros_like(mg1)
    dp = p0 - p1
    sum_g_new = sum_g + 0.5 * (mg_prev + mg1) * dp
    sum_o_new = sum_o + 0.5 * (mo_prev + mo1) * dp * tf.cast(compute_mo, tf.float32)
    return sum_g_new, sum_o_new, mg1, mo1


# ----------------------------------------------------
# 4) Main blocking integral computation
# ----------------------------------------------------

@tf.function
def compute_blocking_integral(
    p_n1=None,
    Sg_n1=None,
    mg_n1=None,
    mo_n1=None,
    well_id=1.0,
    min_bhp=1000.0,
    model_PVT=None,
    relperm_model=None,
    n_intervals=8,
    n_root_iter=20,
    eps=1e-12,
    solver="newton",
    compute_mo=True,
    boil_n1=None,
    bgas_n1=None,
    Rvi=None,
    fluid_type='DG',
    Sg_n1_max=1.,
    enable_logging=True
):
    """
    Compute blocking integral using pure TensorFlow operations.
    Args:
        p_n1: Tensor, pressure at initial condition.
        Sg_n1: Tensor, gas saturation at initial condition.
        mg_n1: Tensor, gas mobility.
        mo_n1: Tensor, oil mobility.
        well_id: Float, well identifier (default: 1.0).
        min_bhp: Float, minimum bottom-hole pressure (default: 1000.0).
        model_PVT: Function, maps pressure to PVT properties tensor [2, n_properties, ...].
        relperm_model: Function, maps gas saturation to relative permeabilities (krog, krgo).
        n_intervals: Int, number of pressure intervals (default: 8).
        n_root_iter: Int, max iterations for root-finding (default: 20).
        eps: Float, small value for numerical stability (default: 1e-12).
        solver: Str, root-finding method ("chandrupatla" or "newton", default: "newton").
        compute_mo: Bool, whether to compute oil mobility integral (default: True).
        boil_n1: Tensor, oil formation volume factor.
        bgas_n1: Tensor, gas formation volume factor.
        Rvi: Initial vapourized oil-gas ratio.
        fluid_type: Str, fluid type ('DG' or 'GC', default: 'DG').
        Sg_n1_max: Maximum gas saturation limit when performing Newton-Raphson/iterative search.
        enable_logging: Bool, whether to log PVT property shapes (default: True).
    Returns:
        Ig, Io: Gas and oil blocking integrals.
    """
    # Validate inputs
    if p_n1 is None or model_PVT is None or relperm_model is None:
        raise ValueError("p_n1, model_PVT, and relperm_model must be provided")

    # Build pressure grid [n_intervals + 1, ...]
    p_grid = tf.linspace(p_n1, min_bhp, n_intervals + 1)
    new_shape = tf.concat([[n_intervals + 1], tf.shape(p_n1)], axis=0)
    p_grid = tf.reshape(p_grid, new_shape)

    # Initial accumulators
    sum_g = tf.zeros_like(p_n1)
    sum_o = tf.zeros_like(p_n1)
    mg_prev = mg_n1
    mo_prev = mo_n1

    i0 = tf.constant(0)

    def outer_cond(i, *_):
        return tf.less(i, n_intervals)

    def outer_body(i, sum_g, sum_o, mg_prev, mo_prev):
        # Debug: Print mg_prev, mo_prev before update
        # tf.print("Interval", i, "mg_prev sample =", mg_prev[0, 0, 0, 0, 0],
        #          "mo_prev sample =", mo_prev[0, 0, 0, 0, 0])
        
        # Use tf.gather to access p_grid elements
        p0 = tf.gather(p_grid, i, axis=0)
        p1 = tf.gather(p_grid, i + 1, axis=0)

        # Get PVT properties: shape [2, n_properties, ...]
        pvt_tensor = model_PVT(p1, fluid_type)
        invBg1, invBo1, invug1, invuo1, Rs1, Rv1 = extract_pvt_properties(
            pvt_tensor, fluid_type, enable_logging
        )

        # Cost function for this cell
        def cost(Sg):
            krog, krgo = relperm_model(Sg)
            if fluid_type == 'DG':
                mgg = krgo * invBg1 * invug1
                mg = mgg
                mo = tf.zeros_like(krgo * invBg1 * invug1 * Rv1)
            else:  # GC
                mgg = krgo * invBg1 * invug1
                mgo = krog * invBo1 * invuo1 * Rs1
                moo = krog * invBo1 * invuo1
                mog = krgo * invBg1 * invug1 * Rv1
                mg = mgg + mgo
                mo = moo + mog if compute_mo else tf.zeros_like(mg)
            c = well_id * (mo * mg_n1 - mo_n1 * mg)
            return c

        # Select solver
        Sg1 = tf.cond(
            tf.equal(solver, "newton"),
            lambda: _solve_newton(cost, Sg_n1, max_iters=n_root_iter, max_value=Sg_n1_max),
            lambda: _solve_chandrupatla(cost, Sg_n1, max_iters=n_root_iter, max_value=Sg_n1_max)
        )

        krog1, krgo1 = relperm_model(Sg1)

        # Compute integrals based on fluid type
        def dg_branch():
            return compute_blocking_integral_dg(
                p0, p1, mg_prev, mo_prev, sum_g, sum_o, Sg_n1, mg_n1, mo_n1,
                invBg1, invug1, krog1, krgo1, well_id, compute_mo, eps
            )

        def gc_branch():
            return compute_blocking_integral_gc(
                p0, p1, mg_prev, mo_prev, sum_g, sum_o, Sg_n1, mg_n1, mo_n1,
                invBg1, invBo1, invug1, invuo1, Rs1, Rv1, krog1, krgo1, well_id, compute_mo, eps
            )

        sum_g_new, sum_o_new, mg1, mo1 = tf.cond(
            tf.equal(fluid_type, 'DG'),
            dg_branch,
            gc_branch
        )

        return i + 1, sum_g_new, sum_o_new, mg1, mo1

    _, Ig, Io, _, _ = tf.while_loop(
        outer_cond, outer_body,
        [i0, sum_g, sum_o, mg_prev, mo_prev],
        shape_invariants=[
            i0.get_shape(),
            p_n1.get_shape(),
            p_n1.get_shape(),
            p_n1.get_shape(),
            p_n1.get_shape(),
        ]
    )
    return Ig, Io


# --------------------------------
# 5) Updated demonstration
# --------------------------------
if __name__ == "__main__":
    # Set global random seed for deterministic results
    tf.random.set_seed(42)

    # Updated dummy PVT function to return [2, n_properties, ...] - similar shape as the SRM
    @tf.function
    def model_PVT_dummy(p, fluid_type):
        p_shape = tf.shape(p)
        invBg = lambda p: 1.0 / (1.0 + 1e-4 * (p - 1500.0))
        invBo = lambda p: 1.0 / (1.0 + 2e-4 * (p - 1500.0))
        a = 0.01      # base viscosity in cP
        b = 0.002     # linear pressure term
        c = 0.0001    # quadratic pressure term
        invug = lambda p: a + b * p + c * p**2
        invuo = lambda p: 5.0 * tf.math.exp(-0.015 * p)
        if fluid_type == 'DG':
            n_properties = 2
            # Construct shape: [3, n_properties, ...]
            pvt_shape = tf.concat([[2, n_properties], p_shape], axis=0)
            pvt = tf.zeros(pvt_shape, dtype=tf.float32)
            # invBg value
            pvt = tf.tensor_scatter_nd_update(pvt, [[0, 0]], [invBg(p)])
            # invug value
            pvt = tf.tensor_scatter_nd_update(pvt, [[0, 1]], [invug(p)])
        else:  # GC
            n_properties = 6
            pvt_shape = tf.concat([[2, n_properties], p_shape], axis=0)
            pvt = tf.zeros(pvt_shape, dtype=tf.float32)
            # invBg value
            pvt = tf.tensor_scatter_nd_update(pvt, [[0, 0]], [invBg(p)])
            # invBo value
            pvt = tf.tensor_scatter_nd_update(pvt, [[0, 1]], [invBo(p)])
            # invug value
            pvt = tf.tensor_scatter_nd_update(pvt, [[0, 2]], [invug(p)])
            # invuo value
            pvt = tf.tensor_scatter_nd_update(pvt, [[0, 3]], [invuo(p)])
            # Rs value
            Rs = tf.fill(p_shape, 100.0)
            pvt = tf.tensor_scatter_nd_update(pvt, [[0, 4]], [Rs])
            # Rv value
            Rv = tf.fill(p_shape, 0.01)
            pvt = tf.tensor_scatter_nd_update(pvt, [[0, 5]], [Rv])
        return pvt

    # Instantiate RelativePermeability
    rel_perm = RelativePermeability(dtype=tf.float32)

    # Test example with shape (32, 1, 39, 39, 1)
    shape = (32, 1, 39, 39, 1)
    p0 = tf.random.uniform(shape, minval=1900.0, maxval=2100.0, seed=50)
    Sg0 = tf.ones(shape, dtype=tf.float32)
    mg0 = tf.ones(shape, dtype=tf.float32) * 0.01
    mo0 = tf.ones(shape, dtype=tf.float32) * 0.02
    well_id0 = tf.ones((1, *shape[1:]), dtype=tf.float32)
    Sg0_max = 0.78
    Rvi0 = 0.094 
    min_b = tf.constant(1000.0)
    
    # Test for both fluid types and logging states
    for fluid_type in ['DG', 'GC']:
        for enable_logging in [False]:
            print(f"\nTesting fluid type: {fluid_type}, enable_logging: {enable_logging}")
            # Run with Chandrupatla solver
            Ig_ch, Io_ch = compute_blocking_integral(
                p_n1=p0,
                Sg_n1=Sg0,
                mg_n1=mg0,
                mo_n1=mo0,
                well_id=well_id0,
                min_bhp=min_b,
                model_PVT=model_PVT_dummy,
                relperm_model=rel_perm.compute_krog_krgo,
                n_intervals=8,
                n_root_iter=15,
                eps=1e-12,
                solver="chandrupatla",
                compute_mo=True,
                fluid_type=fluid_type,
                Sg_n1_max=Sg0_max,
                enable_logging=enable_logging
            )

            # Run with Newton solver
            Ig_nw, Io_nw = compute_blocking_integral(
                p_n1=p0,
                Sg_n1=Sg0,
                mg_n1=mg0,
                mo_n1=mo0,
                well_id=well_id0,
                min_bhp=min_b,
                model_PVT=model_PVT_dummy,
                relperm_model=rel_perm.compute_krog_krgo,
                n_intervals=8,
                n_root_iter=6,
                eps=1e-12,
                solver="newton",
                compute_mo=True,
                fluid_type=fluid_type,
                Sg_n1_max=Sg0_max,
                enable_logging=enable_logging
            )

            # Print shapes and sample values
            tf.print(f"{fluid_type} Chandrupatla (logging={enable_logging}) → I_g shape =", tf.shape(Ig_ch),
                     "  I_o shape =", tf.shape(Io_ch))
            tf.print(f"{fluid_type} Newton (logging={enable_logging}) → I_g shape =", tf.shape(Ig_nw),
                     "  I_o shape =", tf.shape(Io_nw))
            tf.print(f"{fluid_type} Chandrupatla (logging={enable_logging}) → I_g sample =", Ig_ch[0, 0, 0, 0, 0],
                     "  I_o sample =", Io_ch[0, 0, 0, 0, 0])
            tf.print(f"{fluid_type} Newton (logging={enable_logging}) → I_g sample =", Ig_nw[0, 0, 0, 0, 0],
                     "  I_o sample =", Io_nw[0, 0, 0, 0, 0])
            
@tf.function
def compute_phase_rates(
    p_n1, pwf,
    Sg_n1, mg_n1, mo_n1,
    invBg_n1, invBo_n1, invug_n1, invuo_n1, Rs_n1, Rv_n1,
    relperm_model,
    Ck, q_target, q_well_idx,
    fluid_type='DG',
    solver='newton',
    n_intervals=8,
    n_root_iter=20,
    Sg_max=0.78,
    compute_mo=True
):
    """
    Compute gas and (optionally) oil rates at given pwf using blocking factors.
    Returns qg (and qo for GC).
    """
    # 1) Compute blocking integrals at this pwf
    Ig, Io = compute_blocking_integral(
        p_n1=p_n1,
        Sg_n1=Sg_n1,
        mg_n1=mg_n1,
        mo_n1=mo_n1,
        min_bhp=pwf,
        model_PVT=model_PVT_dummy,
        relperm_model=relperm.compute_krog_krgo,
        n_intervals=n_intervals,
        n_root_iter=n_root_iter,
        solver=solver,
        compute_mo=compute_mo,
        fluid_type=fluid_type,
        Sg_n1_max=Sg_max
    )
    dp = p_n1 - pwf + 1e-12

    # 2) compute blocking factors
    blk_fac_g = Ig / (mg_n1 * dp)
    blk_fac_o = Io / (mo_n1 * dp)

    # 3) max rates
    qg_max = q_well_idx * Ck * blk_fac_g * mg_n1 * dp
    qo_max = q_well_idx * Ck * blk_fac_o * mo_n1 * dp

    # 4) apply rate control: here assume target is gas rate for DG, for GC use both
    qg = tf.maximum(tf.minimum(q_target, qg_max), 0.)
    qo = tf.zeros_like(qg)
    if fluid_type == 'GC':
        # oil target based on GOR = 1/Rv
        qo_target = qg * (1.0 / (Rv_n1 + 1e-12))
        qo = tf.maximum(tf.minimum(qo_target, qo_max), 0.)

    return qg, qo

@tf.function
def split_condensate_components(
    qg, qo,
    Sg_n1, invBg_n1, invBo_n1, invug_n1, invuo_n1, Rs_n1, Rv_n1,
    relperm_model
):
    """
    Given qg and qo, split into qgg, qgo, qoo, qog.
    """
    krog, krgo = relperm_model(Sg_n1)
    mgg = krgo * invBg_n1 * invug_n1
    mgo = krog * invBo_n1 * invuo_n1 * Rs_n1
    moo = krog * invBo_n1 * invuo_n1
    mog = krgo * invBg_n1 * invug_n1 * Rv_n1
    denom_g = mgg + mgo + 1e-12
    denom_o = moo + mog + 1e-12
    qgg = qg * (mgg / denom_g)
    qgo = qg * (mgo / denom_g)
    qoo = qo * (moo / denom_o)
    qog = qo * (mog / denom_o)
    return qgg, qgo, qoo, qog

@tf.function
def estimate_pwf_and_rates(
    p_n1, Sg_n1, mg_n1, mo_n1,
    invBg_n1, invBo_n1, invug_n1, invuo_n1, Rs_n1, Rv_n1,
    Ck, q_target, q_well_idx,
    relperm_model,
    fluid_type='DG',
    solver='newton',
    n_intervals=8,
    n_root_iter=20,
    Sg_max=0.78,
    compute_mo=True,
    max_iters=10,
    tol=1e-6
):
    """
    Iteratively estimate pwf so that qg matches q_target, then compute final rates.
    Returns rates and pwf.
    """
    # initial pwf guess
    pwf = 0.5 * (p_n1 + 1000.0)
    it = tf.constant(0)

    def cond(pwf, it):
        qg, _ = compute_phase_rates(
            p_n1, pwf,
            Sg_n1, mg_n1, mo_n1,
            invBg_n1, invBo_n1, invug_n1, invuo_n1, Rs_n1, Rv_n1,
            relperm_model,
            Ck, q_target, q_well_idx,
            fluid_type, solver, n_intervals, n_root_iter, Sg_max, compute_mo
        )
        return tf.logical_and(it < max_iters,
                              tf.reduce_any(tf.abs(qg - q_target) > tol))

    def body(pwf, it):
        qg, _ = compute_phase_rates(
            p_n1, pwf,
            Sg_n1, mg_n1, mo_n1,
            invBg_n1, invBo_n1, invug_n1, invuo_n1, Rs_n1, Rv_n1,
            relperm_model,
            Ck, q_target, q_well_idx,
            fluid_type, solver, n_intervals, n_root_iter, Sg_max, compute_mo
        )
        # derivative d qg / d pwf via finite diff
        eps = 1e-3
        qg_plus, _ = compute_phase_rates(
            p_n1, pwf + eps,
            Sg_n1, mg_n1, mo_n1,
            invBg_n1, invBo_n1, invug_n1, invuo_n1, Rs_n1, Rv_n1,
            relperm_model,
            Ck, q_target, q_well_idx,
            fluid_type, solver, n_intervals, n_root_iter, Sg_max, compute_mo
        )
        dqg = (qg_plus - qg) / eps
        pwf_new = pwf - (qg - q_target) / (dqg + 1e-12)
        pwf_new = tf.clip_by_value(pwf_new, 1000.0, p_n1)
        return pwf_new, it + 1

    pwf_final, _ = tf.while_loop(cond, body, [pwf, it])

    # final rates
    qg, qo = compute_phase_rates(
        p_n1, pwf_final,
        Sg_n1, mg_n1, mo_n1,
        invBg_n1, invBo_n1, invug_n1, invuo_n1, Rs_n1, Rv_n1,
        relperm_model,
        Ck, q_target, q_well_idx,
        fluid_type, solver, n_intervals, n_root_iter, Sg_max, compute_mo
    )
    if fluid_type == 'DG':
        return qg, pwf_final
    else:
        qgg, qgo, qoo, qog = split_condensate_components(
            qg, qo,
            Sg_n1, invBg_n1, invBo_n1, invug_n1, invuo_n1, Rs_n1, Rv_n1,
            relperm_model
        )
        return (qgg, qgo, qoo, qog), pwf_final

# Setup
relperm = RelativePermeability(dtype=tf.float32)
shape=(4,)
p0=tf.constant([2000.,2100.,2200.,2300.])
Sg0=tf.ones(shape)
mg0=tf.ones(shape)*0.01
mo0=tf.ones(shape)*0.02
invBg0=1.0/(1.0+1e-4*(p0-1500.))
invBo0=1.0/(1.0+2e-4*(p0-1500.))
invug0=0.01+0.002*p0+1e-4*p0*p0
invuo0=5.*tf.exp(-0.015*p0)
Rs0=tf.fill(shape,100.)
Rv0=tf.fill(shape,0.01)
Ck=tf.constant(1e-3)
q_target=tf.constant([100.,100.,100.,100.])

# DG test
qg_dg, pwf_dg = estimate_pwf_and_rates(
    p0,Sg0,mg0,mo0,
    invBg0,invBo0,invug0,invuo0,Rs0,Rv0,
    Ck,q_target,q_target,
    relperm.compute_krog_krgo,'DG'
)
# GC test
(qgg,qgo,qoo,qog), pwf_gc = estimate_pwf_and_rates(
    p0,Sg0,mg0,mo0,
    invBg0,invBo0,invug0,invuo0,Rs0,Rv0,
    Ck,q_target,q_target,
    relperm.compute_krog_krgo,'GC'
)

# Print results
tf.print('DG → qg:', qg_dg, ' pwf:', pwf_dg)
tf.print('GC → qgg:',qgg,' qgo:',qgo,' qoo:',qoo,' qog:',qog,' pwf:',pwf_gc)
