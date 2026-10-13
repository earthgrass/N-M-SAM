from .hvp import hvp


def cross_block_coupling(loss,z,q_a,q_b):
    ha=hvp(loss,z,q_a);hb=hvp(loss,z,q_b);q=q_a+q_b
    direct=q@hvp(loss,z,q)
    aa,bb,cross=q_a@ha,q_b@hb,q_a@hb
    return {"a_quadratic":float(aa),"b_quadratic":float(bb),"cross":float(cross),
            "joint_quadratic":float(direct),"decomposed":float(aa+bb+2*cross),
            "identity_error":float(abs(direct-aa-bb-2*cross)),
            "symmetry_error":float(abs(cross-q_b@ha)),"hvp_calls":3}
