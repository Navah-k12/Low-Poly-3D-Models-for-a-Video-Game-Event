# =====================================================================================
# Lukas R-H-07 - forma correctiva del hombro al apuntar (PENDIENTE, desactivada).
# Se carga desde lukas_rh07_rig.py solo si CORRECTIVE no es None. Medido al apuntar: ni la
# relajacion de aristas, ni Corrective Smooth, ni ARAP calcularon una forma que estire menos que la
# malla sin forma; el activador (distancia codo - base de la clavicula, driver LOC_DIFF) si funciona.
# =====================================================================================
import bpy, math
from mathutils import Matrix, Vector

def skin_matrices(mesh, rig):
    """matriz de deformacion (pesado lineal normalizado) de cada vertice en la pose actual"""
    D = {b.name: rig.pose.bones[b.name].matrix @ b.matrix_local.inverted() for b in rig.data.bones}
    gn = {g.index: g.name for g in mesh.vertex_groups}
    S = []
    for v in mesh.data.vertices:
        m = Matrix(((0,) * 4,) * 4); tot = 0.0
        for g in v.groups:
            n = gn[g.group]
            if n in D and g.weight > 0: m = m + D[n] * g.weight; tot += g.weight
        S.append(m * (1.0 / tot) if tot > 0 else Matrix.Identity(4))
    return S

def build_corrective(mesh, rig, G, acts, sc):
    """Forma correctiva del hombro: en la pose de apuntar, la piel del hombro trasero, la axila y el
    deltoides se relaja con ARAP (reparte el estiramiento sin plegar caras); el desplazamiento se pasa
    al espacio de reposo invirtiendo el pesado de cada vertice y se guarda como una shape key por
    lado, activada por la distancia del codo a la base de la clavicula."""
    from mathutils.bvhtree import BVHTree
    import bmesh
    cfg = CORRECTIVE
    act = acts[cfg['action']]
    assign_action(rig, act); sc.frame_set(cfg['frame'])
    me = mesh.data; V = me.vertices
    rest = [v.co.copy() for v in V]
    S = skin_matrices(mesh, rig)
    dg = bpy.context.evaluated_depsgraph_get(); ev = mesh.evaluated_get(dg); mv = ev.to_mesh()
    posed = [v.co.copy() for v in mv.vertices]
    err = max((S[i] @ rest[i] - posed[i]).length for i in range(len(V)))
    ptree = BVHTree.FromPolygons(posed, [list(p.vertices) for p in me.polygons])
    pnorm = [v.normal.copy() for v in mv.vertices]
    ev.to_mesh_clear()
    print(f"correctivo: pesado reproducido con error {err*1000:.3f} mm")
    # zona que se corrige (hombro trasero, axila, deltoides); tubos, cuello alto y pecho quedan fijos
    back_y = G['back_y']
    tube = [i for i, p in enumerate(rest) if p.y > 0.05 and abs(p.x) < 0.22 and 1.14 < p.z < 1.56 and p.y > back_y(p.x, p.z) + 0.0045]
    ttree = BVHTree.FromPolygons([rest[i] for i in tube], [[k] for k in range(len(tube))]) if tube else None
    def near_tube(p):
        if not tube: return False
        loc, n, idx, d = ttree.find_nearest(p)
        return d is not None and d < 0.022
    def in_zone(p):
        return 0.05 < abs(p.x) < 0.33 and p.y > -0.06 and 1.16 < p.z < 1.46
    tset = set(tube)
    zone = [i for i, p in enumerate(rest) if in_zone(p) and i not in tset and not near_tube(p)]
    zs = set(zone)
    nb = {i: [] for i in zone}
    L0 = {}
    for e in me.edges:
        a, b = e.vertices
        if a in zs or b in zs:
            L0[(a, b)] = L0[(b, a)] = (rest[a] - rest[b]).length
            if a in zs: nb[a].append(b)
            if b in zs: nb[b].append(a)
    border = {i for i in zone if any(j not in zs for j in nb[i])}
    move = [i for i in zone if i not in border]
    # forma objetivo en la pose: "lo mas rigido posible" (ARAP) con el borde de la zona fijo. Cada
    # vertice busca la posicion en la que sus aristas se parecen mas a las de reposo (solo giradas):
    # el estiramiento se reparte y no se pliegan caras. 'snap' la devuelve a la superficie deformada.
    import numpy as np
    P = [p.copy() for p in posed]
    Rn = [np.array(p) for p in rest]
    for it in range(cfg['iters']):
        Pn = {i: np.array(P[i]) for i in zone for i in [i] + nb[i]}
        newP = {}
        for i in move:
            Cm = np.zeros((3, 3))
            for j in nb[i]:
                Cm += np.outer(Rn[i] - Rn[j], Pn[i] - Pn[j])
            Uu, ss_, Vt = np.linalg.svd(Cm)
            Rm = Vt.T @ Uu.T
            if np.linalg.det(Rm) < 0:
                Vt[2, :] *= -1; Rm = Vt.T @ Uu.T
            tgt = sum(Pn[j] + Rm @ (Rn[i] - Rn[j]) for j in nb[i]) / len(nb[i])
            q = Pn[i] + (tgt - Pn[i]) * cfg['factor']
            newP[i] = Vector(q.tolist())
        for i, q in newP.items(): P[i] = q
    if cfg.get('snap'):
        for i in move:
            loc, nn, idx, dd = ptree.find_nearest(P[i])
            if loc is not None: P[i] = loc
    # al espacio de reposo: d_reposo = (parte lineal del pesado)^-1 * d_pose
    rest_off = {}
    for i in move:
        dp = P[i] - posed[i]
        if dp.length < 1e-7: continue
        rest_off[i] = S[i].to_3x3().inverted() @ dp
    if not me.shape_keys: mesh.shape_key_add(name='Basis', from_mix=False)
    # activador: distancia del codo a la base de la clavicula (lo que mide el driver LOC_DIFF).
    # Al apuntar el codo sube hacia delante y la distancia se acorta; al caminar o en reposo, no.
    def dist(side):
        return (rig.pose.bones[side + 'LowerArm'].head - rig.pose.bones[side + 'Shoulder'].head).length
    ranges = {}
    for name, a in acts.items():
        assign_action(rig, a)
        vals = []
        for f in range(int(a.frame_range[0]), int(a.frame_range[1]) + 1):
            sc.frame_set(f); vals += [dist('Left'), dist('Right')]
        ranges[name] = (min(vals), max(vals))
    rig.data.pose_position = 'REST'; sc.frame_set(1); d_rest = dist('Left'); rig.data.pose_position = 'POSE'
    aim_lo, aim_hi = ranges[cfg['action']]
    others = [r for n, r in ranges.items() if n != cfg['action']]
    print("correctivo: distancia codo-clavicula (cm) reposo %.1f | " % (d_rest * 100)
          + " | ".join(f"{n} {lo*100:.1f}-{hi*100:.1f}" for n, (lo, hi) in ranges.items()))
    a_on = aim_hi                                              # con el codo del apuntado: forma completa
    a_off = min([d_rest] + [lo for lo, hi in others])          # reposo y demas acciones: sin forma
    assert a_off > a_on + 0.01, "el apuntado no se distingue de las demas acciones"
    a_off -= 0.25 * (a_off - a_on)                             # margen: caminar queda claramente en 0
    print(f"correctivo: rampa {a_off*100:.1f} cm (0) -> {a_on*100:.1f} cm (1)")
    for side, s in (('Left', 1), ('Right', -1)):
        kb = mesh.shape_key_add(name='Corr_Apuntar_' + side[0], from_mix=False)
        n = 0
        for i, d in rest_off.items():
            if rest[i].x * s > 0: kb.data[i].co = rest[i] + d; n += 1
        kb.slider_min, kb.slider_max = 0.0, 1.0
        fc = kb.driver_add('value'); drv = fc.driver; drv.type = 'AVERAGE'
        for m_ in list(fc.modifiers): fc.modifiers.remove(m_)
        var = drv.variables.new(); var.name = 'codo'; var.type = 'LOC_DIFF'
        var.targets[0].id = rig; var.targets[0].bone_target = side + 'LowerArm'
        var.targets[1].id = rig; var.targets[1].bone_target = side + 'Shoulder'
        gen = fc.modifiers.new('GENERATOR'); gen.mode = 'POLYNOMIAL'; gen.poly_order = 1
        gen.coefficients = (a_off / (a_off - a_on), -1.0 / (a_off - a_on))
        print(f"correctivo: shape key {kb.name} con {n} vertices movidos (max {max((d.length for i, d in rest_off.items() if rest[i].x * s > 0), default=0)*1000:.1f} mm en reposo)")
    # comprobacion: valores del driver en cada accion
    for name, a in acts.items():
        assign_action(rig, a); vals = []
        for f in range(int(a.frame_range[0]), int(a.frame_range[1]) + 1, 4):
            sc.frame_set(f)
            kev = me.shape_keys.evaluated_get(bpy.context.evaluated_depsgraph_get())
            vals.append(kev.key_blocks['Corr_Apuntar_L'].value)
        print(f"correctivo: valor en {name}: {min(vals):.2f}-{max(vals):.2f}"
              f" ({len(zone)} vertices en la zona, {len(move)} movibles, {len(tube)} de tubos fijos)")

