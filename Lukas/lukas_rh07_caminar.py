# =====================================================================================
# Lukas R-H-07 - esqueleto + pesos + animaciones (caminar y apuntar con el arma).
# Se ejecuta con Blender sobre Lukas/Player.blend (no lo guarda):
#   blender -b Player.blend --python lukas_rh07_caminar.py
# 1) genera la malla unica (lukas_malla_unica.py)
# 2) crea Lukas_RH07_Rig: los mismos 21 huesos (nombres y jerarquia) que PJ_Rig, con los
#    brazos girados a la pose A de la malla (45 grados)
# 3) pesa la malla con pesos automaticos
# 4) transfiere las acciones de ACTIONS desde PJ_Rig: cada hueso copia, fotograma a fotograma, la
#    transformacion de su hueso equivalente (el brazo sale de la pose A a la de la animacion);
#    la pistola de prueba pasa a la mano derecha del esqueleto nuevo
# 5) guarda Lukas/Lukas_RH07.blend con la malla, el esqueleto, las acciones, la pistola y las referencias
# =====================================================================================
import bpy, os, sys, math, traceback
from mathutils import Matrix, Vector

LUKAS = os.path.dirname(os.path.abspath(bpy.data.filepath))
ACTIONS = [('Lukas_Caminar_Bien', 'Lukas_RH07_Caminar'),      # (accion de PJ_Rig, accion nueva)
           ('Lukas_Apuntar_Bien', 'Lukas_RH07_Apuntar')]
RIG_NAME = 'Lukas_RH07_Rig'
GUN = 'Pistola_Preview'                                      # arma de prueba, emparentada a la mano
# ajuste opcional de claviculas por accion: {'accion': (grados hacia delante, grados hacia arriba)}.
# La clavicula gira sobre su base y el brazo se desplaza con ella sin cambiar de orientacion (las dos
# manos siguen sujetando el arma igual). Medido al apuntar (0, 8, 12 y 16 grados): alivia el pliegue
# de la axila pero no estira menos el hombro trasero y estira algo mas la axila, asi que va vacio.
PROTRACT = {}
# forma correctiva opcional del hombro al apuntar: una shape key por lado, activada por la distancia
# del codo a la base de la clavicula (al caminar vale 0, al apuntar 1). Desactivada: medido al apuntar,
# ni la relajacion de aristas, ni Corrective Smooth, ni ARAP calcularon una forma que estire menos que
# la malla sin forma (hombro trasero 1.30x sin forma, 1.47-2.15x con ellas). Para activarla:
# CORRECTIVE = {'action': 'Lukas_RH07_Apuntar', 'frame': 25, 'factor': 0.5, 'iters': 30, 'snap': False}
CORRECTIVE = None

def protract(P, fwd_deg, up_deg):
    """mueve en P (poses del esqueleto) las claviculas hacia delante/arriba y desplaza cada brazo"""
    for side, s in (('Left', 1), ('Right', -1)):
        base = P[side + 'Shoulder'].translation.copy()
        M = (Matrix.Translation(base) @ Matrix.Rotation(-s * math.radians(fwd_deg), 4, 'Z')
             @ Matrix.Rotation(-s * math.radians(up_deg), 4, 'Y') @ Matrix.Translation(-base))
        J = P[side + 'UpperArm'].translation.copy()
        D = Matrix.Translation((M @ J) - J)
        P[side + 'Shoulder'] = M @ P[side + 'Shoulder']
        for b in ('UpperArm', 'LowerArm', 'Hand'):
            P[side + b] = D @ P[side + b]
    return P
LOG = os.environ.get('LUKAS_LOG')
if LOG:
    _log = open(LOG, 'w', encoding='utf-8'); sys.stdout = _log; sys.stderr = _log

def assign_action(obj, act):
    obj.animation_data_create()
    obj.animation_data.action = act
    try:
        if act.slots: obj.animation_data.action_slot = act.slots[0]
    except AttributeError:
        pass

def main():
    # ---------------------------------------------------------------- 1) malla
    G = {'__name__': '__main__'}
    exec(open(os.path.join(LUKAS, 'lukas_malla_unica.py'), encoding='utf-8').read(), G)
    mesh = bpy.data.objects['Lukas_RH07']
    mesh.data.transform(mesh.matrix_world); mesh.matrix_world = Matrix.Identity(4)
    old = bpy.data.objects['PJ_Rig']
    old.data.pose_position = 'POSE'
    sc = bpy.context.scene

    # ---------------------------------------------------------------- 2) esqueleto en pose A
    ob = old.data.bones
    def arm_rot(side):
        J = ob[side + 'UpperArm'].head_local; E = ob[side + 'LowerArm'].head_local
        s = 1 if J.x > 0 else -1
        ang0 = math.atan2(abs(E.x - J.x), -(E.z - J.z))
        R = Matrix.Rotation(-s * (math.radians(45.0) - ang0), 4, 'Y')       # igual que la malla
        return Matrix.Translation(J) @ R @ Matrix.Translation(-J)
    ARM = {}
    for side in ('Left', 'Right'):
        M = arm_rot(side)
        for part in ('UpperArm', 'LowerArm', 'Hand'): ARM[side + part] = M
    arm_data = bpy.data.armatures.new(RIG_NAME)
    rig = bpy.data.objects.new(RIG_NAME, arm_data)
    bpy.data.collections['Lukas_RH07'].objects.link(rig)
    vl = bpy.context.view_layer
    for o in vl.objects: o.select_set(False)
    vl.objects.active = rig; rig.select_set(True)
    bpy.ops.object.mode_set(mode='EDIT')
    eb = arm_data.edit_bones
    for b in ob:
        e = eb.new(b.name)
        M = ARM.get(b.name, Matrix.Identity(4))
        e.head = M @ b.head_local; e.tail = M @ b.tail_local
        e.matrix = M @ b.matrix_local                    # misma orientacion (roll) que el hueso original
        e.use_deform = b.use_deform
    for b in ob:
        if b.parent:
            eb[b.name].parent = eb[b.parent.name]; eb[b.name].use_connect = b.use_connect
    bpy.ops.object.mode_set(mode='OBJECT')
    print("huesos:", len(arm_data.bones))

    # ---------------------------------------------------------------- 3) pesos automaticos
    for o in vl.objects: o.select_set(False)
    mesh.select_set(True); rig.select_set(True); vl.objects.active = rig
    bpy.ops.object.parent_set(type='ARMATURE_AUTO')
    vg_names = {g.index: g.name for g in mesh.vertex_groups}
    sin_peso = sum(1 for v in mesh.data.vertices if not any(g.weight > 1e-4 for g in v.groups))
    print("grupos de vertices:", len(mesh.vertex_groups), "| vertices sin peso:", sin_peso,
          "| modificadores:", [m.type for m in mesh.modifiers])

    # bisagras: los 3 loops de cada articulacion reparten el giro 25 / 50 / 75 % entre los dos huesos
    # (los anillos vecinos quedan enteros en su hueso para que no choquen con la bisagra)
    LEG_H = [('UpperLeg', 'LowerLeg', [(0.545, 0.0), (0.505, 0.25), (0.492, 0.5), (0.479, 0.75), (0.432, 1.0), (0.392, 1.0)]),
             ('LowerLeg', 'Foot',     [(0.236, 0.0), (0.192, 0.0), (0.141, 0.25), (0.128, 0.5), (0.115, 0.75), (0.090, 1.0)])]
    ARM_H = [('UpperArm', 'LowerArm', [(0.195, 0.0), (0.222, 0.0), (0.250, 0.0), (0.258, 0.0), (0.281, 0.25), (0.294, 0.5),
                                       (0.307, 0.75), (0.338, 1.0), (0.356, 1.0), (0.392, 1.0)]),
             ('LowerArm', 'Hand',     [(0.442, 0.0), (0.494, 0.0), (0.508, 0.0), (0.528, 0.25), (0.540, 0.5), (0.552, 0.75), ('palma', 1.0)])]
    ring_pts = {}
    for _, _, rings in ARM_H:
        for sv, _ in rings:
            ring_pts[sv] = (G['palm_ring'](0.040, 0.0215, 0.046, 0.050) if sv == 'palma' else G['arm_ring_orig'](sv))
    ROTI = G['ROT'].inverted(); J0 = G['J0']
    def set_w(v, side, up, lo, w):
        for g in list(v.groups):
            mesh.vertex_groups[g.group].remove([v.index])
        if w < 1.0: mesh.vertex_groups[side + up].add([v.index], 1.0 - w, 'REPLACE')
        if w > 0.0: mesh.vertex_groups[side + lo].add([v.index], w, 'REPLACE')
    cnt = {}
    for v in mesh.data.vertices:
        p = v.co; side = 'Left' if p.x > 0 else 'Right'
        q = Vector((abs(p.x), p.y, p.z))
        if abs(p.x) > 0.015 and p.z < 0.6:                        # piernas: anillos horizontales
            for up, lo, rings in LEG_H:
                for z, w in rings:
                    if abs(p.z - z) < 1e-5:
                        set_w(v, side, up, lo, w); cnt[up + lo] = cnt.get(up + lo, 0) + 1
        elif abs(p.x) > 0.20 and p.z > 0.8:                       # brazos: se vuelve a la pose original
            qo = J0 + ROTI @ (q - J0)
            for up, lo, rings in ARM_H:
                for sv, w in rings:
                    if min((qo - r).length for r in ring_pts[sv]) < 1e-5:
                        set_w(v, side, up, lo, w); cnt[up + lo] = cnt.get(up + lo, 0) + 1
    print("vertices de bisagra (ambos lados):", cnt)
    # (el hombro conserva los pesos automaticos: medido al apuntar, reparten mejor el estiramiento
    #  por los loops de soporte que un degradado fijo o interpolado)

    # ---------------------------------------------------------------- 4) transferir las animaciones
    names = [b.name for b in ob]
    made = []
    for src_name, new_name in ACTIONS:
        made.append(transfer(src_name, new_name, old, rig, mesh, names, sc))
    assign_action(rig, made[-1][0])                        # la ultima (apuntar) queda activa
    f0, f1 = made[-1][1], made[-1][2]
    if CORRECTIVE:
        build_corrective(mesh, rig, G, {a.name: a for a, _, _ in made}, sc)
        assign_action(rig, made[-1][0])

    # pistola de prueba (la de Player.blend): misma mano y misma colocacion respecto al hueso
    gun = bpy.data.objects.get(GUN)
    if gun:
        pb_name = gun.parent_bone
        mpi = gun.matrix_parent_inverse.copy(); mb = gun.matrix_basis.copy()
        gun.parent = rig; gun.parent_type = 'BONE'; gun.parent_bone = pb_name
        gun.matrix_parent_inverse = mpi; gun.matrix_basis = mb
        print("pistola en", gun.parent.name, gun.parent_bone)

    # ---------------------------------------------------------------- 5) archivo limpio
    keep_actions = {a for a, _, _ in made}
    sc.frame_start, sc.frame_end = f0, f1
    sc.frame_set(f0)
    refs = bpy.data.collections.new('Referencias'); sc.collection.children.link(refs)
    for n in ('Empty', 'Empty.001'):
        o = bpy.data.objects[n]
        for c in list(o.users_collection): c.objects.unlink(o)
        refs.objects.link(o)
    keep = {mesh.name, rig.name, 'Empty', 'Empty.001', GUN}
    for o in list(bpy.data.objects):
        if o.name not in keep: bpy.data.objects.remove(o)
    for c in list(bpy.data.collections):
        if c.name not in ('Lukas_RH07', 'Referencias', 'Arma_Preview'): bpy.data.collections.remove(c)
    for coll in (bpy.data.actions, bpy.data.materials, bpy.data.meshes, bpy.data.armatures, bpy.data.images):
        for idb in coll:
            if idb not in keep_actions: idb.use_fake_user = False
    bpy.data.orphans_purge(do_local_ids=True, do_linked_ids=True, do_recursive=True)
    for s in list(bpy.data.scenes):
        if s != sc: bpy.data.scenes.remove(s)
    print("archivo -> objetos", [o.name for o in bpy.data.objects], "| acciones", [a.name for a in bpy.data.actions],
          "| materiales", [m.name for m in bpy.data.materials], "| uv cuerpo", len(mesh.data.uv_layers),
          "| materiales cuerpo", len(mesh.data.materials))
    out = os.path.join(LUKAS, 'Lukas_RH07.blend')
    bpy.ops.wm.save_as_mainfile(filepath=out, relative_remap=True, compress=False)
    print("GUARDADO", out, os.path.getsize(out))

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

def transfer(src_name, new_name, old, rig, mesh, names, sc):
    """copia la accion src_name de PJ_Rig a rig: cada hueso toma, fotograma a fotograma, la misma
    transformacion que su hueso equivalente; devuelve (accion, primer fotograma, ultimo)"""
    arm_data = rig.data
    src = bpy.data.actions[src_name]
    assign_action(old, src)
    f0, f1 = int(src.frame_range[0]), int(src.frame_range[1])
    act = bpy.data.actions.new(new_name)
    assign_action(rig, act)
    depth = {}
    for b in arm_data.bones:
        d = 0; p = b.parent
        while p: d += 1; p = p.parent
        depth[b.name] = d
    order = sorted(names, key=lambda n: depth[n])
    rest = {n: arm_data.bones[n].matrix_local.copy() for n in names}
    for pb in rig.pose.bones: pb.rotation_mode = 'QUATERNION'
    prevq = {}
    for f in range(f0, f1 + 1):
        sc.frame_set(f)
        P = {n: old.pose.bones[n].matrix.copy() for n in names}      # pose del ciclo (espacio del esqueleto)
        if new_name in PROTRACT: P = protract(P, *PROTRACT[new_name])
        for n in order:
            b = arm_data.bones[n]
            if b.parent:
                pn = b.parent.name
                basis = (rest[pn].inverted() @ rest[n]).inverted() @ P[pn].inverted() @ P[n]
            else:
                basis = rest[n].inverted() @ P[n]
            loc, q, sca = basis.decompose()
            if n in prevq and q.dot(prevq[n]) < 0: q.negate()          # sin saltos de cuaternion
            prevq[n] = q.copy()
            pb = rig.pose.bones[n]
            pb.location = loc if not b.use_connect else Vector()
            pb.rotation_quaternion = q; pb.scale = sca
            pb.keyframe_insert('location', frame=f, group=n)
            pb.keyframe_insert('rotation_quaternion', frame=f, group=n)
            pb.keyframe_insert('scale', frame=f, group=n)
    act.use_frame_range = True; act.frame_start = f0; act.frame_end = f1
    act.use_cyclic = src.use_cyclic; act.use_fake_user = True
    # comprobacion: cada hueso nuevo termina donde el original (o donde lo puso el ajuste de claviculas)
    err = 0.0; moved = 0.0; grip = 0.0
    for f in range(f0, f1 + 1):
        sc.frame_set(f)
        T = {n: old.pose.bones[n].matrix.copy() for n in names}
        T0 = {n: m.copy() for n, m in T.items()}
        if new_name in PROTRACT: T = protract(T, *PROTRACT[new_name])
        for n in names:
            err = max(err, (rig.pose.bones[n].matrix.translation - T[n].translation).length)
            moved = max(moved, (T[n].translation - T0[n].translation).length)
        g_new = (rig.pose.bones['LeftHand'].head - rig.pose.bones['RightHand'].head).length
        g_old = (old.pose.bones['LeftHand'].head - old.pose.bones['RightHand'].head).length
        grip = max(grip, abs(g_new - g_old))
    print(f"accion {new_name}: fotogramas {f0}-{f1}, ciclica {act.use_cyclic} | error max de hueso: {err*1000:.3f} mm"
          + (f" | ajuste de claviculas {PROTRACT[new_name]}: brazos desplazados hasta {moved*1000:.0f} mm,"
             f" distancia entre manos cambia {grip*1000:.1f} mm" if new_name in PROTRACT else ""))

    # estiramiento de la malla durante el ciclo (aristas mas estiradas/encogidas)
    me0 = mesh.data
    E = [(e.vertices[0], e.vertices[1]) for e in me0.edges]
    L0 = [(me0.vertices[a].co - me0.vertices[b].co).length for a, b in E]
    worst = (1.0, 1.0)
    for f in range(f0, f1 + 1, 2):
        sc.frame_set(f)
        dg = bpy.context.evaluated_depsgraph_get()
        mev = mesh.evaluated_get(dg).to_mesh()
        r = [(mev.vertices[a].co - mev.vertices[b].co).length / max(l, 1e-9) for (a, b), l in zip(E, L0)]
        worst = (min(worst[0], min(r)), max(worst[1], max(r)))
        mesh.evaluated_get(dg).to_mesh_clear()
    print(f"   aristas en {new_name}: la mas encogida {worst[0]:.2f}x, la mas estirada {worst[1]:.2f}x")
    return act, f0, f1

ok = True
try:
    main()
except Exception:
    traceback.print_exc(); ok = False
print("FIN", "OK" if ok else "ERROR")
if LOG:
    _log.flush()
    open(LOG + '.done', 'w').write('ok' if ok else 'error')
