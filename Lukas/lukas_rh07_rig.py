# =====================================================================================
# Lukas R-H-07 - esqueleto (Unity Humanoid) + pesos + acciones de animacion.
# Se ejecuta con Blender sobre Lukas/Player.blend (no lo guarda; Player.blend es el molde):
#   blender -b Player.blend --python lukas_rh07_rig.py
# 1) genera la malla unica (lukas_malla_unica.py)
# 2) crea Lukas_RH07_Rig: los 21 huesos de PJ_Rig (nombres de Unity) con los brazos en la pose A de
#    la malla + 15 huesos de dedos por mano (Proximal / Intermediate / Distal), con el roll orientado
#    para que la flexion (giro sobre X local) cierre el dedo hacia la palma
# 3) pesos: automaticos en el cuerpo, bisagras 25/50/75 % en los 3 loops de cada articulacion y
#    pesos por anillo en los dedos
# 4) acciones: traspasa las del juego (PJ_Rig) y crea las nuevas con lukas_rh07_animaciones.py;
#    cada accion tiene su pista de NLA, usuario falso y claves limpias para Unity
# 5) guarda Lukas/Lukas_RH07.blend con la malla, el esqueleto, las acciones, la pistola y las referencias
# =====================================================================================
import bpy, os, sys, math, traceback
from mathutils import Matrix, Vector
from mathutils.kdtree import KDTree

LUKAS = os.path.dirname(os.path.abspath(bpy.data.filepath))
RIG_NAME = 'Lukas_RH07_Rig'
GUN = 'Pistola_Preview'                                      # arma de prueba en el socket Arma_R (mano derecha)
GUN_L = 'Pistola_Preview_L'                                  # su reflejo en el socket Arma_L (mano izquierda)
SOCKETS = ('Arma_R', 'Arma_L')                               # huesos sin deformacion: donde va el arma en Unity
SOCK_LEN = 0.10
CORRECTIVE = None                                            # forma correctiva del hombro (pendiente)
# acciones del juego que se traspasan: (accion de PJ_Rig, accion nueva, dedos {lado: pose})
TRANSFERS = [
    ('Lukas_Idle_Bien',             'Anim_Idle',          {}),
    ('Lukas_Idle_Herido',           'Anim_Idle_Hurt',     {'R': 'clutch', 'L': 'grip'}),     # aprieta la herida, arma en la izquierda
    ('Lukas_Idle_Critico',          'Anim_Idle_Critical', {'R': 'clutch', 'L': 'grip'}),
    ('Lukas_Caminar_Bien',          'Anim_Walk',          {}),
    ('Lukas_Correr_Bien',           'Anim_Run',           {}),
    ('Lukas_GirarDer_Bien',         'Anim_Turn',          {}),
    ('Lukas_Agachado_Bien',         'Anim_Crouch_Idle',   {}),
    ('Lukas_CaminarAgachado_Bien',  'Anim_Crouch_Walk',   {}),
    ('Lukas_ApuntarCaminando_Bien', 'Anim_Aim_Walk',      {'R': 'grip', 'L': 'support'}),
    ('Lukas_Apuntar_Bien',          'Anim_Combat_Aim',    {'R': 'grip', 'L': 'support'}),
]
FINGER_NAMES = ('Thumb', 'Index', 'Middle', 'Ring', 'Little')
SEGS = ('Proximal', 'Intermediate', 'Distal')
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

def fcurves_of(act):
    """curvas de una accion (acciones por capas de Blender 4.4+ o acciones clasicas)"""
    try:
        return list(act.fcurves)
    except AttributeError:
        out = []
        for layer in act.layers:
            for strip in layer.strips:
                for cb in strip.channelbags: out += list(cb.fcurves)
        return out

def remove_fcurve(act, fc):
    try:
        act.fcurves.remove(fc); return
    except AttributeError:
        pass
    for layer in act.layers:
        for strip in layer.strips:
            for cb in strip.channelbags:
                if fc in list(cb.fcurves): cb.fcurves.remove(fc); return

def clean_action(act):
    """claves limpias sin dejar huesos sin curva: en Blender un hueso sin curva conserva la pose de la accion
    anterior al cambiar de accion, asi que cada accion guarda todas sus curvas. Una curva constante se queda
    con 2 claves (inicio y fin) y se quita solo la posicion de los huesos conectados (no tiene efecto)."""
    rig = bpy.data.objects.get(RIG_NAME)
    removed = 0
    for fc in fcurves_of(act):
        dp = fc.data_path
        if dp.endswith('.location') and rig and '"' in dp:
            b = rig.data.bones.get(dp.split('"')[1])
            if b is not None and b.use_connect:
                remove_fcurve(act, fc); removed += 1; continue
        kp = fc.keyframe_points
        if len(kp) > 2 and all(abs(k.co[1] - kp[0].co[1]) < 1e-6 for k in kp):
            while len(kp) > 2: kp.remove(kp[1], fast=True)
            fc.update(); removed += 1
    return removed

def main():
    # ---------------------------------------------------------------- 1) malla
    G = {'__name__': '__main__'}
    exec(open(os.path.join(LUKAS, 'lukas_malla_unica.py'), encoding='utf-8').read(), G)
    mesh = bpy.data.objects['Lukas_RH07']
    mesh.data.transform(mesh.matrix_world); mesh.matrix_world = Matrix.Identity(4)
    old = bpy.data.objects['PJ_Rig']
    old.data.pose_position = 'POSE'
    sc = bpy.context.scene

    # ---------------------------------------------------------------- 2) esqueleto en pose A + dedos
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
    # dedos: 3 huesos por dedo sobre la geometria registrada por el generador; el eje Z local mira al
    # dorso, asi la flexion es un giro sobre X local (cierra el dedo hacia la palma)
    finger_bones = []
    for s, side in ((1, 'Left'), (-1, 'Right')):
        F = G['FINGERS'][s]
        for fname in FINGER_NAMES:
            d = F[fname]
            if fname == 'Thumb':
                joints = [d['palm'], d['pos'][1], d['pos'][2], d['pos'][3]]; dors = [d['dorsal'][0], d['dorsal'][1], d['dorsal'][2]]
            else:
                joints = [d['pos'][0], d['pos'][3], d['pos'][4], d['pos'][5]]; dors = [d['dorsal'][1], d['dorsal'][3], d['dorsal'][4]]
            for k in range(3):
                name = side + fname + SEGS[k]
                e = eb.new(name); e.head = joints[k]; e.tail = joints[k + 1]
                e.align_roll(dors[k])
                e.parent = eb[side + 'Hand'] if k == 0 else eb[side + fname + SEGS[k - 1]]
                e.use_connect = k > 0
                e.use_deform = False                     # fuera de los pesos automaticos; se pesan a mano
                finger_bones.append(name)
    # sockets del arma: Arma_R donde la pistola ya estaba respecto a la mano derecha, Arma_L su reflejo en la izquierda
    SOCK = {}
    gun = bpy.data.objects.get(GUN)
    if gun:
        hb = eb[gun.parent_bone]
        GR = hb.matrix @ Matrix.Translation((0, hb.length, 0)) @ gun.matrix_parent_inverse @ gun.matrix_basis
        loc, rq, _ = GR.decompose()
        Grot = Matrix.Translation(loc) @ rq.to_matrix().to_4x4()
        resid = Grot.inverted() @ GR
        vs = [resid @ v.co for v in gun.data.vertices]
        ext = [max(v[i] for v in vs) - min(v[i] for v in vs) for i in range(3)]
        t = ext.index(min(ext))                              # eje mas delgado = izquierda-derecha de la pistola
        St = Matrix.Diagonal([(-1.0 if i == t else 1.0) for i in range(3)] + [1.0])
        GL = Matrix.Diagonal((-1.0, 1.0, 1.0, 1.0)) @ Grot @ St
        for name, M, par in (('Arma_R', Grot, 'RightHand'), ('Arma_L', GL, 'LeftHand')):
            R3 = M.to_3x3()
            e = eb.new(name); e.head = M.translation; e.tail = M.translation + R3.col[1] * SOCK_LEN
            e.align_roll(R3.col[2]); e.parent = eb[par]; e.use_deform = False
        SOCK = {'GR': GR, 'resid': resid, 'St': St}
        print("sockets del arma: eje delgado", 'XYZ'[t], [round(x, 3) for x in ext])
    bpy.ops.object.mode_set(mode='OBJECT')
    print("huesos:", len(arm_data.bones), "| dedos:", len(finger_bones))

    # ---------------------------------------------------------------- 3) pesos
    for o in vl.objects: o.select_set(False)
    mesh.select_set(True); rig.select_set(True); vl.objects.active = rig
    bpy.ops.object.parent_set(type='ARMATURE_AUTO')
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
    def set_weights(vi, ws):
        for g in list(mesh.data.vertices[vi].groups):
            mesh.vertex_groups[g.group].remove([vi])
        for name, w in ws.items():
            if w > 0: mesh.vertex_groups[name].add([vi], w, 'REPLACE')
    cnt = {}
    for v in mesh.data.vertices:
        p = v.co; side = 'Left' if p.x > 0 else 'Right'
        q = Vector((abs(p.x), p.y, p.z))
        if abs(p.x) > 0.015 and p.z < 0.6:                        # piernas: anillos horizontales
            for up, lo, rings in LEG_H:
                for z, w in rings:
                    if abs(p.z - z) < 1e-5:
                        set_weights(v.index, {side + up: 1 - w, side + lo: w}); cnt[up + lo] = cnt.get(up + lo, 0) + 1
        elif abs(p.x) > 0.20 and p.z > 0.8:                       # brazos: se vuelve a la pose original
            qo = J0 + ROTI @ (q - J0)
            for up, lo, rings in ARM_H:
                for sv, w in rings:
                    if min((qo - r).length for r in ring_pts[sv]) < 1e-5:
                        set_weights(v.index, {side + up: 1 - w, side + lo: w}); cnt[up + lo] = cnt.get(up + lo, 0) + 1
    print("vertices de bisagra (ambos lados):", cnt)
    # dedos: pesos por anillo (nudillo compartido con la mano, articulaciones a medias)
    for name in finger_bones:
        if name not in mesh.vertex_groups: mesh.vertex_groups.new(name=name)
    kd = KDTree(len(mesh.data.vertices))
    for v in mesh.data.vertices: kd.insert(v.co, v.index)
    kd.balance()
    RING_W = [{'H': 0.5, 0: 0.5}, {'H': 0.2, 0: 0.8}, {0: 1.0}, {0: 0.5, 1: 0.5}, {1: 0.5, 2: 0.5}, {2: 1.0}]
    THUMB_W = [{'H': 0.6, 0: 0.4}, {0: 0.5, 1: 0.5}, {1: 0.5, 2: 0.5}, {2: 1.0}]
    nf = 0
    for s, side in ((1, 'Left'), (-1, 'Right')):
        for fname in FINGER_NAMES:
            rings = G['FINGERS'][s][fname]['rings']
            table = THUMB_W if fname == 'Thumb' else RING_W
            for ring, wt in zip(rings, table):
                ws = {(side + 'Hand' if k == 'H' else side + fname + SEGS[k]): w for k, w in wt.items()}
                for p in ring:
                    co, idx, dist = kd.find(p)
                    assert dist < 1e-5, (side, fname, dist)
                    set_weights(idx, ws); nf += 1
    for name in finger_bones: arm_data.bones[name].use_deform = True
    sin_peso = sum(1 for v in mesh.data.vertices if not any(g.weight > 1e-4 for g in v.groups))
    print("vertices de dedos pesados:", nf, "| grupos:", len(mesh.vertex_groups), "| vertices sin peso:", sin_peso)

    # ---------------------------------------------------------------- 4) acciones
    A = {'__name__': 'anim', 'rig': rig, 'old': old, 'sc': sc, 'G': G, 'mesh': mesh,
         'assign_action': assign_action, 'FINGER_NAMES': FINGER_NAMES, 'SEGS': SEGS}
    exec(open(os.path.join(LUKAS, 'lukas_rh07_animaciones.py'), encoding='utf-8').read(), A)
    made = []
    for src_name, new_name, fingers in TRANSFERS:
        made.append(A['transfer'](src_name, new_name, fingers))
    made += A['build_authored']()
    if CORRECTIVE:
        C = dict(A); C.update({'CORRECTIVE': CORRECTIVE})
        exec(open(os.path.join(LUKAS, 'lukas_rh07_correctivo.py'), encoding='utf-8').read(), C)
        C['build_corrective'](mesh, rig, G, {a.name: a for a in made}, sc)
    # pistas de NLA (una por accion, silenciadas para que no se mezclen al ver la accion activa)
    rig.animation_data_create()
    for act in made:
        tr = rig.animation_data.nla_tracks.new(); tr.name = act.name
        st = tr.strips.new(act.name, int(act.frame_range[0]), act)
        try:
            if act.slots: st.action_slot = act.slots[0]
        except AttributeError:
            pass
        tr.mute = True
    removed = sum(clean_action(a) for a in made)
    for a in made: a.use_fake_user = True
    print("curvas simplificadas o quitadas:", removed)
    nb = len([b for b in rig.data.bones if b.name not in SOCKETS])
    for a in made:
        rb = {fc.data_path.split('"')[1] for fc in fcurves_of(a) if fc.data_path.endswith('rotation_quaternion')}
        sk = {fc.data_path.split('"')[1] for fc in fcurves_of(a) if fc.data_path.endswith('.scale')}
        if len(rb) != nb or sk != set(SOCKETS): print("AVISO curvas incompletas:", a.name, len(rb), nb, sorted(sk))
    print("acciones:")
    for a in made:
        print(f"   {a.name:28s} {int(a.frame_range[0]):3d}-{int(a.frame_range[1]):3d}  bucle={'si' if a.use_cyclic else 'no'}  curvas={len(fcurves_of(a))}")
    idle = bpy.data.actions['Anim_Idle']
    assign_action(rig, idle)

    # pistola de prueba (la de Player.blend) en el socket derecho y su reflejo en el izquierdo; la escala de cada
    # socket (clave en cada accion) decide en que mano se ve
    gun = bpy.data.objects.get(GUN)
    if gun and SOCK:
        gl = gun.copy(); gl.name = GUN_L
        for c in gun.users_collection: c.objects.link(gl)
        for o, bn, basis in ((gun, 'Arma_R', SOCK['resid']), (gl, 'Arma_L', SOCK['St'] @ SOCK['resid'] @ SOCK['St'])):
            o.parent = rig; o.parent_type = 'BONE'; o.parent_bone = bn
            o.matrix_parent_inverse = Matrix.Translation((0, -SOCK_LEN, 0)); o.matrix_basis = basis
        rig.data.pose_position = 'REST'; vl.update()
        S = Matrix.Diagonal((-1.0, 1.0, 1.0, 1.0))
        errR = max(abs(a - b) for ra, rb in zip(gun.matrix_world, SOCK['GR']) for a, b in zip(ra, rb))
        errL = max(abs(a - b) for ra, rb in zip(gl.matrix_world, S @ SOCK['GR'] @ SOCK['St']) for a, b in zip(ra, rb))
        rig.data.pose_position = 'POSE'; vl.update()
        print(f"pistolas en Arma_R / Arma_L (desvio en reposo {errR:.2e} / {errL:.2e})")
    if os.environ.get('LUKAS_WORK'):          # copia de trabajo que aun conserva el esqueleto viejo (para iterar poses)
        bpy.ops.wm.save_as_mainfile(filepath=os.environ['LUKAS_WORK'], copy=True, compress=False)
        print("copia de trabajo:", os.environ['LUKAS_WORK'])

    # ---------------------------------------------------------------- 5) archivo limpio
    keep_actions = set(made)
    sc.frame_start, sc.frame_end = int(idle.frame_range[0]), int(idle.frame_range[1])
    sc.frame_set(sc.frame_start)
    refs = bpy.data.collections.new('Referencias'); sc.collection.children.link(refs)
    for n in ('Empty', 'Empty.001'):
        o = bpy.data.objects[n]
        for c in list(o.users_collection): c.objects.unlink(o)
        refs.objects.link(o)
    keep = {mesh.name, rig.name, 'Empty', 'Empty.001', GUN, GUN_L}
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
    print("archivo -> objetos", [o.name for o in bpy.data.objects], "| acciones", len(bpy.data.actions),
          "| materiales", [m.name for m in bpy.data.materials], "| uv cuerpo", len(mesh.data.uv_layers))
    if os.environ.get('LUKAS_NOSAVE'):
        return
    out = os.path.join(LUKAS, 'Lukas_RH07.blend')
    bpy.ops.wm.save_as_mainfile(filepath=out, relative_remap=True, compress=False)
    print("GUARDADO", out, os.path.getsize(out))

ok = True
try:
    main()
except Exception:
    traceback.print_exc(); ok = False
print("FIN", "OK" if ok else "ERROR")
if LOG:
    _log.flush()
    open(LOG + '.done', 'w').write('ok' if ok else 'error')
