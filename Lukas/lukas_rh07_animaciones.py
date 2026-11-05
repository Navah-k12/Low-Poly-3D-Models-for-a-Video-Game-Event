# =====================================================================================
# Lukas R-H-07 - acciones de animacion (se carga desde lukas_rh07_rig.py con rig, old, sc, G).
# - transfer(): copia una accion de PJ_Rig al esqueleto nuevo (misma pose en el mundo, hueso a hueso)
# - sistema de poses para las acciones nuevas: pose base del juego + giros de cadera/columna/cabeza
#   en ejes del personaje + IK de 2 huesos en brazos y piernas + orientacion de mano + dedos
# Ejes del personaje: mira hacia -Y, su izquierda es +X, Z arriba. Giros (grados):
#   pitch > 0 se inclina hacia delante, yaw > 0 gira hacia su izquierda, roll > 0 se ladea a su izquierda
# =====================================================================================
import bpy, math
from mathutils import Matrix, Vector, Quaternion

I4 = Matrix.Identity(4)
T = Matrix.Translation
def Rd(a): return math.radians(a)

bones = rig.data.bones
NAMES = [b.name for b in bones]
PARENT = {b.name: (b.parent.name if b.parent else None) for b in bones}
def _depth(n):
    d = 0
    while PARENT[n]: n = PARENT[n]; d += 1
    return d
ORDER = sorted(NAMES, key=_depth)
REST = {n: bones[n].matrix_local.copy() for n in NAMES}
REL = {n: (REST[PARENT[n]].inverted() @ REST[n]) if PARENT[n] else REST[n].copy() for n in NAMES}
LEN = {n: bones[n].length for n in NAMES}
BODY = [b.name for b in old.data.bones]
SOCKETS = [n for n in ('Arma_R', 'Arma_L') if n in bones]     # sockets del arma (no deforman)
FINGERS = [n for n in NAMES if n not in BODY and n not in SOCKETS]
KEYED = [n for n in NAMES if n not in SOCKETS]
# en que mano se ve el arma en cada accion (las que no salen aqui van sin arma)
WEAPON = {'Anim_Combat_Aim': 'R', 'Anim_Aim_Walk': 'R', 'Anim_Combat_Shoot': 'R', 'Anim_Combat_Reload_Loop': 'R',
          'Anim_Melee_Push': 'R', 'Anim_Idle_Hurt': 'L', 'Anim_Idle_Critical': 'L'}
HIDE = 1e-4                                                 # escala del socket cuando el arma no se ve
# dorso de cada mano en reposo: el eje Z de la falange proximal del medio (orientado al dorso)
HAND_D = {s: REST[s + 'MiddleProximal'].to_3x3().col[2].normalized() for s in ('Left', 'Right')}

# signo de la flexion de cada falange: girar sobre X local debe cerrar el dedo hacia la palma
FLEX = {}
for n in FINGERS:
    M = REST[n]; z = M.to_3x3().col[2]
    t0 = M @ Vector((0, LEN[n], 0)); t1 = (M @ Matrix.Rotation(Rd(20), 4, 'X')) @ Vector((0, LEN[n], 0))
    FLEX[n] = 1.0 if (t1 - t0).dot(-z) > 0 else -1.0

# poses de dedos: grados de flexion (proximal, intermedia, distal)
FPOSE = {
    'relaxed': {'Thumb': (8, 10, 8),   'Index': (12, 14, 8),  'Middle': (15, 18, 10), 'Ring': (18, 20, 12), 'Little': (20, 22, 14)},
    'grip':    {'Thumb': (30, 28, 22), 'Index': (32, 42, 22), 'Middle': (75, 85, 45), 'Ring': (80, 88, 45), 'Little': (85, 90, 45)},
    'trigger': {'Thumb': (30, 28, 22), 'Index': (48, 62, 34), 'Middle': (75, 85, 45), 'Ring': (80, 88, 45), 'Little': (85, 90, 45)},
    'support': {'Thumb': (22, 15, 10), 'Index': (55, 65, 30), 'Middle': (60, 70, 35), 'Ring': (65, 75, 40), 'Little': (70, 78, 40)},
    'pinch':   {'Thumb': (38, 30, 25), 'Index': (42, 48, 26), 'Middle': (58, 62, 30), 'Ring': (62, 70, 35), 'Little': (66, 75, 35)},
    'fist':    {'Thumb': (38, 35, 25), 'Index': (85, 95, 60), 'Middle': (88, 98, 60), 'Ring': (88, 98, 60), 'Little': (85, 95, 60)},
    'open':    {'Thumb': (2, 3, 2),    'Index': (3, 3, 2),    'Middle': (3, 3, 2),    'Ring': (4, 3, 2),    'Little': (5, 4, 3)},
    'grasp':   {'Thumb': (26, 20, 15), 'Index': (50, 55, 30), 'Middle': (55, 60, 32), 'Ring': (58, 62, 34), 'Little': (60, 65, 35)},
    'claw':    {'Thumb': (20, 25, 25), 'Index': (35, 55, 45), 'Middle': (35, 55, 45), 'Ring': (38, 58, 45), 'Little': (40, 60, 45)},
    'clutch':  {'Thumb': (14, 12, 8),  'Index': (22, 26, 14), 'Middle': (24, 28, 14), 'Ring': (26, 30, 16), 'Little': (28, 32, 16)},
}
def finger_bases(fp):
    out = {}
    for side, key in (('Left', 'L'), ('Right', 'R')):
        pose = FPOSE[fp.get(key, 'relaxed')]
        for fname in FINGER_NAMES:
            for k, seg in enumerate(SEGS):
                n = side + fname + seg
                out[n] = Matrix.Rotation(FLEX[n] * Rd(pose[fname][k]), 4, 'X')
    return out

def rot_world(pitch=0.0, yaw=0.0, roll=0.0):
    return Matrix.Rotation(Rd(yaw), 4, 'Z') @ Matrix.Rotation(Rd(pitch), 4, 'X') @ Matrix.Rotation(Rd(roll), 4, 'Y')

# ---------------------------------------------------------------- poses base del juego
SRC = {'idle': 'Lukas_Idle_Bien', 'aim': 'Lukas_Apuntar_Bien', 'crouch': 'Lukas_Agachado_Bien'}
_BASES = {}
def base_bases(key, frame):
    k = (key, frame)
    if k not in _BASES:
        assign_action(old, bpy.data.actions[SRC[key]]); sc.frame_set(frame)
        P = {n: old.pose.bones[n].matrix.copy() for n in BODY}
        _BASES[k] = {n: (REL[n].inverted() @ P[PARENT[n]].inverted() @ P[n]) if PARENT[n] else REL[n].inverted() @ P[n] for n in BODY}
    return _BASES[k]

OVERLAY = {'Hips': 'hips_rot', 'Spine': 'spine', 'Chest': 'chest', 'Neck': 'neck', 'Head': 'head',
           'LeftShoulder': 'shL', 'RightShoulder': 'shR'}
def fk(B, spec):
    P = {}
    for n in ORDER:
        p = PARENT[n]
        M = (P[p] @ REL[n] if p else REL[n]) @ B.get(n, I4)
        if n == 'Hips' and 'hips' in spec: M = T(Vector(spec['hips'])) @ M
        ov = spec.get(OVERLAY.get(n, '_'))
        if ov:
            h = M.translation.copy(); M = T(h) @ rot_world(*ov) @ T(-h) @ M
        P[n] = M
    return P

def _aim_bone(M, head, to):
    y = M.to_3x3().col[1].normalized()
    q = y.rotation_difference((to - head).normalized())
    return T(head) @ q.to_matrix().to_4x4() @ T(-head) @ M

def _two_bone(S, W, L1, L2, pole):
    dv = W - S; d = max(min(dv.length, L1 + L2 - 0.003), abs(L1 - L2) + 0.01)
    dv.normalize(); W = S + dv * d
    a = (L1 * L1 - L2 * L2 + d * d) / (2 * d); h = math.sqrt(max(L1 * L1 - a * a, 0.0))
    n = pole - dv * pole.dot(dv)
    if n.length < 1e-6: n = Vector((0, 0, -1)) - dv * (-dv.z)
    n.normalize()
    return S + dv * a + n * h, W

def _repropagate(P, B, prefix):
    for n in ORDER:
        if n in FINGERS and n.startswith(prefix):
            P[n] = P[PARENT[n]] @ REL[n] @ B.get(n, I4)

def ik_arm(P, Pb, B, side, a):
    s = 1.0 if side == 'Left' else -1.0
    ua, la, hd = side + 'UpperArm', side + 'LowerArm', side + 'Hand'
    W = Vector(a['target']) if 'target' in a else Pb[hd].translation + Vector(a.get('offset', (0, 0, 0)))
    S = P[ua].translation.copy()
    E, W = _two_bone(S, W, LEN[ua], LEN[la], Vector(a.get('pole', (s * 0.5, 0.6, -0.6))))
    P[ua] = _aim_bone(P[ua], S, E)
    P[la] = P[ua] @ REL[la] @ B.get(la, I4)
    P[la] = _aim_bone(P[la], E, W)
    if 'hand' in a:                         # orientacion absoluta: (direccion de los dedos, direccion del dorso)
        yt = Vector(a['hand'][0]).normalized(); dt = Vector(a['hand'][1]); dt = (dt - yt * dt.dot(yt)).normalized()
        Rr = REST[hd].to_3x3(); yr = Rr.col[1].normalized(); dr = HAND_D[side]; dr = (dr - yr * dr.dot(yr)).normalized()
        Fr = Matrix((yr, dr, yr.cross(dr))).transposed(); Ft = Matrix((yt, dt, yt.cross(dt))).transposed()
        rot = Ft @ Fr.transposed() @ Rr
    else:
        rot = Pb[hd].to_3x3().normalized()
    if 'hand_rot' in a: rot = rot_world(*a['hand_rot']).to_3x3() @ rot
    # la mitad del giro de la mano sobre el eje del antebrazo la toma el antebrazo (mejor deformacion)
    q_rest = REST[la].to_quaternion().inverted() @ REST[hd].to_quaternion()
    qd = P[la].to_quaternion().inverted() @ rot.to_quaternion() @ q_rest.inverted()
    tw = 2.0 * math.atan2(qd.y, qd.w)
    tw = math.atan2(math.sin(tw), math.cos(tw))
    P[la] = P[la] @ Matrix.Rotation(0.5 * tw, 4, 'Y')
    P[hd] = T(W) @ rot.to_4x4()
    _repropagate(P, B, side)

def fk_arm(P, B, side, a):
    """brazo desde la pose A de reposo (la que mejor deforma) con giros en ejes del cuerpo:
    abd > 0 lo separa mas del cuerpo, fwd > 0 lo lleva hacia delante, elbow dobla el codo, twist gira el brazo"""
    sg = 1.0 if side == 'Left' else -1.0
    ua, la, hd = side + 'UpperArm', side + 'LowerArm', side + 'Hand'
    Rb = P['Chest'].to_3x3().normalized() @ REST['Chest'].to_3x3().normalized().inverted()
    fwd = Rb @ Vector((0, -1, 0))
    S = P[ua].translation.copy()
    R = (Matrix.Rotation(Rd(a.get('fwd', 0.0)), 4, Rb @ Vector((-1, 0, 0)))
         @ Matrix.Rotation(Rd(a.get('abd', 0.0)), 4, Rb @ Vector((0, -sg, 0))))
    P[ua] = T(S) @ R @ T(-S) @ P[ua]
    if a.get('twist'):
        P[ua] = T(S) @ Matrix.Rotation(Rd(sg * a['twist']), 4, P[ua].to_3x3().col[1].normalized()) @ T(-S) @ P[ua]
    P[la] = P[ua] @ REL[la] @ B.get(la, I4)
    if a.get('elbow'):
        E = P[la].translation.copy(); y = P[la].to_3x3().col[1].normalized()
        ax = (E - S).normalized().cross(fwd)
        if ax.length < 1e-6: ax = P[la].to_3x3().col[0]
        ax.normalize(); ang = Rd(a['elbow'])
        if (Matrix.Rotation(ang, 3, ax) @ y).dot(fwd) < y.dot(fwd): ang = -ang      # el codo cierra hacia delante
        P[la] = T(E) @ Matrix.Rotation(ang, 4, ax) @ T(-E) @ P[la]
    P[hd] = P[la] @ REL[hd] @ B.get(hd, I4)
    _repropagate(P, B, side)

def ik_leg(P, Pb, B, side, l):
    s = 1.0 if side == 'Left' else -1.0
    ul, ll, ft, toe = side + 'UpperLeg', side + 'LowerLeg', side + 'Foot', side + 'Toes'
    W = Vector(l['target']) if 'target' in l else Pb[ft].translation + Vector(l.get('offset', (0, 0, 0)))
    S = P[ul].translation.copy()
    E, W = _two_bone(S, W, LEN[ul], LEN[ll], Vector(l.get('pole', (s * 0.15, -1.0, 0.1))))
    P[ul] = _aim_bone(P[ul], S, E)
    P[ll] = P[ul] @ REL[ll] @ B.get(ll, I4)
    P[ll] = _aim_bone(P[ll], E, W)
    rot = Pb[ft].to_3x3().normalized()
    if 'foot' in l: rot = rot_world(*l['foot']).to_3x3() @ rot
    P[ft] = T(W) @ rot.to_4x4()
    P[toe] = P[ft] @ REL[toe] @ B.get(toe, I4)
    if 'toes' in l:
        h = P[toe].translation.copy(); P[toe] = T(h) @ rot_world(*l['toes']) @ T(-h) @ P[toe]

def compose(spec):
    B = dict(base_bases(*spec.get('base', ('idle', 1))))
    fp = {'L': 'relaxed', 'R': 'relaxed'}; fp.update(spec.get('fingers', {}))
    B.update(finger_bases(fp))
    for side, key in (('Left', 'fkL'), ('Right', 'fkR')):        # brazo en FK: parte de la pose A de reposo
        if key in spec:
            for part in ('Shoulder', 'UpperArm', 'LowerArm', 'Hand'): B[side + part] = I4
    Pb = fk(B, {})
    P = fk(B, spec)
    for side, key in (('Left', 'fkL'), ('Right', 'fkR')):
        if key in spec: fk_arm(P, B, side, spec[key])
    for side, key in (('Left', 'armL'), ('Right', 'armR')):
        if key in spec: ik_arm(P, Pb, B, side, spec[key])
    if not spec.get('legs_fk'):
        for side, key in (('Left', 'legL'), ('Right', 'legR')):
            ik_leg(P, Pb, B, side, spec.get(key, {}))
    return P

def bases_from(P, names):
    return {n: (REL[n].inverted() @ P[PARENT[n]].inverted() @ P[n]) if PARENT[n] else REL[n].inverted() @ P[n] for n in names}

def key_weapon(name, f0, f1):
    """visibilidad del arma por accion: escala 1 en el socket de la mano que la lleva, casi 0 en el otro"""
    hand = WEAPON.get(name)
    for bn, side in (('Arma_R', 'R'), ('Arma_L', 'L')):
        if bn not in bones: continue
        pb = rig.pose.bones[bn]; v = 1.0 if hand == side else HIDE
        pb.scale = (v, v, v)
        for f in (f0, f1): pb.keyframe_insert('scale', frame=f, group=bn)

def key_pose(f, P, prevq, names=None):
    names = names or KEYED
    Bs = bases_from(P, names)
    for n in names:
        loc, q, sca = Bs[n].decompose()
        if n in prevq and q.dot(prevq[n]) < 0: q.negate()
        prevq[n] = q.copy()
        pb = rig.pose.bones[n]
        pb.rotation_mode = 'QUATERNION'
        pb.location = loc if (n == 'Hips' or not bones[n].use_connect) else Vector()
        pb.rotation_quaternion = q; pb.scale = (1, 1, 1)
        pb.keyframe_insert('location', frame=f, group=n)
        pb.keyframe_insert('rotation_quaternion', frame=f, group=n)

def make_action(name, keys, cyclic=False):
    act = bpy.data.actions.new(name); assign_action(rig, act)
    prevq = {}
    for f, spec in keys:
        key_pose(f, compose(spec), prevq)
    key_weapon(name, keys[0][0], keys[-1][0])
    act.use_frame_range = True; act.frame_start = keys[0][0]; act.frame_end = keys[-1][0]; act.use_cyclic = cyclic
    print(f"accion nueva {name}: {keys[0][0]}-{keys[-1][0]}, {len(keys)} poses clave, bucle={cyclic}, arma={WEAPON.get(name, '-')}")
    return act

# ---------------------------------------------------------------- traspaso de acciones del juego
# idles con herida: la mano derecha sigue apretando la herida y la izquierda sostiene el arma lista para disparar
LOW_READY_L = {'offset': (-0.05, -0.17, 0.10), 'hand': ((0.0, -1.0, -0.55), (1.0, 0.0, 0.15)), 'pole': (0.5, 0.7, -0.5)}
ARM_FIX = {'Anim_Idle_Hurt': {'Left': LOW_READY_L}, 'Anim_Idle_Critical': {'Left': LOW_READY_L}}

def transfer(src_name, new_name, fingers):
    src = bpy.data.actions[src_name]
    assign_action(old, src)
    f0, f1 = int(src.frame_range[0]), int(src.frame_range[1])
    act = bpy.data.actions.new(new_name); assign_action(rig, act)
    fb = finger_bases({'L': fingers.get('L', 'relaxed'), 'R': fingers.get('R', 'relaxed')})
    fix = ARM_FIX.get(new_name, {})
    prevq = {}; err = 0.0
    for f in range(f0, f1 + 1):
        sc.frame_set(f)
        P = {n: old.pose.bones[n].matrix.copy() for n in BODY}
        for n in ORDER:
            if n in FINGERS: P[n] = P[PARENT[n]] @ REL[n] @ fb[n]
        if fix:
            Pb = dict(P)
            for side, a in fix.items(): ik_arm(P, Pb, fb, side, a)
        key_pose(f, P, prevq, KEYED if f in (f0, f1) else BODY)     # dedos: pose fija (clave al inicio y al final)
    key_weapon(new_name, f0, f1)
    act.use_frame_range = True; act.frame_start = f0; act.frame_end = f1; act.use_cyclic = src.use_cyclic
    moved = {s + p for s in fix for p in ('LowerArm', 'Hand')}
    for f in (f0, (f0 + f1) // 2, f1):
        sc.frame_set(f)
        for n in BODY:
            if n not in moved: err = max(err, (rig.pose.bones[n].head - old.pose.bones[n].head).length)
    print(f"traspaso {src_name} -> {new_name}: {f0}-{f1}, bucle={act.use_cyclic}, error de hueso {err*1000:.3f} mm")
    return act

# ---------------------------------------------------------------- acciones nuevas
def build_authored():
    acts = []
    AIM = ('aim', 1)
    GR = {'R': 'grip', 'L': 'support'}

    # --- disparo: retroceso en brazos (manos suben y vuelven), muñecas, pecho, columna y cabeza
    def shot(k, idx=False):
        return {'base': AIM, 'hips': (0, 0.015 * k, 0), 'chest': (-8 * k, 0, 0), 'spine': (-3 * k, 0, 0), 'head': (-5 * k, 0, 0),
                'shL': (-5 * k, 0, 0), 'shR': (-5 * k, 0, 0),
                'armR': {'offset': (0, 0.09 * k, 0.11 * k), 'hand_rot': (-30 * k, 0, 0)},
                'armL': {'offset': (0, 0.09 * k, 0.11 * k), 'hand_rot': (-30 * k, 0, 0)},
                'fingers': {'R': 'trigger' if idx else 'grip', 'L': 'support'}}
    acts.append(make_action('Anim_Combat_Shoot', [
        (1, shot(0)), (2, shot(0.15, True)), (3, shot(1.0, True)), (6, shot(0.45)), (10, shot(0.12)), (14, shot(0))]))

    # --- recarga bala a bala: 17 fotogramas (0.71 s) por bala; empieza y acaba en la misma pose
    def reload(lh, ldirs, fl, head_yaw=0.0):
        return {'base': AIM, 'chest': (8, 0, 0), 'spine': (4, 0, 0), 'neck': (8, 0, 0), 'head': (16, head_yaw, 0),
                'armR': {'target': (-0.06, -0.30, 1.12), 'hand_rot': (40, 30, 0), 'pole': (-0.6, 0.4, -0.6)},
                'armL': {'target': lh, 'hand': ldirs, 'pole': (0.6, 0.3, -0.7)},
                'fingers': {'R': 'grip', 'L': fl}}
    AT_GUN = ((0.01, -0.31, 1.15), ((-0.7, -0.3, -0.6), (0.3, -0.2, 0.9)))
    UP_1 = ((0.10, -0.26, 1.08), ((-0.3, -0.4, -0.85), (0.8, -0.1, 0.4)))
    POUCH = ((0.20, -0.16, 1.05), ((0.0, -0.25, -1.0), (1.0, 0.0, 0.0)))
    UP_2 = ((0.12, -0.27, 1.10), ((-0.4, -0.4, -0.8), (0.7, -0.2, 0.5)))
    acts.append(make_action('Anim_Combat_Reload_Loop', [
        (1, reload(*AT_GUN, 'pinch')), (4, reload(*UP_1, 'open', 6)), (8, reload(*POUCH, 'open', 14)),
        (10, reload(*POUCH, 'pinch', 14)), (14, reload(*UP_2, 'pinch', 5)), (17, reload(*AT_GUN, 'pinch'))], cyclic=True))

    # --- abrir una puerta (mano izquierda: la derecha lleva el revolver)
    HANDLE = (0.24, -0.44, 1.00)
    acts.append(make_action('Anim_Interact_Door', [
        (1, {}),
        (10, {'hips': (0, -0.03, -0.01), 'chest': (6, 8, 0), 'spine': (3, 0, 0), 'head': (6, 10, 0),
              'armL': {'target': HANDLE, 'hand': ((0.05, -1, -0.12), (0, -0.1, 1)), 'pole': (0.7, 0.3, -0.5)}, 'fingers': {'L': 'open'}}),
        (15, {'hips': (0, -0.03, -0.01), 'chest': (6, 8, 0), 'spine': (3, 0, 0), 'head': (6, 10, 0),
              'armL': {'target': (0.24, -0.44, 0.98), 'hand': ((0.05, -1, -0.12), (0.35, -0.1, 0.9)), 'pole': (0.7, 0.3, -0.5)}, 'fingers': {'L': 'grasp'}}),
        (25, {'hips': (0, -0.07, -0.02), 'chest': (10, 6, 0), 'spine': (5, 0, 0), 'head': (4, 6, 0),
              'armL': {'target': (0.24, -0.56, 0.98), 'hand': ((0.05, -1, -0.12), (0.35, -0.1, 0.9)), 'pole': (0.7, 0.3, -0.5)},
              'legL': {'offset': (0, -0.10, 0)}, 'fingers': {'L': 'grasp'}}),
        (31, {'hips': (0, -0.06, -0.02), 'chest': (8, 5, 0), 'spine': (4, 0, 0), 'head': (3, 5, 0),
              'armL': {'target': (0.26, -0.52, 1.00), 'hand': ((0.05, -1, -0.12), (0, -0.1, 1)), 'pole': (0.7, 0.3, -0.5)},
              'legL': {'offset': (0, -0.10, 0)}, 'fingers': {'L': 'open'}}),
        (42, {})]))

    # --- recoger un objeto del suelo (mano izquierda)
    def bend(k, lh, ldirs, fl):
        return {'hips': (0, 0.05 * k, -0.40 * k), 'hips_rot': (16 * k, 0, 0), 'spine': (20 * k, 0, 0), 'chest': (16 * k, 0, 0),
                'neck': (-8 * k, 0, 0), 'head': (10 * k, 0, 0),
                'armL': {'target': lh, 'hand': ldirs, 'pole': (0.6, 0.5, -0.4)},
                'armR': {'target': (-0.16, -0.20 + 0.05 * (1 - k), 0.86 - 0.36 * k), 'pole': (-0.6, 0.5, -0.4)},
                'fingers': {'L': fl}}
    FLOOR = (0.15, -0.33, 0.07)
    DOWN = ((0.05, -0.3, -0.95), (0.6, -0.5, 0.2))
    acts.append(make_action('Anim_Interact_Pickup', [
        (1, {}),
        (14, bend(1.0, FLOOR, DOWN, 'open')),
        (19, bend(1.0, (0.15, -0.33, 0.09), DOWN, 'fist')),
        (30, bend(0.5, (0.18, -0.26, 0.62), ((0.0, -0.3, -0.95), (0.9, 0, 0.3)), 'fist')),
        (38, {'armL': {'target': (0.12, -0.24, 1.04), 'hand': ((-0.5, -0.4, 0.75), (0.6, -0.6, 0.4)), 'pole': (0.6, 0.4, -0.6)},
              'head': (12, 8, 0), 'fingers': {'L': 'fist'}}),
        (46, {})]))

    # --- empujon / golpe con la culata (paso al frente con la pierna izquierda)
    acts.append(make_action('Anim_Melee_Push', [
        (1, {'base': AIM, 'fingers': GR}),
        (6, {'base': AIM, 'chest': (-3, -14, 0), 'spine': (0, -6, 0), 'fingers': GR,
             'armR': {'offset': (0.02, 0.20, -0.10), 'hand_rot': (-55, 0, 0)}, 'armL': {'offset': (0.0, 0.20, -0.10), 'hand_rot': (-40, 0, 0)}}),
        (10, {'base': AIM, 'hips': (0, -0.11, -0.04), 'chest': (12, 12, 0), 'spine': (5, 4, 0), 'fingers': GR,
              'armR': {'offset': (0.0, -0.12, 0.02), 'hand_rot': (-60, 0, 0)}, 'armL': {'offset': (0.0, -0.10, 0.0), 'hand_rot': (-45, 0, 0)},
              'legL': {'offset': (0, -0.24, 0)}, 'legR': {'offset': (0, 0.06, 0)}}),
        (13, {'base': AIM, 'hips': (0, -0.11, -0.05), 'chest': (10, 10, 0), 'spine': (5, 4, 0), 'fingers': GR,
              'armR': {'offset': (0.0, -0.10, 0.0), 'hand_rot': (-55, 0, 0)}, 'armL': {'offset': (0.0, -0.08, -0.01), 'hand_rot': (-40, 0, 0)},
              'legL': {'offset': (0, -0.24, 0)}, 'legR': {'offset': (0, 0.06, 0)}}),
        (24, {'base': AIM, 'fingers': GR})]))

    # --- golpe al suelo sin arma (enemigo caido delante): carga el puño junto al hombro (codo abajo) y golpea hacia abajo
    def gstrike(dz=0.0):
        return {'hips': (0, -0.05, -0.36 + dz), 'hips_rot': (22, 0, 0), 'spine': (24, 0, 0), 'chest': (20, 6, 0), 'head': (14, 0, 0),
                'armR': {'target': (-0.08, -0.50, 0.20 + dz), 'hand': ((0, -0.35, -1), (0, -1, 0.35)), 'pole': (-0.8, 0.3, 0.2)},
                'armL': {'target': (0.16, -0.26, 0.50 + dz), 'pole': (0.7, 0.3, -0.3)},
                'legL': {'offset': (0, -0.14, 0)}, 'legR': {'offset': (0, 0.14, 0)}, 'fingers': {'R': 'fist', 'L': 'open'}}
    acts.append(make_action('Anim_Melee_GroundStrike', [
        (1, {}),
        (8, {'hips': (0, -0.02, -0.10), 'hips_rot': (6, 0, 0), 'chest': (6, -14, 0), 'spine': (4, -4, 0), 'head': (16, 0, 0),
             'armR': {'target': (-0.25, -0.04, 1.26), 'hand': ((0, -0.25, 1), (-0.6, 0.8, 0)), 'pole': (-0.3, 0.4, -1.0)},
             'armL': {'target': (0.26, -0.20, 0.95), 'pole': (0.7, 0.3, -0.6)},
             'legL': {'offset': (0, -0.10, 0)}, 'fingers': {'R': 'fist', 'L': 'open'}}),
        (13, gstrike()),
        (16, gstrike(0.03)),
        (32, {})]))

    # --- remate con el punzon de descarga (en el puño derecho): agarra con la izquierda, carga el puño junto a la
    #     cadera (codo atras, sin forzar el hombro), lo clava al frente, descarga (temblor) y suelta
    CLAW_L = {'target': (0.10, -0.60, 1.30), 'hand': ((0.05, -1, 0.05), (0.2, 0, 1)), 'pole': (0.7, 0.4, -0.5)}
    CHAMBER = {'target': (-0.22, 0.06, 1.00), 'hand': ((0.05, -1, -0.2), (-1, 0, 0)), 'pole': (-0.4, 1.0, -0.4)}
    STAB = (-0.04, -0.60, 1.20)
    def fin(hips, chest, armR, lh=None, fl='claw', extra=None):
        d = {'hips': hips, 'chest': chest, 'spine': (chest[0] * 0.4, chest[1] * 0.4, 0), 'armR': armR,
             'armL': dict(CLAW_L, target=lh or CLAW_L['target']), 'legL': {'offset': (0, -0.25, 0)}, 'fingers': {'R': 'fist', 'L': fl}}
        if extra: d.update(extra)
        return d
    def stab(dx=0.0, dz=0.0):
        return {'target': (STAB[0] + dx, STAB[1], STAB[2] + dz), 'hand': ((0.1, -1, 0.05), (-0.9, 0, 0.4)), 'pole': (-0.7, 0.2, -0.7)}
    keys = [(1, {}),
            (9, fin((0, -0.12, -0.04), (4, -16, 0), CHAMBER)),
            (16, fin((0, -0.12, -0.05), (3, -20, 0), dict(CHAMBER, target=(-0.23, 0.09, 1.01)))),
            (21, fin((0, -0.16, -0.06), (12, 12, 0), stab()))]
    for i, f in enumerate(range(24, 41, 2)):         # descarga: temblor
        sgn = 1 if i % 2 == 0 else -1
        keys.append((f, fin((0, -0.16, -0.06), (12, 12, 2.5 * sgn), stab(0.008 * sgn, 0.006 * sgn), lh=(0.10 - 0.006 * sgn, -0.60, 1.30),
                            extra={'head': (4, 4 * sgn, 0)})))
    keys += [(46, fin((0, -0.13, -0.05), (6, -6, 0), {'target': (-0.18, -0.30, 1.10), 'hand': ((0.05, -1, -0.1), (-1, 0, 0.2)), 'pole': (-0.6, 0.6, -0.6)})),
             (52, fin((0, -0.10, -0.04), (8, 0, 0), {'target': (-0.20, -0.20, 1.04), 'hand': ((0.05, -1, -0.2), (-1, 0, 0)), 'pole': (-0.5, 0.8, -0.5)},
                      lh=(0.12, -0.70, 1.26), fl='open')),
             (64, {})]
    acts.append(make_action('Anim_Melee_Finisher', keys))

    # --- muerte por golpe: el impacto lo echa atras, se doblan las rodillas y cae de espaldas. Brazos en FK desde la
    #     pose A (sueltos, sin torcer el antebrazo); en el suelo quedan abiertos con los codos un poco doblados
    def lie_b(dz=0.0, head=(12, 18, 0)):
        return {'hips': (0, 0.46, -0.74 + dz), 'hips_rot': (-82, 0, 0), 'spine': (4, 0, 0), 'chest': (2, 0, 0), 'head': head,
                'fkL': {'abd': 15, 'fwd': -27, 'elbow': 18}, 'fkR': {'abd': 18, 'fwd': -29, 'elbow': 22},
                'legL': {'target': (0.17, -0.18, 0.148), 'pole': (0.2, -0.2, 1), 'foot': (-70, 0, 0)},
                'legR': {'target': (-0.15, -0.10, 0.148), 'pole': (-0.2, -0.2, 1), 'foot': (-70, 0, 0)}}
    acts.append(make_action('Anim_Death_Strike', [
        (1, {}),
        (4, {'hips': (0, 0.04, 0), 'chest': (-25, 8, 0), 'spine': (-10, 0, 0), 'head': (-35, 10, 0),
             'fkL': {'abd': 8, 'fwd': 30, 'elbow': 40}, 'fkR': {'abd': 10, 'fwd': 26, 'elbow': 42}, 'fingers': {'L': 'open', 'R': 'open'}}),
        (10, {'hips': (0, 0.14, -0.08), 'hips_rot': (-10, 0, 0), 'chest': (-15, 6, 0), 'spine': (-6, 0, 0), 'head': (-20, 8, 0),
              'fkL': {'abd': 14, 'fwd': 22, 'elbow': 35}, 'fkR': {'abd': 16, 'fwd': 18, 'elbow': 35},
              'legR': {'offset': (0, 0.20, 0)}, 'fingers': {'L': 'open', 'R': 'open'}}),
        (18, {'hips': (0, 0.20, -0.38), 'hips_rot': (-25, 0, 0), 'chest': (-8, 0, 0), 'spine': (-8, 0, 0), 'head': (-10, 10, 0),
              'fkL': {'abd': 20, 'fwd': 10, 'elbow': 30}, 'fkR': {'abd': 22, 'fwd': 8, 'elbow': 30},
              'legR': {'offset': (0, 0.20, 0)}}),
        (28, lie_b()),
        (33, lie_b(0.03, (2, 14, 0))),
        (40, lie_b()),
        (56, lie_b())]))

    # --- muerte por agarre: un enemigo lo sujeta de frente, forcejea, le rompe el cuello y cae de bruces.
    #     Tras el chasquido los brazos quedan sueltos (FK) y en el suelo descansan junto al cuerpo en pose A
    def grab(k_roll=0.0, k_yaw=0.0, hl=(0.12, -0.34, 1.42), hr=(-0.12, -0.34, 1.42), kick=None):
        d = {'hips': (0, -0.06, 0.05), 'chest': (-10, k_yaw, k_roll), 'spine': (-4, 0, 0), 'head': (-15, k_yaw * 2, 0),
             'armL': {'target': hl, 'pole': (0.7, 0.3, -0.6)}, 'armR': {'target': hr, 'pole': (-0.7, 0.3, -0.6)},
             'legL': {'offset': (0, 0, 0.03), 'foot': (-12, 0, 0)}, 'legR': {'offset': (0, 0, 0.03), 'foot': (-12, 0, 0)},
             'fingers': {'L': 'claw', 'R': 'claw'}}
        if kick: d[kick] = {'offset': (0, -0.06, 0.14)}
        return d
    def lie_f(dz=0.0, head=(-12, 40, 0)):
        return {'hips': (0, -0.36, -0.685 + dz), 'hips_rot': (87, 0, 0), 'chest': (9, 0, 0), 'spine': (3, 0, 0), 'neck': (-6, 0, 0), 'head': head,
                'fkL': {'abd': 0, 'fwd': 15, 'elbow': 6}, 'fkR': {'abd': 0, 'fwd': 15, 'elbow': 6},
                'legL': {'target': (0.15, 0.42, 0.125), 'pole': (0, 0.3, -1), 'foot': (-80, 0, 0)},
                'legR': {'target': (-0.15, 0.40, 0.15), 'pole': (0, 0.3, -1), 'foot': (-80, 0, 0)}}
    acts.append(make_action('Anim_Death_Grapple', [
        (1, {}),
        (6, grab()),
        (10, grab(8, 10, (0.14, -0.30, 1.44), (-0.10, -0.38, 1.40), 'legL')),
        (14, grab(-8, -10, (0.10, -0.38, 1.40), (-0.14, -0.30, 1.44), 'legR')),
        (18, grab(7, 12, (0.15, -0.29, 1.43), (-0.11, -0.37, 1.41), 'legL')),
        (22, grab(-7, -12, (0.11, -0.37, 1.41), (-0.15, -0.29, 1.43), 'legR')),
        (26, grab(4, 6)),
        (32, {'hips': (0, -0.05, 0.03), 'chest': (-6, 10, -6), 'spine': (-2, 0, 0), 'neck': (15, 20, -15), 'head': (20, 25, -20),
              'fkL': {'abd': -28, 'fwd': 12, 'elbow': 20}, 'fkR': {'abd': -28, 'fwd': 10, 'elbow': 20}}),
        (40, {'hips': (0, -0.04, -0.15), 'chest': (10, 6, -4), 'spine': (6, 0, 0), 'neck': (20, 15, -12), 'head': (25, 20, -18),
              'fkL': {'abd': -30, 'fwd': -8, 'elbow': 15}, 'fkR': {'abd': -30, 'fwd': -8, 'elbow': 15}}),
        (50, {'hips': (0, -0.02, -0.44), 'hips_rot': (10, 0, 0), 'chest': (16, 4, -4), 'spine': (8, 0, 0), 'neck': (20, 10, -10), 'head': (25, 15, -15),
              'fkL': {'abd': -30, 'fwd': -24, 'elbow': 10}, 'fkR': {'abd': -30, 'fwd': -24, 'elbow': 10},
              'legL': {'target': (0.15, 0.32, 0.08), 'pole': (0, -1, -0.2), 'foot': (-60, 0, 0)},
              'legR': {'target': (-0.15, 0.32, 0.08), 'pole': (0, -1, -0.2), 'foot': (-60, 0, 0)}}),
        (60, lie_f()),
        (64, lie_f(0.02, (-9, 42, 0))),
        (68, lie_f())]))
    return acts
