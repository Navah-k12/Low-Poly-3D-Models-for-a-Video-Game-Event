# =====================================================================================
# Lukas R-H-07 - malla UNICA construida por extrusion continua desde UNA primitiva
# (circulo de 32 lados en la cintura). Cada anillo se ajusta con rayos a la superficie
# del modelo actual (PJ_Cuerpo en pose de reposo) para conservar su silueta y los
# volumenes de la ropa. Solo geometria: sin UV, sin materiales, sin texturas.
# Convencion: mira hacia -Y, su izquierda es +X, Z arriba, metros, suelas en z = 0.
# =====================================================================================
import bpy, bmesh, math
from mathutils import Vector, Matrix
from mathutils.bvhtree import BVHTree

DEBUG = False
SRC_NAME = 'PJ_Cuerpo'
OUT_NAME = 'Lukas_RH07'
N, NA, NH = 32, 16, 16           # columnas: torso, brazos, cabeza (piernas: 18)
TAU = 2 * math.pi

# ------------------------------------------------------------------ 0) muestreo
src = bpy.data.objects[SRC_NAME]
_me = src.data
_gn = {g.index: g.name for g in src.vertex_groups}
_dom = {}
for v in _me.vertices:
    if v.groups:
        _dom[v.index] = _gn[max(v.groups, key=lambda g: g.weight).group]
_co = [src.matrix_world @ v.co for v in _me.vertices]

# eje aproximado de los tubos viejos de la espalda (lado izquierdo): sus caras no cuentan como espalda
OLD_TUBE = [Vector(p) for p in ((0.054, 0.167, 1.53), (0.054, 0.167, 1.30), (0.062, 0.166, 1.28), (0.085, 0.161, 1.25),
                                (0.105, 0.157, 1.23), (0.120, 0.153, 1.21), (0.140, 0.148, 1.19), (0.155, 0.145, 1.175))]
def _near_old_tube(p, lim=0.024):
    q = Vector((abs(p.x), p.y, p.z))
    for a, b in zip(OLD_TUBE, OLD_TUBE[1:]):
        ab = b - a; t = max(0.0, min(1.0, (q - a).dot(ab) / ab.length_squared))
        if (q - (a + ab * t)).length < lim: return True
    return False
_tube_v = {i for i, c in enumerate(_co) if _near_old_tube(c)}

def _tree(groups, mode='any', no_tubes=False):
    polys = []
    for p in _me.polygons:
        if no_tubes and all(i in _tube_v for i in p.vertices): continue
        ins = [_dom.get(i) in groups for i in p.vertices]
        if (any(ins) if mode == 'any' else all(ins)):
            polys.append(list(p.vertices))
    return BVHTree.FromPolygons(_co, polys)

T_TORSO = _tree({'Hips', 'Spine', 'Chest', 'Neck', 'LeftShoulder', 'RightShoulder'}, no_tubes=True)
T_TORSO_LEGS = _tree({'Hips', 'Spine', 'Chest', 'LeftUpperLeg', 'RightUpperLeg'})
T_HEAD = _tree({'Head', 'Neck'}, 'all')
T_LEG = _tree({'LeftUpperLeg', 'LeftLowerLeg', 'LeftFoot', 'LeftToes'})
T_ARM = _tree({'RightUpperArm', 'RightLowerArm', 'RightHand', 'RightShoulder'})   # brazo limpio (sin dispositivo)

def ray_in(tree, C, d, rmax):
    """rayo desde fuera hacia el eje: distancia a la superficie mas exterior"""
    d = d.normalized()
    h = tree.ray_cast(C + d * rmax, -d, rmax)
    return None if h[0] is None else (h[0] - C).dot(d)

def fill(rs, circular):
    n = len(rs); out = list(rs)
    ok = [i for i, r in enumerate(rs) if r is not None]
    assert ok, 'anillo sin muestras'
    for i in range(n):
        if rs[i] is not None: continue
        if circular:
            a = next((i - j) % n for j in range(1, n) if rs[(i - j) % n] is not None)
            b = next((i + j) % n for j in range(1, n) if rs[(i + j) % n] is not None)
            da, db = (i - a) % n, (b - i) % n
        else:
            pv = [j for j in ok if j < i]; nx = [j for j in ok if j > i]
            if not pv: out[i] = rs[nx[0]]; continue
            if not nx: out[i] = rs[pv[-1]]; continue
            a, b = pv[-1], nx[0]; da, db = i - a, b - i
        out[i] = (rs[a] * db + rs[b] * da) / (da + db)
    return out

def mirror(p): return Vector((-p.x, p.y, p.z))

# ------------------------------------------------------------------ operaciones de malla
bm = bmesh.new()
def key(co): return (round(co.x, 6), round(co.y, 6), round(co.z, 6))

def extrude_ring(verts, positions):
    """extruye las aristas de un anillo de borde; devuelve el anillo nuevo en el mismo orden"""
    n = len(verts)
    edges = [bm.edges.get((verts[i], verts[(i + 1) % n])) for i in range(n)]
    r = bmesh.ops.extrude_edge_only(bm, edges=edges)
    by = {key(v.co): v for v in r['geom'] if isinstance(v, bmesh.types.BMVert)}
    new = [by[key(v.co)] for v in verts]
    for v, p in zip(new, positions): v.co = p
    return new

def extrude_region(faces, par=None):
    """extruye una region de caras. Devuelve (mapa vertice viejo -> nuevo, caras tapa)"""
    oldk = {}
    for f in faces:
        for v in f.verts: oldk[v] = key(v.co)
    pk = {oldk[v]: par[v] for v in oldk} if par is not None else None
    r = bmesh.ops.extrude_face_region(bm, geom=list(faces))
    nv = [g for g in r['geom'] if isinstance(g, bmesh.types.BMVert)]
    by = {key(v.co): v for v in nv}
    mp = {v: by[k] for v, k in oldk.items() if v.is_valid and k in by}
    if pk is not None:
        for v in nv: par[v] = pk[key(v.co)]
    newset = set(nv)
    cap = [f for f in r['geom'] if isinstance(f, bmesh.types.BMFace) and all(v in newset for v in f.verts)]
    left = [f for f in faces if f.is_valid]
    if left: bmesh.ops.delete(bm, geom=left, context='FACES')
    return mp, cap

def face_of(vs):
    s = set(vs)
    for f in vs[0].link_faces:
        if set(f.verts) == s: return f
    raise RuntimeError('cara no encontrada')

def grid_cap(loop, r, c, start, lift=None):
    """tapa un anillo de 2(r+c) vertices con una rejilla de quads (parche de Coons)"""
    n = len(loop); assert n == 2 * (r + c), (n, r, c)
    L = [loop[(start + i) % n] for i in range(n)]
    bot = L[0:c + 1]; right = L[c:c + r + 1]; top = L[c + r:2 * c + r + 1][::-1]; left = (L[2 * c + r:] + [L[0]])[::-1]
    G = [[None] * (c + 1) for _ in range(r + 1)]
    for j in range(c + 1): G[0][j] = bot[j]; G[r][j] = top[j]
    for i in range(r + 1): G[i][0] = left[i]; G[i][c] = right[i]
    for i in range(1, r):
        for j in range(1, c):
            u, v = j / c, i / r
            p = (bot[j].co * (1 - v) + top[j].co * v + left[i].co * (1 - u) + right[i].co * u
                 - (bot[0].co * (1 - u) * (1 - v) + bot[c].co * u * (1 - v) + top[0].co * (1 - u) * v + top[c].co * u * v))
            G[i][j] = bm.verts.new(lift(p, u, v) if lift else p)
    fs = [bm.faces.new((G[i][j], G[i][j + 1], G[i + 1][j + 1], G[i + 1][j])) for i in range(r) for j in range(c)]
    return G, fs

def uniform_angles(phis, base=0.0):
    """reparte n angulos de forma uniforme (base + 2pi m/n) conservando su orden.
    base=None: elige entre 0 y medio paso el reparto que menos gira los vertices"""
    n = len(phis); order = sorted(range(n), key=lambda i: phis[i] % TAU)
    best = None
    for b0 in ((0.0, math.pi / n) if base is None else (base,)):
        for s in range(n):
            tgt = [0.0] * n
            for rank, i in enumerate(order): tgt[i] = b0 + TAU * ((rank - s) % n) / n
            cost = sum(abs(math.remainder(tgt[i] - phis[i], TAU)) for i in range(n))
            if best is None or cost < best[0]: best = (cost, tgt)
    return best[1]

def region_loft(region, par, steps, place):
    cur = region
    for i in range(len(steps)):
        mp, cur = extrude_region(cur, par)
        for v in {v for f in cur for v in f.verts}: v.co = place(i, par[v])
    return cur

# ------------------------------------------------------------------ estilo de ropa (referencias de Blender)
def jit(a, b):
    """ruido fijo en [-1, 1] (mismo resultado en cada ejecucion): facetas irregulares de la tela"""
    x = math.sin(a * 12.9898 + b * 78.233) * 43758.5453
    return 2.0 * (x - math.floor(x)) - 1.0
def fold(pts, C, a, f, o, amp=0.0, tilt=0.0, phase=0.0, lump=0.0, lphase=0.0, back_only=False):
    """pliegue facetado: el anillo se abomba (amp), se inclina sobre su eje (tilt) y se deforma de forma
    irregular (lump), como los pliegues en acordeon de las mangas y perneras de las referencias"""
    out = []
    for p in pts:
        v = p - C; h = v.dot(a); rv = v - a * h
        th = math.atan2(rv.dot(o), rv.dot(f))
        w = 1.0 if not back_only else max(0.0, -math.cos(th)) ** 0.7
        sc = 1.0 + w * (amp + lump * math.cos(2 * th + lphase))
        out.append(C + a * (h + w * tilt * math.cos(th - phase)) + rv * sc)
    return out

# =============================================================== 1) TORSO (primitiva)
TORSO = [  # z, A, B, tipo         (A, B: semiejes de la direccion de muestreo)
    (0.845, 0.190, 0.130, 'fit'),    # 0  base de la cadera (de aqui salen las piernas)
    (0.885, 0.190, 0.130, 'fit'),    # 1
    (0.925, 0.185, 0.135, 'fit'),    # 2
    (0.962, 0.178, 0.135, 'fit'),    # 3  bajo el cinturon
    (0.975, 0.178, 0.135, 'belt'),   # 4  cinturon: borde inferior
    (1.010, 0.178, 0.135, 'belt'),   # 5  cinturon: centro
    (1.045, 0.178, 0.135, 'belt'),   # 6  cinturon: borde superior
    (1.055, 0.178, 0.135, 'fit'),    # 7  sobre el cinturon
    (1.100, 0.168, 0.136, 'fit'),    # 8  cintura  <- PRIMITIVA
    (1.150, 0.168, 0.138, 'fit'),    # 9
    (1.195, 0.172, 0.138, 'fit'),    # 10 soporte bajo la axila
    (1.232, 0.176, 0.138, 'fit'),    # 11 soporte pecho -> axila
    (1.262, 0.180, 0.138, 'fit'),    # 12 axila (fila inferior del hueco del brazo)
    (1.307, 0.185, 0.135, 'fit'),    # 13
    (1.352, 0.190, 0.130, 'fit'),    # 14
    (1.397, 0.200, 0.125, 'fit'),    # 15
    (1.440, 0.180, 0.120, 'fit'),    # 16 hombro (fila superior del hueco del brazo)
    (1.465, 0.160, 0.110, 'fit')]    # 17 base del cuello alto
PRIM = 8
YC = 0.010
HOLE_K = (7, 11)          # columnas del hueco del brazo izquierdo (el derecho es su espejo)
HOLE_R = (12, 16)         # anillos del hueco

def torso_dirs(A, B):
    return [Vector((A * math.sin(TAU * k / N), -B * math.cos(TAU * k / N), 0)).normalized() for k in range(N // 2 + 1)]

def torso_rs(z, A, B):
    tree = T_TORSO_LEGS if z < 0.965 else T_TORSO
    C = Vector((0, YC, z)); rs = []
    for k, d in enumerate(torso_dirs(A, B)):
        t = TAU * k / N; e = Vector((A * math.sin(t), B * math.cos(t))).length
        r = ray_in(tree, C, d, 0.5)
        if r is not None and not (0.55 * e < r < 1.8 * e): r = None
        rs.append(r)
    return fill(rs, False)

def torso_pts(z, A, B, rs):
    C = Vector((0, YC, z)); H = N // 2; pts = [None] * N
    for k, d in enumerate(torso_dirs(A, B)):
        p = C + d * rs[k]
        if k in (0, H): p.x = 0.0
        pts[k] = p
    for k in range(1, H): pts[N - k] = mirror(pts[k])
    return pts

TRS = {}
for i, (z, A, B, kind) in enumerate(TORSO):
    if kind == 'fit': TRS[i] = torso_rs(z, A, B)
# espalda: quita los picos que dejan los tubos viejos y sus enchufes (la espalda nueva queda lisa)
for i in range(9, len(TORSO)):
    rs = TRS[i]
    for _ in range(2):
        for k in range(10, N // 2 + 1):
            nb = rs[k + 1] if k < N // 2 else rs[k - 1]
            m = 0.5 * (rs[k - 1] + nb)
            if rs[k] > m + 0.006: rs[k] = m
for i, (z, A, B, kind) in enumerate(TORSO):
    if kind == 'belt':
        TRS[i] = [max(a, b) + 0.009 for a, b in zip(TRS[3], TRS[7])]
# lado del torso bajo el brazo: en el modelo actual ahi esta el deltoides del brazo caido;
# con el brazo en pose A el costado sigue el perfil de la axila (anillo 12) un poco abombado
SIDE_FAC = {13: 1.030, 14: 1.050, 15: 1.055, 16: 1.030}
for i, fac in SIDE_FAC.items():
    for k in range(5, 14):
        rv = TRS[12][k] * fac
        TRS[i][k] = rv if 6 <= k <= 12 else 0.5 * (min(TRS[i][k], rv * 1.1) + rv)
# estilo: el mono se abolsa sobre el cinturon y lleva la tapeta de la cremallera en el centro del pecho
TRS[PRIM] = [r * 1.035 for r in TRS[PRIM]]
for i in range(PRIM, 16): TRS[i][0] += 0.0045
for i in range(0, 17):                                  # facetas de la tela (el cinturon queda liso)
    if TORSO[i][3] == 'fit':
        TRS[i] = [r * (1.0 + 0.012 * jit(i + 1, k)) for k, r in enumerate(TRS[i])]
TPTS = [torso_pts(z, A, B, TRS[i]) for i, (z, A, B, kind) in enumerate(TORSO)]

# la primitiva: circulo de 32 lados
circ = bmesh.ops.create_circle(bm, cap_ends=False, segments=N, radius=0.17)
ring0 = sorted(circ['verts'], key=lambda v: (math.atan2(v.co.y, v.co.x) + math.pi / 2) % TAU)
for v, p in zip(ring0, TPTS[PRIM]): v.co = p
R = {PRIM: ring0}
for i in range(PRIM + 1, len(TORSO)): R[i] = extrude_ring(R[i - 1], TPTS[i])
for i in range(PRIM - 1, -1, -1): R[i] = extrude_ring(R[i + 1], TPTS[i])

# =============================================================== 2) PIERNAS + BOTAS
LEG_Z = [0.770, 0.725, 0.700, 0.675, 0.632, 0.620, 0.598, 0.588, 0.545,
         0.505, 0.492, 0.479,                       # rodilla: 3 loops juntos (bisagra)
         0.432, 0.392, 0.375, 0.352, 0.330, 0.310, 0.290, 0.272, 0.258, 0.250, 0.236, 0.192,
         0.141, 0.128, 0.115,                       # tobillo: 3 loops juntos (bisagra)
         0.090, 0.068, 0.047, 0.036, 0.033, 0.006, 0.000]
# pliegues de la pernera (como en las referencias: la tela se amontona de la rodillera a la bota)
LEG_FOLD = {0.700: dict(amp=0.035, tilt=0.006, phase=math.pi, lump=0.020, lphase=0.6, back_only=True),
            0.352: dict(amp=0.050, tilt=0.009, phase=0.5, lump=0.055, lphase=1.2),
            0.330: dict(amp=-0.020, lump=0.025, lphase=2.9),
            0.310: dict(amp=0.060, tilt=-0.009, phase=2.8, lump=0.060, lphase=2.3),
            0.290: dict(amp=-0.015, lump=0.025, lphase=1.0),
            0.272: dict(amp=0.050, tilt=0.007, phase=1.6, lump=0.050, lphase=0.2)}
# bota: suela gruesa que sobresale del empeine (labio) y un bisel por debajo
SOLE = {0.036: -0.002, 0.033: 0.008, 0.006: 0.008, 0.000: 0.004}
LEG_BONE = [(0.836, Vector((0.125, 0.011))), (0.490, Vector((0.137, 0.017))), (0.128, Vector((0.139, 0.047)))]
FOOT_C = Vector((0.158, -0.022))
FOOT_ANG = math.radians(11.0)        # la punta del pie gira hacia fuera, como en el modelo actual

def leg_center(z):
    if z >= LEG_BONE[0][0]: return LEG_BONE[0][1].copy()
    for (z0, c0), (z1, c1) in zip(LEG_BONE, LEG_BONE[1:]):
        if z1 <= z <= z0: return c0.lerp(c1, (z0 - z) / (z0 - z1))
    t = min(1.0, (0.128 - z) / 0.058)
    return LEG_BONE[-1][1].lerp(FOOT_C, t)

def leg_ab(z):
    t = 0.0 if z >= 0.128 else min(1.0, (0.128 - z) / 0.058)
    return 1.0 + (0.06 / 0.15 - 1.0) * t, 1.0      # (A, B) relativos: el pie es alargado en Y

hb = R[0]
crotch = bm.verts.new((0.0, 0.016, 0.796))
LEG_LOOP = {+1: [crotch] + hb[0:N // 2 + 1], -1: [crotch] + hb[N // 2:] + [hb[0]]}
leg_face = {s: bm.faces.new(LEG_LOOP[s]) for s in (+1, -1)}

def leg_ring_pts(z, psis, s, uniform_dirs=True):
    c = leg_center(z); C3 = Vector((c.x, c.y, z)); A, B = leg_ab(z)
    t = 0.0 if z >= 0.128 else min(1.0, (0.128 - z) / 0.058)
    rot = Matrix.Rotation(FOOT_ANG * t, 3, 'Z')
    rs, ds = [], []
    for ps in psis:
        d = (rot @ Vector((A * math.sin(ps), -B * math.cos(ps), 0))).normalized()
        zq = max(z, 0.004)
        r = ray_in(T_LEG, Vector((c.x, c.y, zq)), d, 0.30)
        if r is not None and not (0.02 < r < 0.26): r = None
        rs.append(r); ds.append(d)
    rs = fill(rs, True)
    fa = 0.016 if z >= 0.25 else 0.005                  # tela con facetas; la bota, casi lisa
    rs = [r * (1.0 + fa * jit(z * 1000.0, round(ps % TAU, 3) * 100.0)) for ps, r in zip(psis, rs)]   # por angulo: simetrico
    # bota robusta como en las referencias: cana mas ancha, el pie algo mas ancho
    bs = 1.0 if z > 0.252 else (1.16 if z >= 0.13 else 1.16 - (0.13 - z) / 0.13 * 0.10)
    rs = [r * bs for r in rs]
    if z in SOLE: rs = [r + SOLE[z] for r in rs]
    pts = [C3 + d * r for d, r in zip(ds, rs)]
    if z in LEG_FOLD:
        pts = fold(pts, C3, Vector((0, 0, 1)), Vector((0, -1, 0)), Vector((1, 0, 0)), **LEG_FOLD[z])
    if s < 0: pts = [mirror(p) for p in pts]
    return pts

def build_leg(s):
    loop = LEG_LOOP[s]
    c0 = leg_center(0.80)
    phi = [math.atan2(s * (v.co.x - s * c0.x), -(v.co.y - c0.y)) for v in loop]   # 0 = frente, 90 = exterior
    psi = uniform_angles(phi, base=math.pi / len(loop))
    cur = [leg_face[s]]; prev = loop; rings = []
    for i, z in enumerate(LEG_Z):
        mp, cur = extrude_region(cur)
        new = [mp[v] for v in prev]
        w = 0.5 if i == 0 else 1.0
        angs = [p + w * math.remainder(q - p, TAU) for p, q in zip(phi, psi)]
        for v, p in zip(new, leg_ring_pts(z, angs, s)): v.co = p
        rings.append(new); prev = new
    bmesh.ops.delete(bm, geom=cur, context='FACES_ONLY')
    # suela: rejilla 3 x 6
    loop = rings[-1]; n = len(loop); r_, c_ = 3, 6
    cc = sum((v.co for v in loop), Vector()) / n
    def score(st):
        Lp = [loop[(st + i) % n] for i in range(n)]
        cs = [Lp[0], Lp[c_], Lp[c_ + r_], Lp[2 * c_ + r_]]
        best = 0
        tg = [(-1, -1), (-1, 1), (1, 1), (1, -1)]
        for v, (tx, ty) in zip(cs, tg):
            best += ((v.co.x - cc.x) / 0.055 - tx) ** 2 + ((v.co.y - cc.y) / 0.14 - ty) ** 2
        return best
    grid_cap(loop, r_, c_, min(range(n), key=score))
    return rings

# =============================================================== 3) BRAZOS en pose A + MANOS
J0, E0, W0, H0 = Vector((0.206, 0.042, 1.388)), Vector((0.284, 0.059, 1.109)), Vector((0.332, 0.033, 0.869)), Vector((0.337, 0.000, 0.766))
SEG = [(J0, E0), (E0, W0), (W0, H0)]
LEN = [(b - a).length for a, b in SEG]
U = [(b - a).normalized() for a, b in SEG]
_ang0 = math.atan2(E0.x - J0.x, -(E0.z - J0.z))
ROT = Matrix.Rotation(-(math.radians(45.0) - _ang0), 3, 'Y')      # brazo izquierdo a 45 grados
def toA(p): return J0 + ROT @ (p - J0)

def arm_center(sv):
    acc = 0.0
    for (a, b), l in zip(SEG, LEN):
        if sv <= acc + l: return a.lerp(b, (sv - acc) / l)
        acc += l
    return SEG[-1][1] + U[-1] * (sv - acc)

def arm_axis(sv):
    e1 = LEN[0]; e2 = LEN[0] + LEN[1]
    if sv < e1 - 0.04: return U[0]
    if sv < e1 + 0.04: return U[0].lerp(U[1], (sv - (e1 - 0.04)) / 0.08).normalized()
    if sv < e2 - 0.03: return U[1]
    if sv < e2 + 0.03: return U[1].lerp(U[2], (sv - (e2 - 0.03)) / 0.06).normalized()
    return U[2]

def arm_frame(a):
    f = Vector((0, -1, 0)); f = (f - a * f.dot(a)).normalized(); o = f.cross(a).normalized()
    return a, f, o

THETA = [TAU * (k + 0.5) / NA for k in range(NA)]       # k = 0 frente, 4 exterior, 8 espalda, 12 interior

ARM_SHOULDER = {0.085: 1.07, 0.112: 1.05, 0.140: 1.03}   # junto al hombro el brazo caido se funde con la espalda:
                                                         # se usa la seccion limpia de s = 0.195, algo mayor (deltoides)
GLOVE = {0.508: 1.22, 0.528: 1.17, 0.540: 1.15}          # puno grueso del guante: copia del antebrazo (s = 0.494), mas ancha
# pliegues de la manga: acordeon facetado entre el hombro y el puno enrollado; el puno son dos rollos
ARM_FOLD = {0.112: dict(amp=0.045, tilt=0.004, phase=1.57, lump=0.030, lphase=0.2),   # junto a la axila: suave
            0.140: dict(amp=-0.030, lump=0.020, lphase=2.0),
            0.168: dict(amp=0.100, tilt=-0.009, phase=3.5, lump=0.050, lphase=1.4),
            0.195: dict(amp=-0.030, lump=0.020, lphase=0.5),
            0.222: dict(amp=0.095, tilt=0.008, phase=2.1, lump=0.050, lphase=2.6),
            0.258: dict(amp=0.030),
            0.281: dict(amp=0.050, lump=0.015, lphase=0.9),           # codo 1: primer rollo
            0.294: dict(amp=-0.010),                                   # codo 2: entre rollos
            0.307: dict(amp=0.050, lump=0.015, lphase=2.0),           # codo 3: segundo rollo
            0.338: dict(amp=0.020)}

def arm_ring_orig(sv):
    """anillo del brazo (en la pose original, lado izquierdo) muestreado del brazo derecho limpio"""
    C = arm_center(sv); a, f, o = arm_frame(arm_axis(sv))
    if sv in ARM_SHOULDER or sv in GLOVE:
        sb, sc = (0.195, ARM_SHOULDER[sv]) if sv in ARM_SHOULDER else (0.494, GLOVE[sv])
        Cb = arm_center(sb)
        pts = [C + (p - Cb) * sc for p in arm_ring_raw(sb)]
    else:
        pts = arm_ring_raw(sv)
    if sv < 0.345:                                       # manga con facetas; el antebrazo es piel
        pts = [C + (p - C) * (1.0 + 0.016 * jit(sv * 1000.0, k)) for k, p in enumerate(pts)]
    if sv in ARM_FOLD: pts = fold(pts, C, a, f, o, **ARM_FOLD[sv])
    return pts

def arm_ring_raw(sv):
    C = arm_center(sv); a, f, o = arm_frame(arm_axis(sv))
    rs, ds = [], []
    for th in THETA:
        d = f * math.cos(th) + o * math.sin(th)
        r = ray_in(T_ARM, mirror(C), mirror(d), 0.14)
        if r is not None and not (0.02 < r < 0.13): r = None
        rs.append(r); ds.append(d)
    rs = fill(rs, True)
    # rayos que tocaron el costado del pecho en el modelo viejo (brazo pegado al cuerpo): picos aislados
    n = len(rs)
    for _ in range(2):
        for k in range(n):
            m = 0.5 * (rs[k - 1] + rs[(k + 1) % n])
            if rs[k] > m * 1.18: rs[k] = m
    return [C + d * r for d, r in zip(ds, rs)]

ARM_S = [0.085, 0.112, 0.140, 0.168, 0.195, 0.222, 0.250, 0.258,
         0.281, 0.294, 0.307,                      # codo: 3 loops juntos (bisagra)
         0.338, 0.356, 0.392, 0.442, 0.494, 0.508,
         0.528, 0.540, 0.552]                      # muneca: 3 loops juntos (bisagra)

# mano (en la pose original, marco de la mano)
HA, HF, HO = arm_frame(U[2])
PALM_OFF = -0.013        # el centro de la palma esta un poco hacia atras del hueso
def palm_ring(dist, hw, hf, hb, p=2.4):
    C = W0 + HA * dist + HF * PALM_OFF; pts = []
    for th in THETA:
        c, s_ = math.cos(th), math.sin(th)
        cf = math.copysign(abs(c) ** (2 / p), c); so = math.copysign(abs(s_) ** (2 / p), s_)
        pts.append(C + HF * ((hf if cf > 0 else hb) * cf) + HO * (hw * so))
    return pts
FP = [0.024, 0.003, -0.001, -0.022, -0.026, -0.047, -0.051, -0.070]    # nudillos, de delante a atras
def knuckle_ring(dist, hw):
    C = W0 + HA * dist; pts = [None] * NA
    for k in range(NA):
        th = THETA[k]
        if th < math.pi: j = k; pts[k] = C + HF * FP[j] + HO * hw            # cara exterior (dorso)
        else: j = NA - 1 - k; pts[k] = C + HF * FP[j] - HO * hw              # cara de la palma
    return pts

def place_arm(p, s):
    q = toA(p)
    return q if s > 0 else mirror(q)

# loops de soporte del hombro sobre el torso: (escala hacia el centro, salida por la normal,
# corrimiento del centro hacia delante, hacia arriba) -> bandas anchas en escapula y axila
SHOULDER_LOOPS = [(0.84, 0.006, 0.010, 0.008),
                  (0.70, 0.014, 0.016, 0.013)]
TRANS = [(0.33, 0.04), (0.66, 0.07)]        # anillos de transicion hueco -> brazo (t, abombado del deltoides)
N_TRANS = len(TRANS)
FINGERS = {}                                 # geometria de cada dedo (para sus huesos y pesos)
HAND_DORSAL = {}                             # direccion del dorso de cada mano en reposo

def build_arm(s):
    side = [R[i] for i in range(HOLE_R[0], HOLE_R[1] + 1)]
    ks = range(HOLE_K[0], HOLE_K[1]) if s > 0 else range(N - HOLE_K[1], N - HOLE_K[0])
    patch = [face_of([side[i][k], side[i][k + 1], side[i + 1][k + 1], side[i + 1][k]]) for i in range(len(side) - 1) for k in ks]
    pset = set(patch)
    def region_border(fs):
        return list({v for f in fs for e in f.edges if sum(1 for g in e.link_faces if g in fs) == 1 for v in e.verts})
    B = region_border(pset); assert len(B) == NA, len(B)
    hc = sum((v.co for v in B), Vector()) / NA
    f0 = Vector((0, -1, 0)); u0 = Vector((0, 0, 1))
    phi = [math.atan2((v.co - hc).dot(u0), (v.co - hc).dot(f0)) for v in B]
    uni = uniform_angles(phi, base=math.pi / NA)
    byk = [None] * NA
    for v, u in zip(B, uni): byk[int(round((u % TAU) / (TAU / NA) - 0.5)) % NA] = v
    assert all(byk)
    nrm = sum((f.normal for f in patch), Vector()).normalized()
    if nrm.x * s < 0: nrm = -nrm
    # refuerzo del hombro: dos loops concentricos sobre el torso que rodean la raiz del brazo
    # (pecho, deltoides, escapula y axila). Su centro se corre hacia delante y arriba, asi las
    # bandas quedan mas anchas detras (escapula) y debajo (axila), donde tira al apuntar.
    Bco = [byk[k].co.copy() for k in range(NA)]
    for scale, out, shf, shu in SHOULDER_LOOPS:
        prev_k = {byk[k]: k for k in range(NA)}
        bmesh.ops.inset_region(bm, faces=patch, thickness=0.003, depth=0.0, use_even_offset=True)
        inner = region_border(pset); assert len(inner) == NA, len(inner)
        nk = [None] * NA
        for v in inner:
            k = prev_k[min(prev_k, key=lambda w: (w.co - v.co).length)]
            nk[k] = v
        assert all(nk)
        c = hc + f0 * shf + u0 * shu
        for k in range(NA): nk[k].co = c + (Bco[k] - c) * scale + nrm * out
        byk = nk
    face = bmesh.ops.dissolve_faces(bm, faces=patch, use_verts=True)['region'][0]
    assert len(face.verts) == NA, len(face.verts)
    # anillos: transicion del hombro + anillos ajustados + palma
    first = [place_arm(p, s) for p in arm_ring_orig(ARM_S[0])]
    # dos anillos de transicion (soporte de la articulacion): solo la mitad superior se abomba
    # (deltoides); la axila queda limpia, sin pliegue por debajo
    trans_rings = []
    for t, bulge in TRANS:
        tr = [byk[k].co.lerp(first[k], t) for k in range(NA)]
        tc = sum(tr, Vector()) / NA
        for k in range(NA):
            up = max(0.0, math.sin(THETA[k]))          # 0 en frente/espalda/axila, 1 arriba
            tr[k] = tc + (tr[k] - tc) * (1.0 + bulge * up)
        trans_rings.append(tr)
    stages = trans_rings + [[place_arm(p, s) for p in arm_ring_orig(sv)] for sv in ARM_S]
    stages += [[place_arm(p, s) for p in palm_ring(0.040, 0.0215, 0.046, 0.050)],
               [place_arm(p, s) for p in palm_ring(0.075, 0.0210, 0.047, 0.052)],
               [place_arm(p, s) for p in knuckle_ring(0.105, 0.0160)]]
    cur = [face]; prev = byk; rings = []
    for pts in stages:
        mp, cur = extrude_region(cur)
        new = [mp[v] for v in prev]
        for v, p in zip(new, pts): v.co = p
        rings.append(new); prev = new
    return cur, rings

def build_hand(s, end_faces, rings):
    bmesh.ops.delete(bm, geom=end_faces, context='FACES_ONLY')
    K = rings[-1]
    G, _ = grid_cap(K, 1, 7, 0)            # K[0] = esquina dorso-frente: fila de 7 quads (nudillos)
    a = (place_arm(W0 + HA, s) - place_arm(W0, s)).normalized()
    f = (place_arm(W0 + HF, s) - place_arm(W0, s)).normalized()
    o = (place_arm(W0 + HO, s) - place_arm(W0, s)).normalized()
    FING = [(0.086, 10), (0.094, 12), (0.089, 14), (0.074, 17)]     # largo, curvatura (medidos de la mano actual)
    for j, (ln, curl) in zip((0, 2, 4, 6), FING):
        fc = face_of([G[0][j], G[0][j + 1], G[1][j + 1], G[1][j]])
        vs = list(fc.verts); cc = sum((v.co for v in vs), Vector()) / 4
        par = {v: ((v.co - cc).dot(f), (v.co - cc).dot(o)) for v in vs}
        # guante sin dedos: el borde del guante abomba el primer tramo; luego asoma el dedo, mas fino
        ST = [(0.006, 1.00, 0.92, 0.0), (0.030 * ln / 0.09, 1.04, 0.98, 0.8), (0.040 * ln / 0.09, 0.90, 0.80, 1.1),
              (0.064 * ln / 0.09, 0.88, 0.78, 1.8), (ln, 0.80, 0.70, 2.6)]
        pos = [cc]; axes = []
        for i, (dist, su, sw, cf) in enumerate(ST):
            ang = math.radians(curl * cf); ax = (a * math.cos(ang) - o * math.sin(ang)).normalized()
            axes.append(ax); pos.append(pos[-1] + ax * (dist - (ST[i - 1][0] if i else 0.0)))
        def place(i, pr, ST=ST, pos=pos, axes=axes):
            dist, su, sw, cf = ST[i]; ax = axes[i]; oo = (o - ax * o.dot(ax)).normalized()
            return pos[i + 1] + f * (pr[0] * su) + oo * (pr[1] * sw)
        region_loft([fc], par, ST, place)
        # geometria del dedo para su esqueleto: anillos (base + uno por tramo), articulaciones y dorso
        FINGERS.setdefault(s, {})[('Index', 'Middle', 'Ring', 'Little')[j // 2]] = {
            'rings': [[v.co.copy() for v in vs]] + [[place(i, par[v]) for v in vs] for i in range(len(ST))],
            'pos': [p.copy() for p in pos], 'dorsal': [(o - ax * o.dot(ax)).normalized() for ax in axes]}
    # pulgar: 2 caras del lado frontal-palmar entre los dos anillos de la palma
    P1, P2 = rings[-3], rings[-2]
    region = [face_of([P1[14], P1[15], P2[15], P2[14]]), face_of([P1[15], P1[0], P2[0], P2[15]])]
    vs = {v for fc in region for v in fc.verts}; cc = sum((v.co for v in vs), Vector()) / len(vs)
    nrm = sum((fc.normal for fc in region), Vector()).normalized()
    if nrm.dot(f - o) < 0: nrm = -nrm
    # orden de los 6 vertices leido en el plano de la region -> seccion hexagonal del pulgar
    side_r = a.cross(nrm).normalized(); up_r = nrm.cross(side_r).normalized()
    ang = {v: math.atan2((v.co - cc).dot(up_r), (v.co - cc).dot(side_r)) for v in vs}
    par = dict(zip(ang.keys(), uniform_angles(list(ang.values()), base=None)))
    Tdir = (nrm * 0.45 + a * 0.85 + f * 0.10).normalized()
    ST = [(0.009, 0.0125, 0.0), (0.037, 0.0115, 0.25), (0.065, 0.0090, 0.50)]     # distancia, radio, giro hacia los dedos
    pos = [cc, cc + nrm * 0.009]; axes = [nrm]                 # la base sale recta del lateral de la palma
    for i, (dist, rad, bend) in enumerate(ST[1:], 1):
        ax = (Tdir * (1 - bend) + a * bend).normalized(); axes.append(ax)
        pos.append(pos[-1] + ax * (dist - ST[i - 1][0]))
    def tplace(i, th):
        ax = axes[i]; sd = (side_r - ax * side_r.dot(ax)).normalized(); upv = ax.cross(sd).normalized()
        if upv.dot(up_r) < 0: upv = -upv
        return pos[i + 1] + (sd * math.cos(th) + upv * math.sin(th)) * ST[i][1]
    vsl = list(vs)
    region_loft(region, par, ST, tplace)
    FINGERS[s]['Thumb'] = {
        'rings': [[v.co.copy() for v in vsl]] + [[tplace(i, par[v]) for v in vsl] for i in range(len(ST))],
        'pos': [p.copy() for p in pos], 'dorsal': [(o - ax * o.dot(ax)).normalized() for ax in axes],
        'palm': place_arm(W0 + HA * 0.035 + HF * 0.022 - HO * 0.004, s)}       # base del pulgar dentro de la palma
    HAND_DORSAL[s] = o
    return rings

def build_device(rings):
    """dispositivo del antebrazo izquierdo (como en la hoja): panel extruido del propio antebrazo"""
    ra, rb, rc = (rings[N_TRANS + ARM_S.index(sv)] for sv in (0.392, 0.442, 0.494))
    region = [face_of([x[k], x[k + 1], y[k + 1], y[k]]) for x, y in ((ra, rb), (rb, rc)) for k in (2, 3, 4)]
    nrm = sum((f.normal for f in region), Vector()).normalized()
    vs = {v for f in region for v in f.verts}; cc = sum((v.co for v in vs), Vector()) / len(vs)
    axis_pt = place_arm(arm_center(0.442), +1)
    if nrm.dot(cc - axis_pt) < 0: nrm = -nrm
    par = {v: v.co.copy() for v in vs}
    def place(i, p0):
        out = 0.004 if i == 0 else 0.009
        sc = 0.97 if i == 0 else 0.90
        return cc + (p0 - cc) * sc + nrm * out
    region_loft(region, par, [0, 1], place)

# =============================================================== 4) CUELLO ALTO + CUELLO + CABEZA
COLLAR_H = [1.478, 1.479, 1.483, 1.491, 1.502, 1.510, 1.516, 1.522, 1.530, 1.540, 1.548, 1.554, 1.560, 1.562, 1.562, 1.561, 1.560]
NECK_C = Vector((0.0, -0.017)); NECK_R = (0.047, 0.040); NECK_Z = 1.484

def build_collar_head():
    top = R[17]; H = N // 2
    A, B = TORSO[17][1], TORSO[17][2]; dirs = torso_dirs(A, B)
    base = TRS[17]
    def ring_at(zs, rs):
        pts = [None] * N
        for k in range(H + 1):
            p = Vector((0, YC, zs[k])) + dirs[k] * rs[k]
            if k in (0, H): p.x = 0.0
            pts[k] = p
        for k in range(1, H): pts[N - k] = mirror(pts[k])
        return pts
    z1 = [1.465 + 0.45 * (h - 1.465) for h in COLLAR_H]
    z2 = list(COLLAR_H)
    r1 = [ray_in(T_TORSO, Vector((0, YC, z)), d, 0.4) for z, d in zip(z1, dirs)]
    r2 = [ray_in(T_TORSO, Vector((0, YC, z - 0.004)), d, 0.4) for z, d in zip(z2, dirs)]
    def clean(rs):
        out = []
        for k, r in enumerate(rs):
            lo = base[k] - 0.02
            out.append(r if (r is not None and lo < r < base[k] + 0.045) else None)
        out = fill(out, False)
        for k in range(1, H):         # quita picos (el enchufe del tubo)
            m = 0.5 * (out[k - 1] + out[k + 1])
            if out[k] > m + 0.012: out[k] = m
        return out
    r1 = clean(r1); r2 = clean(r2)
    # radio del cuello en cada direccion (el interior del cuello alto no puede pasarlo)
    def neck_r(k):
        d = dirs[k]
        # distancia desde (0, YC) al borde del cuello en la direccion d
        best = 0.0
        for i in range(200):
            t = i * 0.001; q = Vector((d.x * t, YC + d.y * t))
            if ((q.x - NECK_C.x) / NECK_R[0]) ** 2 + ((q.y - NECK_C.y) / NECK_R[1]) ** 2 <= 1.0: best = t
        return best
    thick = [0.016] * (H + 1)          # cuello alto grueso, con el borde redondeado (anillo de cresta)
    # pared interior: dentro de la exterior en toda su altura (el cuello alto se abre hacia arriba)
    r3 = [max(min(r2[k] - thick[k], r1[k] - 0.009, base[k] - 0.009), neck_r(k) + 0.010) for k in range(H + 1)]
    r4 = [r3[k] - 0.002 for k in range(H + 1)]
    z4 = [1.470] * (H + 1)
    C1, C2, C3, C4 = ring_at(z1, r1), ring_at(z2, r2), ring_at(z2, r3), ring_at(z4, r4)
    C2b = ring_at([z + 0.006 for z in z2], [0.5 * (a_ + b_) for a_, b_ in zip(r2, r3)])     # cresta del borde
    # escote en V: delante no hay cuello alto, el pecho sube liso hasta el cuello
    def wk(k):
        kk = min(k, N - k)
        return max(0.0, min(1.0, (COLLAR_H[kk] - 1.465 - 0.012) / 0.024))
    for k in range(N):
        w = wk(k)
        if w >= 1.0: continue
        kk = min(k, N - k)
        p0 = top[k].co.copy()
        d = dirs[kk].copy()
        if k > H: d.x = -d.x
        pn = Vector((0, YC, NECK_Z)) + d * neck_r(kk)
        for ring, t in ((C1, 0.17), (C2, 0.33), (C2b, 0.5), (C3, 0.67), (C4, 0.83)):
            ring[k] = p0.lerp(pn, t).lerp(ring[k], w)
    c1 = extrude_ring(top, C1)
    c2 = extrude_ring(c1, C2)
    c2b = extrude_ring(c2, C2b)
    c3 = extrude_ring(c2b, C3)
    c4 = extrude_ring(c3, C4)
    # suelo del cuello alto: reduccion 32 -> 16 solo con quads (3:1 + 1:1 en cada grupo de 4)
    neck = [bm.verts.new((NECK_C.x + NECK_R[0] * math.sin(TAU * m / 16), NECK_C.y - NECK_R[1] * math.cos(TAU * m / 16), NECK_Z)) for m in range(16)]
    def quad(a, b, c, d): bm.faces.new((a, b, c, d))
    for g in range(8):
        t = [c4[(4 * g + i) % N] for i in range(5)]; b = [neck[(2 * g + i) % 16] for i in range(3)]
        if g < 4:   # lado izquierdo: 3:1 sobre t0..t3, recto t3..t4
            x = bm.verts.new(t[1].co * 0.5 + (b[0].co * 0.7 + b[1].co * 0.3) * 0.5)
            y = bm.verts.new(t[2].co * 0.5 + (b[0].co * 0.3 + b[1].co * 0.7) * 0.5)
            quad(t[0], t[1], x, b[0]); quad(t[1], t[2], y, x); quad(t[2], t[3], b[1], y); quad(x, y, b[1], b[0])
            quad(t[3], t[4], b[2], b[1])
        else:       # lado derecho (espejo): recto t0..t1, 3:1 sobre t1..t4
            quad(t[0], t[1], b[1], b[0])
            x = bm.verts.new(t[2].co * 0.5 + (b[1].co * 0.7 + b[2].co * 0.3) * 0.5)
            y = bm.verts.new(t[3].co * 0.5 + (b[1].co * 0.3 + b[2].co * 0.7) * 0.5)
            quad(t[1], t[2], x, b[1]); quad(t[2], t[3], y, x); quad(t[3], t[4], b[2], y); quad(x, y, b[2], b[1])
    # cabeza: anillos de 16 ajustados a la cabeza actual
    HEAD_Z = [1.496, 1.507, 1.517, 1.535, 1.560, 1.585, 1.598, 1.609, 1.635, 1.662, 1.690, 1.715, 1.738, 1.753, 1.763]
    HR = [neck]
    for z in HEAD_Z:
        C = Vector((0, -0.020, z)); rs, ds = [], []
        for m in range(NH):
            t = TAU * m / NH; d = Vector((math.sin(t), -math.cos(t), 0))
            # bajo la mandibula solo la barbilla sobresale; el resto es cuello (no el interior del cuello alto)
            rmax = 0.2 if (z >= 1.53 or (z >= 1.505 and math.cos(t) > 0.85)) else 0.07
            r = ray_in(T_HEAD, C, d, rmax)
            if r is not None and not (0.02 < r < 0.19): r = None
            rs.append(r); ds.append(d)
        rs = fill(rs, True)
        for m in range(1, NH // 2):          # simetria
            a_, b_ = rs[m], rs[NH - m]; rs[m] = rs[NH - m] = 0.5 * (a_ + b_)
        pts = [C + d * r for d, r in zip(ds, rs)]
        pts[0].x = 0.0; pts[NH // 2].x = 0.0
        HR.append(extrude_ring(HR[-1], pts))
    topz = 1.780
    def lift(p, u, v):
        w = 1 - min(1.0, (math.hypot(u - 0.5, v - 0.5) / 0.7071) ** 2)
        return Vector((p.x, p.y, HEAD_Z[-1] + (topz - HEAD_Z[-1]) * w))
    grid_cap(HR[-1], 4, 4, 2, lift=lift)
    return c1, c2

# =============================================================== 5) TUBOS DE LA ESPALDA
def socket(region, center, nrm, r_out, depth, r_tube, mouth=None):
    """enchufe hexagonal extruido desde una region de 2 caras; devuelve la tapa (boca del tubo).
    mouth = (centro, direccion): la boca se gira hacia esa direccion (codo)"""
    vs = {v for f in region for v in f.verts}
    c0 = sum((v.co for v in vs), Vector()) / len(vs)
    # el orden de los vertices se lee en el plano de la propia region y se lleva al plano del enchufe
    nreg = sum((f.normal for f in region), Vector()).normalized()
    if nreg.dot(nrm) < 0: nreg = -nreg
    upw = Vector((0, 0, 1)) if abs(nreg.z) < 0.9 else Vector((1, 0, 0))
    side_r = nreg.cross(upw).normalized(); up_r = side_r.cross(nreg).normalized()
    ang = {v: math.atan2((v.co - c0).dot(up_r), (v.co - c0).dot(side_r)) for v in vs}
    side = (side_r - nrm * side_r.dot(nrm)).normalized(); up = side.cross(nrm).normalized()
    uni = uniform_angles(list(ang.values()), base=None)
    ua = dict(zip(ang.keys(), uni))
    if mouth is not None:
        mc, md = mouth
        mside = (side - md * side.dot(md)).normalized(); mup = mside.cross(md).normalized()
        if mup.dot(up) < 0: mup = -mup
    def place(i, th):
        if i == 0: return center + nrm * (depth * 0.25) + (side * math.cos(th) + up * math.sin(th)) * r_out
        if i == 1: return center + nrm * depth + (side * math.cos(th) + up * math.sin(th)) * r_out
        if mouth is not None: return mc + (mside * math.cos(th) + mup * math.sin(th)) * (r_tube + 0.002)
        return center + nrm * (depth + 0.003) + (side * math.cos(th) + up * math.sin(th)) * r_tube
    cap = region_loft(region, ua, [0, 1, 2], place)
    if mouth is not None: return cap, ua, mside, mup
    return cap, ua, side, up

def tube_path_frames(P, t_start, t_end, side0, up0):
    T = []
    for i in range(len(P)):
        a = P[max(i - 1, 0)][0]; b = P[min(i + 1, len(P) - 1)][0]
        T.append((b - a).normalized())
    T[0] = t_start.normalized(); T[-1] = t_end.normalized()
    Uu = [(side0 - T[0] * side0.dot(T[0])).normalized()]
    for i in range(1, len(P)):
        u = Uu[-1] - T[i] * Uu[-1].dot(T[i]); Uu.append(u.normalized())
    sg = 1.0 if T[0].cross(Uu[0]).dot(up0) > 0 else -1.0      # mismo sentido de giro que el enchufe
    Wd = [T[i].cross(Uu[i]).normalized() * sg for i in range(len(P))]
    return T, Uu, Wd

def back_y(x, z):
    """y de la espalda del torso NUEVO en (x, z), interpolada en sus anillos"""
    ax = abs(x)
    zs = [TORSO[i][0] for i in range(len(TORSO))]
    z = max(zs[0], min(zs[-1], z))
    i = max(j for j in range(len(zs) - 1) if zs[j] <= z) if z >= zs[0] else 0
    i = min(i, len(zs) - 2)
    def ring_y(ii):
        pts = TPTS[ii]
        for k in range(N // 4, N // 2):
            a, b = pts[k], pts[k + 1]           # de la espalda lateral (x grande) al centro (x = 0)
            if b.x <= ax <= a.x:
                u = (a.x - ax) / max(a.x - b.x, 1e-6); return a.y + (b.y - a.y) * u
        return pts[N // 2].y
    t = (z - zs[i]) / (zs[i + 1] - zs[i])
    return ring_y(i) * (1 - t) + ring_y(i + 1) * t

def bridge_by_angle(A, B, axis, side, up):
    """une dos anillos de igual numero de vertices con quads, emparejados por angulo"""
    def srt(L):
        c = sum((v.co for v in L), Vector()) / len(L)
        return sorted(L, key=lambda v: math.atan2((v.co - c).dot(up), (v.co - c).dot(side)))
    A = srt(A); B = srt(B); n = len(A)
    assert n == len(B), (n, len(B))
    j = min(range(n), key=lambda j: sum((A[i].co - B[(i + j) % n].co).length for i in range(n)))
    if DEBUG: print('bridge', n, [tuple(round(c, 3) for c in v.co) for v in A], [tuple(round(c, 3) for c in v.co) for v in B], j)
    for i in range(n):
        bm.faces.new((A[i], A[(i + 1) % n], B[(i + 1 + j) % n], B[(i + j) % n]))

def build_tube(s, c1):
    # enchufe superior: 2 caras del cuello alto (espalda), enchufe inferior: 2 caras de la espalda baja
    top_reg = [face_of([R[17][k], R[17][k + 1], c1[k + 1], c1[k]]) for k in ((13, 14) if s > 0 else (N - 15, N - 14))]
    bot_reg = [face_of([R[10][k], R[10][k + 1], R[11][k + 1], R[11][k]]) for k in ((11, 12) if s > 0 else (N - 13, N - 12))]
    RT, RC, GAP = 0.0135, 0.021, 0.007
    X = 0.055 * s
    def on_back(x, z, r): return Vector((x, back_y(x, z) + r + GAP, z))
    def clear(p, ymin): return Vector((p.x, max(p.y, ymin), p.z))
    path = [(clear(on_back(X, 1.470, RT), 0.176), RT), (clear(on_back(X, 1.452, RT), 0.171), RT),
            (on_back(X, 1.430, RT), RT), (on_back(X, 1.428, RC), RC), (on_back(X, 1.407, RC), RC), (on_back(X, 1.405, RT), RT),   # abrazadera
            (on_back(X, 1.360, RT), RT), (on_back(X * 1.02, 1.314, RT), RT), (on_back(X * 1.12, 1.287, RT), RT),
            (on_back(0.079 * s, 1.262, RT), RT), (on_back(0.097 * s, 1.242, RT), RT)]
    # enchufe superior asentado sobre la espalda del cuello alto
    rv = {v for f in top_reg for v in f.verts}
    top_n = Vector((0, 1, 0))
    top_c = Vector((0.055 * s, max(v.co.y for v in rv) - 0.004, 1.492))
    # enchufe inferior: codo que sale de la espalda y abre su boca hacia el tubo que llega
    rv = {v for f in bot_reg for v in f.verts}
    c_r = sum((v.co for v in rv), Vector()) / len(rv)
    n_s = sum((f.normal for f in bot_reg), Vector()).normalized()
    if n_s.dot(Vector((s * 0.5, 1, 0))) < 0: n_s = -n_s
    D = Vector((0.70 * s, 0, -0.71)); D = (D - n_s * D.dot(n_s)).normalized()
    mdir = (n_s * 0.5 - D * 0.866).normalized()
    bot_c = c_r + n_s * 0.002
    M = bot_c + n_s * 0.030 - D * 0.010
    path.append((M + mdir * 0.028, RT))
    top_cap, ua_t, side_t, up_t = socket(top_reg, top_c, top_n, 0.018, 0.028, RT)
    bot_cap, ua_b, side_b, up_b = socket(bot_reg, bot_c, n_s, 0.017, 0.012, RT, mouth=(M, mdir))
    mouth_t = top_c + top_n * 0.031
    # codo de salida: arco de radio RB desde la boca del enchufe hasta bajar en vertical
    RB = 0.019; p0 = mouth_t + top_n * 0.006; cc = p0 - Vector((0, 0, RB))
    elbow = [(cc + Vector((0, RB * math.sin(a), RB * math.cos(a))), RT) for a in (0.0, math.radians(30), math.radians(60), math.radians(90))]
    zc = elbow[-1][0].z
    path = [pp for pp in path if pp[0].z < zc - 0.012]
    P = elbow + path
    P.append((M + mdir * 0.007, RT))
    T, Uu, Wd = tube_path_frames(P, top_n, -mdir, side_t, up_t)
    vs = {v for f in top_cap for v in f.verts}; c0 = sum((v.co for v in vs), Vector()) / len(vs)
    par = {v: math.atan2((v.co - c0).dot(up_t), (v.co - c0).dot(side_t)) for v in vs}
    def place(i, th):
        c, r = P[i]; return c + (Uu[i] * math.cos(th) + Wd[i] * math.sin(th)) * r
    end = region_loft(top_cap, par, P, place)
    A = list({v for f in end for v in f.verts}); B = list({v for f in bot_cap for v in f.verts})
    bmesh.ops.delete(bm, geom=end, context='FACES')
    bmesh.ops.delete(bm, geom=bot_cap, context='FACES')
    bridge_by_angle(A, B, mdir, side_b, up_b)

# =============================================================== 6) CINTURON: hebilla + bolsas
def belt_box(ks, rows, center_deg, half_deg, r_front, z_lo, z_hi, s=+1, lid=0.0):
    """extruye caras de la banda del cinturon y aplana la tapa en una caja (lid: solapa que sobresale arriba)"""
    region = [face_of([R[i][k % N], R[i][(k + 1) % N], R[i + 1][(k + 1) % N], R[i + 1][k % N]]) for i in rows for k in ks]
    vs = {v for f in region for v in f.verts}
    th = math.radians(center_deg)
    n = Vector((s * math.sin(th), -math.cos(th), 0)); t = Vector((math.cos(th) * s, math.sin(th), 0))
    C = Vector((0, YC, 0))
    lat = {v: (v.co - C).dot(t) for v in vs}
    lmax = max(abs(x) for x in lat.values())
    hw = r_front * math.tan(math.radians(half_deg))
    zs = sorted({round(v.co.z, 4) for v in vs}); zlo0, zhi0 = zs[0], zs[-1]
    par = {v: (lat[v] / lmax, (v.co.z - zlo0) / (zhi0 - zlo0)) for v in vs}
    def place(i, pr):
        u, w = pr
        p = C + n * (r_front + (lid if w > 0.8 else 0.0)) + t * (u * hw * (1.06 if w > 0.8 and lid else 1.0))
        p.z = z_lo + (z_hi - z_lo) * w
        return p
    region_loft(region, par, [0], place)

def panel_box(region, depth, lid=0.0, inset=0.10):
    """bolsillo: caja extruida de una region de caras; la fila de arriba sobresale (solapa)"""
    vs = {v for f in region for v in f.verts}
    cc = sum((v.co for v in vs), Vector()) / len(vs)
    n = sum((f.normal for f in region), Vector()).normalized()
    if n.dot(cc - Vector((0, YC, cc.z))) < 0: n = -n
    ztop = max(v.co.z for v in vs)
    par = {v: v.co.copy() for v in vs}
    def place(i, p0):
        q = cc + (p0 - cc) * (1 - inset)
        return q + n * (depth + (lid if p0.z > ztop - 1e-4 else 0.0))
    region_loft(region, par, [0], place)

# =============================================================== construir
for s in (+1, -1): build_leg(s)
arm_rings = {}
for s in (+1, -1):
    ef, rings = build_arm(s)
    build_hand(s, ef, rings)
    arm_rings[s] = rings
build_device(arm_rings[+1])
c1, c2 = build_collar_head()
for s in (+1, -1): build_tube(s, c1)
belt_box((30, 31, 0, 1), (4, 5), 0.0, 15.0, 0.170, 0.984, 1.041)                  # hebilla
for s in (+1, -1):
    ks = (5, 6) if s > 0 else (N - 7, N - 6)
    belt_box(ks, (3, 4, 5, 6), 66.0, 8.0, 0.222, 0.962, 1.055, s, lid=0.006)        # bolsa delantera con solapa
    ks = (10, 11) if s > 0 else (N - 12, N - 11)
    belt_box(ks, (3, 4, 5, 6), 122.0, 8.0, 0.215, 0.962, 1.055, s, lid=0.006)       # bolsa trasera con solapa
    # bolsillos del pecho con solapa (hoja de Lukas)
    ks = (2, 3) if s > 0 else (N - 4, N - 3)
    panel_box([face_of([R[i][k], R[i][k + 1], R[i + 1][k + 1], R[i + 1][k]]) for i in (13, 14) for k in ks], 0.007, lid=0.005)

bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-6)
bmesh.ops.recalc_face_normals(bm, faces=bm.faces)

old = bpy.data.objects.get(OUT_NAME)
if old:
    m0 = old.data; bpy.data.objects.remove(old)
    if m0.users == 0: bpy.data.meshes.remove(m0)
me = bpy.data.meshes.new(OUT_NAME); bm.to_mesh(me); bm.free()
for pl in me.polygons: pl.use_smooth = False
obj = bpy.data.objects.new(OUT_NAME, me)
coll = bpy.data.collections.get('Lukas_RH07') or bpy.data.collections.new('Lukas_RH07')
if coll.name not in bpy.context.scene.collection.children: bpy.context.scene.collection.children.link(coll)
coll.objects.link(obj)
