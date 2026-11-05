# Lukas R-H-07 - blockout de UNA sola malla a partir de UNA primitiva (circulo de 32 lados en la cintura),
# construido solo con extrusiones (de anillos de aristas y de regiones de caras). Pose A (brazos a 45 grados).
# Convencion: mira hacia -Y, su izquierda es +X, Z arriba, metros, pies en z = 0.
import bpy, bmesh, math
from mathutils import Vector, Matrix

N = 32
MATS = ['Suit', 'Skin', 'Glove', 'Boot', 'Sole', 'RedTrim', 'Shirt', 'Metal']
SUIT, SKIN, GLOVE, BOOT, SOLE, RED, SHIRT, METAL = range(8)
COLORS = {'Suit': (0.16, 0.29, 0.26), 'Skin': (0.78, 0.56, 0.44), 'Glove': (0.07, 0.08, 0.08), 'Boot': (0.33, 0.21, 0.14),
          'Sole': (0.08, 0.07, 0.07), 'RedTrim': (0.55, 0.11, 0.11), 'Shirt': (0.06, 0.06, 0.07), 'Metal': (0.36, 0.38, 0.41)}

def lerp(a, b, t): return a + (b - a) * t
def se(t, p):
    c, s = math.cos(t), math.sin(t)
    return math.copysign(abs(c) ** (2 / p), c), math.copysign(abs(s) ** (2 / p), s)
def body_pt(k, z, X, Yf, Yb, p, n=N, yc=0.0):
    cx, sy = se(math.radians(-90 + 360.0 * k / n), p)
    return Vector((X * cx, yc + (Yf if sy < 0 else Yb) * sy, z))

bm = bmesh.new()
def key(co): return (round(co.x, 6), round(co.y, 6), round(co.z, 6))

# ---------------- operaciones de extrusion ----------------
def extrude_ring(verts, positions, mat=SUIT):
    """extruye las aristas de un anillo abierto; devuelve el anillo nuevo en el mismo orden"""
    n = len(verts)
    edges = [bm.edges.get((verts[i], verts[(i + 1) % n])) for i in range(n)]
    r = bmesh.ops.extrude_edge_only(bm, edges=edges)
    by = {key(v.co): v for v in r['geom'] if isinstance(v, bmesh.types.BMVert)}
    new = [by[key(v.co)] for v in verts]
    for f in r['geom']:
        if isinstance(f, bmesh.types.BMFace): f.material_index = mat
    for v, p in zip(new, positions): v.co = p
    return new

def extrude_region(faces, mat=SUIT, par=None):
    """extruye una region de caras. Devuelve (mapa vertice_borde_viejo -> nuevo, caras tapa).
    par: dict vertice -> parametro; se copia a los vertices nuevos (incluidos los interiores)."""
    oldk = {}
    for f in faces:
        for v in f.verts: oldk[v] = key(v.co)
    pk = {oldk[v]: par[v] for v in oldk} if par is not None else None
    r = bmesh.ops.extrude_face_region(bm, geom=list(faces))
    nv = [g for g in r['geom'] if isinstance(g, bmesh.types.BMVert)]
    by = {key(v.co): v for v in nv}
    mp = {v: by[k] for v, k in oldk.items() if v.is_valid}
    if pk is not None:
        for v in nv: par[v] = pk[key(v.co)]
    newset = set(nv)
    cap = [f for f in r['geom'] if isinstance(f, bmesh.types.BMFace) and all(v in newset for v in f.verts)]
    left = [f for f in faces if f.is_valid]
    if left: bmesh.ops.delete(bm, geom=left, context='FACES')
    capset = set(cap)
    for v in newset:                       # paredes laterales nuevas = caras que tocan vertices nuevos y viejos
        for f in v.link_faces:
            if f not in capset and any(w not in newset for w in f.verts): f.material_index = mat
    return mp, cap

def face_of(vs):
    s = set(vs)
    for f in vs[0].link_faces:
        if set(f.verts) == s: return f
    raise RuntimeError('cara no encontrada')

def grid_cap(loop, r, c, start, lift=None, mat=SUIT):
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
    fs = []
    for i in range(r):
        for j in range(c):
            f = bm.faces.new((G[i][j], G[i][j + 1], G[i + 1][j + 1], G[i + 1][j])); f.material_index = mat; fs.append(f)
    return G, fs

def uniform_angles(phis):
    n = len(phis); order = sorted(range(n), key=lambda i: phis[i] % (2 * math.pi))
    best = None
    for s in range(n):
        tgt = [0.0] * n
        for rank, i in enumerate(order): tgt[i] = 2 * math.pi * ((rank - s) % n) / n
        cost = sum(abs(math.remainder(tgt[i] - phis[i], 2 * math.pi)) for i in range(n))
        if best is None or cost < best[0]: best = (cost, tgt)
    return best[1]

def frame(axis, s):
    """a: eje del miembro, f: frente, o: exterior/arriba (igual en ambos lados)"""
    a = Vector(axis).normalized(); f = Vector((0, -1, 0)); f = (f - a * f.dot(a)).normalized(); o = f.cross(a).normalized()
    if s < 0: o = -o
    return a, f, o

def region_loft(region, par, steps, mat_of_step, place):
    """extruye una region varias veces; place(i, param) -> posicion"""
    cur = region
    for i, st in enumerate(steps):
        mp, cur = extrude_region(cur, mat_of_step(i), par)
        for v in {v for f in cur for v in f.verts}: v.co = place(i, par[v])
    return cur

# ================= 1) PRIMITIVA: circulo de 32 lados en la cintura =================
TORSO = [  # z,    X,     Yf,    Yb,    p
    (0.950, 0.176, 0.132, 0.124, 2.7),
    (1.040, 0.170, 0.144, 0.134, 2.7),
    (1.130, 0.167, 0.143, 0.146, 2.7),
    (1.180, 0.168, 0.144, 0.148, 2.7),   # loop de soporte pecho -> axila
    (1.225, 0.170, 0.144, 0.150, 2.8),   # axila (base del hueco del brazo)
    (1.280, 0.171, 0.140, 0.152, 2.8),   # loops de soporte que pasan por la axila
    (1.335, 0.172, 0.134, 0.153, 2.8),
    (1.390, 0.171, 0.121, 0.147, 2.8),
    (1.448, 0.168, 0.104, 0.136, 2.6),   # parte alta del hombro
    (1.472, 0.130, 0.086, 0.110, 2.4)]   # base del cuello alto
circ = bmesh.ops.create_circle(bm, cap_ends=False, segments=N, radius=0.17)
ring0 = sorted(circ['verts'], key=lambda v: (math.atan2(v.co.y, v.co.x) + math.pi / 2) % (2 * math.pi))
z, X, Yf, Yb, p = TORSO[0]
for k, v in enumerate(ring0): v.co = body_pt(k, z, X, Yf, Yb, p)
R = [ring0]
for z, X, Yf, Yb, p in TORSO[1:]:
    R.append(extrude_ring(R[-1], [body_pt(k, z, X, Yf, Yb, p) for k in range(N)]))
HIPS = [(0.875, 0.184, 0.135, 0.132, 2.6), (0.800, 0.198, 0.128, 0.126, 2.5)]
Hr = [ring0]
for z, X, Yf, Yb, p in HIPS:
    Hr.append(extrude_ring(Hr[-1], [body_pt(k, z, X, Yf, Yb, p) for k in range(N)]))
hb = Hr[-1]
crotch = bm.verts.new((0.0, 0.010, 0.752))
LEG_LOOP = {+1: [crotch] + hb[0:N // 2 + 1], -1: [crotch] + hb[N // 2:] + [hb[0]]}
leg_face = {s: bm.faces.new(LEG_LOOP[s]) for s in (+1, -1)}

def torso_back_y(x, z):
    for a, b in zip(TORSO, TORSO[1:]):
        if a[0] <= z <= b[0]:
            t = (z - a[0]) / (b[0] - a[0]); X = lerp(a[1], b[1], t); Yb = lerp(a[3], b[3], t); p = lerp(a[4], b[4], t)
            return Yb * max(0.0, 1 - min(abs(x) / X, 0.999) ** p) ** (1 / p)
    return 0.12

# ================= 2) PIERNAS + BOTAS =================
LEG = [  # z,     cx,    cy,     hw,    hdf,   hdb,  material
    (0.725, 0.110, -0.008, 0.100, 0.112, 0.110, SUIT),
    (0.655, 0.115, -0.007, 0.094, 0.104, 0.103, SUIT),
    (0.585, 0.118, -0.006, 0.087, 0.097, 0.097, SUIT),
    (0.500, 0.127,  0.000, 0.082, 0.093, 0.092, SUIT),   # rodilla: 3 loops juntos
    (0.472, 0.129,  0.007, 0.079, 0.092, 0.092, SUIT),   #   (bisagra)
    (0.444, 0.131,  0.016, 0.078, 0.090, 0.091, SUIT),
    (0.405, 0.135,  0.032, 0.080, 0.084, 0.092, SUIT),
    (0.345, 0.137,  0.036, 0.074, 0.083, 0.085, SUIT),
    (0.300, 0.140,  0.034, 0.079, 0.087, 0.087, SUIT),   # pliegue del pantalon sobre la bota
    (0.268, 0.139,  0.038, 0.081, 0.093, 0.093, SUIT),
    (0.262, 0.141,  0.040, 0.067, 0.076, 0.076, SUIT),
    (0.250, 0.141,  0.040, 0.069, 0.079, 0.079, BOOT),   # boca de la bota
    (0.205, 0.141,  0.040, 0.061, 0.071, 0.071, BOOT),
    (0.175, 0.141,  0.040, 0.057, 0.067, 0.067, BOOT),   # tobillo: 3 loops juntos
    (0.150, 0.141,  0.040, 0.054, 0.064, 0.064, BOOT),   #   (bisagra)
    (0.125, 0.141,  0.040, 0.055, 0.066, 0.066, BOOT),
    (0.090, 0.141,  0.040, 0.058, 0.070, 0.076, BOOT),   # empeine
    (0.034, 0.141,  0.040, 0.060, 0.072, 0.078, BOOT),   # canto de la suela
    (0.000, 0.141,  0.040, 0.059, 0.070, 0.076, SOLE)]
def build_leg(s):
    prev = LEG_LOOP[s]
    c0 = Vector((0.100 * s, 0.0))
    ang = {v: math.atan2(v.co.y - c0.y, v.co.x - c0.x) for v in prev}
    cur = [leg_face[s]]; rings = []
    for z, cx, cy, hw, hdf, hdb, m in LEG:
        mp, cur = extrude_region(cur, m)
        new = [mp[v] for v in prev]
        for vo, v in zip(prev, new):
            t = ang[vo]; ang[v] = t
            v.co = Vector((s * cx + hw * math.cos(t), cy + (hdf if math.sin(t) < 0 else hdb) * math.sin(t), z))
        rings.append(new); prev = new
    bmesh.ops.delete(bm, geom=cur, context='FACES_ONLY')
    loop = rings[-1]; n = len(loop); r_, c_ = 4, n // 2 - 4
    cx, cy = s * LEG[-1][1], LEG[-1][2]
    def score(st):
        Lp = [loop[(st + i) % n] for i in range(n)]
        cs = [Lp[0], Lp[c_], Lp[c_ + r_], Lp[2 * c_ + r_]]
        tg = [(-1, -1), (1, -1), (1, 1), (-1, 1)]
        return sum(((v.co.x - cx) * 12 - tx) ** 2 + ((v.co.y - cy) * 12 - ty) ** 2 for v, (tx, ty) in zip(cs, tg))
    grid_cap(loop, r_, c_, min(range(n), key=score), mat=SOLE)
    # pie: extrusion hacia delante de las caras frontales de las dos filas de abajo
    TH = math.radians(14.0) * s
    Fd = Vector((math.sin(TH), -math.cos(TH), 0)); LAT = Vector((math.cos(TH), math.sin(TH), 0)); C = Vector((cx, cy, 0))
    region = []
    for a, b in ((rings[-3], rings[-2]), (rings[-2], rings[-1])):
        for k in range(n):
            f = face_of([a[k], a[(k + 1) % n], b[(k + 1) % n], b[k]])
            d = f.calc_center_median() - C; d.z = 0
            if d.length > 1e-6 and d.normalized().dot(Fd) > 0.42: region.append(f)
    par = {}
    for f in region:
        for v in f.verts:
            d = v.co - C; par[v] = (d.dot(LAT), d.dot(Fd), v.co.z)
    STEPS = [(0.100, 1.04, 0.086), (0.150, 1.08, 0.070), (0.188, 1.00, 0.056), (0.209, 0.78, 0.046)]
    def place(i, pr):
        fwd, ws, top = STEPS[i]; l, f0, h = pr
        zz = h if h <= 0.036 else 0.034 + (h - 0.034) * (top - 0.034) / (0.090 - 0.034)
        return C + LAT * (l * ws) + Fd * (fwd + 0.25 * (f0 - 0.06)) + Vector((0, 0, zz))
    toe = region_loft(region, par, STEPS, lambda i: BOOT, place)
    for f in toe: f.material_index = BOOT
    for f in bm.faces:
        cc = f.calc_center_median()
        if cc.z < 0.003 and abs(cc.x - cx) < 0.13 and cc.z >= -0.001: f.material_index = SOLE

# ================= 3) TUBOS DE LA ESPALDA (extruidos de la espalda y soldados abajo) =================
def socket(region, s, out1, out2, r):
    """extruye un enchufe corto (caja) desde una region de la espalda y lo cierra a seccion de tubo"""
    vs = {v for f in region for v in f.verts}
    c0 = sum((v.co for v in vs), Vector()) / len(vs)
    nrm = sum((f.normal for f in region), Vector()).normalized()
    if nrm.y < 0: nrm = -nrm
    up = Vector((0, 0, 1)); side_ = nrm.cross(up).normalized()
    par = {v: ((v.co - c0).dot(side_), (v.co - c0).dot(up)) for v in vs}
    def place(i, pr):
        u, w = pr
        if i == 0: return c0 + nrm * out1 + side_ * (u * 0.80) + up * (w * 0.80)
        th = math.atan2(w, u); return c0 + nrm * out2 + (side_ * math.cos(th) + up * math.sin(th)) * r
    cap = region_loft(region, par, [0, 1], lambda i: METAL, place)
    for f in cap: f.material_index = METAL
    return cap, c0 + nrm * out2, nrm, side_

def build_tube(s):
    # enchufe de arriba (2 caras juntas bajo el cuello) y enchufe de abajo (2 caras en la espalda baja)
    ks = (14, 15) if s > 0 else (16, 17)
    top_reg = [face_of([R[7][k], R[7][k + 1], R[8][k + 1], R[8][k]]) for k in ks]
    ks2 = (11, 12) if s > 0 else (19, 20)
    bot_reg = [face_of([R[2][k], R[2][k + 1], R[3][k + 1], R[3][k]]) for k in ks2]
    top_cap, t0, tn, tside = socket(top_reg, s, 0.010, 0.020, 0.013)
    bot_cap, b0, bn, bside = socket(bot_reg, s, 0.010, 0.020, 0.013)
    # el tubo sale del enchufe de arriba, baja por la espalda y entra en el de abajo
    region = top_cap
    vs = {v for f in region for v in f.verts}
    c0 = sum((v.co for v in vs), Vector()) / len(vs)
    up = Vector((0, 0, 1))
    par = {v: math.atan2((v.co - c0).dot(up), (v.co - c0).dot(tside)) for v in vs}
    path = []
    for x, zz in ((0.048, 1.362), (0.050, 1.318), (0.053, 1.278), (0.062, 1.245), (0.080, 1.222), (0.100, 1.205)):
        path.append(((x, zz), 0.013))
    P = [(t0 + tn * 0.010 - up * 0.014, 0.013)]
    for (x, zz), r in path:
        P.append((Vector((s * x, torso_back_y(x, zz) + 0.026, zz)), r))
    P.append((b0 + bn * 0.030 + up * 0.018, 0.013))      # baja por delante del enchufe
    P.append((b0 + bn * 0.012, 0.013))                    # queda frente a la boca del enchufe
    hc = b0; hn = bn
    hole = bot_cap
    # marcos por transporte paralelo
    T = []
    for i in range(len(P)):
        a = P[max(i - 1, 0)][0]; b = P[min(i + 1, len(P) - 1)][0]
        T.append((b - a).normalized())
    T[0] = (tn * 0.5 + T[0] * 0.5).normalized()
    T[-1] = -bn
    T[-2] = (T[-2] - bn).normalized()
    # marco que sigue la curva (transporte paralelo); empieza con tside / arriba como el enchufe
    U = [(tside - T[0] * tside.dot(T[0])).normalized()]
    for i in range(1, len(P)):
        u = U[-1] - T[i] * U[-1].dot(T[i]); U.append(u.normalized())
    Wd = []
    for i in range(len(P)):
        w = T[i].cross(U[i]).normalized()
        if i == 0: w = w if w.dot(up) > 0 else -w
        else: w = w if w.dot(Wd[-1]) > 0 else -w
        Wd.append(w)
    def place2(i, th):
        c, r = P[i]
        return c + (U[i] * math.cos(th) + Wd[i] * math.sin(th)) * r
    end = region_loft(region, par, P, lambda i: METAL, place2)
    # abre el final del tubo y el enchufe de abajo, y los suelda con un puente de quads
    bmesh.ops.delete(bm, geom=end, context='FACES')
    bmesh.ops.delete(bm, geom=hole, context='FACES')
    edges = [e for e in bm.edges if e.is_boundary]
    near = [e for e in edges if min((e.verts[0].co - hc).length, (e.verts[1].co - hc).length) < 0.06]
    r = bmesh.ops.bridge_loops(bm, edges=near)
    for f in r['faces']: f.material_index = METAL

# ================= 4) BRAZOS en pose A + MANOS =================
D = lambda s: Vector((0.70 * s, -0.04, -0.71)).normalized()
ARM = [  # distancia sobre el brazo, hw(exterior/interior), hdf, hdb, material
    (0.050, 0.066, 0.066, 0.066, SUIT),   # deltoides
    (0.100, 0.058, 0.061, 0.061, SUIT),
    (0.170, 0.055, 0.058, 0.058, SUIT),
    (0.225, 0.056, 0.058, 0.058, SUIT),
    (0.245, 0.068, 0.068, 0.067, SUIT),   # manga enrollada
    (0.272, 0.070, 0.069, 0.067, SUIT),   # codo: 3 loops juntos
    (0.298, 0.070, 0.068, 0.066, SUIT),   #   (bisagra)
    (0.324, 0.067, 0.065, 0.063, RED),    # borde rojo de la manga
    (0.329, 0.050, 0.048, 0.046, RED),
    (0.400, 0.048, 0.046, 0.044, SKIN),
    (0.465, 0.042, 0.040, 0.038, SKIN),
    (0.508, 0.036, 0.034, 0.032, SKIN),   # muñeca: 3 loops juntos
    (0.530, 0.034, 0.032, 0.031, SKIN),   #   (bisagra)
    (0.552, 0.040, 0.037, 0.035, GLOVE),  # borde del guante
    (0.592, 0.024, 0.045, 0.045, GLOVE),  # palma
    (0.627, 0.023, 0.048, 0.048, GLOVE),
    (0.657, 0.019, 0.048, 0.048, GLOVE)]  # nudillos: anillo rectangular para los dedos
FPOS = [0.046, 0.038, 0.019, 0.0, -0.019, -0.038, -0.046]
def build_arm(s):
    side = N // 4 if s > 0 else 3 * N // 4
    # hueco del hombro: 3 filas (z 1.280 -> 1.448) x 5 columnas = 16 vertices de borde.
    # Los anillos 1.180 y 1.225 pasan por debajo de la axila (soporte continuo pecho -> axila).
    ks = range(side - 3, side + 2) if s > 0 else range(side - 2, side + 3)
    patch = [face_of([R[i][k % N], R[i][(k + 1) % N], R[i + 1][(k + 1) % N], R[i + 1][k % N]])
             for i in range(5, 8) for k in ks]
    face = bmesh.ops.dissolve_faces(bm, faces=patch, use_verts=True)['region'][0]
    loop = list(face.verts)
    hc = sum((v.co for v in loop), Vector()) / len(loop)
    J = Vector((0.196 * s, 0.000, 1.360)); d = D(s)
    a0, f0, o0 = frame((s, 0, 0), s)
    o0 = Vector((0, 0, 1))
    phi = {v: math.atan2((v.co - hc).dot(o0), (v.co - hc).dot(f0)) for v in loop}
    uni = uniform_angles([phi[v] for v in loop]); uphi = {v: u for v, u in zip(loop, uni)}
    stages = [(hc + Vector((0.028 * s, 0, 0.0)), Vector((s, 0, 0)) * 0.6 + d * 0.4, 0.075, 0.072, 0.072, SUIT)]
    for dist, hw, hdf, hdb, m in ARM: stages.append((J + d * dist, d, hw, hdf, hdb, m))
    cur = [face]; prev = loop; rings = []
    for i, (c, ax, hw, hdf, hdb, m) in enumerate(stages):
        mp, cur = extrude_region(cur, m)
        new = [mp[v] for v in prev]
        a, f, o = frame(ax, s)
        for vo, v in zip(prev, new):
            phi[v] = phi[vo]; uphi[v] = uphi[vo]
            ph = (phi[vo] + 0.5 * math.remainder(uphi[vo] - phi[vo], 2 * math.pi)) if i == 0 else uphi[vo]
            if i == len(stages) - 1:
                dg = round(math.degrees(uphi[vo]) / 22.5) % 16
                if dg == 0: fu, ov = 0.050, 0.0
                elif dg == 8: fu, ov = -0.050, 0.0
                elif dg < 8: fu, ov = FPOS[dg - 1], hw
                else: fu, ov = FPOS[16 - dg - 1], -hw
                v.co = c + f * fu + o * ov
            else:
                cf, so = se(ph, 2.3)
                v.co = c + f * ((hdf if cf > 0 else hdb) * cf) + o * (hw * so)
        rings.append(new); prev = new
    return cur, rings, uphi, d

def build_hand(s, end_faces, rings, uphi, d):
    a, f, o = frame(d, s)
    bmesh.ops.delete(bm, geom=end_faces, context='FACES_ONLY')
    loop = rings[-1]
    byd = {round(math.degrees(uphi[v]) / 22.5) % 16: v for v in loop}
    L = [byd[k] for k in (0, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1)]
    G, _ = grid_cap(L, 2, 6, 1, lift=lambda p, u, v: p + a * 0.006, mat=GLOVE)
    # dedos: columnas 1..4
    FING = [(0.082, 14), (0.090, 16), (0.086, 18), (0.070, 22)]
    for j, (ln, curl) in zip(range(1, 5), FING):
        region = [face_of([G[0][j], G[0][j + 1], G[1][j + 1], G[1][j]]), face_of([G[1][j], G[1][j + 1], G[2][j + 1], G[2][j]])]
        vs = {v for fc in region for v in fc.verts}
        cc = sum((v.co for v in vs), Vector()) / len(vs)
        par = {v: ((v.co - cc).dot(f), (v.co - cc).dot(o)) for v in vs}
        ST = [(0.010, 1.00, 1.00, 0.0), (0.042 * ln / 0.09, 0.96, 0.95, 0.7), (0.069 * ln / 0.09, 0.90, 0.88, 1.5), (ln, 0.72, 0.70, 2.5)]
        pos = [cc]; axes = []
        for i, (dist, su, sv, cfac) in enumerate(ST):
            ang = math.radians(curl * cfac)
            ax = (a * math.cos(ang) - o * math.sin(ang)).normalized()      # se curva hacia la palma
            axes.append(ax)
            pos.append(pos[-1] + ax * (dist - (ST[i - 1][0] if i else 0.0)))
        def place(i, pr, ST=ST, pos=pos, axes=axes):
            dist, su, sv, cfac = ST[i]; ax = axes[i]; oo = f.cross(ax).normalized()
            if oo.dot(o) < 0: oo = -oo
            u, w = pr
            uu = max(-1.0, min(1.0, u / 0.0095)) * 0.0095 * su; ww = max(-1.0, min(1.0, w / 0.019)) * 0.0085 * sv
            return pos[i + 1] + f * uu + oo * ww
        tip = region_loft(region, par, ST, lambda i: GLOVE if i == 0 else SKIN, place)
        for fc in tip: fc.material_index = SKIN
    # pulgar: 2 caras del lado del pulgar entre el borde del guante y la palma
    ra, rb = rings[14], rings[15]
    da = {round(math.degrees(uphi[v]) / 22.5) % 16: v for v in ra}; db = {round(math.degrees(uphi[v]) / 22.5) % 16: v for v in rb}
    region = [face_of([da[15], da[0], db[0], db[15]]), face_of([da[0], da[1], db[1], db[0]])]
    vs = {v for fc in region for v in fc.verts}
    cc = sum((v.co for v in vs), Vector()) / len(vs)
    nrm = sum((fc.normal for fc in region), Vector()).normalized()
    par = {v: v.co - cc for v in vs}
    Tdir = (nrm * 0.55 + a * 0.70 + f * 0.25).normalized()
    ST = [(0.010, 1.00, 0.00), (0.036, 0.95, 0.20), (0.060, 0.72, 0.45)]   # (distancia, escala, giro hacia los dedos)
    pos = [cc]; axes = []
    for i, (dist, sc, bend) in enumerate(ST):
        ax = (Tdir * (1 - bend) + a * bend).normalized()
        axes.append(ax); pos.append(pos[-1] + ax * (dist - (ST[i - 1][0] if i else 0.0)))
    def tplace(i, off):
        if i == 0: return cc + off * 0.80 + nrm * 0.010          # primer anillo: misma forma que la base, despegada
        ax = axes[i]; O = (o - ax * o.dot(ax)).normalized(); A = O.cross(ax).normalized() * s   # *s: mismo sentido en el espejo
        u = off.dot(a); w = off.dot(o); sc = ST[i][1]
        return pos[i + 1] + A * (max(-1.0, min(1.0, u / 0.02)) * 0.011 * sc) + O * (max(-1.0, min(1.0, w / 0.014)) * 0.0095 * sc)
    tip = region_loft(region, par, ST, lambda i: GLOVE if i == 0 else SKIN, tplace)
    for fc in tip: fc.material_index = SKIN

# ================= 5) CUELLO ALTO + CUELLO + CABEZA =================
def build_collar_head():
    COL = [(1.500, 0.124, 0.104, 0.100, 2.2), (1.540, 0.128, 0.114, 0.096, 2.2),
           (1.540, 0.114, 0.100, 0.082, 2.2), (1.500, 0.104, 0.090, 0.080, 2.2), (1.478, 0.096, 0.082, 0.078, 2.2)]
    cur = R[-1]
    for z, X, Yf, Yb, p in COL:
        pts = []
        for k in range(N):
            q = body_pt(k, z, X, Yf, Yb, p)
            w = max(0.0, math.cos(math.radians(360.0 * k / N))) ** 3       # 1 en el frente
            q.z = 1.472 + (q.z - 1.472) * (1 - 0.80 * w)                   # escote en V
            pts.append(q)
        cur = extrude_ring(cur, pts, SUIT)
    neck = [bm.verts.new(body_pt(k, 1.480, 0.050, 0.048, 0.048, 2.0, n=16, yc=-0.010)) for k in range(16)]
    for k in range(16):       # suelo del cuello alto: 32 -> 16
        bm.faces.new((cur[2 * k], cur[(2 * k + 1) % N], neck[k])).material_index = SUIT
        bm.faces.new((cur[(2 * k + 1) % N], cur[(2 * k + 2) % N], neck[(k + 1) % 16], neck[k])).material_index = SUIT
    HEAD = [  # z,    ax,    ayf,   ayb,    yc,   pf,  pb,  jaw,   eye,   brow
        (1.495, 0.052, 0.052, 0.050, -0.012, 2.0, 2.0, 0.000, 0.000, 0.000),
        (1.508, 0.056, 0.084, 0.058, -0.040, 2.6, 2.0, 0.000, 0.000, 0.004),
        (1.530, 0.063, 0.100, 0.074, -0.040, 3.0, 2.1, 0.006, 0.000, 0.004),
        (1.560, 0.068, 0.100, 0.094, -0.040, 3.2, 2.3, 0.008, 0.000, 0.000),
        (1.595, 0.074, 0.106, 0.104, -0.038, 3.2, 2.4, 0.004, 0.000, 0.000),
        (1.640, 0.077, 0.105, 0.110, -0.037, 3.0, 2.5, 0.000, 0.011, 0.000),
        (1.680, 0.076, 0.112, 0.112, -0.036, 2.8, 2.5, 0.000, 0.000, 0.004),
        (1.720, 0.074, 0.106, 0.108, -0.034, 2.6, 2.4, 0.000, 0.000, 0.000),
        (1.750, 0.066, 0.090, 0.096, -0.032, 2.3, 2.3, 0.000, 0.000, 0.000)]
    def hring(z, ax, ayf, ayb, yc, pf, pb, jaw, eye, brow, n=16):
        pts = []
        for k in range(n):
            t = math.radians(-90 + 360.0 * k / n); front = math.sin(t) < 0
            cx, sy = se(t, pf if front else pb); x = ax * cx; y = (ayf if front else ayb) * sy
            ang = abs(math.degrees(math.atan2(math.cos(t), -math.sin(t))))
            if front and 35 < ang < 80: x += math.copysign(jaw, x)
            if front and 15 < ang < 45: y += eye
            if front and ang < 20: y -= brow
            pts.append(Vector((x, yc + y, z)))
        return pts
    HR = [neck]
    for h in HEAD: HR.append(extrude_ring(HR[-1], hring(*h), SKIN))
    grid_cap(HR[-1], 4, 4, 2, lift=lambda p, u, v: Vector((p.x, p.y, 1.750 + 0.026 * (1 - min(1.0, (math.hypot(u - 0.5, v - 0.5) / 0.7071) ** 2)))), mat=SKIN)
    # nariz: 2 caras del centro de la cara
    reg = [face_of([HR[5][15], HR[5][0], HR[6][0], HR[6][15]]), face_of([HR[5][0], HR[5][1], HR[6][1], HR[6][0]])]
    vs = {v for f in reg for v in f.verts}; cc = sum((v.co for v in vs), Vector()) / len(vs)
    par = {v: (v.co.x - cc.x, v.co.z - cc.z) for v in vs}
    NS_ = [(Vector((0, -0.150, 1.615)), 0.80, 0.85), (Vector((0, -0.169, 1.598)), 0.40, 0.45)]
    def nplace(i, pr):
        c, sx, sz = NS_[i]; return c + Vector((pr[0] * sx, 0, pr[1] * sz + (0.008 if pr[1] > 0 else 0) * (i == 1)))
    region_loft(reg, par, NS_, lambda i: SKIN, nplace)
    # orejas
    for s, ks in ((1, (3, 4, 5)), (-1, (11, 12, 13))):
        reg = [face_of([HR[5][ks[i]], HR[5][ks[i + 1]], HR[6][ks[i + 1]], HR[6][ks[i]]]) for i in range(2)]
        vs = {v for f in reg for v in f.verts}; cc = sum((v.co for v in vs), Vector()) / len(vs)
        par = {v: (v.co.y - cc.y, v.co.z - cc.z) for v in vs}
        ES = [(0.008, 1.10), (0.014, 1.00)]
        def eplace(i, pr, s=s, cc=cc, ES=ES):
            out, sc = ES[i]; return Vector((cc.x + s * out, cc.y + 0.030 + pr[0] * 0.75 * sc, cc.z - 0.008 + pr[1] * 1.15 * sc))
        region_loft(reg, par, ES, lambda i: SKIN, eplace)

# ================= ejecutar =================
for s in (+1, -1): build_leg(s)
for s in (+1, -1): build_tube(s)
for s in (+1, -1):
    ef, rings, uphi, d = build_arm(s)
    build_hand(s, ef, rings, uphi, d)
build_collar_head()
for f in bm.faces:       # camiseta oscura visible en el escote
    c = f.calc_center_median()
    if 1.43 < c.z < 1.49 and c.y < -0.07 and abs(c.x) < 0.045 and f.material_index == SUIT: f.material_index = SHIRT
bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-6)
bmesh.ops.recalc_face_normals(bm, faces=bm.faces)

name = 'Lukas_R-H-07'
old = bpy.data.objects.get(name)
if old:
    m0 = old.data; bpy.data.objects.remove(old)
    if m0.users == 0: bpy.data.meshes.remove(m0)
me = bpy.data.meshes.new(name); bm.to_mesh(me); bm.free()
for pl in me.polygons: pl.use_smooth = False
for mn in MATS:
    m = bpy.data.materials.get('M_' + mn) or bpy.data.materials.new('M_' + mn)
    m.use_nodes = True
    bsdf = next(nd for nd in m.node_tree.nodes if nd.type == 'BSDF_PRINCIPLED')
    c = COLORS[mn]; bsdf.inputs['Base Color'].default_value = (*c, 1); bsdf.inputs['Roughness'].default_value = 0.85
    m.diffuse_color = (*c, 1); me.materials.append(m)
obj = bpy.data.objects.new(name, me)
coll = bpy.data.collections.get('Lukas_RH07') or bpy.data.collections.new('Lukas_RH07')
if coll.name not in bpy.context.scene.collection.children: bpy.context.scene.collection.children.link(coll)
coll.objects.link(obj)
