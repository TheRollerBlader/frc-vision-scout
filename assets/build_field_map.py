"""Tessellate the REBUILT field STEP, tag every leaf part by field element,
and build 2D maps (1 cm grid) used by the randomizer:
  floor_z      : height of the drivable surface (carpet, bump tops/ramps, tower base, depot)
  obstacle_low : lowest underside of any obstacle geometry above the floor (inf = clear)
Coordinates written in FIELD frame (meters, Z-up): origin at the blue alliance wall,
right-hand corner as seen from the blue driver stations, +X toward red, +Y to the left.
"""
import sys, json, re, numpy as np
from OCP.STEPCAFControl import STEPCAFControl_Reader
from OCP.TDocStd import TDocStd_Document
from OCP.TCollection import TCollection_ExtendedString
from OCP.XCAFDoc import XCAFDoc_DocumentTool, XCAFDoc_ColorType
from OCP.TDF import TDF_Label
from OCP.collections import Sequence_TDF_Label
from OCP.TDataStd import TDataStd_Name
from OCP.TopLoc import TopLoc_Location
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.TopExp import TopExp_Explorer
from OCP.TopAbs import TopAbs_FACE, TopAbs_REVERSED
from OCP.BRep import BRep_Tool
from OCP.TopoDS import TopoDS
from OCP.Quantity import Quantity_Color
import trimesh

STEP=sys.argv[1]; OUT=sys.argv[2]
doc=TDocStd_Document(TCollection_ExtendedString("doc"))
r=STEPCAFControl_Reader(); r.SetNameMode(True); r.SetColorMode(True)
r.ReadFile(STEP); r.Transfer(doc)
ST=XCAFDoc_DocumentTool.ShapeTool_s(doc.Main())
CT=XCAFDoc_DocumentTool.ColorTool_s(doc.Main())
def name(l):
    a=TDataStd_Name()
    return a.Get().ToExtString() if l.FindAttribute(TDataStd_Name.GetID_s(), a) else "?"
def color(l, shape):
    c=Quantity_Color()
    for t in (XCAFDoc_ColorType.XCAFDoc_ColorSurf, XCAFDoc_ColorType.XCAFDoc_ColorGen):
        if CT.GetColor_s(l,t,c) or CT.GetColor(shape,t,c):
            return (c.Red(),c.Green(),c.Blue())
    return None

def category(top, path):
    t=top.lower(); p=" / ".join(path).lower()
    if 'fuel' in t: return 'fuel'
    if 'tape' in t: return 'tape'
    if 'carpet' in t: return 'carpet'
    if 'scoring table' in t: return 'scoring_table'
    if 'bump' in t: return 'bump'
    if 'trench' in t: return 'trench'
    if 'hub' in t: return 'hub'
    if 'depot' in t: return 'depot'
    if 'tower' in t: return 'tower_base' if 'tower base' in p else 'tower'
    if 'outpost' in t: return 'outpost'
    if 'perimeter' in t: return 'perimeter'
    return 'other'

leaves=[]  # (category, top, located shape, color)
def walk(label, loc, path, inherited_color):
    shape=ST.GetShape_s(label)
    col=color(label, shape) or inherited_color
    childloc = loc.Multiplied(ST.GetLocation_s(label)) if ST.IsComponent_s(label) else loc
    ref=label
    if ST.IsReference_s(label):
        ref=TDF_Label(); ST.GetReferredShape_s(label,ref)
        col=color(ref, ST.GetShape_s(ref)) or col
    nm=name(label); path=path+[nm]
    if ST.IsAssembly_s(ref):
        comps=Sequence_TDF_Label(); ST.GetComponents_s(ref,comps)
        for i in range(1,comps.Length()+1): walk(comps.Value(i), childloc, path, col)
    else:
        s=shape.Moved(loc) if not loc.IsIdentity() else shape
        top=path[1] if len(path)>1 else path[0]
        leaves.append((category(top,path), top, s, col))
roots=Sequence_TDF_Label(); ST.GetFreeShapes(roots)
for i in range(1,roots.Length()+1): walk(roots.Value(i), TopLoc_Location(), [], None)
print("leaves", len(leaves), file=sys.stderr)


meshes={}
for cat,top,s,col in leaves:
    if cat=='fuel': continue
    BRepMesh_IncrementalMesh(s, 2.0, False, 0.35, True)
    exp=TopExp_Explorer(s, TopAbs_FACE)
    V=[];F=[];off=0
    while exp.More():
        f=TopoDS.Face(exp.Current()); l=TopLoc_Location()
        tri=BRep_Tool.Triangulation_s(f,l)
        if tri is not None:
            tr=l.Transformation()
            n=tri.NbNodes()
            pts=np.array([[p.X(),p.Y(),p.Z()] for p in (tri.Node(i).Transformed(tr) for i in range(1,n+1))])
            tris=np.array([[t.Value(1),t.Value(2),t.Value(3)] for t in (tri.Triangle(i) for i in range(1,tri.NbTriangles()+1))])-1
            if f.Orientation()==TopAbs_REVERSED: tris=tris[:,[0,2,1]]
            V.append(pts); F.append(tris+off); off+=n
        exp.Next()
    if not V: continue
    key=(cat,col)
    meshes.setdefault(key,[]).append((np.vstack(V),np.vstack(F)))

# save triangle soups per (category,color) in CAD mm
out={}; meta=[]
for i,((cat,col),lst) in enumerate(meshes.items()):
    tris=np.concatenate([v[f] for v,f in lst]).astype(np.float32)  # (T,3,3)
    out[f"m{i}"]=tris; meta.append({"key":f"m{i}","category":cat,"color":col,"tris":int(len(tris))})
np.savez_compressed(OUT+"/field_tris_cad_mm.npz", **out)
json.dump(meta, open(OUT+"/field_tris_meta.json","w"), indent=1)
print("saved", sum(m["tris"] for m in meta), "triangles", file=sys.stderr)
