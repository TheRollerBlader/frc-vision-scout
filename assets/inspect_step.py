import sys, json
from OCP.STEPCAFControl import STEPCAFControl_Reader
from OCP.TDocStd import TDocStd_Document
from OCP.TCollection import TCollection_ExtendedString
from OCP.XCAFDoc import XCAFDoc_DocumentTool
from OCP.TDF import TDF_Label
from OCP.collections import Sequence_TDF_Label as TDF_LabelSequence
from OCP.TDataStd import TDataStd_Name
from OCP.Bnd import Bnd_Box
from OCP.BRepBndLib import BRepBndLib
from OCP.TopLoc import TopLoc_Location
from OCP.Interface import Interface_Static
path=sys.argv[1]; maxdepth=int(sys.argv[2]) if len(sys.argv)>2 else 2
doc=TDocStd_Document(TCollection_ExtendedString("doc"))
r=STEPCAFControl_Reader(); r.SetNameMode(True)
st=r.ReadFile(path); print("read",st, file=sys.stderr)
r.Transfer(doc)
st_tool=XCAFDoc_DocumentTool.ShapeTool_s(doc.Main())
def name(l):
    a=TDataStd_Name()
    if l.FindAttribute(TDataStd_Name.GetID_s(), a): return a.Get().ToExtString()
    return "?"
out=[]
def walk(label, loc, depth, prefix):
    shape=st_tool.GetShape_s(label)
    s=shape.Moved(loc) if not loc.IsIdentity() else shape
    b=Bnd_Box(); BRepBndLib.Add_s(s,b,False)
    if b.IsVoid(): bb=None
    else:
        mn=b.CornerMin(); mx=b.CornerMax(); bb=[round(v,1) for v in (mn.X(),mn.Y(),mn.Z(),mx.X(),mx.Y(),mx.Z())]
    out.append({"depth":depth,"name":prefix+name(label),"bbox":bb})
    if depth>=maxdepth: return
    childloc = loc.Multiplied(st_tool.GetLocation_s(label)) if st_tool.IsComponent_s(label) else loc
    ref=label
    if st_tool.IsReference_s(label):
        ref=TDF_Label(); st_tool.GetReferredShape_s(label,ref)
    if st_tool.IsAssembly_s(ref):
        comps=TDF_LabelSequence(); st_tool.GetComponents_s(ref,comps)
        for i in range(1,comps.Length()+1):
            c=comps.Value(i)
            walk(c, childloc, depth+1, "")
roots=TDF_LabelSequence(); st_tool.GetFreeShapes(roots)
for i in range(1,roots.Length()+1): walk(roots.Value(i), TopLoc_Location(), 0, "")
json.dump(out,open(sys.argv[3],"w"))
