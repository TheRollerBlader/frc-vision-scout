"""Find where the FRC field goes in each stadium GLB and write anchors.
Output: models/stadiums/anchors/<name>.json + <name>_anchor.png
Stadium frame: meters, Z-up (glTF Y-up converted back: scene=(x, -z_gltf, y_gltf)).
The anchor maps field frame -> stadium frame: field center (8.27, 4.035) goes to `center_xy`,
field +X along `yaw` rad, field floor at `floor_z`."""
import trimesh, numpy as np, json, glob, os, sys
from scipy import ndimage
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT='/workspace/frc/models/stadiums'
CELL=0.5; ZB=0.5
NEED_L, NEED_W = 20.0, 11.0   # field 16.54x8.07 + driver stations/guardrail/working space
CLEAR_LO, CLEAR_HI = 0.5, 8.0 # band above floor that must be empty (thin boards/hoops drop out cells; holes get filled)
REMOVE_HI = 3.5               # stadium geometry inside the field box up to this height is deleted
ROOF_MIN = 10.0

def load(p):
    s=trimesh.load(p,force='scene'); m=s.to_geometry() if hasattr(s,'to_geometry') else s.dump(concatenate=True)
    v=m.vertices.copy(); m.vertices=np.c_[v[:,0],-v[:,2],v[:,1]]
    return m

def raster(m):
    T=m.triangles; b0=m.bounds[0]-1; b1=m.bounds[1]+1
    nx,ny=int((b1[0]-b0[0])/CELL)+1,int((b1[1]-b0[1])/CELL)+1; nz=int((b1[2]-b0[2])/ZB)+2
    occ=np.zeros((nx,ny,nz),bool); upo=np.zeros((nx,ny,nz),bool)
    up=m.face_normals[:,2]>0.9
    # sample points on triangles densely enough for CELL (area-based count)
    area=m.area_faces; n=np.clip((area/(CELL*CELL/4)).astype(int)+3,3,200000)
    rng=np.random.default_rng(0)
    for s in range(0,len(T),20000):
        idx=np.arange(s,min(s+20000,len(T))); cnt=n[idx]; rep=np.repeat(idx,cnt)
        r1=np.sqrt(rng.random(len(rep))); r2=rng.random(len(rep))
        P=(1-r1)[:,None]*T[rep,0]+(r1*(1-r2))[:,None]*T[rep,1]+(r1*r2)[:,None]*T[rep,2]
        # include vertices too
        ix=((P[:,0]-b0[0])/CELL).astype(int); iy=((P[:,1]-b0[1])/CELL).astype(int); iz=((P[:,2]-b0[2])/ZB).astype(int)
        occ[ix,iy,iz]=True
        u=up[rep]; upo[ix[u],iy[u],iz[u]]=True
    return occ,upo,b0

def anchor(name):
    m=load(f'{ROOT}/glb/{name}.glb'); occ,upo,b0=raster(m)
    nx,ny,nz=occ.shape
    # per cell: lowest up-facing level with an empty band (CLEAR_LO,CLEAR_HI] above it and something (roof) above ROOF_MIN
    cs=np.concatenate([np.zeros((nx,ny,1),int),np.cumsum(occ,2)],2)   # cs[...,k]=count of occ in bins <k
    lo=int(CLEAR_LO/ZB)+1; hi=int(CLEAR_HI/ZB)+1; rf=int(ROOF_MIN/ZB)+1
    floor=np.full((nx,ny),np.nan); roof=np.zeros((nx,ny),bool)
    for k in range(nz):
        a=min(k+lo,nz); b=min(k+hi,nz); r=min(k+rf,nz)
        band=cs[...,b]-cs[...,a]; above=cs[...,nz]-cs[...,r]
        ok=upo[...,k]&(band==0)&(k+hi<=nz)&np.isnan(floor)
        floor[ok]=b0[2]+(k+0.5)*ZB; roof|=ok&(above>0)
    has=np.isfinite(floor); clear=has
    # enclosed = inside the ring of tall structure (seating bowl / walls), so parking lots and site ground don't count
    zmin=b0[2]+1; tall2=occ[...,int((zmin+8-b0[2])/ZB):].any(2)
    roof=ndimage.binary_fill_holes(ndimage.binary_closing(tall2,iterations=6))
    cand=has&clear&roof
    # building center = middle of the tall structure (seating bowl + roof), not the site/parking
    tall=(occ[...,int((ROOF_MIN+ (np.nanmin(floor) if has.any() else b0[2]) - b0[2])/ZB):]).any(2)
    ti=np.argwhere(tall); bc=(ti.min(0)+ti.max(0))/2
    best=None
    levels=np.round(floor[cand]/0.25)*0.25
    for L in np.unique(levels):
        mask=cand&(np.abs(floor-L)<0.3)
        mask=ndimage.binary_opening(mask,iterations=3)   # cut thin links (doorways, aisles) to concourses/outside
        lab,nl=ndimage.label(mask)
        if nl==0: continue
        # component closest to building center, must be at least 300 m2
        d,ind=ndimage.distance_transform_edt(lab==0,return_indices=True)
        cxi,cyi=np.clip(bc.astype(int),0,np.array(mask.shape)-1)
        if d[cxi,cyi]*CELL>30: continue
        j=lab[ind[0][cxi,cyi],ind[1][cxi,cyi]]
        sz=(lab==j).sum()*CELL*CELL
        if sz<300: continue
        key=(-L,)  # prefer the lowest qualifying level (the arena floor, not seating tiers)
        if best is None or L<best[1]: best=(sz,L,lab==j)
    if best is None: raise RuntimeError('no floor near building center')
    area,L,comp=best
    comp=ndimage.binary_fill_holes(ndimage.binary_closing(comp,iterations=2))
    pts=np.argwhere(comp)*CELL+b0[:2]+CELL/2
    c=pts.mean(0); w,V=np.linalg.eigh(np.cov((pts-c).T)); ax=V[:,np.argmax(w)]; yaw=float(np.arctan2(ax[1],ax[0]))
    # how far can the 20x11 box slide/rotate and stay inside comp?
    dist=ndimage.distance_transform_edt(comp)*CELL
    def fits(cx,cy,yw):
        ca,sa=np.cos(yw),np.sin(yw)
        u,v=np.meshgrid(np.linspace(-NEED_L/2,NEED_L/2,41),np.linspace(-NEED_W/2,NEED_W/2,23))
        X=cx+u*ca-v*sa; Y=cy+u*sa+v*ca
        ix=((X-b0[0])/CELL).astype(int); iy=((Y-b0[1])/CELL).astype(int)
        ok=(ix>=0)&(iy>=0)&(ix<nx)&(iy<ny)
        return ok.all() and comp[ix,iy].all()
    # refine center: maximize min clearance along the box
    ok0=fits(c[0],c[1],yaw)
    # exact floor height: drop rays onto the floor inside the field box, starting just under the clear band
    ca_,sa_=np.cos(yaw),np.sin(yaw); g=np.array([(u,v) for u in np.linspace(-8,8,9) for v in np.linspace(-4,4,5)])
    O=np.c_[c[0]+g[:,0]*ca_-g[:,1]*sa_, c[1]+g[:,0]*sa_+g[:,1]*ca_, np.full(len(g),L+CLEAR_HI-0.5)]
    loc,ri,_=m.ray.intersects_location(O,np.tile([0,0,-1.0],(len(O),1)),multiple_hits=False)
    if len(loc): L=float(np.median(loc[:,2])); floor_spread=float(np.ptp(loc[:,2]))
    else: floor_spread=None
    jit={}
    for k_,d in (('dx',np.array([np.cos(yaw),np.sin(yaw)])),('dy',np.array([-np.sin(yaw),np.cos(yaw)]))):
        lim=0
        for s in np.arange(0.25,10.01,0.25):
            if fits(*(c+s*d),yaw) and fits(*(c-s*d),yaw): lim=s
            else: break
        jit[k_]=min(lim,3.0)
    rot=0.0
    for a in np.arange(1,30.1,1):
        if a<=10 and fits(c[0],c[1],yaw+np.radians(a)) and fits(c[0],c[1],yaw-np.radians(a)): rot=a
        else: break
    out=dict(name=name,glb=f'models/stadiums/glb/{name}.glb',frame='stadium meters Z-up (glTF converted: x, -z, y)',
        floor_z=round(float(L),3),floor_spread_m=floor_spread,floor_hits=int(len(loc)),center_xy=[round(float(c[0]),3),round(float(c[1]),3)],yaw=round(yaw,4),
        open_floor_m2=round(float(comp.sum()*CELL*CELL),1),needed_box_m=[NEED_L,NEED_W],fits=bool(ok0),
        jitter_m={'along':jit['dx'],'across':jit['dy']},jitter_deg=rot,allow_flip_180=True,
        clear_box={'size_m':[NEED_L+2,NEED_W+2],'z_from_floor':[-0.05,REMOVE_HI],'note':'delete stadium faces inside this box (boards, hoops, ice edges) before placing field'},
        field_to_stadium='p_st = R(yaw) @ (p_field - [8.27,4.035,0]) + [cx,cy,floor_z]')
    os.makedirs(f'{ROOT}/anchors',exist_ok=True); json.dump(out,open(f'{ROOT}/anchors/{name}.json','w'),indent=1)
    # plot
    fig,a=plt.subplots(figsize=(9,8))
    top=np.where(occ.any(2),b0[2]+ZB*(nz-1-np.argmax(occ[:,:,::-1],2)),np.nan)
    a.imshow(top.T,origin='lower',extent=[b0[0],b0[0]+nx*CELL,b0[1],b0[1]+ny*CELL],cmap='gray')
    a.contour(np.arange(nx)*CELL+b0[0],np.arange(ny)*CELL+b0[1],comp.T.astype(float),[0.5],colors='lime')
    ca,sa=np.cos(yaw),np.sin(yaw)
    for (Lx,Wy,col) in ((NEED_L,NEED_W,'yellow'),(16.54,8.07,'red')):
        cor=np.array([[-1,-1],[1,-1],[1,1],[-1,1],[-1,-1]])*[Lx/2,Wy/2]
        P=c+cor@np.array([[ca,sa],[-sa,ca]]); a.plot(P[:,0],P[:,1],col,lw=1.5)
    r=max(NEED_L,40); a.set_xlim(c[0]-r*1.6,c[0]+r*1.6); a.set_ylim(c[1]-r*1.3,c[1]+r*1.3)
    a.set_title(f'{name}: floor z={L:.2f} m, open floor {out["open_floor_m2"]:.0f} m2, fits={ok0}, jitter {jit} rot±{rot}°',fontsize=8)
    fig.savefig(f'{ROOT}/anchors/{name}_anchor.png',dpi=90); plt.close(fig)
    print(name,json.dumps({k:out[k] for k in ('floor_z','floor_spread_m','floor_hits','center_xy','yaw','open_floor_m2','fits','jitter_m','jitter_deg')}))

if __name__=="__main__":
 for n in (sys.argv[1:] or [os.path.basename(p)[:-4] for p in sorted(glob.glob(f'{ROOT}/glb/*.glb'))]):
    try: anchor(n)
    except Exception as e: import traceback; traceback.print_exc(); print(n,'FAILED',e)
