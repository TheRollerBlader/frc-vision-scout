import trimesh, numpy as np, glob, os, json, sys
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
out={}
for d in sorted(glob.glob('/workspace/frc/models/stadiums/raw/*')):
    n=os.path.basename(d)
    try:
        s=trimesh.load(f'{d}/model.dae', force='scene')
        m=s.dump(concatenate=True) if hasattr(s,'dump') else s
    except Exception as e:
        print(n,'FAIL',e); continue
    m.apply_scale(0.0254)  # inches -> m (trimesh doesn't apply collada unit)
    b=m.bounds; ext=b[1]-b[0]
    out[n]=dict(tris=int(len(m.faces)),min=b[0].round(2).tolist(),ext=ext.round(2).tolist())
    print(n, len(m.faces), 'ext m', ext.round(1), 'min', b[0].round(1))
    # z histogram of upward-facing area to find the floor
    up=m.face_normals[:,2]>0.95; z=m.triangles_center[up,2]; a=m.area_faces[up]
    h,e=np.histogram(z,bins=200,weights=a); top=np.argsort(h)[::-1][:4]
    print('   big flat levels (z m, area m2):',[(round(float((e[i]+e[i+1])/2),2),round(float(h[i]))) for i in top])
    fig,ax=plt.subplots(1,2,figsize=(14,6))
    T=m.triangles; sel=np.random.default_rng(0).choice(len(T),min(len(T),150000),replace=False)
    c=m.triangles_center[sel]
    ax[0].scatter(c[:,0],c[:,1],s=0.2,c=c[:,2],cmap='viridis'); ax[0].set_aspect('equal'); ax[0].set_title(f'{n} top (color=z)')
    ax[1].scatter(c[:,0],c[:,2],s=0.2,c='k'); ax[1].set_aspect('equal'); ax[1].set_title('side x-z')
    fig.savefig(f'/workspace/frc/models/stadiums/preview_{n}.png',dpi=80); plt.close(fig)
json.dump(out,open('/workspace/frc/models/stadiums/inspect.json','w'),indent=1)
