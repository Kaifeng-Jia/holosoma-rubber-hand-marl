"""Offline metric geometry: exposed box-union faces and signed Procrustes.

Independent pilot utilities; no training imports, solver modification or scaling.
"""
from itertools import product, permutations
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation


def origin(element):
    out = np.eye(4)
    if element is not None:
        out[:3, 3] = np.fromstring(element.get('xyz', '0 0 0'), sep=' ')
        out[:3, :3] = Rotation.from_euler('xyz', np.fromstring(element.get('rpy', '0 0 0'), sep=' ')).as_matrix()
    return out


class BoxUnion:
    """Exact finite external boundary of the actual axis-aligned URDF boxes.

    Split every face at all box edges, discard cells whose outward probe lies
    inside another box. Distances are to those exposed rectangular patches.
    Sign is union occupancy, never the minimum of internal per-box surfaces.
    """
    def __init__(self, urdf):
        self.path = Path(urdf)
        root = ET.parse(urdf).getroot()
        if len(root.findall('link')) != 1:
            raise ValueError('Pilot requires single-link object; do not ignore articulated transforms')
        self.parts = []
        for col in root.findall('./link/collision'):
            box = col.find('./geometry/box')
            if box is None:
                raise ValueError('Pilot supports exact box union only, not mesh-to-box substitution')
            tf = origin(col.find('origin'))
            if not np.allclose(tf[:3, :3], np.eye(3), atol=1e-12):
                raise ValueError('Non-axis-aligned object boxes require oriented boundary implementation')
            size = np.fromstring(box.get('size'), sep=' ')
            self.parts.append(dict(id=col.get('name'), center=tf[:3, 3], size=size,
                                   lo=tf[:3, 3]-size/2, hi=tf[:3, 3]+size/2,
                                   kind='tabletop' if 'tabletop' in col.get('name', '') else 'leg'))
        self.parts.sort(key=lambda p:p['id']); self.faces=[]; self.patches=[]
        for part in self.parts:
            for axis, sign in product(range(3), [-1, 1]):
                other=[k for k in range(3) if k!=axis]
                normal=np.eye(3)[axis]*sign
                plane=part['lo' if sign<0 else 'hi'][axis]
                coords=[]
                for ax in other:
                    cuts=[part['lo'][ax],part['hi'][ax]]
                    for p in self.parts:
                        cuts.extend(v for v in (p['lo'][ax],p['hi'][ax]) if part['lo'][ax]<v<part['hi'][ax])
                    coords.append(sorted(set(cuts)))
                patches=[]
                for i,j in product(range(len(coords[0])-1),range(len(coords[1])-1)):
                    lo=part['lo'].copy();hi=part['hi'].copy();lo[axis]=hi[axis]=plane
                    lo[other]=[coords[0][i],coords[1][j]];hi[other]=[coords[0][i+1],coords[1][j+1]]
                    probe=(lo+hi)/2+normal*1e-8
                    if any(np.all(probe>p['lo']-1e-12) and np.all(probe<p['hi']+1e-12) for p in self.parts if p['id']!=part['id']):
                        continue
                    patches.append(dict(lo=lo,hi=hi))
                if patches:
                    corners=np.array([np.where(np.array(s),part['hi'],part['lo']) for s in product([0,1],repeat=3) if s[axis]==(sign>0)])
                    f=dict(id=f'{part["id"]}/axis{axis}/{sign:+d}',part_id=part['id'],axis=axis,sign=sign,normal=normal,
                           corners=corners,patches=patches)
                    idx=len(self.faces);self.faces.append(f)
                    self.patches.extend(dict(**p,face_index=idx) for p in patches)
        self.los=np.array([p['lo'] for p in self.patches]);self.his=np.array([p['hi'] for p in self.patches])
        self.face_indices=np.array([p['face_index'] for p in self.patches])

    def distances(self, points, face_ids=None, with_ties=False):
        p=np.asarray(points);shape=p.shape[:-1];flat=p.reshape(-1,3)
        ids=np.arange(len(self.patches)) if face_ids is None else np.flatnonzero(np.isin(self.face_indices,face_ids))
        if len(ids)==0: raise ValueError('No exposed patches for requested faces')
        distances=[];faces=[];closest=[];ties=[]
        for start in range(0,len(flat),2048):
            x=flat[start:start+2048]
            near=np.clip(x[:,None,:],self.los[ids],self.his[ids])
            squared=np.sum((x[:,None,:]-near)**2,axis=-1);idx=np.argmin(squared,axis=1)
            distances.extend(np.sqrt(squared[np.arange(len(x)),idx]));faces.extend(self.face_indices[ids[idx]])
            closest.extend(near[np.arange(len(x)),idx])
            if with_ties:
                if len(self.faces)>63:raise ValueError('Face bitset requires <=63 faces')
                tied=np.abs(np.sqrt(squared)-np.sqrt(squared[np.arange(len(x)),idx])[:,None])<=1e-7
                bits=np.left_shift(np.uint64(1),self.face_indices[ids].astype(np.uint64))
                ties.extend(np.bitwise_or.reduce(np.where(tied,bits[None],np.uint64(0)),axis=1))
        d=np.array(distances);inside=np.zeros(len(flat),dtype=bool)
        for part in self.parts:inside|=np.all((flat>=part['lo'])&(flat<=part['hi']),axis=-1)
        result=(d.reshape(shape),np.where(inside,-d,d).reshape(shape),np.array(faces).reshape(shape),np.array(closest).reshape(p.shape))
        return (*result,np.array(ties,dtype=np.uint64).reshape(shape)) if with_ties else result

    def manifest(self):
        return dict(path=str(self.path),parts=self.parts,faces=self.faces,patch_count=len(self.patches),
                    boundary='Exact external rectangular patches of collision-box union; negative means inside union')


def map_robot(qpos,object_qpos,Q,c,parity,joint_map,sign_mask):
    """Apply one segment-wide object-local O(3) map using proper root rotations."""
    qpos=np.asarray(qpos);obj=np.asarray(object_qpos);Q=np.asarray(Q);c=np.asarray(c)
    if not np.allclose(Q.T@Q,np.eye(3),atol=1e-8) or not np.isclose(np.linalg.det(Q),parity,atol=1e-8):raise ValueError('Invalid O(3) map')
    ro=Rotation.from_quat(obj[:,3:7][:,[1,2,3,0]]).as_matrix()
    rs=Rotation.from_quat(qpos[:,3:7][:,[1,2,3,0]]).as_matrix()
    local=np.einsum('tji,tj->ti',ro,qpos[:,:3]-obj[:,:3])
    positions=obj[:,:3]+np.einsum('tij,tj->ti',ro,local@Q.T+c)
    F=ro@Q@ro.transpose(0,2,1)
    J=np.diag([1.,-1.,1.]) if parity<0 else np.eye(3)
    roots=F@rs@J
    if not np.allclose(np.linalg.det(roots),1,atol=1e-8):raise ValueError('Improper robot root rotation')
    quat=Rotation.from_matrix(roots).as_quat()[:,[3,0,1,2]]
    for t in range(1,len(quat)):
        if np.dot(quat[t],quat[t-1])<0:quat[t]*=-1
    joints=qpos[:,7:][:,joint_map]*sign_mask if parity<0 else qpos[:,7:].copy()
    return np.c_[positions,quat,joints]

def surface_samples(mesh, count, seed):
    rng=np.random.default_rng(seed)
    triangles=mesh.triangles
    areas=mesh.area_faces
    ids=rng.choice(len(triangles),count,p=areas/areas.sum())
    uv=rng.random((count,2));root=np.sqrt(uv[:,0])
    weights=np.column_stack([1-root,root*(1-uv[:,1]),root*uv[:,1]])
    return np.einsum('ni,nij->nj',weights,triangles[ids])


def collision_meshes(urdf):
    """Preserve collision mesh/primitive origin transforms in each link frame."""
    result={};path=Path(urdf)
    for link in ET.parse(path).getroot().findall('link'):
        meshes=[]
        for col in link.findall('collision'):
            geom=col.find('geometry');m=geom.find('mesh');box=geom.find('box');sphere=geom.find('sphere');cyl=geom.find('cylinder')
            if m is not None:
                f=Path(m.get('filename'));f=f if f.is_absolute() else path.parent/f
                mesh=trimesh.load(f,force='mesh',process=False)
                mesh.apply_scale(np.fromstring(m.get('scale','1 1 1'),sep=' '))
            elif box is not None:mesh=trimesh.creation.box(np.fromstring(box.get('size'),sep=' '))
            elif sphere is not None:mesh=trimesh.creation.icosphere(subdivisions=2,radius=float(sphere.get('radius')))
            elif cyl is not None:mesh=trimesh.creation.cylinder(radius=float(cyl.get('radius')),height=float(cyl.get('length')),sections=32)
            else:raise ValueError('Unsupported collision geometry')
            mesh.apply_transform(origin(col.find('origin')));meshes.append(mesh)
        if meshes:result[link.get('name')]=trimesh.util.concatenate(meshes)
    return result


def procrustes(a,b,w,parity):
    a,b,w=np.asarray(a,float),np.asarray(b,float),np.asarray(w,float)
    if a.shape!=b.shape or a.ndim!=2 or a.shape[1]!=3 or w.shape!=(len(a),) or parity not in (-1,1):raise ValueError('Invalid correspondence')
    if not all(np.isfinite(x).all() for x in (a,b,w)) or np.any(w<0) or w.sum()<=0:raise ValueError('Invalid finite weights/points')
    w=w/w.sum();ac=w@a;bc=w@b;h=((a-ac)*w[:,None]).T@(b-bc)
    u,s,vt=np.linalg.svd(h)
    if s[0]<=1e-12 or s[1]/s[0]<1e-4:
        return dict(status='degenerate',singular_values=s)
    v=vt.T;d=np.diag([1.,1.,parity*np.linalg.det(v@u.T)])
    q=v@d@u.T;c=bc-q@ac;errors=(a@q.T+c)-b
    return dict(status='solved',Q=q,c=c,singular_values=s,determinant=float(np.linalg.det(q)),
                weighted_rmse_m=float(np.sqrt(np.sum(w*np.sum(errors**2,axis=1)))),
                inverse_error=float(np.max(np.abs((a@q.T+c-c)@q-a))))


def signed_permutations(parity):
    matrices=[]
    for perm in permutations(range(3)):
        for signs in product([-1,1],repeat=3):
            q=np.eye(3)[:,perm]*np.array(signs)[None,:]
            if round(np.linalg.det(q))==parity:matrices.append(q)
    return matrices


def continuous_windows(mask,min_frames):
    mask=np.array(mask,bool);mask[1:-1]|=mask[:-2]&mask[2:]
    edges=np.diff(np.r_[False,mask,False].astype(int));starts=np.flatnonzero(edges==1);ends=np.flatnonzero(edges==-1)
    return [(int(a),int(b)) for a,b in zip(starts,ends) if b-a>=min_frames]
