"""Offline actual-FK surface checks and analytic separating-axis proxies.

No mj_step, control, force inference or policy inference. Robot-object uses
link OBB vs exact box SAT; cross-robot uses link OBB SAT. Separate selected
frame refinement may use hull_box_sat. OBB overlap is not a mesh contact label.
"""
from itertools import product
import xml.etree.ElementTree as ET
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
from reference_geometry_math import continuous_windows
from reference_geometry_pilot import quaternion_wxyz_to_matrix


def normalized_axes(axes):
    axes=np.asarray(axes);length=np.linalg.norm(axes,axis=1);axes=axes[length>1e-10]/length[length>1e-10,None]
    return axes


def hull_box_sat(vertices,face_normals,edges,center,halfsize,box_rotation):
    """Signed SAT margin; positive is a separation lower bound, not Euclidean distance."""
    cross=np.cross(edges[:,None,:],box_rotation.T[None,:,:]).reshape(-1,3)
    axes=normalized_axes(np.vstack([face_normals,box_rotation.T,cross]))
    projected=vertices@axes.T;bc=center@axes.T;radius=halfsize@abs(box_rotation.T@axes.T)
    return float(np.maximum(projected.min(0)-(bc+radius),(bc-radius)-projected.max(0)).max())


def obb_sat(ac,ar,ah,bc,br,bh):
    axes=normalized_axes(np.vstack([ar.T,br.T,np.cross(ar.T[:,None],br.T[None]).reshape(-1,3)]))
    delta=abs((bc-ac)@axes.T);ra=ah@abs(ar.T@axes.T);rb=bh@abs(br.T@axes.T)
    return float(np.max(delta-ra-rb))

V0=dict(version='round2_v0_before_candidate_checks',ground_z_m=0.,
    numeric_tolerance=1e-8,normal_error_deg=1e-4,fk_origin_error_m=.0005,
    additional_hand_penetration_m=.001,additional_nonhand_proxy_penetration_m=.01,
    pair_proxy_penetration_m=.01,additional_foot_ground_penetration_m=.01,
    additional_region_min_distance_m=.01,region_near_fraction_drop=.05,
    relative_gravity_deviation_deg=5.,support_speed_m_s=.15,support_height_m=.04,
    support_min_duration_s=.1,additional_support_speed_m_s=.01,additional_support_drift_m=.01,
    max_added_root_step_m=.02,max_added_root_rotation_step_rad=.02,
    additional_root_speed_m_s=1.,additional_root_acceleration_m_s2=5.,
    fit_tie_tolerance_m=1e-10,geometry_distance_cap_m=.1,
    provenance='1cm penetration/sliding-increment scales reuse retargeting evaluator tolerances; other values are explicit v0 engineering source-relative screens, not validated paper thresholds. Source support is robot bottom <=4cm, XY speed <=0.15m/s, >=0.1s; no human contact labels available.',
    acceptance_scope='Source-relative sampled/convex-proxy geometry only; not dynamics or exact mesh collision certification',
    selection_data='Independent of 256 fitting samples but 1024 validation samples ARE used for screening; not heldout evaluation')


def rect_distance(points,faces):
    """Exact point-to-union-of-finite-rectangles, no infinite planes."""
    patches=[p for f in faces for p in f['patches']]
    # Current source/target/clipped faces are filled rectangles. Merge only
    # when area verifies no hole; otherwise retain all actual external cells.
    merged=[]
    for f in faces:
        lo=np.min([p['lo'] for p in f['patches']],0);hi=np.max([p['hi'] for p in f['patches']],0)
        tangential=[i for i in range(3) if i!=f['axis']]
        area=sum(np.prod((p['hi']-p['lo'])[tangential]) for p in f['patches'])
        if np.isclose(area,np.prod((hi-lo)[tangential]),atol=1e-12):merged.append(dict(lo=lo,hi=hi))
        else:merged.extend(f['patches'])
    flat=np.asarray(points).reshape(-1,3);out=np.empty(len(flat))
    lo=np.array([p['lo'] for p in merged]);hi=np.array([p['hi'] for p in merged])
    if not len(lo):raise ValueError('No actual finite surface')
    for start in range(0,len(flat),2048):
        x=flat[start:start+2048,None];delta=x-np.clip(x,lo,hi)
        out[start:start+len(x)]=np.sqrt(np.min(np.sum(delta*delta,axis=-1),axis=1))
    return out.reshape(np.asarray(points).shape[:-1])


def point_surface(points,geometry):
    d=rect_distance(points,geometry.faces);inside=np.zeros(d.shape,bool)
    for p in geometry.parts:inside|=np.all((points>=p['lo'])&(points<=p['hi']),axis=-1)
    return d,np.where(inside,-d,d)


class CollisionProxy:
    def __init__(self,fk,geometry):
        self.names=list(fk.meshes);self.n=len(self.names);self.geometry=geometry
        xml=ET.Element('mujoco',model='round2_offline_convex_proxies')
        ET.SubElement(xml,'compiler',angle='radian')
        asset=ET.SubElement(xml,'asset');world=ET.SubElement(xml,'worldbody');assets={}
        self.local_bounds={};self.mesh_info=[]
        self.hulls={}
        for i,name in enumerate(self.names):
            mesh=fk.meshes[name];hull=mesh.convex_hull
            self.local_bounds[name]=mesh.bounds
            adjacent=hull.face_adjacency
            sharp=np.sum(hull.face_normals[adjacent[:,0]]*hull.face_normals[adjacent[:,1]],axis=1)<1-1e-10
            edges=np.diff(hull.vertices[hull.face_adjacency_edges[sharp]],axis=1)[:,0]
            normals=np.unique(np.round(hull.face_normals,12),axis=0)
            self.hulls[name]=(hull.vertices,normals,normalized_axes(edges))
            assets[f'm{i}.stl']=hull.export(file_type='stl');ET.SubElement(asset,'mesh',name=f'm{i}',file=f'm{i}.stl')
            self.mesh_info.append(dict(link=name,original_vertices=len(mesh.vertices),convex_hull_vertices=len(hull.vertices),
                mesh_watertight=bool(mesh.is_watertight),representation='Per-link convex hull of declared collision geometry; conservative proxy'))
        for a in range(2):
            for i,name in enumerate(self.names):
                body=ET.SubElement(world,'body',name=f'a{a}_{i}',mocap='true')
                ET.SubElement(body,'geom',name=f'g{a}_{i}',type='mesh',mesh=f'm{i}',contype='0',conaffinity='0')
        for i,p in enumerate(geometry.parts):
            body=ET.SubElement(world,'body',name=f'o{i}',mocap='true')
            ET.SubElement(body,'geom',name=f'og{i}',type='box',size=' '.join(map(str,p['size']/2)),contype='0',conaffinity='0')
        self.model=mujoco.MjModel.from_xml_string(ET.tostring(xml,encoding='unicode'),assets)
        self.data=mujoco.MjData(self.model)
        self.body_ids=np.array([mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_BODY,f'a{a}_{i}') for a in range(2) for i in range(self.n)]+[mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_BODY,f'o{i}') for i in range(len(geometry.parts))])
        self.mocap=self.model.body_mocapid[self.body_ids]
        self.geom_ids=np.array([mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_GEOM,f'g{a}_{i}') for a in range(2) for i in range(self.n)]+[mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_GEOM,f'og{i}') for i in range(len(geometry.parts))])
        self.hand_indices=[i for i,n in enumerate(self.names) if 'rubber_hand' in n]
    def _minimum(self,first,second,bounds):
        ii,jj=np.meshgrid(first,second,indexing='ij');ii=ii.ravel();jj=jj.ravel()
        delta=np.maximum(np.maximum(bounds[ii,0]-bounds[jj,1],bounds[jj,0]-bounds[ii,1]),0)
        lower=np.linalg.norm(delta,axis=1);best=.1;witness=None
        for k in np.argsort(lower,kind='stable'):
            if lower[k]>max(best,0)+1e-10:break
            i,j=int(ii[k]),int(jj[k]);a=self.current_tf[i];b=self.current_tf[j]
            name=self.names[i%self.n]
            if j>=2*self.n:
                ba=self.local_bounds[name];part=self.geometry.parts[j-2*self.n]
                d=obb_sat(a[:3,:3]@ba.mean(0)+a[:3,3],a[:3,:3],np.diff(ba,axis=0)[0]/2,
                          b[:3,3],b[:3,:3],part['size']/2)
            else:
                ba=self.local_bounds[name];bb=self.local_bounds[self.names[j%self.n]]
                d=obb_sat(a[:3,:3]@ba.mean(0)+a[:3,3],a[:3,:3],np.diff(ba,axis=0)[0]/2,
                          b[:3,:3]@bb.mean(0)+b[:3,3],b[:3,:3],np.diff(bb,axis=0)[0]/2)
            if d<best:best=float(d);witness=[int(ii[k]),int(jj[k])]
        return best,witness
    def evaluate(self,poses,obj):
        T=len(obj);A=len(poses);R=quaternion_wxyz_to_matrix(obj[:,3:]);corners=np.array(list(product([0,1],repeat=3)))
        bounds=[];tf=[]
        for a in range(2):
            for name in self.names:
                v=poses[a][name].copy() if a<A else np.broadcast_to(np.eye(4),(T,4,4)).copy()
                if a>=A:v[:,:3,3]=1000
                b=self.local_bounds[name];local=np.where(corners,b[1],b[0]);p=np.einsum('tij,pj->tpi',v[:,:3,:3],local)+v[:,:3,3][:,None]
                bounds.append(np.stack([p.min(1),p.max(1)],axis=1));tf.append(v)
        for part in self.geometry.parts:
            v=np.broadcast_to(np.eye(4),(T,4,4)).copy();v[:,:3,:3]=R;v[:,:3,3]=obj[:,:3]+np.einsum('tij,j->ti',R,part['center'])
            local=np.where(corners,part['size']/2,-part['size']/2);p=np.einsum('tij,pj->tpi',R,local)+v[:,:3,3][:,None]
            bounds.append(np.stack([p.min(1),p.max(1)],axis=1));tf.append(v)
        bounds=np.stack(bounds,1);tf=np.stack(tf,1);quat=Rotation.from_matrix(tf[:,:,:3,:3].reshape(-1,3,3)).as_quat().reshape(T,-1,4)[:,:,[3,0,1,2]]
        nonhand=np.full((T,A),.1);hand=np.full((T,A),np.nan);pair=np.full(T,.1);witnesses=[]
        oi=np.arange(2*self.n,2*self.n+len(self.geometry.parts))
        for t in range(T):
            self.current_tf=tf[t]
            for a in range(A):
                ids=np.arange(a*self.n,(a+1)*self.n);h=ids[self.hand_indices];nh=np.delete(ids,self.hand_indices)
                nonhand[t,a],who=self._minimum(nh,oi,bounds[t])
                if nonhand[t,a]<0:witnesses.append(dict(frame=t,agent=a,kind='nonhand_object',geom_indices=who,distance_m=float(nonhand[t,a])))
            if A==2:
                pair[t],who=self._minimum(np.arange(self.n),np.arange(self.n,2*self.n),bounds[t])
                if pair[t]<0:witnesses.append(dict(frame=t,kind='robot_robot',geom_indices=who,distance_m=float(pair[t])))
        return dict(nonhand_object_clearance_m=nonhand,hand_object_convex_clearance_m=hand,robot_robot_clearance_m=pair),witnesses


def surface_curves(q,obj,fk,geometry,spec,role=None,candidate=None):
    poses=fk.poses(q,list(fk.meshes));ro=quaternion_wxyz_to_matrix(obj[:,3:]);out={};windows=spec['weighted_source']['windows']
    for h,name in enumerate(fk.hands):
        actual_name=fk.hands[1-h] if candidate is not None and candidate['parity']<0 else name
        pose=poses[actual_name];world=np.einsum('tij,pj->tpi',pose[:,:3,:3],fk.dense[actual_name])+pose[:,:3,3][:,None]
        local=np.einsum('tji,tpj->tpi',ro,world-obj[:,:3][:,None]);d,s=point_surface(local,geometry)
        prefix=name;out[prefix+'_min_distance_m']=d.min(1);out[prefix+'_mean_distance_m']=d.mean(1)
        out[prefix+'_p95_distance_m']=np.quantile(d,.95,axis=1);out[prefix+'_near_fraction']=(d<=.03).mean(1)
        out[prefix+'_penetration_m']=np.maximum(0,-s.min(1))
        groups={}
        for t in range(len(q)):
            active=tuple(sorted({w['face_id'] for w in windows if w['hand']==name and w['start']<=t<w['end_exclusive']}))
            if active:groups.setdefault(active,[]).append(t)
        region_min=np.full(len(q),np.nan);region_frac=region_min.copy()
        for active,frames in groups.items():
            if candidate is None:faces=[f for f in geometry.faces if f['id'] in active]
            else:
                ids={candidate['face_map'][fid]['target_face_id'] for fid in active}
                faces=[f for f in spec['regions'][role]['faces'] if f['id'] in ids]
            distance=rect_distance(local[frames],faces)
            region_min[frames]=distance.min(1);region_frac[frames]=(distance<=.03).mean(1)
        out[prefix+'_region_min_distance_m']=region_min;out[prefix+'_region_near_fraction']=region_frac
    for h,side in enumerate(['left','right']):
        actual_side=['right','left'][h] if candidate is not None and candidate['parity']<0 else side
        name=actual_side+'_ankle_roll_link';pose=poses[name];vertices=fk.meshes[name].vertices
        p=np.einsum('tij,pj->tpi',pose[:,:3,:3],vertices)+pose[:,:3,3][:,None]
        out[side+'_foot_bottom_z_m']=p[:,:,2].min(1)
        out[side+'_foot_position_m']=pose[:,:3,3]
    return out,poses


def support_metrics(curves,source,fps):
    result={}
    for side in ['left','right']:
        position=curves[side+'_foot_position_m'];p0=source[side+'_foot_position_m']
        speed=np.linalg.norm(np.gradient(position,1/fps,axis=0,edge_order=2)[:,:2],axis=1)
        s0=np.linalg.norm(np.gradient(p0,1/fps,axis=0,edge_order=2)[:,:2],axis=1)
        eligible=(source[side+'_foot_bottom_z_m']<=V0['support_height_m'])&(s0<=V0['support_speed_m_s'])
        windows=continuous_windows(eligible,int(np.ceil(fps*V0['support_min_duration_s'])))
        mask=np.zeros(len(position),bool);drift=np.full(len(position),np.nan);d0=drift.copy()
        for a,b in windows:
            mask[a:b]=True;drift[a:b]=np.linalg.norm(position[a:b,:2]-position[a,:2],axis=1);d0[a:b]=np.linalg.norm(p0[a:b,:2]-p0[a,:2],axis=1)
        result[side+'_source_support_mask']=mask;result[side+'_support_speed_m_s']=np.where(mask,speed,np.nan)
        result[side+'_support_drift_m']=drift;result[side+'_support_added_speed_m_s']=np.where(mask,speed-s0,np.nan)
        result[side+'_support_added_drift_m']=drift-d0
    return result


def motion_metrics(q,source,obj,fps,Q):
    r=quaternion_wxyz_to_matrix(q[:,3:7]);rs=quaternion_wxyz_to_matrix(source[:,3:7]);ro=quaternion_wxyz_to_matrix(obj[:,3:7]);F=ro@Q@ro.transpose(0,2,1)
    gravity=np.degrees(np.arccos(np.clip(F[:,2,2],-1,1)))
    speed=np.gradient(q[:,:3],1/fps,axis=0,edge_order=2);s0=np.gradient(source[:,:3],1/fps,axis=0,edge_order=2)
    rotation_step=np.r_[0,Rotation.from_matrix(r[1:]@r[:-1].transpose(0,2,1)).magnitude()]
    source_step=np.r_[0,Rotation.from_matrix(rs[1:]@rs[:-1].transpose(0,2,1)).magnitude()]
    return dict(relative_gravity_deviation_deg=gravity,root_tilt_deg=np.degrees(np.arccos(np.clip(r[:,2,2],-1,1))),
        root_position_step_m=np.r_[0,np.linalg.norm(np.diff(q[:,:3],axis=0),axis=1)],
        root_rotation_step_rad=rotation_step,root_rotation_step_added_rad=rotation_step-source_step,
        root_position_step_added_m=np.r_[0,np.linalg.norm(np.diff(q[:,:3],axis=0),axis=1)-np.linalg.norm(np.diff(source[:,:3],axis=0),axis=1)],
        root_speed_m_s=np.linalg.norm(speed,axis=1),root_speed_added_m_s=np.linalg.norm(speed,axis=1)-np.linalg.norm(s0,axis=1),
        root_acceleration_m_s2=np.linalg.norm(np.gradient(speed,1/fps,axis=0,edge_order=2),axis=1),
        root_acceleration_added_m_s2=np.linalg.norm(np.gradient(speed,1/fps,axis=0,edge_order=2),axis=1)-np.linalg.norm(np.gradient(s0,1/fps,axis=0,edge_order=2),axis=1),
        joint_step_max_rad=np.r_[0,np.max(abs(np.diff(q[:,7:],axis=0)),axis=1)],
        quaternion_norm_error=np.abs(np.linalg.norm(q[:,3:7],axis=1)-1),root_rotation_det_error=np.abs(np.linalg.det(r)-1))


def finite_max(x,default=None):
    a=np.asarray(x);a=a[np.isfinite(a)];return float(a.max()) if len(a) else default


def summaries(curves):
    out={}
    for key,value in curves.items():
        if np.asarray(value).ndim!=1:continue
        a=np.asarray(value);a=a[np.isfinite(a)]
        out[key]=dict(count=len(a),min=float(a.min()) if len(a) else None,max=float(a.max()) if len(a) else None,
            mean=float(a.mean()) if len(a) else None,rms=float(np.sqrt(np.mean(a*a))) if len(a) else None)
    return out
