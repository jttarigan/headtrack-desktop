"""Test-process figure for the side paper: from a session recorded with
run_session.sh --photos, picks one webcam still per chosen protocol step (the
one where the ARKit reference shows the largest angle for that step), crops a
480x480 square around the head (MediaPipe landmarks; the front pose's position
when no face is found) into <session>/figure_crops/ (with angles.csv: source still and ARKit angles per crop),
and writes a labelled grid. headtrack/analysis/photo_figure.py builds the paper's figure from them.

    headtrack/.venv/bin/python headtrack/analysis/photo_crops.py runs/<session> grid.jpg

Run from the repository root.
"""
import sys,os,glob,cv2,numpy as np,datetime as dt,math,csv
sys.path.insert(0,'headtrack/analysis'); import htlib as H
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision
d,outf=sys.argv[1],sys.argv[2]
lm=vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='headtrack/models/face_landmarker.task'),num_faces=1))
P=lambda s:dt.datetime.strptime(s,'%Y-%m-%dT%H:%M:%S.%f').replace(tzinfo=dt.timezone.utc)
log=open(d+'/log.txt').read().splitlines()
steps=[(P(l[:23]),l.split('step: ')[1]) for l in log if '  step: ' in l]
end=[P(l[:23]) for l in log if 'protocol: done' in l][0]
ev=[r for r in csv.DictReader(open(d+'/events.csv')) if r['kind']=='step']
off=int(ev[0]['t_ns'])/1e9-steps[0][0].timestamp()
rows=list(csv.DictReader(open(d+'/ref.csv')))
t=np.array([int(r['recv_ns'])/1e9 for r in rows]); ok=np.array([r['tracked']=='1' for r in rows])
q=np.array([[float(r[k]) for k in ('qx','qy','qz','qw')] for r in rows])
mac=lambda w:w.timestamp()+off
s1=(t>mac(steps[0][0])+2)&(t<mac(steps[1][0])-1)&ok
ang=np.full((len(q),3),np.nan); ang[ok]=H.head_angles(q[ok],H.qaverage(q[s1]))
def ang_at(w):
    j=np.argmin(np.abs(t-mac(w))); return ang[j],abs(t[j]-mac(w))
W=sorted((dt.datetime.strptime(os.path.basename(p)[7:-4],'%Y%m%dT%H%M%S.%fZ').replace(tzinfo=dt.timezone.utc),p) for p in glob.glob(d+'/photos/webcam_*.jpg'))
# yaw + = right, pitch + = down (checked against steps 11-14, 17, 20)
want=[(1,'front',lambda a:-np.abs(a).sum()),
      (9,'left_screen_edge',lambda a:-a[0]),
      (14,'right_screen_edge',lambda a:a[0]),
      (4,'tilt',lambda a:abs(a[2])),
      (17,'up',lambda a:-a[1]),
      (20,'down',lambda a:a[1]),
      (22,'up_left',lambda a:-a[0]-a[1]),
      (25,'down_right',lambda a:a[0]+a[1])]
S=480; out=[]; centre=None; arows=[]
os.makedirs(d+'/figure_crops',exist_ok=True)
for f in glob.glob(d+'/figure_crops/*.jpg'): os.remove(f)
for n,name,score in want:
    t0,txt=steps[n-1]; t1=steps[n][0] if n<len(steps) else end
    best=None
    for tw,p in W:
        if not t0<=tw<=t1: continue
        a,gap=ang_at(tw)
        if gap>0.05 or np.isnan(a).any(): continue
        if best is None or score(a)>best[0]: best=(score(a),a,p)
    _,a,p=best
    img=cv2.imread(p); r=lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB,data=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)))
    if r.face_landmarks:
        pts=np.array([[u.x*img.shape[1],u.y*img.shape[0]] for u in r.face_landmarks[0]])
        c=(int(pts[:,0].mean()),int(pts[:,1].mean())-30); note=''
    else:
        c=centre; note=' (no face found; front crop position)'
    if centre is None: centre=c
    x0=min(max(c[0]-S//2,0),img.shape[1]-S); y0=min(max(c[1]-S//2,0),img.shape[0]-S)
    crop=img[y0:y0+S,x0:x0+S]
    cv2.imwrite('%s/figure_crops/%02d_%s.jpg'%(d,n,name),crop,[cv2.IMWRITE_JPEG_QUALITY,95])
    lab=np.full((62,S,3),255,np.uint8)
    cv2.putText(lab,txt.split('s ',1)[1][:42],(6,24),cv2.FONT_HERSHEY_SIMPLEX,0.58,(0,0,0),2)
    cv2.putText(lab,'ARKit yaw %+.0f  pitch %+.0f  roll %+.0f deg'%tuple(a),(6,52),cv2.FONT_HERSHEY_SIMPLEX,0.55,(60,60,60),1)
    arows.append('%02d_%s,%s,%.1f,%.1f,%.1f'%(n,name,os.path.basename(p),*a))
    out.append(np.vstack([lab,crop])); print('%2d %-18s %s yaw %+5.1f pitch %+5.1f roll %+5.1f%s'%(n,name,os.path.basename(p),*a,note))
open(d+'/figure_crops/angles.csv','w').write('crop,still,yaw_deg,pitch_deg,roll_deg\n'+'\n'.join(arows)+'\n')
cv2.imwrite(outf,np.vstack([np.hstack(out[:4]),np.hstack(out[4:])]),[cv2.IMWRITE_JPEG_QUALITY,88])
