"""SILENT MAZE  |  pip install pygame numpy  |  WASD+mouse move · R restart · ESC quit"""
import pygame, math, random, sys
from collections import deque
import numpy as np

W,H=1280,720; FPS=60; RAYS=320; SCALE=W//RAYS
FOV=math.pi/3; HFOV=FOV/2; MAXD=30.0
PDIST=(W/2)/math.tan(HFOV); HORIZON=H//2+int(H*0.08)
SPD,SPR=0.07,0.13; ROT=0.040
SAN_MAX,SAN_R=100.0,0.04; MON_CATCH=0.72
MW=MH=75
CWL,CWD=(155,55,35),(55,15,8)
SR=22050

DIFF={
    "EASY":  dict(spd=0.013,sight=26,n=1,bfs=40,san_d=0.05,fog=0.25,flk=0.4,passive=0.0, anger=1.2,approach=False,dark_vig=False,
                  desc="1 slow monster  ·  light fog  ·  forgiving"),
    "HARD":  dict(spd=0.040,sight=62,n=3,bfs=15,san_d=0.30,fog=0.50,flk=1.5,passive=0.008,anger=1.9,approach=True, dark_vig=False,
                  desc="3 fast monsters  ·  warning sounds  ·  heavy fog"),
    "NIGHTMARE":dict(spd=0.058,sight=99,n=4,bfs=7, san_d=0.52,fog=0.63,flk=2.4,passive=0.022,anger=2.3,approach=True, dark_vig=True,
                  desc="4 near-sprint monsters  ·  darkness closes in  ·  near-blind"),
}

RC=[math.cos(-HFOV+r*(FOV/RAYS)) for r in range(RAYS)]
RS=[math.sin(-HFOV+r*(FOV/RAYS)) for r in range(RAYS)]
WLUT=[(int(CWL[0]*v+CWD[0]*(1-v)),int(CWL[1]*v+CWD[1]*(1-v)),int(CWL[2]*v+CWD[2]*(1-v)))
      for v in (i/255 for i in range(256))]

# ── AUDIO ─────────────────────────────────────────────────────────────────────
def _snd(a): s=(np.clip(a,-1,1)*32767).astype(np.int16); return pygame.sndarray.make_sound(np.column_stack([s,s]))
def _nz(d,v=0.3): return (np.random.rand(int(SR*d))*2-1)*v
def _t(d): return np.linspace(0,d,int(SR*d),False)

def build_sounds():
    S={}
    def thump(d,f,dc=22):
        t=_t(d); return np.sin(2*np.pi*f*t)*np.exp(-t*dc)*.95+np.sin(2*np.pi*f*.5*t)*np.exp(-t*dc*1.4)*.35
    S['heartbeat']=_snd(np.concatenate([thump(.20,58,20),np.zeros(int(SR*.08)),thump(.14,46,28),np.zeros(int(SR*.60))]))
    t=_t(1.1); w=np.sin(2*np.pi*3.5*t)*4
    S['growl']=_snd(np.clip((np.sin(2*np.pi*(28+w)*t)*.55+np.sin(2*np.pi*(56+w)*t)*.28+_nz(1.1,.18))*np.exp(-t*1.5)*np.linspace(0,1,int(SR*1.1))**.3,-1,1))
    t=_t(.8); sc2=_nz(.8,.60)*np.exp(-t*2.2)+np.sin(2*np.pi*38*t)*np.exp(-t*3)*.35+np.sin(2*np.pi*19*t)*np.exp(-t*1.8)*.20
    e=np.ones(len(sc2)); e[:int(SR*.06)]=np.linspace(0,1,int(SR*.06)); S['stalk']=_snd(np.clip(sc2*e,-1,1))
    t=_t(.5); ev=np.linspace(0,1,int(SR*.5))**1.5
    S['hiss']=_snd(np.clip(_nz(.5,.70)*ev+np.sin(2*np.pi*220*t)*.15*ev,-1,1))
    t=_t(.9)
    S['sting']=_snd(np.clip(np.sin(2*np.pi*55*t)*np.exp(-t*12)*.9+(np.sin(2*np.pi*1100*t)*.5+np.sin(2*np.pi*1480*t)*.35)*np.exp(-t*5)+_nz(.9,.40)*np.exp(-t*8),-1,1))
    w2=np.zeros(int(SR*2.5))
    for _ in range(4):
        off=random.randint(0,int(SR*.8)); n2=int(SR*random.uniform(1.0,1.8))
        seg=_nz(n2/SR,.07)*np.sin(np.pi*np.linspace(0,1,n2))**2; w2[off:off+min(n2,len(w2)-off)]+=seg[:len(w2)-off]
    t=_t(2.5); w2+=np.sin(2*np.pi*180*t)*.015*np.sin(np.pi*t/2.5); S['whisper']=_snd(np.clip(w2,-1,1))
    t=_t(.08); S['step']=_snd(np.clip((_nz(.08,.55)+np.sin(2*np.pi*95*t)*.3)*np.exp(-t*100),-1,1))
    t=_t(1.4); ev2=np.sin(np.pi*t/1.4)**2.5
    S['breathe']=_snd(np.clip(_nz(1.4,.22)*ev2+np.sin(2*np.pi*80*t)*.04*ev2,-1,1))
    t=_t(3.0)
    S['rumble']=_snd(np.clip((np.sin(2*np.pi*22*t)*.45+np.sin(2*np.pi*33*t)*.25+np.sin(2*np.pi*18*t)*.20+_nz(3.0,.12))*np.sin(np.pi*t/3.0)**.5,-1,1))
    return S

# ── MAZE ──────────────────────────────────────────────────────────────────────
def generate_maze(cols,rows):
    g=[[1]*cols for _ in range(rows)]
    lw,lh=(cols-1)//2,(rows-1)//2

    def carve(mx,my):
        for dy in range(2):
            for dx in range(2):
                if 0<mx+dx<cols-1 and 0<my+dy<rows-1: g[my+dy][mx+dx]=0

    def connect(lx1,ly1,lx2,ly2):
        x1,y1=lx1*2+1,ly1*2+1
        if   lx2==lx1+1: [g[y1+dy].__setitem__(x1+2,0) for dy in range(2)]
        elif lx2==lx1-1: [g[y1+dy].__setitem__(x1-1,0) for dy in range(2)]
        elif ly2==ly1+1: [g[y1+2].__setitem__(x1+dx,0) for dx in range(2)]
        elif ly2==ly1-1: [g[y1-1].__setitem__(x1+dx,0) for dx in range(2)]

    vis=[[False]*lw for _ in range(lh)]
    stk=[(0,0)]; vis[0][0]=True; carve(1,1)
    while stk:
        lx,ly=stk[-1]; dirs=[(1,0),(-1,0),(0,1),(0,-1)]; random.shuffle(dirs); moved=False
        for dx,dy in dirs:
            nx,ny=lx+dx,ly+dy
            if 0<=nx<lw and 0<=ny<lh and not vis[ny][nx]:
                vis[ny][nx]=True; carve(nx*2+1,ny*2+1); connect(lx,ly,nx,ny); stk.append((nx,ny)); moved=True; break
        if not moved: stk.pop()

    # Punch open rooms for variety
    for _ in range(18):
        rw=random.randint(3,8); rh=random.randint(3,8)
        rx=random.randint(4,cols-rw-4); ry=random.randint(4,rows-rh-4)
        for y in range(ry,ry+rh):
            for x in range(rx,rx+rw): g[y][x]=0

    # Extra long straight corridors to break monotony
    for _ in range(12):
        if random.random()<.5:
            y=random.randint(3,rows-4); x0=random.randint(2,cols//3); x1=random.randint(2*cols//3,cols-3)
            for x in range(x0,x1): g[y][x]=g[min(rows-1,y+1)][x]=0
        else:
            x=random.randint(3,cols-4); y0=random.randint(2,rows//3); y1=random.randint(2*rows//3,rows-3)
            for y in range(y0,y1): g[y][x]=g[y][min(cols-1,x+1)]=0

    # Scatter small pillars (obstacles)
    for ry in range(6,rows-6):
        for rx in range(6,cols-6):
            if g[ry][rx]==0 and random.random()<0.035:
                w2,h2=random.choice([(1,1),(1,2),(2,1)])
                if all(g[ry+dy][rx+dx]==0 for dy in range(h2) for dx in range(w2) if 0<rx+dx<cols-1 and 0<ry+dy<rows-1):
                    for dy in range(h2):
                        for dx in range(w2): g[ry+dy][rx+dx]=1

    for x in range(cols): g[0][x]=g[rows-1][x]=1
    for y in range(rows): g[y][0]=g[y][cols-1]=1
    for y in range(1,7):
        for x in range(1,7): g[y][x]=0
    return g

# ── EMBERS ────────────────────────────────────────────────────────────────────
_NE=200
_EX=np.random.randint(0,W,_NE).astype(np.float32); _EY=np.random.randint(0,H,_NE).astype(np.float32)
_EVX=np.random.uniform(-.6,.6,_NE).astype(np.float32); _EVY=np.random.uniform(-2.2,-.5,_NE).astype(np.float32)
_ESZ=np.random.randint(1,4,_NE); _EA=np.random.randint(80,220,_NE)
_EC=np.array([random.choice([(255,80,10),(255,140,20),(255,200,60),(180,40,5)]) for _ in range(_NE)])
_ESURF=None
def _init_embers(): global _ESURF; _ESURF=pygame.Surface((W,H),pygame.SRCALPHA)
def draw_embers(scr):
    global _EX,_EY; _EX+=_EVX; _EY+=_EVY
    m=_EY<-4; _EY[m]=float(H)+4; _EX[m]=np.random.randint(0,W,int(m.sum()))
    m2=(_EX<0)|(_EX>W); _EX[m2]=np.mod(_EX[m2],W); _ESURF.fill((0,0,0,0))
    for i in range(_NE): _ESURF.fill((*_EC[i].tolist(),int(_EA[i])),(int(_EX[i]),int(_EY[i]),int(_ESZ[i]),int(_ESZ[i])))
    scr.blit(_ESURF,(0,0))

# ── BAKED ASSETS ──────────────────────────────────────────────────────────────
def bake_border(col):
    s=pygame.Surface((W,H),pygame.SRCALPHA)
    for i in range(0,55,5): pygame.draw.rect(s,(*col,max(0,200-i*4)),(i,i,W-2*i,H-2*i),5)
    return s

def bake_door():
    DW,DH=128,200; s=pygame.Surface((DW,DH),pygame.SRCALPHA); s.fill((0,0,0,0)); cx=DW//2
    for r in range(70,0,-5): pygame.draw.ellipse(s,(0,255,80,int((1-r/70)**1.5*100)),(cx-r,DH//2-int(r*1.4),r*2,int(r*2.8)))
    FM=(28,32,28); FH=(55,65,50); pad=8
    pygame.draw.rect(s,FM,(0,20,DW,DH-20)); pygame.draw.rect(s,FH,(0,20,4,DH-20)); pygame.draw.rect(s,FH,(0,20,DW,3))
    pygame.draw.rect(s,(35,42,32),(pad,28,DW-pad*2,DH-32))
    for rx2,ry2 in((5,26),(DW-5,26),(5,DH-8),(DW-5,DH-8)): pygame.draw.circle(s,FH,(rx2,ry2),4)
    GC=(0,255,80,160); my2=DH//2
    for line in[((pad,28),(pad,DH-8)),((DW-pad,28),(DW-pad,DH-8)),((cx,28),(cx,DH-8)),((pad,my2),(DW-pad,my2))]:
        pygame.draw.line(s,GC,*line,1)
    pygame.draw.rect(s,(0,60,20),(cx-22,22,44,14),border_radius=2); pygame.draw.rect(s,(0,200,60),(cx-22,22,44,14),2,border_radius=2)
    lbl=pygame.font.SysFont("couriernew",9,bold=True).render("EXIT",True,(0,255,80))
    s.blit(lbl,(cx-lbl.get_width()//2,25)); return s

def bake_js_face():
    s=pygame.Surface((W,H)); s.fill((0,0,0)); cx,cy=W//2,H//2
    hrx,hry=int(W*.30),int(H*.42)
    pygame.draw.ellipse(s,(18,14,12),(cx-hrx,cy-hry,hrx*2,hry*2))
    for side in(-1,1):
        ex=cx+side*int(W*.13); ey=cy-int(H*.06); erx,ery=int(W*.095),int(H*.075)
        pygame.draw.ellipse(s,(120,0,0),(ex-erx-12,ey-ery-10,(erx+12)*2,(ery+10)*2))
        pygame.draw.ellipse(s,(220,10,5),(ex-erx,ey-ery,erx*2,ery*2))
        pygame.draw.ellipse(s,(255,180,60),(ex-erx//2,ey-ery//2,erx,ery))
    my2=cy+int(H*.14); mw=int(W*.28); mh=int(H*.20); tc=(195,188,170)
    pygame.draw.ellipse(s,(4,2,2),(cx-mw,my2,mw*2,mh))
    for i in range(8):
        tx=cx-mw+int(mw*2*i/8)+mw//16; th=random.randint(int(mh*.35),int(mh*.65)); tw=max(4,mw//4-6)
        pygame.draw.polygon(s,tc,[(tx-tw//2,my2+4),(tx+tw//2,my2+4),(tx+tw//4,my2+th),(tx-tw//4,my2+th)])
    vig=pygame.Surface((W,H),pygame.SRCALPHA)
    for r in range(max(W,H),0,-8): pygame.draw.circle(vig,(0,0,0,int((1-r/max(W,H))**1.8*200)),(cx,cy),r)
    s.blit(vig,(0,0)); return s

def bake_monster():
    TW,TH=256,512; s=pygame.Surface((TW,TH),pygame.SRCALPHA); s.fill((0,0,0,0))
    tc=TW//2; u=TH/10; bc=(28,28,24); dc=(18,18,16); cc=(12,12,10)
    lt,lb=int(6*u),TH; lw=max(3,int(.10*TW))
    for side,off in((-1,int(.28*TW)),(1,int(.72*TW))):
        sp=side*int(.04*TW)
        pygame.draw.polygon(s,dc,[(off-lw//2,lt),(off+lw//2,lt),(off+lw//2+sp,lb),(off-lw//2+sp,lb)])
    tt,tb=int(2.2*u),int(6.2*u); ttw,tbw=int(.95*TW),int(.30*TW)
    pygame.draw.polygon(s,bc,[(tc-ttw//2,tt),(tc+ttw//2,tt),(tc+tbw//2,tb),(tc-tbw//2,tb)])
    for i in range(1,4):
        ry=tt+int((tb-tt)*i/4); fr=i/4; rw=int((ttw*(1-fr)+tbw*fr)*.38)
        pygame.draw.line(s,(10,10,8),(tc-rw,ry),(tc+rw,ry),max(1,int(u*.18)))
    at2,am,ab=int(2.4*u),int(5.5*u),int(8.2*u); aw=max(3,int(.07*TW))
    for side in(-1,1):
        sx=tc+side*(ttw//2-int(.05*TW)); ex2=tc+side*int(.52*TW); cx2=tc+side*int(.44*TW)
        pygame.draw.line(s,dc,(sx,at2),(ex2,am),aw); pygame.draw.line(s,dc,(ex2,am),(cx2,ab),aw)
        for fi in range(-1,2):
            fx=cx2+fi*max(2,int(.04*TW))
            pygame.draw.line(s,dc,(fx,ab),(fx+side*fi*2,ab+int(.6*u)),max(1,aw-1))
    hcx,hcy=tc,int(1.1*u); hrx,hry=int(.30*TW),int(1.15*u)
    pygame.draw.ellipse(s,bc,(hcx-hrx,hcy-hry,hrx*2,hry*2))
    pygame.draw.line(s,cc,(hcx,hcy-hry+4),(hcx+int(hrx*.3),hcy+int(hry*.4)),2)
    pygame.draw.line(s,cc,(hcx-int(hrx*.2),hcy-int(hry*.3)),(hcx-int(hrx*.5),hcy+int(hry*.5)),2)
    esx=int(hrx*.45); ecy=int(hcy-hry*.05); erx=max(4,int(hrx*.32)); ery=max(2,int(hry*.22))
    eyes=[]
    for side in(-1,1):
        ex3=hcx+side*esx; eyes.append((ex3,ecy,erx,ery))
        pygame.draw.ellipse(s,(0,0,0,255),(ex3-erx-2,ecy-ery-2,(erx+2)*2,(ery+2)*2))
    return s,eyes,TW,TH

# ── GAME ──────────────────────────────────────────────────────────────────────
class Game:
    def __init__(self):
        pygame.init(); pygame.mixer.init(frequency=SR,size=-16,channels=2,buffer=512)
        _init_embers(); pygame.display.set_caption("SILENT MAZE")
        self.scr=pygame.display.set_mode((W,H)); self.clock=pygame.time.Clock()
        self.FB=pygame.font.SysFont("couriernew",72,bold=True)
        self.FM=pygame.font.SysFont("couriernew",28)
        self.FS=pygame.font.SysFont("couriernew",16)
        self.SND=build_sounds()
        self._bg=self._bake_bg(); self._anger=bake_border((165,8,8)); self._escape=bake_border((20,200,80))
        self._jsface=bake_js_face(); self._door=bake_door()
        self._mont,self._meyes,self._mtw,self._mth=bake_monster()
        self._grain=pygame.Surface((W,H),pygame.SRCALPHA)
        self._dvig=self._bake_vig(); self._hvig=self._bake_hell_vig()
        self._tip=self.FS.render("WASD:MOVE  SHIFT:SPRINT  A/D:TURN  M:MAP  R:RESTART  ESC:QUIT",True,(68,58,48))
        self._mlbl=self.FS.render("MAP",True,(125,105,85))
        self._anlbl=self.FS.render("SANITY",True,(88,72,52))
        self.zbuf=np.full(RAYS,MAXD,dtype=np.float32); self._mmdir=True
        self.title=True; self.diff="HARD"; self.snd_on=True; self.embers_on=True
        self.show_mm=True; self._menu="title"; self._rumble_cd=0
        self.reset()

    def _bake_bg(self):
        s=pygame.Surface((W,H)); sp=max(1,H-HORIZON)
        for y in range(HORIZON): t=y/HORIZON; s.fill((int(35+25*t),int(5+5*t),int(3+4*t)),(0,y,W,1))
        for y in range(HORIZON,H): t=(y-HORIZON)/sp; s.fill((int(28-18*t),int(5-4*t),int(3-2*t)),(0,y,W,1))
        return s

    def _bake_vig(self):
        s=pygame.Surface((W,H),pygame.SRCALPHA)
        for r in range(max(W,H)//2,0,-12): t=r/(max(W,H)//2); pygame.draw.circle(s,(180,0,0,int((1-t)**2*140)),(W//2,H//2),r)
        return s

    def _bake_hell_vig(self):
        s=pygame.Surface((W,H),pygame.SRCALPHA)
        for i in range(0,60,3): a=max(0,int((1-(i/60))**2*90)); pygame.draw.rect(s,(160,20,0,a),(i,i,W-i*2,H-i*2),3)
        return s

    def reset(self):
        d=DIFF[self.diff]
        self.mon_spd=d['spd']; self.mon_sight=d['sight']; self.n_mons=d['n']
        self.bfs_max=d['bfs']; self.san_d=d['san_d']; self.fog_base=d['fog']
        self.flk_scale=d['flk']; self.passive=d['passive']; self.anger_mult=d['anger']
        self.approach=d['approach']; self.dark_vig=d['dark_vig']
        self.maze=generate_maze(MW,MH); self.state="playing"
        self.px,self.py,self.angle=2.5,2.5,math.pi/4
        self.shake=self.mon_anger=0; self.sanity=SAN_MAX
        self.bob_t=self.bob_y=0.0; self.flicker=1.0; self.flicker_cd=30
        self.whisper_cd=random.randint(400,1000); self.growl_cd=200
        self.step_cd=self.hb_cd=self.js=self.approach_cd=0
        self._mmdir=True; self.moving=self.sprinting=False
        # Door: random far corner
        corners=[(range(MW-2,MW//2,-1),range(MH-2,MH//2,-1)),(range(1,MW//2),range(MH-2,MH//2,-1)),
                 (range(MW-2,MW//2,-1),range(1,MH//2)),(range(1,MW//2),range(1,MH//2))]
        random.shuffle(corners[:3]); self.ex=self.ey=None
        for xr,yr in corners:
            for ey in yr:
                for ex in xr:
                    if self.maze[ey][ex]==0 and math.hypot(ex-2.5,ey-2.5)>32: self.ex,self.ey=ex,ey; break
                if self.ex is not None: break
            if self.ex is not None: break
        if self.ex is None: self.ex,self.ey=MW-3,MH-3
        # Monsters
        self.mons=[]; used=set()
        for _ in range(self.n_mons):
            for _ in range(5000):
                mx=random.randint(2,MW-3); my=random.randint(2,MH-3)
                if(self.maze[my][mx]==0 and math.hypot(mx-2.5,my-2.5)>30 and (mx,my) not in used
                        and all(math.hypot(mx-m[0],my-m[1])>12 for m in self.mons)):
                    self.mons.append([mx+.5,my+.5,0.0,[],0]); used.add((mx,my)); break
            else: self.mons.append([MW-3.5,MH-3.5,0.0,[],0])
        self.zbuf=np.full(RAYS,MAXD,dtype=np.float32)

    def solid(self,x,y):
        gx,gy=int(x),int(y); return not(0<=gx<MW and 0<=gy<MH) or self.maze[gy][gx]==1

    def mv(self,nx,ny):
        if not self.solid(nx,self.py): self.px=nx
        if not self.solid(self.px,ny): self.py=ny

    def bfs(self,sx,sy,tx,ty):
        s,g=(int(sx),int(sy)),(int(tx),int(ty))
        if s==g: return []
        cf={s:None}; q=deque([s])
        while q:
            cx,cy=q.popleft()
            if(cx,cy)==g: break
            for dx,dy in((1,0),(-1,0),(0,1),(0,-1)):
                nb=(cx+dx,cy+dy)
                if nb not in cf and 0<=nb[0]<MW and 0<=nb[1]<MH and self.maze[nb[1]][nb[0]]==0:
                    cf[nb]=(cx,cy); q.append(nb)
        if g not in cf: return []
        p=[]; n=g
        while n!=s: p.append(n); n=cf[n]
        return p[::-1]

    def inp(self):
        k=pygame.key.get_pressed()
        spd=SPR if(k[pygame.K_LSHIFT] or k[pygame.K_RSHIFT]) else SPD
        ca=math.cos(self.angle)*spd; sa=math.sin(self.angle)*spd; mv=False
        if k[pygame.K_w]: self.mv(self.px+ca,self.py+sa); mv=True
        if k[pygame.K_s]: self.mv(self.px-ca,self.py-sa); mv=True
        if k[pygame.K_a]: self.angle-=ROT
        if k[pygame.K_d]: self.angle+=ROT
        self.moving=mv; self.sprinting=(spd==SPR)

    def horror(self):
        dist=min(math.hypot(self.px-m[0],self.py-m[1]) for m in self.mons)
        self.sanity=max(0,min(SAN_MAX,self.sanity+(-self.san_d if dist<self.mon_sight else SAN_R)-self.passive))
        sf=self.sanity/SAN_MAX
        self.flicker_cd-=1
        if self.flicker_cd<=0:
            p=max(3,int(35*sf)); self.flicker_cd=random.randint(p,p*4)
            sev=min(1.0,(1-sf)*self.flk_scale)
            self.flicker=random.uniform(.1,.45)*sev if sev>.3 and random.random()<.45 else random.uniform(.80,1.0)
        self.flicker+=(1-self.flicker)*.14
        if self.moving:
            self.bob_t+=.18 if self.sprinting else .10
            self.bob_y=math.sin(self.bob_t)*(5 if self.sprinting else 2.5)
        else: self.bob_y*=.82
        if not self.snd_on: return
        self.step_cd-=1
        if self.moving and self.step_cd<=0:
            self.step_cd=11 if self.sprinting else 22; self.SND['step'].set_volume(.30); self.SND['step'].play()
        self.hb_cd-=1; prx=max(0,1-dist/self.mon_sight)
        if prx>.08 and self.hb_cd<=0:
            self.hb_cd=max(18,int(75-prx*55)); self.SND['heartbeat'].set_volume(min(1,prx*1.7)); self.SND['heartbeat'].play()
        self.growl_cd-=1
        if dist<self.mon_sight and self.growl_cd<=0:
            self.growl_cd=random.randint(90,260); self.SND['growl'].set_volume(min(1,(1-dist/self.mon_sight)*1.4)); self.SND['growl'].play()
        self.approach_cd=max(0,self.approach_cd-1)
        if self.approach:
            cd=min(math.hypot(self.px-m[0],self.py-m[1]) for m in self.mons)
            if cd<6.0 and self.approach_cd<=0:
                vol=min(1.0,(1-cd/6.0)*1.5)
                if cd<2.5: self.SND['hiss'].set_volume(vol); self.SND['hiss'].play(); self.approach_cd=25
                else:       self.SND['stalk'].set_volume(vol*.8); self.SND['stalk'].play(); self.approach_cd=55
        if self.dark_vig:
            self._rumble_cd=max(0,self._rumble_cd-1)
            if self._rumble_cd<=0:
                self.SND['rumble'].set_volume(0.18+(1-sf)*.28); self.SND['rumble'].play(); self._rumble_cd=random.randint(160,300)
        self.whisper_cd-=1
        if self.whisper_cd<=0:
            self.whisper_cd=random.randint(320,900); self.SND['whisper'].set_volume(.40+(1-sf)*.50); self.SND['whisper'].play()
        if sf<.40 and random.random()<.007:
            self.SND['breathe'].set_volume((1-sf)*.70); self.SND['breathe'].play()

    def monster(self):
        for m in self.mons:
            mx,my=m[0],m[1]; dx,dy=self.px-mx,self.py-my; dist=math.hypot(dx,dy)
            if dist<MON_CATCH:
                if self.snd_on: self.SND['sting'].set_volume(1); self.SND['sting'].play()
                self.js=210; self.state="dead"; self.shake=65; return
            m[2]=anger=min(1,max(0,(1-dist/self.mon_sight)*1.4))
            if dist>self.mon_sight: continue
            spd=self.mon_spd*(1+anger*self.anger_mult); m[4]-=1
            if m[4]<=0: m[4]=self.bfs_max; m[3]=self.bfs(mx,my,self.px,self.py)
            if m[3]:
                wx,wy=m[3][0]; wtx,wty=wx+.5,wy+.5; wdx,wdy=wtx-mx,wty-my; wd=math.hypot(wdx,wdy)
                if wd<.25: m[3].pop(0)
                else:
                    step=min(spd,wd); nx,ny=mx+(wdx/wd)*step,my+(wdy/wd)*step
                    if   not self.solid(nx,ny): m[0],m[1]=nx,ny
                    elif not self.solid(nx,my): m[0]=nx
                    elif not self.solid(mx,ny): m[1]=ny
            else:
                nx,ny=mx+(dx/dist)*spd,my+(dy/dist)*spd
                if   not self.solid(nx,ny): m[0],m[1]=nx,ny
                elif not self.solid(nx,my): m[0]=nx
                elif not self.solid(mx,ny): m[1]=ny
        self.mon_anger=max(m[2] for m in self.mons)

    def rays(self):
        bob=int(self.bob_y); eH=HORIZON+bob
        self.scr.blit(self._bg,(0,bob))
        px,py=self.px,self.py; bc=math.cos(self.angle); bs=math.sin(self.angle)
        zb=self.zbuf; fl=self.flicker; sf=self.sanity/SAN_MAX
        # Numpy floor cast
        n_rows=max(0,H-eH-1)
        if n_rows>0:
            lc=math.cos(self.angle-HFOV); ls=math.sin(self.angle-HFOV)
            rc2=math.cos(self.angle+HFOV); rs2=math.sin(self.angle+HFOV)
            sy=np.arange(eH+1,H,dtype=np.float32)
            rd=PDIST/np.maximum(1.0,sy-eH)
            dt=np.minimum(1.0,rd/MAXD); bl=np.maximum(0.04,(1.0-dt)*0.58*fl); ht=0.35+(1-sf)*0.3
            R=np.minimum(255,60*bl*(1+ht*.9)).astype(np.uint8)
            G=np.maximum(0,20*bl*(1-ht*.5)).astype(np.uint8)
            B=np.maximum(0,12*bl*(1-ht*.8)).astype(np.uint8)
            xs=np.arange(W,dtype=np.float32)/W
            fx=px+rd[:,None]*(lc+xs[None,:]*(rc2-lc)); fy=py+rd[:,None]*(ls+xs[None,:]*(rs2-ls))
            fr=np.broadcast_to(R[:,None],(n_rows,W)).copy(); fg=np.broadcast_to(G[:,None],(n_rows,W)).copy(); fb=np.broadcast_to(B[:,None],(n_rows,W)).copy()
            seam=(fx%1.0<0.055)|(fy%1.0<0.055)
            fr[seam]=np.maximum(0,fr[seam].astype(np.int16)-18).astype(np.uint8)
            fg[seam]=np.maximum(0,fg[seam].astype(np.int16)-6).astype(np.uint8)
            fb[seam]=np.maximum(0,fb[seam].astype(np.int16)-4).astype(np.uint8)
            arr=pygame.surfarray.pixels3d(self.scr); arr[:,eH+1:H,0]=fr.T; arr[:,eH+1:H,1]=fg.T; arr[:,eH+1:H,2]=fb.T; del arr
        # Wall cast
        for ray in range(RAYS):
            rc,rs=RC[ray],RS[ray]; cra=bc*rc-bs*rs or 1e-10; sra=bs*rc+bc*rs or 1e-10
            mx,my=int(px),int(py); ddx,ddy=abs(1/cra),abs(1/sra)
            if cra<0: sx=-1; sdx=(px-mx)*ddx
            else:      sx= 1; sdx=(mx+1-px)*ddx
            if sra<0: sy2=-1; sdy=(py-my)*ddy
            else:      sy2= 1; sdy=(my+1-py)*ddy
            hs=0
            for _ in range(int(MAXD*3)):
                if sdx<sdy: sdx+=ddx; mx+=sx; hs=0
                else:        sdy+=ddy; my+=sy2; hs=1
                if 0<=mx<MW and 0<=my<MH and self.maze[my][mx]==1: break
            d=max(.001,((sdx-ddx) if hs==0 else (sdy-ddy))*RC[ray]); zb[ray]=d
            wh=int(PDIST/d); yt=max(0,eH-wh//2); yb=min(H,eH+wh//2)
            lit=max(self.fog_base+sf*.08,1-d/MAXD)*fl*(.55 if hs else 1)
            col=list(WLUT[min(255,int(lit*255))]); ht=0.35+(1-sf)*0.3
            col[0]=min(255,int(col[0]*(1+ht*.8))); col[1]=max(0,int(col[1]*(1-ht*.6))); col[2]=max(0,int(col[2]*(1-ht*.9)))
            pygame.draw.rect(self.scr,tuple(col),(ray*SCALE,yt,SCALE,yb-yt))

    def blit_sprite(self,surf,draw_x,top_y,sw,sh,dist):
        c0=max(0,draw_x//SCALE); c1=min(RAYS,(draw_x+sw)//SCALE+1)
        for ci in range(c0,c1):
            if dist>=self.zbuf[ci]: continue
            bx0=max(0,ci*SCALE-draw_x); bx1=min(sw,bx0+SCALE)
            if bx0>=bx1: continue
            scx=draw_x+bx0
            if scx>=W or scx+(bx1-bx0)<=0: continue
            self.scr.blit(surf,(scx,top_y),(bx0,0,bx1-bx0,sh))

    def draw_door(self):
        dx,dy=self.ex+.5-self.px,self.ey+.5-self.py; dist=math.hypot(dx,dy)
        if dist<.2: return
        at=math.atan2(dy,dx)-self.angle; at=(at+math.pi)%(2*math.pi)-math.pi
        if abs(at)>HFOV+.4: return
        eH=HORIZON+int(self.bob_y); proj=min(PDIST/dist,H*2)
        dh=max(4,int(proj*1.8)); dw=max(4,int(proj*.9))
        cx=int((at/FOV+.5)*W); draw_x=cx-dw//2; top_y=eH+int(proj*.5)-dh
        if draw_x+dw<0 or draw_x>=W: return
        pulse=.75+.25*math.sin(pygame.time.get_ticks()*.004)
        sc=pygame.transform.scale(self._door,(dw,dh))
        tint=pygame.Surface((dw,dh),pygame.SRCALPHA); tint.fill((0,int(pulse*55),int(pulse*10),0))
        sc.blit(tint,(0,0),special_flags=pygame.BLEND_RGBA_ADD); self.blit_sprite(sc,draw_x,top_y,dw,dh,dist)

    def draw_monster(self):
        for m in self.mons:
            mx,my,anger=m[0],m[1],m[2]; dx,dy=mx-self.px,my-self.py; dist=math.hypot(dx,dy)
            if dist<.15: continue
            at=math.atan2(dy,dx)-self.angle; at=(at+math.pi)%(2*math.pi)-math.pi
            if abs(at)>HFOV+.5: continue
            eH=HORIZON+int(self.bob_y); proj=min(PDIST/dist,H*2.5)
            th=max(4,int(proj*2.2)); tw=max(4,int(proj*1.1))
            cx=int((at/FOV+.5)*W); draw_x=cx-tw//2; top_y=eH+int(proj*.5)-th
            if draw_x+tw<0 or draw_x>=W: continue
            pulse=.7+.3*math.sin(pygame.time.get_ticks()*.006)
            sc=pygame.transform.scale(self._mont,(tw,th))
            if anger>0:
                ts=pygame.Surface((tw,th),pygame.SRCALPHA); ts.fill((int(anger*pulse*80),0,0,0))
                sc.blit(ts,(0,0),special_flags=pygame.BLEND_RGBA_ADD)
            eb=min(255,int((.6+anger*.4)*pulse*255)); ecol=(eb,int(eb*.08),int(eb*.04)); eglow=(min(255,int(eb*.7)),0,0)
            rsx,rsy=tw/self._mtw,th/self._mth
            for ex2,ey2,erx,ery in self._meyes:
                sex,sey=int(ex2*rsx),int(ey2*rsy); srx=max(2,int(erx*rsx)); sry=max(1,int(ery*rsy))
                pygame.draw.ellipse(sc,eglow,(sex-srx-2,sey-sry-2,(srx+2)*2,(sry+2)*2))
                pygame.draw.ellipse(sc,ecol,(sex-srx,sey-sry,srx*2,sry*2))
                pygame.draw.ellipse(sc,(max(0,eb//4),0,0),(sex-max(1,srx//3),sey-max(1,sry//2),max(1,srx//3)*2,max(1,sry//2)*2))
            self.blit_sprite(sc,draw_x,top_y,tw,th,dist)

    def minimap(self):
        if not self.show_mm: return
        if self._mmdir:
            cell=max(3,210//max(MW,MH)); dw=cell*MW; dh=cell*MH
            ms=pygame.Surface((dw,dh),pygame.SRCALPHA); ms.fill((0,0,0,175))
            for row in range(MH):
                for col in range(MW): ms.fill((80,68,55,220) if self.maze[row][col] else (18,15,12,100),(col*cell,row*cell,cell,cell))
            pygame.draw.rect(ms,(130,110,85,220),(0,0,dw,dh),1)
            self._mm=ms; self._mmc=cell; self._mmdw=dw; self._mmdh=dh; self._mmdir=False
        cell=self._mmc; dw=self._mmdw; dh=self._mmdh; MM=210
        cw=min(dw,MM); ch=min(dh,MM); ox=W-cw-14; oy=14
        ppx=int(self.px*cell); ppy=int(self.py*cell)
        cx=max(0,min(ppx-cw//2,dw-cw)); cy=max(0,min(ppy-ch//2,dh-ch))
        self.scr.blit(self._mm,(ox,oy),(cx,cy,cw,ch))
        def mp(wx,wy): return(ox+int(wx*cell)-cx,oy+int(wy*cell)-cy)
        dot=max(3,cell-1)
        pygame.draw.circle(self.scr,(30,220,80),mp(self.ex+.5,self.ey+.5),dot)
        for m in self.mons: pygame.draw.circle(self.scr,(220,30,30),mp(m[0],m[1]),dot)
        pp=mp(self.px,self.py); pygame.draw.circle(self.scr,(255,220,50),pp,dot+1)
        al=max(8,cell*3)
        pygame.draw.line(self.scr,(255,220,50),pp,(int(pp[0]+math.cos(self.angle)*al),int(pp[1]+math.sin(self.angle)*al)),2)
        pygame.draw.rect(self.scr,(130,110,85),(ox,oy,cw,ch),1); self.scr.blit(self._mlbl,(ox+4,oy+ch+4))

    def hud(self):
        cx=W//2
        pygame.draw.line(self.scr,(185,165,145),(cx-8,HORIZON),(cx+8,HORIZON),1)
        pygame.draw.line(self.scr,(185,165,145),(cx,HORIZON-8),(cx,HORIZON+8),1)
        bx,by,bw,bh=12,H-36,160,9; sf=self.sanity/SAN_MAX
        pygame.draw.rect(self.scr,(35,25,18),(bx,by,bw,bh))
        pygame.draw.rect(self.scr,(int(30+170*(1-sf)),int(200*sf),int(180*sf)),(bx,by,int(bw*sf),bh))
        pygame.draw.rect(self.scr,(100,85,65),(bx,by,bw,bh),1); self.scr.blit(self._anlbl,(bx,by-17))
        if self.mon_anger>.25: self._anger.set_alpha(int(self.mon_anger**2*255)); self.scr.blit(self._anger,(0,0))
        if self.sanity<45:
            intensity=1-self.sanity/45
            if random.random()<intensity*.28:
                for _ in range(random.randint(1,3)):
                    ly=random.randint(0,H-4); sh=random.randint(-10,10)
                    try: self.scr.blit(self.scr.subsurface((0,ly,W,min(random.randint(1,4),H-ly))).copy(),(sh,ly))
                    except: pass
        # Nightmare darkness vignette
        if self.dark_vig:
            vs=int((1.0-sf)**1.6*220)
            if vs>8:
                vig=pygame.Surface((W,H),pygame.SRCALPHA); steps=max(1,vs//14)
                for i in range(steps):
                    ins=i*(min(W,H)//(steps*2)); a=max(0,int(vs*(1-(i/steps))**1.4))
                    if a>0: pygame.draw.rect(vig,(0,0,0,a),(ins,ins,W-ins*2,H-ins*2),max(1,min(W,H)//(steps*3)+4))
                self.scr.blit(vig,(0,0))
        d=math.hypot(self.px-self.ex-.5,self.py-self.ey-.5)
        if d<5:
            g2=int(255*(1-d/5)); ht=self.FM.render("EXIT NEARBY",True,(20,g2,40))
            self.scr.blit(ht,(W//2-ht.get_width()//2,H-65))
        self.scr.blit(self._tip,(10,H-20))

    def grain(self):
        sf=self.sanity/SAN_MAX; alpha=int(15+(1-sf)*65); n=max(30,int(alpha*2.5))
        self._grain.fill((0,0,0,0))
        xs=np.random.randint(0,W-4,n); ys=np.random.randint(0,H-4,n); vs=np.random.randint(25,110,n)
        for i in range(n): self._grain.fill((int(vs[i]),int(vs[i]),int(vs[i]),alpha),(int(xs[i]),int(ys[i]),4,4))
        self.scr.blit(self._grain,(0,0))

    # ── MENU ──────────────────────────────────────────────────────────────────
    def _btn(self,label,cx,y,w=260,h=46,active=False,col=None):
        mx,my=pygame.mouse.get_pos(); r=(cx-w//2,y,w,h); hov=r[0]<=mx<=r[0]+w and r[1]<=my<=r[1]+h
        if col: bg=tuple(min(255,c+30) for c in col) if(hov or active) else col; bd=tuple(min(255,c+60) for c in col) if(hov or active) else tuple(min(255,c+20) for c in col)
        else:   bg=(70,18,8) if active else((55,14,6) if hov else(35,8,4)); bd=(220,60,30) if active else((180,50,20) if hov else(120,35,15))
        tc=(255,220,200) if(hov or active) else(200,160,140)
        pygame.draw.rect(self.scr,bg,r,border_radius=5); pygame.draw.rect(self.scr,bd,r,2,border_radius=5)
        lbl=self.FM.render(label,True,tc); self.scr.blit(lbl,(cx-lbl.get_width()//2,y+h//2-lbl.get_height()//2))
        return r

    def _hit(self,r): mx,my=pygame.mouse.get_pos(); return r[0]<=mx<=r[0]+r[2] and r[1]<=my<=r[1]+r[3]

    def _hdr(self,txt,y,col=(200,80,30)):
        p=.82+.18*math.sin(pygame.time.get_ticks()*.002); c=tuple(int(v*p) for v in col)
        s=self.FB.render(txt,True,c); self.scr.blit(s,(W//2-s.get_width()//2,y))

    def _mbg(self):
        self.scr.fill((8,2,2))
        if self.embers_on: draw_embers(self.scr)
        self.scr.blit(self._hvig,(0,0))

    def title_scr(self,events):
        if self._menu=="title":
            self._mbg(); self._hdr("SILENT MAZE",H//2-160)
            sub=self.FM.render("survive the maze  —  find the exit",True,(140,80,60)); self.scr.blit(sub,(W//2-sub.get_width()//2,H//2-70))
            r1=self._btn("▶  PLAY",W//2,H//2-10); r2=self._btn("⚙  DIFFICULTY",W//2,H//2+55)
            r3=self._btn("◈  SETTINGS",W//2,H//2+120); r4=self._btn("✕  QUIT",W//2,H//2+185,col=(45,10,5))
            for ev in events:
                if ev.type==pygame.MOUSEBUTTONDOWN:
                    if self._hit(r1): self.title=False
                    elif self._hit(r2): self._menu="diff"
                    elif self._hit(r3): self._menu="settings"
                    elif self._hit(r4): pygame.quit(); sys.exit()
                if ev.type==pygame.KEYDOWN:
                    if ev.key in(pygame.K_SPACE,pygame.K_RETURN): self.title=False
                    elif ev.key==pygame.K_ESCAPE: pygame.quit(); sys.exit()

        elif self._menu=="diff":
            self._mbg(); self._hdr("DIFFICULTY",H//2-200)
            base_y=H//2-110
            for i,dk in enumerate(DIFF):
                active=self.diff==dk
                r=self._btn(dk,W//2,base_y+i*72,w=380,active=active,col=(10,45,10) if active else None)
                dc=self.FS.render(DIFF[dk]['desc'],True,(160,220,120) if active else(120,90,70))
                self.scr.blit(dc,(W//2-dc.get_width()//2,base_y+i*72+30))
                for ev in events:
                    if ev.type==pygame.MOUSEBUTTONDOWN and self._hit(r): self.diff=dk
            rb=self._btn("◀  BACK",W//2,H//2+158,w=200)
            for ev in events:
                if ev.type==pygame.MOUSEBUTTONDOWN and self._hit(rb): self._menu="title"
                if ev.type==pygame.KEYDOWN and ev.key==pygame.K_ESCAPE: self._menu="title"

        elif self._menu=="settings":
            self._mbg(); self._hdr("SETTINGS",H//2-180)
            toggles=[("SOUND",self.snd_on,"snd_on"),("EMBERS",self.embers_on,"embers_on"),("MINIMAP",self.show_mm,"show_mm")]
            for i,(name,val,attr) in enumerate(toggles):
                r=self._btn(f"{name}  [ {'ON' if val else 'OFF'} ]",W//2,H//2-80+i*68,w=340,active=val,col=(10,45,10) if val else(45,10,10))
                for ev in events:
                    if ev.type==pygame.MOUSEBUTTONDOWN and self._hit(r): setattr(self,attr,not getattr(self,attr))
            info=self.FS.render(f"difficulty: {self.diff}  ·  monsters: {DIFF[self.diff]['n']}",True,(100,70,50))
            self.scr.blit(info,(W//2-info.get_width()//2,H//2+136))
            rb=self._btn("◀  BACK",W//2,H//2+158,w=200)
            for ev in events:
                if ev.type==pygame.MOUSEBUTTONDOWN and self._hit(rb): self._menu="title"
                if ev.type==pygame.KEYDOWN and ev.key==pygame.K_ESCAPE: self._menu="title"

    def dead_scr(self):
        js=self.js
        if js>140:
            prog=(210-js)/70; fw,fh=int(W*(.15+prog*.85)),int(H*(.15+prog*.85))
            self.scr.fill((0,0,0)); self.scr.blit(pygame.transform.scale(self._jsface,(fw,fh)),((W-fw)//2,(H-fh)//2))
        elif js>60:
            self.scr.blit(self._jsface,(0,0))
            t=(140-js)/80; ov=pygame.Surface((W,H),pygame.SRCALPHA); ov.fill((200,0,0,int(abs(math.sin(t*math.pi*4))*160))); self.scr.blit(ov,(0,0))
            for ry in np.random.randint(0,H,80):
                try: self.scr.blit(self.scr.subsurface((0,int(ry),W,2)).copy(),(random.randint(-40,40),int(ry)))
                except: pass
        elif js>0:
            self.scr.blit(self._jsface,(0,0))
            fd=pygame.Surface((W,H),pygame.SRCALPHA); fd.fill((0,0,0,255-int(js/60*255))); self.scr.blit(fd,(0,0))
        else:
            self.scr.fill((10,0,0))
            if self.embers_on: draw_embers(self.scr)
            self.scr.blit(self._dvig,(0,0))
            for txt,y in[(self.FB.render("YOU DIED",True,(212,26,26)),H//2-100),(self.FM.render("it found you",True,(128,52,52)),H//2-18)]:
                self.scr.blit(txt,(W//2-txt.get_width()//2,y))
            bx,by=W//2-110,H//2+60; bw,bh=220,48
            mx,my=pygame.mouse.get_pos(); hov=bx<=mx<=bx+bw and by<=my<=by+bh
            pygame.draw.rect(self.scr,(140,25,25) if hov else(90,15,15),(bx,by,bw,bh),border_radius=6)
            pygame.draw.rect(self.scr,(240,80,80) if hov else(180,40,40),(bx,by,bw,bh),2,border_radius=6)
            lb=self.FM.render("RESTART",True,(255,200,200) if hov else(220,120,120))
            self.scr.blit(lb,(bx+bw//2-lb.get_width()//2,by+bh//2-lb.get_height()//2))
            self.scr.blit(self.FS.render("or press  R",True,(80,40,40)),(W//2-40,by+bh+10))
        self.js=max(0,self.js-1)

    def win_scr(self):
        self.scr.fill((0,15,5))
        if self.embers_on: draw_embers(self.scr)
        for txt,y in[(self.FB.render("ESCAPED",True,(68,212,108)),H//2-80),(self.FM.render("the fog parts… but for how long",True,(52,128,68)),H//2),(self.FM.render("[ R — PLAY AGAIN ]",True,(68,148,88)),H//2+60)]:
            self.scr.blit(txt,(W//2-txt.get_width()//2,y))
        self._escape.set_alpha(int(abs(math.sin(pygame.time.get_ticks()*.003))*220)); self.scr.blit(self._escape,(0,0))

    def run(self):
        while True:
            self.clock.tick(FPS); events=pygame.event.get()
            for ev in events:
                if ev.type==pygame.QUIT: pygame.quit(); sys.exit()
                if ev.type==pygame.KEYDOWN:
                    if ev.key==pygame.K_ESCAPE and not self.title and self.state!="dead": pygame.quit(); sys.exit()
                    if ev.key==pygame.K_r and not self.title and self.js==0: self.reset()
                    if ev.key==pygame.K_m and not self.title: self.show_mm=not self.show_mm
                if ev.type==pygame.MOUSEBUTTONDOWN and self.state=="dead" and self.js==0:
                    mx,my=pygame.mouse.get_pos()
                    if W//2-110<=mx<=W//2+110 and H//2+60<=my<=H//2+108: self.reset()
            if self.title: self.title_scr(events); pygame.display.flip(); continue
            if self.state=="playing":
                self.inp(); self.horror(); self.monster()
                if math.hypot(self.px-self.ex-.5,self.py-self.ey-.5)<.9: self.state="win"
                ox=oy=0
                if self.shake>0: ox=random.randint(-self.shake,self.shake)//3; oy=random.randint(-self.shake,self.shake)//3; self.shake=max(0,self.shake-1)
                if ox or oy: buf=pygame.Surface((W,H)); prev,self.scr=self.scr,buf
                self.rays(); self.draw_door(); self.draw_monster(); self.grain()
                if self.embers_on: draw_embers(self.scr)
                self.scr.blit(self._hvig,(0,0)); self.minimap(); self.hud()
                if ox or oy: self.scr=prev; self.scr.blit(buf,(ox,oy))
            elif self.state=="dead":
                if self.js==0: self.rays()
                self.dead_scr()
            elif self.state=="win": self.rays(); self.win_scr()
            pygame.display.flip()

if __name__=="__main__": Game().run()
