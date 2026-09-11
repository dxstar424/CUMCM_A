import numpy as np
from src.solver import Config, step_1d, solve_1d, _surface_state

def test_zero_flux_uniform_state():
    c=Config(kind='q1',n=12,dt=10,h=0,hm=0)
    T,C,it,_,_=step_1d(c,0,np.full(12,28.),np.full(12,2.55),10)
    assert np.max(abs(T-28))<1e-12 and np.max(abs(C-2.55))<1e-12

def test_moving_fixed_domain_degeneracy():
    a=solve_1d(Config(kind='q4',n=16,dt=30),300,np.arange(0,301,60.))
    b=solve_1d(Config(kind='q4',n=16,dt=30,moving=True,gamma=0),300,np.arange(0,301,60.))
    assert np.isfinite(b['C']).all() and b['C'].shape == a['C'].shape

def test_latent_zero_matches_baseline():
    t=np.arange(0,601,60.)
    a=solve_1d(Config(kind='q2',n=16,dt=30),600,t)
    b=solve_1d(Config(kind='q2',n=16,dt=30,latent=True,Lv=0),600,t)
    assert np.max(abs(a['T']-b['T']))<1e-10 and np.max(abs(a['C']-b['C']))<1e-10

def test_surface_reconstruction_between_values():
    c=Config(kind='q2',n=16,dt=30)
    T=np.full(16,40.); C=np.linspace(2.0,1.0,16)
    _,_,Cs,_=_surface_state(c,T[-1],C[-1],0,0.02/16)
    assert min(C[-1],0.05)<=Cs<=max(C[-1],0.05)
