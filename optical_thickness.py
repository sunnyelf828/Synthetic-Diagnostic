import numpy as np
from scipy.integrate import trapz, cumtrapz
import numpy.fft as fft
import matplotlib.pyplot as plt
from matplotlib import rcParams, animation
import ipyparallel as ipp
from IPython.display import HTML

from sdp.settings.unitsystem import cgs
import sdp.plasma.analytic.testparameter as tp 
from sdp.plasma.character import PlasmaCharProfile
from sdp.diagnostic.ecei.ecei2d.imaging import ECE2D,ECEImagingSystem
from sdp.diagnostic.ecei.ecei2d.CodeV_detector2d import Code5Antenna
import matplotlib
from sdp.plasma.m3dc1.loader_fluc import M3DC1_Loader as MCL 

plt.ion()
rcParams['figure.figsize'] = 8,10
rcParams['font.size'] = 12
color_array = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

c = cgs['c']
keV = cgs['keV']
e = cgs['e']
me = cgs['m_e']

f=MCL(m3dpath='/Users/chenm/Box Sync/Research Report/SyntheticMIR-FVK/JOREK vs M3D-C1 vs exp/fullrng_1.0_2.6/',tor_slice=[1,2])

p2d_m3dc1=f.create_profile('ecei2d')
p2d_m3dc1.setup_interps()
pcp_m3dc1=PlasmaCharProfile(p2d_m3dc1)
pcp_m3dc1.set_coords([p2d_m3dc1.grid.Z2D,p2d_m3dc1.grid.R2D])
omega_ce=pcp_m3dc1.omega_ce/(2*np.pi*1e9)
omega_pe=pcp_m3dc1.omega_pe/(2*np.pi*1e9)
omega_R=pcp_m3dc1.omega_R/(2*np.pi*1e9)

Z_mid=p2d_m3dc1.grid.NZ/2
# Show the analytic equlibrium profile

fig,(ax,bx,cx)=plt.subplots(3,sharex=True)
ax.plot(p2d_m3dc1.grid.R1D,p2d_m3dc1.Te0[Z_mid,:]/keV,'r')
ax.set_ylabel('$T_e [keV]$',color='r')
ax.tick_params('y',colors='r')
#ax.set_xlabel('R [cm]')
ax.set_title('(a) Equlibrium profile at midplane')
ax2=ax.twinx()
ax2.plot(p2d_m3dc1.grid.R1D,p2d_m3dc1.ne0[Z_mid,:],'b--')
ax2.set_ylabel('$n_e [cm^{-3}]$',color='b')
ax2.tick_params('y',colors='b')

#plt.figure()

bx.plot(p2d_m3dc1.grid.R1D,omega_ce[Z_mid,:],'r--',label='$\omega_{ce}$')
bx.plot(p2d_m3dc1.grid.R1D,omega_ce[Z_mid,:]*2,'r',label='$\omega_{2ce}$')
bx.plot(p2d_m3dc1.grid.R1D,omega_R[Z_mid,:],label='$\omega_R$')
bx.plot(p2d_m3dc1.grid.R1D,omega_pe[Z_mid,:],'k',label='$\omega_{pe}$')
bx.legend(loc=1)
#bx.xlabel('R [cm]');
bx.set_ylabel('Frequency [GHz]')
bx.set_title('(b) Characteristic frequencies [GHz] at midplane')
bx.grid()

Te = p2d_m3dc1.Te0[Z_mid,:]*1000/keV # unit in eV
omega_p = omega_pe[Z_mid,:]*(2*np.pi*1e9)
omega_c = omega_ce[Z_mid,:]*(2*np.pi*1e9)
pi = np.pi
q = (omega_p/omega_c)**2
#kB = 1.38064852e-16
kB = 1.602e-12	
LB = 167

Fo = np.sqrt(1-q)
C1 = kB*Te/(me*c**2)
C2 = q*omega_c*LB/c
tau_o1 = (pi/2)*Fo*C1*C2
Fo2 = Fo**3
tau_o2 = pi*Fo2*(C1**2)*C2

Fx2 = np.sqrt((12-8*q+q**2)/(12-4*q))*((6-q)/(6-2*q))**2
tau_x2 = pi*Fx2*C1*C2

#plt.figure()
cx.plot(p2d_m3dc1.grid.R1D,tau_x2,label=u'$\u03C4_{x2}$') #cannot use $\tau_{x2} --> going to treat \t as tab
cx.plot(p2d_m3dc1.grid.R1D,tau_o2,label=u'$\u03C4_{o2}$')
cx.plot(p2d_m3dc1.grid.R1D,tau_o1,label=u'$\u03C4_{o1}$')
cx.axhline(1,color='k',ls='dashed',label=u'\u03C4 =1')
cx.legend(loc=1)
cx.set_title(u'(c) Optical thickness \u03C4 of Characteristic modes')
cx.set_xlabel('R [cm]');
cx.set_ylabel(u'Optical thickness \u03C4')
cx.grid()