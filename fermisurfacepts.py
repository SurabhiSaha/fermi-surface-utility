import pyprocar
from numpy import zeros, loadtxt, asarray,meshgrid, mgrid, reshape, subtract, array, dot,add
import numpy as np
from numpy.linalg import norm
from datetime import datetime
import subprocess, os
import pyvista as pv
from scipy.interpolate import LinearNDInterpolator
from skimage.measure import marching_cubes
#__________________________________
import warnings                   #  
warnings.filterwarnings("ignore") #
#_________________________________#

#-----------------------------------------------------------------------------
start_time = datetime.now()

#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#This part is to run a linux command using python and extract output
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
string = ['BANDGRID_3D_BANDS','BANDGRID_3D_fermi']
#________________________________________________________Class Structure_____________________________________________________________
class data_extract:
	def __init__(self,prefix, tag):
		self.prefix = prefix
		self.tag = tag

	def fermi_energy(self):
		e_f = subprocess.check_output("cat %s | grep 'Fermi Energy:'| awk '{print $3}'"%(self.prefix+'.bxsf'), shell=True)
		e_f = float(e_f.decode('utf-8'))			
		return(e_f)
		
	def nbands(self):
		if self.tag == 1:
			sss = string[0]
		elif self.tag ==2:
			sss = string[1]
		nband_f = subprocess.check_output("cat %s | grep -A6 %s| sed -n '2p'"%(self.prefix+'.bxsf',sss), shell=True)
		nband_f = int(nband_f.decode('utf-8'))			
		return(nband_f)


	def kmesh(self):
		if self.tag == 1:
			sss = string[0]
		elif self.tag ==2:
			sss = string[1]

		kx = subprocess.check_output("cat %s | grep -A6 %s| sed -n '3p'| awk '{print $1}'"%(self.prefix+'.bxsf',sss), 		shell=True)
		kx = int(kx.decode('utf-8'))
		ky = subprocess.check_output("cat %s | grep -A6 '%s'| sed -n '3p'| awk '{print $2}'"%(self.prefix+'.bxsf',sss), shell=True)
		ky = int(ky.decode('utf-8'))
		kz = subprocess.check_output("cat %s | grep -A6 '%s'| sed -n '3p'| awk '{print $3}'"%(self.prefix+'.bxsf',sss), 		shell=True)
		kz = int(kz.decode('utf-8'))
		return (kx,ky,kz)
		
	def center(self):
		if self.tag == 1:
			sss = string[0]
		elif self.tag ==2:
			sss = string[1]

		cent_1 = subprocess.check_output("cat %s | grep -A6 '%s'| sed -n '4p'| awk '{print $1}'"%(self.prefix+'.bxsf',sss), shell=True)
		cent_1 = float(cent_1.decode('utf-8'))
		cent_2 = subprocess.check_output("cat %s | grep -A6 %s| sed -n '4p'| awk '{print $2}'"%(self.prefix+'.bxsf',sss), shell=True)
		cent_2 = float(cent_2.decode('utf-8'))
		cent_3 = subprocess.check_output("cat %s | grep -A6 %s| sed -n '4p'| awk '{print $3}'"%(self.prefix+'.bxsf',sss), shell=True)
		cent_3 = float(cent_3.decode('utf-8'))
		cent = asarray([cent_1,cent_2, cent_3])
		return(cent)

	def lattice_vectors(self):
		if self.tag == 1:
			sss = string[0]
		elif self.tag ==2:
			sss = string[1]

		a1 = subprocess.check_output("cat %s | grep -A6 %s| sed -n '5p'| awk '{print $1}'"%(self.prefix+'.bxsf',sss), shell=True)
		a2 = subprocess.check_output("cat %s | grep -A6 %s | sed -n '5p'| awk '{print $2}'"%(self.prefix+'.bxsf',sss), shell=True)
		a3 = subprocess.check_output("cat %s | grep -A6 %s| sed -n '5p'| awk '{print $3}'"%(self.prefix+'.bxsf',sss), shell=True)
	
		b1 = subprocess.check_output("cat %s | grep -A6 '%s'| sed -n '6p'| awk '{print $1}'"%(self.prefix+'.bxsf',sss), shell=True)
		b2 = subprocess.check_output("cat %s | grep -A6 '%s'| sed -n '6p'| awk '{print $2}'"%(self.prefix+'.bxsf',sss), shell=True)
		b3 = subprocess.check_output("cat %s | grep -A6 '%s'| sed -n '6p'| awk '{print $3}'"%(self.prefix+'.bxsf',sss), shell=True)

		c1 = subprocess.check_output("cat %s | grep -A6 %s| sed -n '7p'| awk '{print $1}'"%(self.prefix+'.bxsf',sss), shell=True)
		c2 = subprocess.check_output("cat %s | grep -A6 %s| sed -n '7p'| awk '{print $2}'"%(self.prefix+'.bxsf',sss), shell=True)
		c3 = subprocess.check_output("cat %s | grep -A6 %s| sed -n '7p'| awk '{print $3}'"%(self.prefix+'.bxsf',sss), shell=True)

		a1, a2, a3 = float(a1.decode('utf-8')), float(a2.decode('utf-8')), float(a3.decode('utf-8'))
		b1, b2, b3 = float(b1.decode('utf-8')), float(b2.decode('utf-8')), float(b3.decode('utf-8'))
		c1, c2, c3 = float(c1.decode('utf-8')), float(c2.decode('utf-8')), float(c3.decode('utf-8'))

		a = asarray([a1, a2, a3])
		b = asarray([b1, b2, b3])
		c = asarray([c1, c2, c3])
		return(a,b,c)
		


class energy_of_bands(data_extract):
	def __init__(self, prefix, tag, ks:np.ndarray , nband_f):
		super().__init__(prefix,tag)
		#self.prefix = prefix
		self.ks = ks
		self.nband_f = nband_f
	
	def kmesh_points(self):
		ks = self.ks				
		return(np.prod(ks))  
	
	def bands_energy(self):
		ks = self.ks  	
		#print (self.prefix)
		if self.tag == 1:
			kgrid_pts = np.prod(ks)
			nband_f = self.nband_f
			band_start = subprocess.check_output("grep -n -m1 'BAND: ' %s | cut -f1 -d: "%(self.prefix+'.bxsf'), shell=True)
			band_start = int(band_start.decode('utf-8'))
			band_en = zeros((nband_f, kgrid_pts))
			row_no = int(kgrid_pts/6.0)
		if self.tag == 2:
			kgrid_pts = np.prod(ks)
			nband_f = self.nband_f
			band_start = subprocess.check_output("grep -n -m1 'BAND: ' %s | cut -f1 -d: "%(self.prefix+'.bxsf'), shell=True)
			band_start = int(band_start.decode('utf-8'))
			band_en = zeros((nband_f, kgrid_pts))
			row_no = int(kgrid_pts)-1

		for i in range(nband_f):
			start = band_start + i*(row_no + 2) + 1
			end = start + row_no
			prefix_temp = self.prefix + '_BAND_' + str(i+1) + '_temp'
			subprocess.check_output("sed -n '%d,%dp' %s | tr -d '\n' > %s"%(start, end, self.prefix+'.bxsf', prefix_temp), shell=True)
			band_en[i][:] = loadtxt(prefix_temp)	
		subprocess.check_output("rm *_temp", shell=True) # removes all temp files
		return band_en
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

name = input(str("\nEnter the prefix: "))

tag = int(input("\nBxsf file Created using QE(1) or wannier90(2): "))

test = data_extract(name, tag)
cent = test.center()
nb = test.nbands()
kx, ky, kz = test.kmesh()
a, b, c = test.lattice_vectors()
en = energy_of_bands(name, tag , (kx,ky,kz), nb).bands_energy()



#_________________________________Determine faces and edges of the BZ_____________________________________
BZ = pyprocar.fermisurface3d.brillouin_zone.BrillouinZone((a,b,c)).wigner_seitz()
f = []
for i in range(len(BZ[1])):
	f.append([len(BZ[1][i])])
	f.append(BZ[1][i])
faces = np.concatenate(asarray(f))
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

#__________________________________Important Functions_________________________________________________
def G(n1,n2,n3):
	return (n1*a[0]+n2*b[0]+n3*c[0], n1*a[1]+n2*b[1]+n3*c[1], n1*a[2]+n2*b[2]+n3*c[2])


Gset = [G(1,0,0),G(-1,0,0),G(0,0,1),G(0,0,-1),G(0,1,0),G(0,-1,0),G(1,1,0),G(-1,-1,0),G(1,-1,0),
G(-1,1,0),G(1,0,1),G(-1,0,-1),G(1,0,-1),G(-1,0,1),G(0,1,1),G(0,-1,-1),G(0,1,-1),G(0,-1,1),
G(1,1,1),G(-1,-1,-1),G(1,1,-1),G(-1,-1,1),G(1,-1,1),G(-1,1,-1),G(-1,1,1),G(1,-1,-1)]

Gset_r = [G(1,0,0),G(-1,0,0),G(0,0,1),G(0,0,-1),G(0,1,0),G(0,-1,0),G(1,1,0),G(-1,-1,0),G(1,-1,0),
G(-1,1,0),G(1,0,1),G(-1,0,-1),G(1,0,-1),G(-1,0,1),G(0,1,1),G(0,-1,-1),G(0,1,-1),G(0,-1,1),
G(1,1,1),G(-1,-1,-1),G(1,1,-1),G(-1,-1,1),G(1,-1,1),G(-1,1,-1),G(-1,1,1),G(1,-1,-1)]

Gset_r.reverse()

def locate(k):
	fail=0
	for GG in Gset:
		if norm(k) > norm(subtract(k,GG)):
			fail=fail+1
	if fail==0:
		return 0 # Inside BZ
	else:
		return 1 # Outside BZ

def cuc2bz(klist):
	outk = []
	for kk in klist:
		if locate(kk) == 0:
			outk.append(kk)
		else:
			for GG in Gset_r:
				k_test = subtract(kk, GG)
				if locate(k_test) == 0:
					outk.append(k_test)
					break
	return outk
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


#_____________________________Determine FermiSurface_____________________________________________________
x=[]
y=[]
z=[]

for n1 in range(kx):
	for n2 in range(ky):
		for n3 in range(kz):
			x.append(n1*a[0]/(kx-1) + n2*b[0]/(ky-1) + n3*c[0]/(kz-1))
			y.append(n1*a[1]/(kx-1) + n2*b[1]/(ky-1) + n3*c[1]/(kz-1))
			z.append(n1*a[2]/(kx-1) + n2*b[2]/(ky-1) + n3*c[2]/(kz-1))

points =  np.array([x,y,z]).T		#the data is in the format (length, 3)


#____________________________Fermi Energy___________________________


isovalue = test.fermi_energy()	#Fermi Energy

#____________________________________Start Plotting_____________________________

band_no = int(input("\nEnter the band number (0-%d): "%(nb-1)))

inter_n = float(input("\nEnter the desired interpolation factor: "))

visualize = int(input("\nVisualize, yes(1) or no(2): "))

kxc = complex(0,round(inter_n*kx))
kyc = complex(0,round(inter_n*ky))
kzc = complex(0,round(inter_n*kz))

xa, ya, za = mgrid[min(x):max(x):kxc, min(y):max(y):kyc, min(z):max(z):kzc]

dx = 1.0/kx/inter_n
dy = 1.0/ky/inter_n
dz = 1.0/kz/inter_n

band_int = LinearNDInterpolator((x, y, z), en[band_no])

#________________________USING SCIpy______________________________
#'''

en_band = band_int(xa,ya,za)

vertsf, facesf, normalsf, valuesf = marching_cubes(en_band, level = isovalue, spacing = (dx, dy, dz))

verts_fil = []

for i in range(len(vertsf[:,0])):
	if np.isnan(vertsf[i,0]) or np.isnan(vertsf[i,1]) or np.isnan(vertsf[i,2]):
		pass
	else:
		verts_fil.append(vertsf[i])

verts_fil = asarray(verts_fil)

verts_fil[:,0] = (max(x)-min(x))*verts_fil[:,0] + min(x)
verts_fil[:,1] = (max(y)-min(y))*verts_fil[:,1] + min(y)
verts_fil[:,2] = (max(z)-min(z))*verts_fil[:,2] + min(z)

bzpts = np.array(cuc2bz(verts_fil))


np.savetxt('pts_band_%d_inter_f_%3.2f.dat'%(band_no,inter_n), [p for p in zip(bzpts[:,0],bzpts[:,1],bzpts[:,2])], delimiter='\t', fmt='%s')

#'''
if visualize ==1:
	#_____Start using pyvista_________
	pv.set_plot_theme('document')
	p = pv.Plotter(notebook=0)
	p.background_color='white'
	p.window_size
	#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
	p.add_points(bzpts, render_points_as_spheres=True, point_size=5.0, color ='red')
	#____________________________________BZ cage___________________________________________________
	surf = pv.PolyData(BZ[0],faces, n_faces = len(BZ[1]))
	p.add_mesh(surf, style='wireframe',show_edges=True, line_width=5)
	#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

	p.show_grid()
	p.link_views()
	p.view_isometric()
	p.show(screenshot='fermisurface.png')

print(datetime.now()-start_time)
