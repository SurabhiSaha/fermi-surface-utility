import pyprocar
from numpy import zeros, loadtxt, asarray, reshape, subtract, array, dot, add
import numpy as np
from numpy.linalg import norm
from datetime import datetime
import pyvista as pv

#__________________________________
import warnings                   #  
warnings.filterwarnings("ignore") #
#_________________________________#

#-----------------------------------------------------------------------------
start_time = datetime.now()

#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
print("-----------------------------------------------")
print("This code will create a n*n*n size k-grid...")
print("-----------------------------------------------")
nx = int(input("\nNumber of points along kx (n): "))
ny = nx
nz = nx
print("\n")
print("Now enter the reciprocal lattice vectors a,b,c...")
a0 = float(input("\nEnter a[0]: "))
a1 = float(input("\nEnter a[1]: "))
a2 = float(input("\nEnter a[2]: "))
b0 = float(input("\nEnter b[0]: "))
b1 = float(input("\nEnter b[1]: "))
b2 = float(input("\nEnter b[2]: "))
c0 = float(input("\nEnter c[0]: "))
c1 = float(input("\nEnter c[1]: "))
c2 = float(input("\nEnter c[2]: "))

a = asarray([a0,a1,a2])
b = asarray([b0,b1,b2])
c = asarray([c0,c1,c2])
#a = asarray([-1.0, -1.0,  1.0])
#b = asarray([ 1.0,  1.0,  1.0])
#c = asarray([-1.0,  1.0, -1.0])


#_____Start using pyvista_________
pv.set_plot_theme('document')    #
p = pv.Plotter(notebook=0)       #
p.background_color='white'       #
p.window_size                    #
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


#________________________Determine faces and edges of the BZ_______________________
BZ = pyprocar.fermisurface3d.brillouin_zone.BrillouinZone((a,b,c)).wigner_seitz() #
f = []                                                                            #
for i in range(len(BZ[1])):                                                       #
	f.append([len(BZ[1][i])])                                                 #
	f.append(BZ[1][i])                                                        #
faces = np.concatenate(asarray(f))                                                #
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^



#__________________________________Important Functions____________________________________________
def G(n1,n2,n3):                                                                                 #
	return (n1*a[0]+n2*b[0]+n3*c[0], n1*a[1]+n2*b[1]+n3*c[1], n1*a[2]+n2*b[2]+n3*c[2])       #
                                                                                                 #
Gset = [G(1,0,0),G(-1,0,0),G(0,0,1),G(0,0,-1),G(0,1,0),G(0,-1,0),G(1,1,0),G(-1,-1,0),G(1,-1,0),  #
        G(-1,1,0),G(1,0,1),G(-1,0,-1),G(1,0,-1),G(-1,0,1),G(0,1,1),G(0,-1,-1),G(0,1,-1),         #
        G(0,-1,1),G(1,1,1),G(-1,-1,-1),G(1,1,-1),G(-1,-1,1),G(1,-1,1),G(-1,1,-1),G(-1,1,1),      #
        G(1,-1,-1)]                                                                              #
                                                                                                 #
def locate(k):                                                                                   #
  fail=0                                                                                         #
  for GG in Gset:                                                                                #
    if norm(k) > norm(subtract(k,GG)):                                                           #
      fail=fail+1                                                                                #
  if fail==0:                                                                                    #
    return 0      # Inside BZ                                                                    #
  else:                                                                                          #
    return 1      # Outside BZ                                                                   #
                                                                                                 #
def cuc2bz(klist):                                                                               #
  outk = []                                                                                      #
  for kk in klist:                                                                               #
    if locate(kk) == 0:                                                                          #
      outk.append(kk)                                                                            #
    else:                                                                                        #
      for GG in Gset:                                                                            #
        k_test = subtract(kk, GG)                                                                #
        if locate(k_test) == 0:                                                                  #
          outk.append(k_test)                                                                    #
          break                                                                                  #
  return outk                                                                                    #
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


#_______________________________Points Inside BZ_________________________________
x=[]                                                                            #
y=[]                                                                            #
z=[]                                                                            #
for n1 in range(nx):                                                            #
  for n2 in range(ny):                                                          #
    for n3 in range(nz):                                                        #
      x.append(n1*a[0]/(nx-1) + n2*b[0]/(ny-1) + n3*c[0]/(nz-1))                #
      y.append(n1*a[1]/(nx-1) + n2*b[1]/(ny-1) + n3*c[1]/(nz-1))                #
      z.append(n1*a[2]/(nx-1) + n2*b[2]/(ny-1) + n3*c[2]/(nz-1))                #
                                                                                #
points = np.array([x,y,z]).T		#the data is in the format (length, 3)  #
                                                                                #
bz = np.array(cuc2bz(points))                                                   #
                                                                                #
p.add_points(bz, render_points_as_spheres=True, point_size=10.0, color ='red')  #
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

np.savetxt('pts_inside_bz.dat', [p for p in zip(bz[:,0],bz[:,1],bz[:,2])], delimiter='\t', fmt='%s')

#___________________________BZ cage________________________________
surf = pv.PolyData(BZ[0],faces, n_faces = len(BZ[1]))             #
p.add_mesh(surf, style='wireframe',show_edges=True, line_width=5) #
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

p.show_grid()
p.link_views()
p.view_isometric()
p.show(screenshot='BZ.png')

print(datetime.now()-start_time)
