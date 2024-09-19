from mayavi import mlab
import numpy as np

    
    
def visualize_3d_volume_def(data, size=(800, 800)):
    #Visualize a 3D volume using Mayavi 
    mlab.figure(size=size)
    src = mlab.pipeline.scalar_field(data)
    mlab.pipeline.volume(src, vmin=np.min(data), vmax=np.max(data))
    mlab.colorbar()
    mlab.show()    
    
    

    