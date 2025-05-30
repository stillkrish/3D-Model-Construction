import numpy as np
from scipy.spatial import Delaunay

def meshgeneration(pts2L,pts2R,pts3,bvalues,blim,trithresh):

    goodpts = np.nonzero((pts3[0,:]>blim[0])&(pts3[0,:]<blim[1]) & \
                         (pts3[1,:]>blim[2])&(pts3[1,:]<blim[3])& \
                         (pts3[2,:]>blim[4])&(pts3[2,:]<blim[5]))
                         
    pts3 = pts3[:,goodpts[0]]
    pts2L = pts2L[:,goodpts[0]]
    pts2R = pts2R[:,goodpts[0]]
    bvalues = bvalues[:,goodpts[0]]


    Triangles = Delaunay(pts2L.T)
    tri = Triangles.simplices


    def find_neighbors(pindex, triang):
        return triang.vertex_neighbor_vertices[1]\
    [triang.vertex_neighbor_vertices[0][pindex]:triang.vertex_neighbor_vertices[0][pindex+1]]

    for x in range (pts3.shape[1]):
        pts3[:,x] = np.mean(pts3[:,find_neighbors(x,Triangles)],axis=1)

    for x in range (pts3.shape[1]):
        pts3[:,x] = np.mean(pts3[:,find_neighbors(x,Triangles)],axis=1)

    for x in range (pts3.shape[1]):
        pts3[:,x] = np.mean(pts3[:,find_neighbors(x,Triangles)],axis=1)

    d01 = np.sqrt(np.sum(np.power(pts3[:,tri[:,0]]-pts3[:,tri[:,1]],2),axis=0))
    d02 = np.sqrt(np.sum(np.power(pts3[:,tri[:,0]]-pts3[:,tri[:,2]],2),axis=0))
    d12 = np.sqrt(np.sum(np.power(pts3[:,tri[:,1]]-pts3[:,tri[:,2]],2),axis=0))

    goodtri = (d01<trithresh)&(d02<trithresh)&(d12<trithresh)
    tri = tri[goodtri,:]

 
    tokeep=np.unique(tri)
    map = np.zeros(pts3.shape[1])
    pts3=pts3[:,tokeep]
    bvalues = bvalues[:,tokeep]

    map[tokeep] = np.arange(0,tokeep.shape[0])
    tri=map[tri]

    return pts3,tri,bvalues,Triangles



def writeply(X,color,tri,filename):
    """
    Save out a triangulated mesh to a ply file
    
    Parameters
    ----------
    pts3 : 2D numpy.array (dtype=float)
        vertex coordinates shape (3,Nvert)
        
    color : 2D numpy.array (dtype=float)
        vertex colors shape (3,Nvert)
        should be float in range (0..1)
        
    tri : 2D numpy.array (dtype=float)
        triangular faces shape (Ntri,3)
        
    filename : string
        filename to save to    
    """
    f = open(filename,"w");
    f.write('ply\n');
    f.write('format ascii 1.0\n');
    f.write('element vertex %i\n' % X.shape[1]);
    f.write('property float x\n');
    f.write('property float y\n');
    f.write('property float z\n');
    f.write('property uchar red\n');
    f.write('property uchar green\n');
    f.write('property uchar blue\n');
    f.write('element face %d\n' % tri.shape[0]);
    f.write('property list uchar int vertex_indices\n');
    f.write('end_header\n');

    C = (255*color).astype('uint8')
    
    for i in range(X.shape[1]):
        f.write('%f %f %f %i %i %i\n' % (X[0,i],X[1,i],X[2,i],C[0,i],C[1,i],C[2,i]));
    
    for t in range(tri.shape[0]):
        f.write('3 %d %d %d\n' % (tri[t,1],tri[t,0],tri[t,2]))

    f.close();