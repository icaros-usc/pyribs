import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from scipy.spatial import Voronoi

from ribs.archives import CVTArchive
from ribs.visualize import cvt_archive_heatmap_3d

archive = CVTArchive(
    solution_dim=3,
    cells=100,
    ranges=[(-1, 1)] * 3,
    samples=10_000,
)

x = y = z = np.linspace(-1, 1, 30)
xxs, yys, zzs = np.meshgrid(x, y, z)
xxs, yys, zzs = xxs.flatten(), yys.flatten(), zzs.flatten()
archive.add(solution_batch=np.stack((xxs, yys, zzs), axis=1),
            objective_batch=-(xxs**2 + yys**2 + zzs**2),
            measures_batch=np.stack((xxs, yys, zzs), axis=1))

cvt_archive_heatmap_3d(
    archive,
    plot_centroids=False,
    ms=100,
)
plt.show()

#  points = np.array(
#      [
#          [1, 1, 1],
#          [0, 0, 0],
#          [0, 1, 0],
#          [0, 2, 0],
#          [1, 0, 0],
#          [1, 1, 0],
#          [1, 2, 0],
#          [2, 0, 0],
#          [2, 1, 0],
#          [2, 2, 0],
#          [0, 0, 1],
#          [0, 1, 1],
#          [0, 2, 1],
#          [1, 0, 1],
#          [1, 2, 1],
#          [2, 0, 1],
#          [2, 1, 1],
#          [2, 2, 1],
#          [0, 0, 2],
#          [0, 1, 2],
#          [0, 2, 2],
#          [1, 0, 2],
#          [1, 1, 2],
#          [1, 2, 2],
#          [2, 0, 2],
#          [2, 1, 2],
#          [2, 2, 2],
#      ],
#      dtype=np.float32,
#  )

#  vor = Voronoi(points)

#  #  idx_of_111 = 0
#  #  pt_region_of_111 = vor.point_region[idx_of_111]
#  #  pts_of_111 = vor.regions[pt_region_of_111]
#  #  vertices_111 = vor.vertices[pts_of_111]
#  #  print(vertices_111)

#  #  points = vertices_111

#  # Plot
#  fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
#  # TODO: You have to plot all edges that exist between the points
#  # Could use a graph here, but just plotting the edges is easier.
#  #  plt.plot(points[:, 0], points[:, 1], points[:, 2])
#  #  for a, b in zip(points[:-1], points[1:]):

#  for ridge in vor.ridge_vertices:
#      if -1 in ridge:
#          continue
#      p = vor.vertices[ridge]
#      plt.plot(p[:, 0], p[:, 1], p[:, 2], color="black")
#      #  plt.plot(vor.vertices[a], vor.vertices[b])

#  #  ax.plot_trisurf(points[:, 0], points[:, 1], points[:, 2])

#  #  ax.set(xticklabels=[], yticklabels=[], zticklabels=[])

#  plt.show()

#  # 3D CVT:
#  # - Construct a voronoi diagram in 3D
#  # - [x] Construct a triangular mesh -> automatic
#  #   - Where does the shading go? Maybe the cells should be spaced far apart?
#  #     - Can apply some transformation to shrink it here
#  #   - Are surfaces in a 3D voronoi tesselation always convex? -> I sure hope so
#  # - Plot the surfaces with plot_trisurf
