"""Angular extent of the SHiP decay volume as viewed from the centre of the target front face."""
from math import atan, hypot
from sensitivity import Geometry

def corner_angle(geometry=None):
    g = Geometry() if geometry is None else geometry
    return max(atan(hypot(x/2, y/2)/(z-g.target_z))
               for z, x, y in [(g.z_min,g.x_in,g.y_in), (g.z_max,g.x_out,g.y_out)])

THETA_SHIP = corner_angle()
