"""Top-down sanity images of field_maps.npz -> field/map_check.png"""
import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
D = '/workspace/frc/field'
M = np.load(f'{D}/field_maps.npz'); c = M['clearance']; f = M['floor_z']; z = M['zone']; L, W = M['size']
fig, ax = plt.subplots(3, 1, figsize=(15, 21))
img = np.where(np.isfinite(c), np.clip(c, 0, 1.5), 1.6)
a = ax[0].imshow(img.T, origin='lower', extent=[0, L, 0, W], cmap='turbo', vmin=0, vmax=1.6, interpolation='nearest')
ax[0].set_title('vertical clearance above floor (m); dark red (1.6) = open, 0 = blocked; trench arms ~0.565')
plt.colorbar(a, ax=ax[0], fraction=0.025)
b = ax[1].imshow(f.T, origin='lower', extent=[0, L, 0, W], cmap='magma', vmin=0, vmax=0.2, interpolation='nearest')
ax[1].set_title('floor height (m) (clipped to 0.2)'); plt.colorbar(b, ax=ax[1], fraction=0.025)
zc = ListedColormap(['#4060ff', '#ff4040', '#cccccc', '#9fb0ff', '#ffa0a0'])
ax[2].imshow(z.T, origin='lower', extent=[0, L, 0, W], cmap=zc, vmin=-0.5, vmax=4.5, interpolation='nearest')
ax[2].contour(np.isfinite(c).T & (np.nan_to_num(c, posinf=9).T < 0.3), levels=[0.5], extent=[0, L, 0, W], colors='k', linewidths=0.5)
ax[2].set_title('zones: blue alliance | blue barrier band | neutral | red barrier band | red alliance (black = blocked <0.3 m)')
for a_ in ax:
    a_.set_xlabel('x (m) from blue wall'); a_.set_ylabel('y (m)'); a_.set_xticks(np.arange(0, 17, 1)); a_.grid(alpha=0.25)
plt.tight_layout(); plt.savefig(f'{D}/map_check.png', dpi=60)
