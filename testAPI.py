import numpy as np
sample={'x': np.random.randint(low=0, high=3, size=(4,3)), 'y': np.random.randint(low=0, high=3, size=(4,3)),
        'uu': np.random.randint(low=0, high=3, size=(4,3)),  'vv': np.random.randint(low=0, high=3, size=(4,3))}
print(sample)
corr_xyz = np.concatenate([
    sample['x'][sample['uu'][0], :2],
    sample['y'][sample['vv'][0], :2]], axis=1)
print(corr_xyz)