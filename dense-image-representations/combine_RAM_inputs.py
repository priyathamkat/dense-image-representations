import numpy as np
import glob

files_list = glob.glob('/cmlscratch/snawathe/dense-image-representations/ram_dataset/*')
data_list = []

for i, filename in enumerate(files_list):
	contents = np.load(filename, allow_pickle=True)
	data_list.append({**contents})
	contents.close()
	if (i+1) % 1000 == 0:
		print(f"{i+1=}")

np.savez(f'/cmlscratch/snawathe/dense-image-representations/ram_dataset_full.npz', data_list)
