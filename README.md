# linear_fft
The relevant files are in the `odyssey` folder. 
`run_fouriernetwork_odyssey.py` is the main file which trains the networks. Multiple runs using SLURM are controlled by `launch_fouriernetwork.py`, which calls `run_odyssey_array.py` and `run_fouriernetwork_odyssey.py`.
