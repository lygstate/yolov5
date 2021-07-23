set PATH=%UserProfile%/.conda/envs/test-env;%UserProfile%/.conda/envs/test-env/Library/mingw-w64/bin;%UserProfile%/.conda/envs/test-env/Library/usr/bin;%UserProfile%/.conda/envs/test-env/Library/bin;%UserProfile%/.conda/envs/test-env/Scripts;%UserProfile%/.conda/envs/test-env/bin;C:/ProgramData/Anaconda3/condabin;%PATH%
call conda activate test-env
python calculate_small.py
