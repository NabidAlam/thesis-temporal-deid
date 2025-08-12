import os

datasets = ['ted', 'tragic_talkers', 'team_ten']
base_video = 'videos'
base_output = 'output'

for d in datasets:
    os.makedirs(os.path.join(base_video, d), exist_ok=True)
    os.makedirs(os.path.join(base_output, 'tspsam', d), exist_ok=True)
    os.makedirs(os.path.join(base_output, 'samurai', d), exist_ok=True)

# python setup_folders.py
