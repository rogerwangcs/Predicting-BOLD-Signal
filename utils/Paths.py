import os

machine = 'csci'

forrest_path = ''
project_path = ''
if machine == 'csci':
    forrest_path = '/home/wangbrh/BrainProject/datasets/forrest'
    project_path = '/home/wangbrh/BrainProject'
else:
    forrest_path = '/gsfs0/data/anzellos/data/forrest'
    project_path = '/gsfs0/data/wangbrh/BrainProject'

forrest_bids_path = os.path.join(forrest_path,'forrest_bids')
forrest_derivatives_path = os.path.join(forrest_path,'derivatives')

datasets_path = os.path.join(project_path, 'datasets')
models_path = os.path.join(project_path, 'savedModels')

objects_dataset_path = os.path.join(datasets_path, '101Objects')

def movie_run_path(run):
    return os.path.join(forrest_path, 'movie', 'small_fg_av_ger_seg{}.mkv'.format(run))

def movie_run_snapshots_path(run):
    return os.path.join(forrest_path, 'movie_snapshots/run-{}'.format(run))

def ffa_mask_path(sub): return os.path.join(forrest_derivatives_path, 'fmriprep/sub-{:02d}_complete/sub-{:02d}_ROIs/rFFA_final_mask_sub-{:02d}_bin.nii.gz'.format(sub, sub, sub))

def func_path(sub, run): return os.path.join(forrest_bids_path,'sub-{:02d}/ses-movie/func/sub-{:02d}_ses-movie_task-movie_run-{}_bold.nii.gz').format(sub, sub, run + 1)

def encoded_features(model, sub, run=None):
    if run is not None:
        return os.path.join(forrest_path, 'encoded_features/{}/sub-{:02d}/features-{}.pt'.format(model, sub, run))
    else:
        return os.path.join(forrest_path, 'encoded_features/{}/sub-{:02d}'.format(model, sub))

def testPaths():
    print(os.path.isdir(forrest_path))
    print(os.path.isdir(forrest_bids_path))
    print(os.path.isdir(forrest_derivatives_path))
    print(os.path.isdir(project_path))
    print(os.path.isdir(datasets_path))
    print(os.path.isdir(models_path))
    
    print(os.path.isfile(ffa_mask_path(1)))
    print(os.path.isfile(func_path(1,1)))
    print(os.path.isfile(features_path('SimpleConvAE', 1,1)))
    
# testPaths()