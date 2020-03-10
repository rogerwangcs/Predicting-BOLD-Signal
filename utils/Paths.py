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
objects_dataset_path = os.path.join(datasets_path, '101Objects')
models_path = os.path.join(project_path, 'savedModels')
results_path = os.path.join(project_path, 'results')

def movie_run_path(run):
    return os.path.join(forrest_path, 'movie', 'small_fg_av_ger_seg{}.mkv'.format(run))

def movie_run_snapshots_path(run):
    return os.path.join(forrest_path, 'movie_snapshots/run-{}'.format(run))

def ffa_mask_path(sub): return os.path.join(forrest_derivatives_path, 'fmriprep/sub-{:02d}_complete/sub-{:02d}_ROIs/rFFA_final_mask_sub-{:02d}_bin.nii.gz'.format(sub, sub, sub))

def func_path(sub, run): return os.path.join(forrest_derivatives_path,'fmriprep/sub-{:02d}_complete/ses-movie/func/sub-{:02d}_ses-movie_task-movie_run-{}_bold_space-MNI152NLin2009cAsym_preproc.nii.gz').format(sub, sub, run + 1)

def encoded_features(model, run=None, save_name=''):
    if run is not None:
        return os.path.join(forrest_path, 'encoded_features/{}/features-{}-{}.pt'.format(model, run, save_name))
    else:
        return os.path.join(forrest_path, 'encoded_features/{}/'.format(model))

def testPaths():
    print(os.path.isdir(forrest_path), end=' ')
    print(os.path.isdir(forrest_bids_path), end=' ')
    print(os.path.isdir(forrest_derivatives_path), end=' ')
    print(os.path.isdir(project_path), end=' ')
    print(os.path.isdir(datasets_path), end=' ')
    print(os.path.isdir(models_path), end=' ')
    
    print(os.path.isfile(ffa_mask_path(1)), end=' ')
    print(os.path.isfile(func_path(1,1)), end=' ')
    print(os.path.isfile(encoded_features('SimpleConvAE',run=1)), end=' ')
    
testPaths()