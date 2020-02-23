import os

forrest_path = '/gsfs0/data/anzellos/data/forrest'
forrest_bids_path = os.path.join(forrest_path,'forrest_bids')
forrest_derivatives_path = os.path.join(forrest_path,'derivatives')

project_path = '/gsfs0/data/wangbrh/BrainProject/'
datasets_path = os.path.join(project_path, 'datasets')
models_path = os.path.join(project_path, 'savedModels')

def ffa_mask_path(sub): return os.path.join(forrest_derivatives_path, 'fmriprep/sub-{:02d}_complete/sub-{:02d}_ROIs/rFFA_final_mask_sub-{:02d}_bin.nii.gz'.format(sub, sub, sub))

def func_path(sub, run): return os.path.join(forrest_bids_path,'sub-{:02d}/ses-movie/func/sub-{:02d}_ses-movie_task-movie_run-{}_bold.nii.gz').format(sub, sub, run)

def features_path(sub, run): return os.path.join(
    forrest_path, 'encoded_features/features-{}.pt'.format(run))

def features_path(sub, run): return os.path.join(
    forrest_path, 'encoded_features/sub-{:02d}/run-{:02d}.pt'.format(sub, run))

def testPaths():
    print(os.path.isdir(forrest_path))
    print(os.path.isdir(forrest_bids_path))
    print(os.path.isdir(forrest_derivatives_path))
    print(os.path.isdir(project_path))
    print(os.path.isdir(datasets_path))
    print(os.path.isdir(models_path))
    
    print(os.path.isfile(ffa_mask_path(1)))
    print(os.path.isfile(func_path(1,1)))
    print(os.path.isfile(features_path(1,1)))
    
# print(testPaths())