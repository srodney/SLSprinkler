import os
import sys
import argparse
from importlib import import_module
import numpy as np
import pandas as pd
# Lenstronomy modules
import lenstronomy
print("Lenstronomy path being used: {:s}".format(lenstronomy.__path__[0]))
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver
from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.SimulationAPI.data_api import DataAPI
import lenstronomy.Util.util as util
import lenstronomy.Util.param_util as param_util
from baobab.sim_utils import instantiate_PSF_models, get_PSF_model, generate_image
import matplotlib.pyplot as plt

#/home/jwp/stage/sl/SLSprinkler/img_pos_validation/lsst_cfg.py
#4077543

def parse_args():
    """Parse command-line arguments

    """
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='config file path')
    parser.add_argument('sys_id', type=int, help='ID of system')
    args = parser.parse_args()
    return args

def get_cfg_from_file(user_cfg_path):
    """Alternative constructor that accepts the path to the user-defined configuration python file
    
    Parameters
    ----------
    user_cfg_path : str or os.path object
        path to the user-defined configuration python file
    
    """
    dirname, filename = os.path.split(os.path.abspath(user_cfg_path))
    module_name, ext = os.path.splitext(filename)
    sys.path.insert(0, dirname)
    #user_cfg_file = map(__import__, module_name)
    #user_cfg = getattr(user_cfg_file, 'cfg')
    user_cfg_script = import_module(module_name)
    user_cfg = getattr(user_cfg_script, 'cfg')
    return user_cfg

def main():
    hostgal_csv_path = '/home/jwp/stage/sl/SLSprinkler/image_verification/lens2_data.csv'
    args = parse_args()
    cfg = get_cfg_from_file(args.config)
    # Instantiate PSF models
    psf_models = instantiate_PSF_models(cfg.psf, cfg.instrument.pixel_scale)
    psf_model = get_PSF_model(psf_models, len(psf_models), current_idx=0)
    # Which components end up on the image
    cfg.components = ['lens_mass', 'external_shear', 'src_light']
    # Instantiate density models
    kwargs_model = dict(
                    lens_model_list=['SIE', 'SHEAR_GAMMA_PSI'],
                    source_light_model_list=['SERSIC_ELLIPSE'],
                    )       
    lens_mass_model = LensModel(lens_model_list=kwargs_model['lens_model_list'])
    src_light_model = LightModel(light_model_list=kwargs_model['source_light_model_list'])
    lens_eq_solver = LensEquationSolver(lens_mass_model)
    # Detector and observation conditions
    kwargs_detector = util.merge_dicts(cfg.instrument, cfg.bandpass, cfg.observation)
    kwargs_detector.update(psf_type=cfg.psf.type)
    data_api = DataAPI(cfg.image.num_pix, **kwargs_detector)
    # Gather input params
    # Lens mass
    q, phi = 0.7735461, -117.194
    phi = np.deg2rad(phi)
    e1, e2 = param_util.phi_q2_ellipticity(phi, q)
    cfg.lens_mass = dict(
                          center_x=0.0,
                          center_y=0.0,
                          #s_scale=0.0,
                          theta_E=1.426983229,
                          e1=e1,
                          e2=e2
                          )
    # External shear
    psi_ext = 156.991
    psi_ext = np.deg2rad(psi_ext)
    cfg.external_shear = dict(
                              gamma_ext=0.03576569,
                              psi_ext=psi_ext
                              )
    # Source light
    src_light_df = pd.read_csv(hostgal_csv_path, index_col=None)
    src_light_df = src_light_df[src_light_df['lens_cat_sys_id'] == args.sys_id].T.squeeze()
    bulge_or_disk = 'bulge'
    
    src_light_df['src_center_x'] = (src_light_df['ra_host'] - src_light_df['ra_lens'])*3600.0 # arcsec
    src_light_df['src_center_y'] = (src_light_df['dec_host'] - src_light_df['dec_lens'])*3600.0 # arcsec
    bandpass = 'r'
    magnitude_src = src_light_df['magnorm_{:s}_{:s}'.format(bulge_or_disk, bandpass)]
    #magnitude_src = 15.0
    n_sersic = src_light_df['sindex_{:s}'.format(bulge_or_disk)]
    R_sersic = (src_light_df['major_axis_{:s}'.format(bulge_or_disk)]*src_light_df['minor_axis_{:s}'.format(bulge_or_disk)])**0.5
    q_src = src_light_df['minor_axis_{:s}'.format(bulge_or_disk)]/src_light_df['major_axis_{:s}'.format(bulge_or_disk)]
    phi_src = np.deg2rad(src_light_df['position_angle'])
    e1_src, e2_src = param_util.phi_q2_ellipticity(phi_src, q_src)

    cfg.src_light = dict(
                         magnitude=magnitude_src,
                         n_sersic=n_sersic,
                         R_sersic=R_sersic,
                         center_x=src_light_df['src_center_x'],
                         center_y=src_light_df['src_center_y'],
                         e1=e1_src,
                         e2=e2_src,
                         )
    params = dict(
                  lens_mass=cfg.lens_mass,
                  external_shear=cfg.external_shear,
                  src_light=cfg.src_light,
                  )
    print(params)
    # Generate the image
    img, img_features = generate_image(params, psf_model, data_api, lens_mass_model, src_light_model, lens_eq_solver, cfg.instrument.pixel_scale, cfg.image.num_pix, cfg.components, cfg.numerics, min_magnification=0.0, lens_light_model=None, ps_model=None)
    # Save image file
    #img_filename = 'validation_{0:07d}.npy'.format(args.sys_id)
    #np.save(img_filename, img)
    img = (img - np.min(img))/(np.max(img) - np.min(img))
    np.save('img_features.npy', img_features)
    np.save('validation_{0:07d}.npy'.format(args.sys_id), img)
    plt.imshow(img, origin='lower')
    plt.colorbar()
    plt.savefig('validation_{0:07d}.png'.format(args.sys_id))
    plt.close()

if __name__ == '__main__':
    main()
