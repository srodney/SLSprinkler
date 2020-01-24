from addict import Dict

cfg = Dict()

# exposure_time, num_exposures, ccd_gain,m magnitude_zero_point do not matter for this validation study because the lensed image pixel values become normalized between 0 and 1.

cfg.instrument = Dict(
              pixel_scale=0.01, # scale (in arcseonds) of pixels
              ccd_gain=100.0, # electrons/ADU (analog-to-digital unit). A gain of 8 means that the camera digitizes the CCD signal so that each ADU corresponds to 8 photoelectrons.
              )

cfg.bandpass = Dict(
                magnitude_zero_point=30.0, # (effectively, the throuput) magnitude in which 1 count per second per arcsecond square is registered (in ADUs)
                )

cfg.observation = Dict(
                  exposure_time=9600.0, # exposure time per image (in seconds)
                  num_exposures=10, # number of exposures that are combined
                  background_noise=0.0, # overrides exposure_time, sky_brightness, read_noise, num_exposures
                  )

cfg.psf = Dict(
           type='NONE', # string, type of PSF
           )

cfg.numerics = Dict(
                supersampling_factor=1)

cfg.image = Dict(
             num_pix=1000, # cutout pixel size
             inverse=False, # if True, coord sys is ra to the left, if False, to the right 
             )