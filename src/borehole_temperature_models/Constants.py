yr_to_s: float                              = 365.25*24*60*60 # Seconds in a year
dt_years: int                               = 1 # TODO: Is this a param?
dt: float                                   = dt_years * yr_to_s

c_i: float                                  = 2e3                           # Head capacity of ice
c_r: float                                  = 7.9e2                         # Heat capacity of basal rock
k_i: float                                  = 2.3                           # Thermal conductivity of ice
k_r: float                                  = 2.8                           # Thermal conductivity of basal rock
rho_i: float                                = 918                           # Density of ice
rho_r: float                                = 2750                          # Density of basal rock
alpha_i: float                              = k_i / rho_i / c_i             # Alpha for ice
alpha_r: float                              = k_r / rho_r / c_r             # Alpha for basal rock
n: float                                    = 3                             # Glen's flow law exponent
g: float                                    = 9.81                          # Gravitational acceleration
A: float                                    = 2e-16                         # Rheological constant determining soft ice as a function of temperature
