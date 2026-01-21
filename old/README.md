# Inference of the Hurst parameter

File DataHandler.py contains all classes dealing with the different files involved in the estimation. Here are the conventions used:
-


dates.py contains special dates that need to be treated carefuylly: two variables. FOMC_announcement and trading halt (explain)


estimator_H.py contains several functions used for the estimation of H. Here are the mail functions:
- Phi_Hl

files_make_daily.py accessory file that takes an input_folder and an output_folder and split csv containing 1 year of date into csv containing 1 day of data
