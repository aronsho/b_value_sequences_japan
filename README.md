# b_value_japan

With this repository, you can recreate all the results of the article.

Software needed:
- Python>=3.10
- Python packages: rft1d, seismostats

Files needed:
The only additional file that is needed is an earthquake catalog of japan. This catalog should be a cvs with the following collumns (case sensitive): time, latitude, longitude, depth, magnitude, event_type.
The csv file is read as dataframe (df). Some notes:
- time: includes data and time, has to be such that this works: pd.to_datetime(df["time"], format="mixed")
- event_type: only the earthquakes that are 'earthquake' are considered
- latitude/ longitude: in degree with decimal points (not minutes/seconds)
- depth: in km

Recreate:
Run the following scripts, in order. They are designed to be run with a slurm workload manager. The way to run each script is commented at the top of each document.
- 0_parameters.py (set the parameters that will be used later) 
- 1_prepare_catalogs.py (filter the catalogs to the given buffer)
- 2a_b_significant.py (estimate the general significance of variation)
- 2b_parameter_variation (estimate the b-values of sequences with different parameters)
- 2c_parameter_variation_agg.py (aggregates the results of all variations)
- 2d_map_high.py (estimate maps with different length scales)
- 2d_map_low.py (estimate maps with different length scales)
- 2e_map_agg.py (aggregate results of 2c)
- 2f_map_full.py (for a chosen length-scale, estimate the full grid of b-values)
- 2g_map_full_agg.py (aggregate results fro 2e)
- 2h_sequence_plots.py (create overview plots of all sequences, for supplement)
- 2i_synthetic_shuffle.py (synthetic tests)
- 2j_synthetic shuffle_agg.py (aggregates synthetic tests)
- 3_all_plots.ipynb (jupyternotebook that recreates all plots shown in the article including the supplement)
