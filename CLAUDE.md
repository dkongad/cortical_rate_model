You are a scientific software developer, always document your code in a clear manner, check for consistency, follow principles of object-oriented-programming. 

You have three previous projects that are written by three scientists (2D_network_StructConn-main, Influence_maping-main, and models_cortical_development-main), and they all make use of firing rate model of the cortex, but with different vairations. 

Please write a easy-to-read, easy-to-maintain pakage of the firing rate model, do not modify any files in the three folders of previous projects, please do not import anything from those folders. The model should work even after deleting all three folders. Please use naming conventions similar to the project Influence_mapping-main. Use pytorch instead of tensorflow. Use parent/child classes if needed. 

The model should combine features in all three projects. The user should be able to set up model in 1D-ring or 2D-cortical sheet (with periodic boundary cnoditions or not) with desired size. The connectivity can be set up with different options (mainly the Mexican-hat type of connectivity, with different heterogeneity mentioned in those projects. It should also should include other types mentioned in the projects, like the random network in Influence_maping). The model can be linear or non-linear, compatible with different nonlinear functions. The model should come with functions allow getting parameters. The model should have different inputs implemented as in those projects, and can simulate(or analytically calculate if possible) responses. There should be saving and loading functions in the model but with a different file setting up paths. There should be some pre-set(default) parameter sets that the users can just load. here should be some helper plotting code to visualize things. 

There are a lot of non-relavant code in these project folders that you should not include. Do not include data analysis code (such as GLM stuff in Influence_maping-main). Do not include code about the analysis done to the model results (For example, there are subfolders called 'analysis' or 'figs'). 

You should include a file detailing the requirement for environment, like packages and versions. an instruction for using: All the possible parameters/choices the user can make and what they mean. another instruction for working on the code: code organization. 

