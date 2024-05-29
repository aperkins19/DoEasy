import json
import pandas as pd
import math
import numpy as np


############

# Intro

# Assigns Master mix tubes to plate wells depending on if the aqueous and or component master mixes are being modulated.

###########


# import the base_rxn_dict
base_rxn_path = "/app/settings/base_rxn.json"
base_rxn_dict = json.load(open(base_rxn_path, 'r'))


# import the design
experiment_design_df = pd.read_csv(project_path + "/Experiment_Designs/design_real_values.csv", index_col=0)
#get the number of experiments
number_of_experiments = experiment_design_df.shape[0]

##### Compatibility checks

# import experimental design parameters
with open('/app/settings/design_parameters.json') as json_file:
    design_parameters = json.load(json_file)


total_reaction_volume = design_parameters["Reaction_Volume"]

# Check Technical replicates is <= 12 if reaction volume is 20 or Technical replicates is <= 24 if reaction volume is 10
### each 200ul PCR tube can supply:
# 25x 10ul reactions - progress to next every 24 (1x row of nunc plate)
# 12x 20ul reactions - progress to next every 12 (0.5x row of nunc plate)

if design_parameters["Reaction_Volume"] == 20:

    Rxns_Per_MasterMix = 12

elif design_parameters["Reaction_Volume"] == 10:

    Rxns_Per_MasterMix = 24


# import experiment_variables
with open('/app/settings/experiment_variables.json') as json_file:
    experiment_variables = json.load(json_file)
experiment_variables = pd.DataFrame(experiment_variables)



### Determine if aqueous and or Components are modulated.

if ('Aqueous' in experiment_variables["Type"].values) and ('Components' in experiment_variables["Type"].values):
    MasterMixesModulated = "Both"

elif 'Aqueous' in experiment_variables["Type"].values:
    MasterMixesModulated = "Aqueous"

elif 'Components' in experiment_variables["Type"].values:
    MasterMixesModulated = "Components"

else:
    raise Exception("Please check the Type of Reaction Elements to be modulated in experiemnt_variables.json")





#### Assign Master Mixes to experiments
## retrieve variable types:

if MasterMixesModulated == "Both":

    # get the experiment_variables == "Aqueous" and get the list of names
    aqueous_lookup_table = experiment_variables[experiment_variables["Type"] == "Aqueous"]
    aqueous_variables = list(aqueous_lookup_table["Variable"])

    # get the experiment_variables == "Components" and get the list of names
    Components_lookup_table = experiment_variables[experiment_variables["Type"] == "Components"]
    Components_variables = list(Components_lookup_table["Variable"])

elif MasterMixesModulated == "Aqueous":

    # get the experiment_variables == "Aqueous" and get the list of names
    aqueous_lookup_table = experiment_variables[experiment_variables["Type"] == "Aqueous"]
    aqueous_variables = list(aqueous_lookup_table["Variable"])

elif MasterMixesModulated == "Components":

    # get the experiment_variables == "Components" and get the list of names
    Components_lookup_table = experiment_variables[experiment_variables["Type"] == "Components"]
    Components_variables = list(Components_lookup_table["Variable"])

else:
    raise Exception("MasterMixesModulated is neither Aqueous, Components or Both: MasterMixesModulated = " + MasterMixesModulated)





############ Add master mix tubes to experimental design 

print(" ")
print("Assigning reagent sources and wells...")
print(" ")

def AllocatingMasterMixTubesBasedOnComparision(master_mix_df, Trimmed_plate_df, variables, col_name, plate_df):

    """
    1. Takes in the master_mix_df and the Trimmed_plate_df
    2. Converts both rows of each to dicts for comparision
    3. if it pings then adds the master mix tube to the list
    4. Assigns list as col_name column on the plate df
    """

    # initialise list of zeros of length plate df rows to be populated later.
    MasterMix_Tube_List = list(np.zeros(plate_df.shape[0]))

    # iterate over the rows
    for mastermix_idx, mastermix_row in master_mix_df.iterrows():

        # extract and segregate key info
        # Have to convert series to dictionaries for comparison bc pandas has thrown a tantrum.
        MasterMixElements = mastermix_row[variables].to_dict()
        Tubes = mastermix_row["Tubes"]

        # initialise counter
        mastermix_allocation_counter = 0
        Working_Tube_idx = 0


        for run_idx, run_row in Trimmed_plate_df.iterrows():

            # Have to convert series to dictionaries for comparison bc pandas has thrown a tantrum.
            run_row = run_row.to_dict()

            # Check if Working_Tube_idx needs progressing
            if mastermix_allocation_counter == Rxns_Per_MasterMix:
                # reset
                mastermix_allocation_counter = 0
                Working_Tube_idx += 1

            if run_row == MasterMixElements:

                # look up the current tube using the mastermix_allocation_counter
                Working_Tube = Tubes[Working_Tube_idx]


                # look up the correct index in the array using run_idx and insert the tube at the position.
                MasterMix_Tube_List[run_idx] = Working_Tube

                #progress the counter
                mastermix_allocation_counter += 1



    plate_df.loc[:, col_name] = MasterMix_Tube_List

    return plate_df



def AddMasterMixTubesToExperimentDesign(plate_df, MasterMixesModulated = MasterMixesModulated):

    # do Aqueous and then Components

    AqueousComponents = ["Aqueous", "Components"]
    for MasterMixType in AqueousComponents:

        
        if (MasterMixesModulated == "Both" and MasterMixType == "Aqueous"):

            # if its a normal modulated allocation, assign the variables
            # and continue to the big comparison loop past this if statement.

            plate_df = AllocatingMasterMixTubesBasedOnComparision(
                master_mix_df = aqueous_master_mixes,
                Trimmed_plate_df = plate_df[aqueous_variables],
                variables = aqueous_variables,
                col_name = "AqueousMasterMixTube",
                plate_df = plate_df
            )

        elif (MasterMixesModulated == "Aqueous" and MasterMixType == "Aqueous"):

            # if its a normal modulated allocation, assign the variables
            # and continue to the big comparison loop past this if statement.

            plate_df = AllocatingMasterMixTubesBasedOnComparision(
                master_mix_df = aqueous_master_mixes,
                Trimmed_plate_df = plate_df[aqueous_variables],
                variables = aqueous_variables,
                col_name = "AqueousMasterMixTube",
                plate_df = plate_df
            )

        elif (MasterMixesModulated == "Both" and MasterMixType == "Components"):

            # if its a normal modulated allocation, assign the variables
            # and continue to the big comparison loop past this if statement.

            plate_df = AllocatingMasterMixTubesBasedOnComparision(
                master_mix_df = Components_master_mixes,
                Trimmed_plate_df = plate_df[Components_variables],
                variables = Components_variables,
                col_name = "ComponentsMasterMixTube",
                plate_df = plate_df
            )

        elif (MasterMixesModulated == "Components" and MasterMixType == "Components"):

            # if its a normal modulated allocation, assign the variables
            # and continue to the big comparison loop past this if statement.

            plate_df = AllocatingMasterMixTubesBasedOnComparision(
                master_mix_df = Components_master_mixes,
                Trimmed_plate_df = plate_df[Components_variables],
                col_name = "ComponentsMasterMixTube",
                plate_df = plate_df
            )

        else:

            # This means the MasterMixType is not being modulated. 
            # I.e just a normal homogenous master mix that needs doleing out.
            # set the variables as appropriate and then continue.

            if (MasterMixesModulated == "Aqueous" and MasterMixType == "Components"):

                master_mix_df = Components_master_mixes
                col_name = "ComponentsMasterMixTube"

            elif (MasterMixesModulated == "Components" and MasterMixType == "Aqueous"):

                master_mix_df = aqueous_master_mixes
                col_name = "AqueousMasterMixTube"
            
            else:
                raise Exception("Unknown MasterMixesModulated: "+ MasterMixesModulated+ " and MasterMixType: "+MasterMixType+" combination.")

            ## This section is a simple counter and progression loop
            # Assigns the MasterMix tubes blindly to the experiment.

            # concatenate all of the tubes together into a 1D list:  Master_Mix_Tube_list
            Master_Mix_Tube_list = []
            for Tubes in master_mix_df["Tubes"]:
                Master_Mix_Tube_list = Master_Mix_Tube_list + Tubes

            # initialise counter
            mastermix_allocation_counter = 0
            Working_Tube_idx = 0

            TubeListToAppend = []

            for i, row in plate_df.iterrows():

                # Check if Working_Tube_idx needs progressing
                if mastermix_allocation_counter == Rxns_Per_MasterMix:
                    # reset
                    mastermix_allocation_counter = 0
                    Working_Tube_idx += 1

                TubeListToAppend.append(Master_Mix_Tube_list[Working_Tube_idx])

                mastermix_allocation_counter += 1

            plate_df.loc[:, col_name] = TubeListToAppend

            # end of else.      

    return plate_df



# use this list to concat together when it's done.
plate_df_list = []

plates_list = list(experiment_design_df["Plate"].unique())

# iterate over and feed the number to reading in the pickled dfs
for plate_number in plates_list:


    aqueous_master_mixes = pd.read_pickle("/app/tmp/MasterMixes/"+str(plate_number)+"_plate_Aqueous_MasterMix_Working_Concs.pkl")
    Components_master_mixes = pd.read_pickle("/app/tmp/MasterMixes/"+str(plate_number)+"_plate_Components_MasterMix_Working_Concs.pkl")

    # slice the experimental design
    plate_df = experiment_design_df[experiment_design_df["Plate"] == plate_number].reset_index(drop=True)

    # execute the function defined above
    plate_df = AddMasterMixTubesToExperimentDesign(plate_df, MasterMixesModulated = MasterMixesModulated)


    ####### Shuffle runs

    if design_parameters["Randomise"] == 1:
        plate_df = plate_df.sample(
                                    frac=1,
                                    random_state=123
                                    ).reset_index(names="Original_Order")


    plate_df_list.append(plate_df)

## Concat plate_dfs together

experiment_design_df = pd.concat(plate_df_list, axis=0).reset_index(drop=True)


print()
print("Assignment Complete.")
print()


### save the full design
experiment_design_df.to_csv(project_path + "/Experiment_Designs/design_real_values.csv", index=False)


#############################################################################################################################################


