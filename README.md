# Discrimination of inherent characteristics of susceptible and resistant strains of Anopheles gambiae by explainable Artificial Intelligence Analysis of Flight Trajectories

This repository contains all materials for the paper **Discrimination of inherent characteristics of susceptible and resistant strains of Anopheles gambiae by explainable Artificial Intelligence Analysis of Flight Trajectories**

## About the Project

Understanding mosquito behaviours is vital for development of insecticide-treated bednets (ITNs), which have been successfully deployed in sub-Saharan Africa to reduce disease transmission, particularly malaria. However, rising insecticide resistance (IR) among mosquito populations, owing to genetic and behavioural changes, poses a significant challenge. We present a machine learning pipeline that successfully distinguishes between IR and insecticide-susceptible (IS) mosquito behaviours by analysing trajectory data. Data driven methods are introduced to accommodate common tracking system shortcomings that occur due to mosquito positions being occluded by the bednet or other objects. Trajectories, obtained from room-scale tracking of two IR and two IS strains around a human-baited, untreated bednet, were analysed using features such as velocity, acceleration, and geometric descriptors. Using these features, an XGBoost model achieved a balanced accuracy of 0.743 and a ROC AUC of 0.813 in classifying IR from IS mosquitoes. SHAP analysis helped decipher that IR mosquitoes tend to fly slower with more directed flight paths and lower variability than IS—traits that are likely a fitness advantage by enhancing their ability to respond more quickly to bloodmeal cues. This approach provides valuable insights based on flight behaviour that can reveal the action of interventions and insecticides on mosquito physiology.

## Set-Up

Install the dependencies using the command:
```
pip3 install -r requirements.txt
```
