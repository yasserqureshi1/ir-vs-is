# Discrimination of inherent characteristics of susceptible and resistant strains of Anopheles gambiae by explainable Artificial Intelligence Analysis of Flight Trajectories

This repository contains all materials for the paper **Discrimination of inherent characteristics of susceptible and resistant strains of Anopheles gambiae by explainable Artificial Intelligence Analysis of Flight Trajectories**

## About the Project

Understanding mosquito behaviours is vital for the development of insecticide-treated bednets (ITNs), which have been successfully deployed in sub-Saharan Africa to reduce disease transmission, including malaria. However, rising insecticide resistance (IR) among mosquito populations, due to genetic and behavioural changes, poses a significant challenge. We present a machine learning pipeline that successfully distinguishes between insecticide-resistant (IR) and insecticide-susceptible (IS) mosquito behaviours by analysing trajectory data. Data driven methods are introduced to accommodate trajectory issues such as interpolated positions within a track that occur due to mosquito position occlusion by the bednet or other objects. The trajectories, obtained from room-scale tracking of two IR and two IS strains around a human-baited, untreated bednet, were analysed using features like such as velocity, acceleration, and geometric descriptors. Using these features, an XGBoost model achieved a balanced accuracy of 0.743 and a ROC AUC of 0.813 in classifying IR from IS mosquitoes. SHAP analysis revealed that IR mosquitoes tend to fly slower with more directed flight paths—traits that are likely a fitness advantage by enhancing their ability to respond more quickly to bloodmeal cues. This approach provides valuable insights that can reveal the action of interventions and insecticides on mosquito flight behaviour.

## Set-Up

Install the dependencies using the command:
```
pip3 install -r requirements.txt
```