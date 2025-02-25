# Discrimination of inherent characteristics of susceptible and resistant strains of Anopheles gambiae by explainable artificial intelligence analysis of flight trajectories

This repository contains all materials for the paper **Discrimination of inherent characteristics of susceptible and resistant strains of Anopheles gambiae by explainable artificial intelligence analysis of flight trajectories**:

Qureshi, Y.M., Voloshin, V., Gleave, K. et al. Discrimination of inherent characteristics of susceptible and resistant strains of Anopheles gambiae by explainable artificial intelligence analysis of flight trajectories. Sci Rep 15, 6759 (2025). https://doi.org/10.1038/s41598-025-91191-w

## About the Project

Understanding mosquito behaviours is vital for the development of insecticide-treated nets (ITNs), which have been successfully deployed in sub-Saharan Africa to reduce disease transmission, particularly malaria. However, rising insecticide resistance (IR) among mosquito populations, owing to genetic and behavioural changes, poses a significant challenge. We present a machine learning pipeline that successfully distinguishes between innate IR and insecticide-susceptible (IS) mosquito flight behaviours independent of insecticidal exposure by analysing trajectory data. Data-driven methods are introduced to accommodate common tracking system shortcomings that occur due to mosquito positions being occluded by the bednet or other objects. Trajectories, obtained from room-scale tracking of two IR and two IS strains around a human-baited, untreated bednet, were analysed using features such as velocity, acceleration, and geometric descriptors. Using these features, an XGBoost model achieved a balanced accuracy of 0.743 and a ROC AUC of 0.813 in classifying IR from IS mosquitoes. SHAP analysis helped decipher that IR mosquitoes tend to fly slower with more directed flight paths and lower variability than IS—traits that are likely a fitness advantage by enhancing their ability to respond more quickly to bloodmeal cues. This approach provides valuable insights based on flight behaviour that can reveal the action of interventions and insecticides on mosquito physiology.

## Set-Up

Install the dependencies using the command:
```
pip3 install -r requirements.txt
```

## Authors
Yasser M. Qureshi, Vitaly Voloshin, Katherine Gleave, Hilary Ranson, Philip J. McCall, James A. Covington, Catherine E. Towers & David P. Towers 

## License
Distributed under the BSD-3 Clause license
