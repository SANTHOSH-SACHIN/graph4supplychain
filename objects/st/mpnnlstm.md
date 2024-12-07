# Message Passing Neural Network LSTM (MPNNLSTM)


*Year*: 2021  
*Publication*: AAAI Conference on Artificial Intelligence  
*Paper Link*: Not provided  

#### Task:
- *COVID-19 Spread Prediction*

#### Architecture:
- *Message Passing Neural Network (MPNN)* with LSTM and Transfer Learning (TL)

#### Spatial Module:
- *MPNN* with neighborhood aggregation for regional transmission patterns

#### Temporal Module:
- *LSTM* for temporal sequence modeling, enabling long-term trend capture

#### Missing Values:
- *Assumed uniform sampling* via mobile app data (e.g., Facebook)

#### Input Graph:
- *Required* as daily mobility-based weighted graphs

#### Learned Graph Relations:
- *Dynamic* (captured via transfer learning across asynchronous outbreaks)

#### Graph Heuristics:
- *Transfer Learning* with Model-Agnostic Meta-Learning (MAML), *Temporal Dependencies* through MPNN-LSTM

## 1. Introduction
MPNN-LSTM forecasts COVID-19 spread by leveraging spatial-temporal mobility data and limited case reporting. Using transfer learning, the model adapts from countries with historical outbreak data to those in early stages, supporting early and accurate pandemic forecasting across regions.

## 2. Methodology
The model includes:
- *Spatial Dependency*: Modeled by MPNN to capture local interactions in each snapshot graph.
- *Temporal Dependency*: LSTM layers handle sequence learning, while transfer learning transfers insights from prior outbreak phases.
- *Mobility & Dynamics*: Explores mobility-based infection transfer across regions, considering the timing of secondary outbreaks.

## 3. Model Components
- *Spatial Aggregation*: MPNN aggregates mobility-based features per region.
- *Temporal Aggregation*: LSTM layers predict long-term trends across daily mobility patterns.
- *Transfer Learning*: MAML facilitates inter-country adaptation for more accurate early-stage predictions.

## 4. Hyperparameters
- *Learning Rate*: 0.001 (Adam optimizer)
- *Hidden Dimensions*: 64 for MPNN and LSTM
- *Batch Size*: 8
- *Evaluation Metrics*: Mean Absolute Error (MAE)

## 5. Performance Metrics

| Dataset           | Up to 3 Days | Up to 7 Days | Up to 14 Days |
|-------------------|--------------|--------------|---------------|
| England           | MAE: 6.05    | MAE: 6.33    | MAE: 6.84     |
| France            | MAE: 5.83    | MAE: 5.90    | MAE: 6.13     |
| Italy             | MAE: 14.08   | MAE: 14.61   | MAE: 16.69    |
| Spain             | MAE: 29.61   | MAE: 31.55   | MAE: 34.65    |

Predictions for next-day cases demonstrate MPNN-LSTM’s advantage in inter-regional transfer learning, improving accuracy in short-to-mid-term pandemic
**Dataset**: Chickenpox

MPNNLSTM integrates message-passing mechanisms with LSTM to handle spatial-temporal data, capturing complex relationships within graph-structured time series.

## Results on Chickenpox Dataset:
- **MSE**: 1.1404
- **MAE**: 0.7670
- **MAPE**: 2138.4280%
- **RMSE**: 1.0679
- **R-squared**: -0.1326

## Version History:
- v1.0
