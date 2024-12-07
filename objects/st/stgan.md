# STGAN

Year: 2023  
Publication: IEEE Transactions on Big Data  
Paper Link: [https://ieeexplore.ieee.org/document/10064498](https://ieeexplore.ieee.org/document/10064498)

#### Task:

- Traffic Data Imputation

#### Architecture:

- GAN-based  
- Encoder-Decoder Architecture

#### Spatial Module:

- Dilated Convolutional Layers

#### Temporal Module:

- Dilated Convolution for Temporal Correlations

#### Missing Values:

- Handled with Generative and Center Loss

#### Input Graph:

- Not Required

#### Learned Graph Relations:

- Spatio-Temporal Relations (ST)

#### Graph Heuristics:

- Spatio-Temporal Correlation, Local and Global Distributions

---

# STGAN: Spatio-Temporal Generative Adversarial Network for Traffic Data Imputation

## 1. Methodology

The **STGAN** model introduces a GAN-based framework specifically designed for traffic data imputation. This model effectively handles missing traffic data using a combination of adversarial learning and spatial-temporal correlations. The generator and discriminator work together to accurately reconstruct missing traffic data while preserving both local and global traffic patterns.

- **Spatial Dependencies**: Captured using dilated convolutional layers to ensure that traffic data at different points in space is connected.
- **Temporal Dependencies**: Managed by using dilated convolutions in the temporal dimension to capture long-term dependencies in traffic data sequences.
- **Center Loss**: Ensures that each imputed entry conforms to its surrounding neighbors, preserving local spatial-temporal relationships.
- **Generative Loss**: Minimizes the reconstruction error between the imputed and real values for missing entries.

## 2. Layers Used

The architecture consists of the following layers:

- **Dilated Convolutional Layers**: Used to capture both spatial and temporal features. The dilated convolution ensures a broader receptive field, allowing the network to learn dependencies over larger spatial and temporal windows.
- **Skip-Connections**: Employed to retain uncorrupted data during imputation, ensuring the integrity of observed values.
- **Fully Connected Layer**: At the end, a fully connected layer is applied for the final traffic data reconstruction.

The network uses a total of:

- **6 Convolutional Layers**
- **4 Dilated Convolutional Layers** with increasing dilation rates
- **1 Fully Connected Layer** for final imputation

## 3. Model Architecture

The **STGAN** architecture can be described as:

1. **Encoder**: Takes the incomplete traffic data as input and applies multiple convolutional and dilated convolutional layers to extract spatial-temporal features.
2. **Decoder**: Reconstructs the imputed data from the learned feature representations.
3. **Discriminator**: A CNN-based discriminator checks if the imputed data conforms to the distribution of real traffic data.
4. **Loss Functions**:
    - **Adversarial Loss**: Ensures the imputed data conforms to global data distributions.
    - **Generative Loss**: Reduces the error between imputed and real values.
    - **Center Loss**: Ensures that imputed data fits the local spatio-temporal distribution.

## 4. Hyperparameters

- **Learning Rate**: 0.001
- **Batch Size**: 8
- **Dropout Rate**: 0.3
- **Optimizer**: Adam
- **Training Epochs**: 100
- **Kernel Size**: (3,3) for convolution layers
- **Dilation Rates**: 2, 4, 8, 16 for dilated convolution layers

## 5. Performance Metrics

The STGAN model was evaluated using metrics like RMSE (Root Mean Squared Error) and MAE (Mean Absolute Error) for various data missing rates across different datasets. The model consistently outperformed traditional methods such as GAIN and MICE in terms of imputation accuracy.

### Example Results:

| Dataset         | Missing Rate | Metric | GAIN | MICE | STGAN  |
|-----------------|--------------|--------|------|------|--------|
| Beijing Road    | 0.2          | RMSE   | 10.12| 14.37| **5.99** |
| Beijing Subway  | 0.5          | RMSE   | 15.79| 20.39| **8.56** |

## 6. Datasets Used

The **STGAN** model was evaluated on two real-world traffic datasets:

1. **Beijing Road Dataset**:
   - **Data**: Average speed data from GPS floating cars.
   - **Coverage**: 9760 roads, with data from July to October 2016.
   
2. **Beijing Subway Dataset**:
   - **Data**: Subway station swipe data.
   - **Coverage**: More than 300 stations from July to August 2015.

These datasets include rich spatio-temporal information, making them ideal for testing the imputation capabilities of the model.
