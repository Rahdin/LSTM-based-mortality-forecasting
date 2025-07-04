# Time-Varying Lee–Carter Mortality Forecasting with LSTM
 
This repository implements and compares three approaches to forecasting the **gap** in life expectancy between Bulgaria (target) and the United States (benchmark) using:
 
1. A **hybrid LSTM + time-varying Lee–Carter** rotation (“WeightNet” model)  
2. A **basic 1-layer LSTM** regressor (vanilla baseline)  
3. A **generalized linear model** (GLM) on flattened gap windows
 
All code is in Python 3.10 / PyTorch and Pandas.
 
---
 
## Paper
 
A unified LSTM model is proposed in  
**A Unified LSTM Model for Coherent Mortality Forecasting in Developing Regions**  
Ran Xu, Jose Garrido, Yuxiang Shang  
Risks 2024, 12(2), 27  
<https://www.mdpi.com/2227-9091/12/2/27>
 
---
 
## What We Did
 
1. **Data Preprocessing**  
   - Read age- and year-specific Exposures (Eₓ,ₜ) and Deaths (Dₓ,ₜ) for USA (1933–2023) and Bulgaria (1947–2021).  
   - Compute central death rates:  
     ```
     mₓ,ₜ = Dₓ,ₜ / Eₓ,ₜ
     ```  
   - Build life tables to extract life expectancy e₀,ₜ.  
   - Form the gap series:  
     ```
     gₜ = e₀,ₜ^(USA) – e₀,ₜ^(BUL)
     ```  
   - Drop age 110+, clamp tiny rates to ε=1e-8, interpolate any missing gaps, then standardize to zero mean/unit variance.
 
2. **Models**  
   - **WeightNet (hybrid LSTM + rotation)**  
     - 2-layer LSTM → dense → sigmoid to predict weight ωₜ₊₁ from the last L=10 gaps.  
     - Rotate Lee–Carter parameters:  
       ```
       bₓ,ₜ₊₁ = (1 – ωₜ₊₁) · b̂ₓ + ωₜ₊₁ · Bᵇₓ  
       dₜ₊₁ = (1 – ωₜ₊₁) · d̂ + ωₜ₊₁ · d⁰
       ```  
     - Forecast log-mortality:  
       ```
       ln mₓ,ₜ₊₁ = aₓ + bₓ,ₜ₊₁ · kₜ₊₁  
       where kₜ₊₁ = kₜ + dₜ₊₁
       ```  
 
   - **Basic LSTM baseline**: same network but only 1 LSTM layer, trained directly on ωₜ₊₁.  
   - **GLM baseline**: flatten each length-10 gap window into a feature vector and fit ordinary linear regression to predict ω.
 
3. **Training & Evaluation**  
   - Sliding windows of length L = 10 years, MSE loss on ω.  
   - 50 epochs, Adam (lr = 1e-3), batch size = 16.  
   - Record MSE and RMSE each epoch for all three models.
 
---
 
## Results
 
### Final-Epoch Errors
 
| Model                | Final MSE  | Final RMSE |
|----------------------|----------:|----------:|
| **WeightNet (ours)** | 0.00085   | 0.0291    |
| Basic LSTM           | 0.00112   | 0.0335    |
| GLM                  | 0.00240   | 0.0490    |
 
### Relative Improvements
 
| Comparison           | MSE ↓ vs baseline | RMSE ↓ vs baseline |
|----------------------|------------------:|-------------------:|
| vs. Basic LSTM       | 24.1 %            | 13.2 %             |
| vs. GLM              | 64.6 %            | 40.6 %             |
 
> *All values are from the final training epoch (50) on the training set.  
> See `comparison_df` in [training.ipynb] for the full history.*
 
---
 
## What Worked
 
- **Smooth rotation** between Bulgarian and benchmark parameters captured the catch-up dynamics.  
- **Sigmoid-bounded outputs** (ω in (0,1)) plus **input normalization** eliminated NaNs and stabilized training.  
- The hybrid model consistently **outperformed** both vanilla LSTM and GLM baselines.
 
## What Didn’t
 
- A pure LSTM on raw gaps tended to over- or under-smooth the learned weight, producing jagged forecasts.  
- A straight Lee–Carter on Bulgaria alone failed to adapt as the gap narrowed.
 
---
 
