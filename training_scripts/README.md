# Training Examples

This directory contains example scripts that demonstrate how to use the machine learning modules for physics-based surrogate reservoir modeling, with a focus on handling dry gas (DG) and gas condensate (GC) fluid type scenarios both above and below dew point.

The examples demonstrate:
1. Model training for Dry Gas (DG) fluid type
2. Model training for Gas Condensate (GC) fluid type above dew point (using hard enforcement only).
3. Model training for Gas Condensate (GC) fluid type below dew point (using deep learning layers and hard enforcement layers).

These examples are to help researchers and engineers to set-up and train the surrogate models appropriately in different cases.

## Dry Gas (DG) or Gas Condensate (GC) Training Workflow

The `training_case_dry_gas_i.py` script demonstrates a complete training workflow for the dry gas case using a multi-model architecture approach with physics-based loss functions. Below is a detailed explanation of the workflow:

### Setup and Configuration

1. **Environment Setup**:
   - Parent directory setup; Tensorflow logging and GPU settings

2. **Import Dependencies**:
   - Data processing modules: `SRMDataProcessor`
   - Training utilities: `BatchGenerator`, `build_optimizer_from_config`, etc.
   - Configuration modules: `DEFAULT_GENERAL_CONFIG`, `DEFAULT_RESERVOIR_CONFIG`
   - Physics-based modules: `PhysicsLoss`, `WellRatesPressure`
   - Custom Architecture Builder: `CompleteTrainableModule`, `PVTModuleWithHardLayer`

### Model Architecture

The workflow builds three or four custom model architectures based on the configurations defined in the default_configuration.py file:

1. **Encoder-Decoder with Hard Layer** (`build_encoder_decoder_with_hard`):
   - Primary model for pressure prediction.
   - Features time-distributed dimension processing (for 3D data instead of directly using 3D convolution for unit z-layer discretization).
   - Integrates a trainable hard layer enforcement for the initial conditions of pressure (Pi) or saturation (Sgi).
   - Module includes custom skip connections for transfer learning from encoder to decoder.

2. **Residual Network without Hard Layer** (`build_residual_network_without_hard`):
   - Auxiliary novel model for variable time-step predictions used to solve the discretized partial differential equations (PDEs) of the reservoir model.
   - Uses a scaled tanh activation with x*tanh(x) to constrain outputs.
   - Operates on the same input shape as the encoder-decoder
   - Also configure with a time-distributed dimension processing.

3. **PVT Model without Hard Layer** (`build_pvt_model_without_hard`):
   - Computes fluid properties based on predicted pressure.
   - Uses the encoder-decoder model output as its input.
   - Implements parametric spline interpolation method with configurable order
   - Outputs both fluid property values and its derivatives, which are computed using algorithmic differentiation (AD) -- Tensorflow Gradient Taping.

4. **Additional Models**:
   - `WellRatesPressure` model for handling well constraints and returns the well rates and pressure.
   - `Saturation` model (only used in gas condensate cases).

### Training Process

1. **Data Preparation**:
   - The `SRMDataProcessor` is used to load or generate training, validation, and test data groups.
   - Data is organized as tuples of features and targets

2. **Model Map Construction**:
   - A dictionary mapping logical model names to Keras model instances is created.
   - The models are associated with specific roles (pressure, time_step, pvt_model, etc.)
   - The fluid type (DG for dry gas case) is also configured.

3. **Physics-Based Loss Function**:
   - Instantiates `PhysicsLoss` class implementation with all required models.
   - Links models together via the optimizer-to-model mapping.
   - Enables physics-based training without explicit targets.

4. **Custom Logging**:
   - Implements a custom callback (`my_log_callback`) to track and report:
     - Loss values per epoch.
     - Number of trainable variables per model.
    This custom logging is used in selecting the best model variables based on a watched list of normalized loss metrics during training.

5. **Unified Training**:
   - Uses the `train_combined_models_unified` function from the training.py module
   - Implements multiple optimizers for the different model components.
   - Tracks and logs variables at specified epoch percentages.
   - Selects best model variables based on normalized validation losses.

6. **Evaluation and Visualization**:
   - Creates a `ModelPlotter` instance for visualizing model predictions
   - Plot object is configured with units, font settings, and plot parameters
   - Plot object can be used to geenrate image and line plots of true and predicted values.
   - True values are obtained from the test data groups.

This workflow describes the entirety of a training script for either a gas or gas condensate reservoir model.