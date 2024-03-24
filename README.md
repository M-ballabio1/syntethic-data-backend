# Synthetic Data Generation API Project

## Overview

The Synthetic Data Generation API Project provides an API for generating synthetic tabular data. It offers functionalities for training generative models (CT-GAN and T-VAE), performing inference, and reporting bugs. This README provides detailed documentation of the backend methods used for API key security and CORS configuration.

## Startup

To start the Synthetic Data Generation API Project, follow these steps:

1. **Clone the Repository**: Clone the project repository from GitHub using the following command: https://github.com/M-ballabio1/syntethic-data-backend.git
2. **Install Dependencies**: Navigate to the project directory and install the required dependencies using pip: cd synthetic-data-api | pip install -r requirements.txt
3. **Set Environment Variables**: Set the required environment variables, including the API key (`API_KEY`), if necessary.
4. **Run the Application**: Start the FastAPI application by running the following command: uvicorn main:app --reload
5. **Access the API**: Once the application is running, you can access the API endpoints at the specified URLs.

## API Key Security

The API implements API key security to restrict access to authorized users. Here's how it works:

1. **API Key Management**: The API key is stored securely as an environment variable (`API_KEY`). It is loaded into the application at startup.
2. **Authentication**: Each API request must include an API key for authentication. The API compares the provided API key with the stored key to verify the user's identity.
3. **Unauthorized Access Handling**: If the provided API key does not match the stored key, the API returns a `401 Unauthorized` error, indicating unauthorized access.

## CORS Configuration

Cross-Origin Resource Sharing (CORS) is configured to allow secure communication between the frontend and backend of the application. Here's how CORS is configured:

1. **Middleware Integration**: The FastAPI framework provides a middleware for CORS handling (`CORSMiddleware`). It is added to the application to enable CORS support.
2. **Allowed Origins**: The API specifies a list of allowed origins from which requests can originate. This prevents unauthorized cross-origin requests.
3. **Allowed Methods and Headers**: The API allows specific HTTP methods and headers in CORS requests. This ensures that only permitted operations are allowed.

## Endpoints

### `/training_model_ctgan` (Training ctgan)

- **Method**: POST
- **Description**: Initiates the training process for the CTGAN model using the provided training data.
- **Security**: Requires API key authentication.
- **Parameters**:
  - `background_tasks`: BackgroundTasks object for executing tasks in the background.
  - `epochs`: Number of epochs for CTGAN model training.
  - `file_training_data`: File containing the training data.
  - `api_key`: API key for authentication.
- **Response**: Returns a confirmation of the successful initiation of the training process.

### `/inference_ctgan_tvae_metrics` (Inference CPU/GPU)

- **Method**: POST
- **Description**: Performs inference using the CTGAN and TVAE models to generate synthetic data.
- **Security**: Requires API key authentication.
- **Parameters**:
  - `model_id`: ID of the model used for inference (CTGAN or TVAE).
  - `sample_num`: Number of samples to generate.
  - `api_key`: API key for authentication.
- **Response**: Returns the synthetic data in CSV format.

### `/train_model_tvae_adults_dataset` (Training tvae)

- **Method**: POST
- **Description**: Initiates the training process for the TVAE model using the adult dataset.
- **Security**: Requires API key authentication.
- **Parameters**:
  - `background_tasks`: BackgroundTasks object for executing tasks in the background.
  - `epochs`: Number of epochs for TVAE model training.
  - `api_key`: API key for authentication.
- **Response**: Returns a confirmation of the successful initiation of the training process.

### `/inference_tvae_gpu` (Inference GPU)

- **Method**: POST
- **Description**: Performs inference using the TVAE model (GPU required) to generate synthetic data.
- **Security**: Requires API key authentication.
- **Parameters**:
  - `unique_id`: Unique ID of the TVAE model.
  - `num_rows`: Number of synthetic data rows to generate.
  - `api_key`: API key for authentication.
- **Response**: Returns the synthetic data in CSV format.

## License

This project is licensed under the Apache License 2.0. For more details, see the [LICENSE](LICENSE) file.
