# Image Classification using AWS SageMaker

Use AWS Sagemaker to train a pretrained model that can perform image classification by using the Sagemaker profiling, debugger, hyperparameter tuning and other good ML engineering practices. This can be done on either the provided dog breed classication data set or one of your choice.

## Project Set Up and Installation
Enter AWS through the gateway in the course and open SageMaker Studio. 
Download the starter files.
Download/Make the dataset available. 

## Dataset
The provided dataset is the dogbreed classification dataset which can be found in the classroom.
The project is designed to be dataset independent so if there is a dataset that is more interesting or relevant to your work, you are welcome to use it to complete the project.

### Access
Upload the data to an S3 bucket through the AWS Gateway so that SageMaker has access to the data. 
![1](https://user-images.githubusercontent.com/85734497/192898433-d6a8574f-f427-4013-9148-b5cd42bbeb19.png)

## Hyperparameter Tuning
What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search
I got those hyperparameters = {"batch_size": "128", "epochs":"3", "lr": "0.0003685428692205665"}

![2022-09-29_00h19_58](https://user-images.githubusercontent.com/85734497/192898892-fe516e45-e4ef-48bd-9e80-eff9d02783c2.png)
![2022-09-29_00h20_28](https://user-images.githubusercontent.com/85734497/192898927-1975a153-87ba-4e8c-a7aa-240c9f5aff31.png)


Remember that your README should:
- Include a screenshot of completed training jobs
- Logs metrics during the training process
- Tune at least two hyperparameters
- Retrieve the best best hyperparameters from all your training jobs

## Debugging and Profiling
**TODO**: Give an overview of how you performed model debugging and profiling in Sagemaker

### Results
**TODO**: What are the results/insights did you get by profiling/debugging your model?

**TODO** Remember to provide the profiler html/pdf file in your submission.


## Model Deployment
**TODO**: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.

**TODO** Remember to provide a screenshot of the deployed active endpoint in Sagemaker.
![2022-09-29_00h22_26](https://user-images.githubusercontent.com/85734497/192899183-572c1a1c-e599-45fc-98df-81d4aff46b86.png)

## Standout Suggestions
**TODO (Optional):** This is where you can provide information about any standout suggestions that you have attempted.
