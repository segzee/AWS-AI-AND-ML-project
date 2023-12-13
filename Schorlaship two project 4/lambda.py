import json
import boto3
import base64
    
s3 = boto3.client('s3')

def lambda_handler(event, context):
    """A function to serialize target data from S3"""
    
    # Get the s3 address from the Step Function event input
    key = event.get('s3_key') 
    bucket = event.get('s3_bucket') 
    
    # Download the data from s3 to /tmp/image.png
    
    s3.download_file(bucket, key, "/tmp/image.png")
    
    # We read the data from a file
    with open("/tmp/image.png", "rb") as f:
        image_data = base64.b64encode(f.read())

    # Pass the data back to the Step Function
    print("Event:", event.keys())
    return {
        'statusCode': 200,
        'body':{
            "image_data": image_data,
            "s3_bucket": bucket,
            "s3_key": key,
            "inferences": []
        }
    }






import subprocess
import sys

# https://stackoverflow.com/questions/60311148/pip-install-python-package-within-aws-lambda
# pip install custom package to /tmp/ and add to path
subprocess.call('pip install sagemaker -t /tmp/ --no-cache-dir'.split(), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
sys.path.insert(1, '/tmp/')

import json
import sagemaker
import base64
from sagemaker.serializers import IdentitySerializer


# Fill this in with the name of your deployed model
ENDPOINT = 'image-classification-2023-10-17-23-33-42-834'

def lambda_handler(event, context):

    # Decode the image data
    get_body = event.get('body')
    get_image = get_body.get('image_data')
    
    if get_image is None:
        return {
                'statusCode': 400,  # Return a bad request status code
                'body': "Missing image_data field in the event."
            }
    image = base64.b64decode(get_image)

    # Instantiate a Predictor
    predictor = sagemaker.predictor.Predictor(ENDPOINT)

    # For this model the IdentitySerializer needs to be "image/png"
    predictor.serializer = IdentitySerializer("image/png")
    
    # Make a prediction:
    inferences = predictor.predict(image)
    
    # We return the data back to the Step Function    
    inference = inferences.decode('utf-8')
    event["inferences"] = eval(inferences)
    return {
        'statusCode': 200,
        'body': event
    }





import json


THRESHOLD = .93


def lambda_handler(event, context):
    
    # Grab the inferences from the event
    first_body = event.get("body")
    
    inferences = first_body.get("inferences",[0,0])  
    
    # Check if any values in our inferences are above THRESHOLD
    meets_threshold = inferences[0]>= THRESHOLD or inferences[1]>= THRESHOLD
    
    # If our threshold is met, pass our data back out of the
    # Step Function, else, end the Step Function with an error
    if meets_threshold:
        pass
    else:
        raise("THRESHOLD_CONFIDENCE_NOT_MET")

    return {
        'statusCode': 200,
        'body': json.dumps(event)
    }