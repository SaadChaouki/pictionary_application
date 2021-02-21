import boto3
from urllib.parse import unquote

def lambda_handler(event, context):
    runtime = boto3.Session().client('sagemaker-runtime')
    response = runtime.invoke_endpoint(EndpointName='sagemaker-pytorch-2021-02-20-19-56-08-829',
                                       ContentType='text/plain',
                                       Body=unquote(event['body'][5:]))
    result = response['Body'].read().decode('utf-8')

    return {
        'statusCode': 200,
        'headers': {'Content-Type': 'text/plain', 'Access-Control-Allow-Origin': '*'},
        'body': result
    }