import boto3
from urllib.parse import unquote

def lambda_handler(event, context):
    endpoint_name = 'endpoint name'

    runtime = boto3.Session().client('sagemaker-runtime')
    response = runtime.invoke_endpoint(EndpointName=endpoint_name,
                                       ContentType='text/plain',
                                       Body=unquote(event['body'][5:]))
    result = response['Body'].read().decode('utf-8')

    return {
        'statusCode': 200,
        'headers': {'Content-Type': 'text/plain', 'Access-Control-Allow-Origin': '*'},
        'body': result
    }