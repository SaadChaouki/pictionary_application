import requests
import json


class RequestsAPI():
    def __init__(self):
        settings = json.load(open('resources/settings.json'))
        self.url = settings['url_api']

    def request_prediction(self, body):
        requestBody = {"body": body}
        result = requests.post(self.url, data=requestBody)
        return result.text
