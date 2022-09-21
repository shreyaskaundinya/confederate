from sys import api_version
import ipfsApi
api = ipfsApi.Client('localhost', 5001)
res = api.add('chest_xray/train/NORMAL', match="*.jpeg")
# api.get('12D3KooWLMEU6xpawpEiVGVUnK7T37zJCYV5sSbxoDZzeSDQ4p9b')
