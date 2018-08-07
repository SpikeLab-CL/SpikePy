##############
# Slack Webhook
#############

import requests
import json
import os



def slack_code_ready(message, username, webhook_env='code_ready_webhook'):
	webhook_num = os.environ[webhook_env]
	webhook = "https://hooks.slack.com/services/" + webhook_num
	username_num = os.environ['slack_' + username]
	message = "<@U{0}> {1}".format(username_num, message)
	payload = {"text": message}
	headers = {'Content-type': 'application/json'}

	return requests.post(url=webhook, data=json.dumps(payload), headers=headers)
