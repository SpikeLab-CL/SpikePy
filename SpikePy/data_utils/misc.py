##############
# Slack Webhook
#############

import requests
import json
import os



def slack_code_ready(message, username, webhook_env='code_ready_webhook'):
	"""
	Sends a message to a username on slack
	"""
	webhook_num = os.environ[webhook_env]
	webhook = "https://hooks.slack.com/services/" + webhook_num
	username_num = os.environ['slack_' + username]
	message = "<@U{0}> {1}".format(username_num, message)
	payload = {"text": message}
	headers = {'Content-type': 'application/json'}

	return requests.post(url=webhook, data=json.dumps(payload), headers=headers)


def aws_kill_stack(stack_name, delete_hook_env='lambda_delete_webhook'):
	"""
	Kills a stack on AWS

	delete_hook_env: name of environmental variable that points to https endpoint. Starts with 'https'
	"""
	lambda_address = os.environ[delete_hook_env]

	payload = {'stack_name': stack_name}
	r = requests.post(lambda_address, params=payload)
	return r

