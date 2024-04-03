import requests
import json

# Constants
#HOST = "http://127.0.0.1:8000"
HOST = "http://ec2-34-252-212-32.eu-west-1.compute.amazonaws.com:8000"

def send_request(url, body=None):
    full_url = f"{HOST}/{url}"
    headers = {"Content-Type": "application/json"}

    if body is None:
        response = requests.post(full_url)
    else:
        response = requests.post(full_url, headers=headers, data=json.dumps(body))

    if response.status_code != 200:
        return "HTTP error status!"

    return response.json()


def check_connection():
    res = send_request("test")
    if res["message"] == "OK":
        return "OK"
    else:
        return "Something is wrong!"


def load_data(user, password, filename, x_variable, y_variable):
    body = {
        "user": user,
        "password": password,
        "file_name": filename,
        "x_variable": x_variable,
        "y_variable": y_variable
    }
    res = send_request("load", body)
    return res["message"]


def train(user, password, batch_size, max_itr, alfa, opt, ansatz, params=None, beta=None, beta2=None):
    if ansatz not in ["SU2", "SU2U4"]:
        return "ERROR: Invalid ansatz."
    if opt not in ["Eva", "Adam", "RMSprop", "Momentum", "StandardGD"]:
        return "ERROR: Invalid optimization method."
    if opt in ["Adam", "RMSprop", "Momentum"] and beta is None:
        return f"ERROR: For optimizer {opt} parameter β has to be set."
    if opt == "Adam" and beta2 is None:
        return "ERROR: For optimizer Adam parameter β2 has to be set."

    body = {
        "user": user,
        "password": password,
        "batch_size": batch_size,
        "max_itr": max_itr,
        "alfa": alfa,
        "opt": opt,
        "ansatz": ansatz
    }
    if params is not None:
        body["params"] = params
    if beta is not None:
        body["beta"] = beta
    if beta2 is not None:
        body["beta2"] = beta2

    res = send_request("train", body)
    return res["message"]


def check_status(user, password):
    body = {"user": user, "password": password}
    res = send_request("status", body)
    return res["message"]


def get_parameters(user, password):
    body = {"user": user, "password": password}
    res = send_request("get_parameters", body)
    return [float(e) for e in res["message"]]


def score(user, password, ansatz, x, params):
    if ansatz not in ["SU2", "SU2U4"]:
        return "ERROR: Invalid ansatz."

    body = {
        "user": user,
        "password": password,
        "ansatz": ansatz,
        "input_vec": x,
        "params": params
    }
    res = send_request("score", body)
    return res["message"]


def validate(user, password, filename, x_variable, y_variable, ansatz, params):
    if ansatz not in ["SU2", "SU2U4"]:
        return "ERROR: Invalid ansatz."

    body = {
        "user": user,
        "password": password,
        "file_name": filename,
        "x_variable": x_variable,
        "y_variable": y_variable,
        "ansatz": ansatz,
        "params": params
    }
    res = send_request("validate", body)
    return res["message"]


